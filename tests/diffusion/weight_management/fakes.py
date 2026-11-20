# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Production-shaped test doubles for the unified weight-management driver.

PR 01-5 of RFC 01 (child of umbrella RFC #9536).
    - Spec: RFC 01 "Job 1 delivered independently" and "Test plan":
      ``tests/diffusion/weight_management/fakes.py exercises no-offload driver,
      confirmations, ledger, cleanup, worker shutdown, sleep/wake, and
      reinitialization. Its plan keeps all weights resident, host storage is
      empty, and runtime registers one accelerator slot.``

Why this file exists
--------------------
The real Loader / HostWeightManager / Offloader are RFCs 02-04 and do not exist
yet. But RFC 01's whole job -- the state machine (section 4), the ownership
chain (section 5.1), and the resource ledger (section 5.2) -- can be driven to
completion against *doubles* that honour the Protocol contracts in
``interfaces.py``. These doubles do exactly that.

Unlike a minimal unit-test stub (which returns ``object()``), these doubles
model the **no-offload** path faithfully, so the lifecycle you can watch here is
the lifecycle the real stack runs:

    * The plan keeps every weight resident on the execution device
      (``PlacementPlan.collective_signature() == {}``): no AllGather, no host
      copies. This is RFC 01's "Job 1" configuration -- the only one unified
      mode accepts before RFC 05 lands.
    * Host storage is therefore **empty** (nothing is host-resident).
    * ``LoadTransaction.target(key)`` returns ``None`` for every weight, which
      the contract defines as "allocate on the execution device inside
      ``ctx.memory_pools.weights()``".
    * The transaction owns a **real** ``ResourceLedger`` (startup.py). Device
      weights register as ``DEVICE`` resources; ``commit()`` ``transfer()``s the
      whole ledger to the runtime; the runtime registers **one accelerator
      slot** (``EXECUTION``) on top. ``runtime.close()`` / ``abort()`` then
      release everything, in ledger order, exactly once.

That last point is the teaching core: run ``python -m ... fakes`` (see
``__main__``) and the printed log is the ownership baton passing from
transaction -> runtime and the ledger closing in ``ResourceKind`` order.

These are TEST doubles. The filename does not start with ``test_`` so pytest
does not collect it (RFC 01 section 9.4); tests import ``FakeStackFactory`` from
here.
"""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from types import SimpleNamespace

import torch
from torch import nn

from vllm_omni.diffusion.weight_management.errors import LifecycleError
from vllm_omni.diffusion.weight_management.startup import (
    ResourceKind,
    ResourceLedger,
    StackFactory,
)

# --------------------------------------------------------------------------- #
# A log sink. The demo and some tests want to SEE the ordered sequence of
# driver events; passing a list-append as ``log`` records them. Tests that only
# assert final state can pass the default no-op.
# --------------------------------------------------------------------------- #
Log = "list[str] | None"


def _sink(log):
    if log is None:
        return lambda *_a: None
    return lambda msg: log.append(msg)


# --------------------------------------------------------------------------- #
# Declarations the Loader "builds" (RFC 02's ComponentCatalog, ExpectedManifest,
# FinalManifest in miniature). A real catalog is the component tree; here it is
# a name -> spec map big enough to show manifest sizing and per-tensor writes.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FakeTensorSpec:
    """One weight's expected shape/dtype/placement (RFC 01 ``expected_manifest``)."""

    shape: tuple[int, ...]
    dtype: torch.dtype
    placement: str = "device"  # no-offload fake: every weight stays resident


# A tiny but realistic diffusion-ish manifest: a text encoder + transformer.
_DEFAULT_MANIFEST: dict[str, FakeTensorSpec] = {
    "text_encoder.layer0.q_proj.weight": FakeTensorSpec((16, 16), torch.float32),
    "text_encoder.layer0.k_proj.weight": FakeTensorSpec((16, 16), torch.float32),
    "transformer.block0.qkv.weight": FakeTensorSpec((48, 16), torch.float32),
    "transformer.block0.mlp.fc1.weight": FakeTensorSpec((64, 16), torch.float32),
}


class FakeComponentCatalog:
    """RFC 02's ``ComponentCatalog`` stand-in: the declared components.

    Declarations stay queryable after ``Loader.close()`` (RFC 01 section 5.1);
    this object holds only plain data, so nothing breaks when the loader closes.
    """

    def __init__(self, manifest: Mapping[str, FakeTensorSpec]) -> None:
        self._manifest = dict(manifest)

    def names(self) -> tuple[str, ...]:
        return tuple(self._manifest)

    def components(self) -> frozenset[str]:
        # The component each tensor belongs to (prefix before the first dot).
        return frozenset(name.split(".", 1)[0] for name in self._manifest)


@dataclass(frozen=True)
class FakePreflight:
    """RFC 02's ``Preflight`` result: identities + the expected manifest, with
    no bytes written yet (RFC 01 Loader.preflight contract)."""

    content_identities: Mapping[str, str]
    expected_manifest: Mapping[str, FakeTensorSpec]


def _build_fake_pipeline(catalog: FakeComponentCatalog) -> nn.Module:
    """What ``Loader.build_structure`` returns: a real (tiny) module graph.

    The real loader builds this on the meta device (structure-only) and
    materialises bytes later in ``write``. For a CPU study build we skip the
    meta dance and just construct a small real module -- enough to be a genuine
    ``nn.Module`` the handle can carry.
    """
    model = nn.Module()
    for comp in sorted(catalog.components()):
        model.add_module(comp, nn.Identity())
    return model


# --------------------------------------------------------------------------- #
# Host side (RFC 03): layout, sharing, election, host storage. In the no-offload
# fake, host storage is EMPTY -- every weight is device-resident -- so these are
# light. They still honour the lifecycle guards (seal, note-after-seal).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FakeLocalRecord:
    rank: int


@dataclass(frozen=True)
class FakeElectionRecord:
    rank: int
    is_writer: bool


class FakeElection:
    """RFC 03's ``Election``: owns this rank's election resources until
    ``allocate`` takes them (or the driver closes it on failure)."""

    def __init__(self, rank: int, is_writer: bool, log) -> None:
        self._rank = rank
        self._is_writer = is_writer
        self._log = log
        self.closed = False

    def record(self) -> FakeElectionRecord:
        return FakeElectionRecord(self._rank, self._is_writer)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self._log(f"election.close (rank {self._rank})")


class FakeSharePlan:
    """RFC 03's ``SharePlan``: who shares which host buffer. Empty here (no host
    residency), but present so the driver's calls type-check end to end."""


class FakeHostLayout:
    """RFC 03's ``HostLayout``: sizing + group binding. ``plan_fingerprint``
    feeds the point-1 fingerprint; ``bind_groups`` is called after point 1."""

    def __init__(self, plan_fingerprint: str = "host-empty") -> None:
        self.plan_fingerprint = plan_fingerprint
        self.bound_groups = None

    def bind_groups(self, groups) -> None:
        self.bound_groups = groups


class FakeHostStorage:
    """RFC 03's ``HostStorage``. Empty in the no-offload fake: nothing is
    host-resident, so ``view`` is never reached and ``note_written`` has nothing
    to reclaim. It still enforces the seal guard (RFC 01 note_written contract).
    """

    def __init__(self, log) -> None:
        self._log = log
        self._sealed = False

    def view(self, tensor_key: str) -> torch.Tensor:  # pragma: no cover
        raise KeyError(f"{tensor_key!r} is device-resident; host storage is empty")

    def ranges(self, unit):  # pragma: no cover - no host units in no-offload
        return []

    def seal(self, final_manifest) -> None:
        self._sealed = True
        self._log("storage.seal (host storage frozen)")

    def connect_writer(self) -> None:
        # After point 2, waiters connect to writers. Nothing to connect when
        # host storage is empty, but the call site still runs.
        self._log("storage.connect_writer")

    def note_written(self, tensor_key: str) -> None:
        if self._sealed:
            raise LifecycleError(
                f"note_written({tensor_key!r}) after seal",
                where="FakeHostStorage.note_written",
            )
        # Empty host storage: no out-of-slice pages to reclaim.

    def close(self) -> None:
        self._log("storage.close (this process's mappings)")


# --------------------------------------------------------------------------- #
# Offload side (RFC 04): plan, transaction, runtime. The transaction owns a REAL
# ResourceLedger; this is where the ownership chain becomes observable.
# --------------------------------------------------------------------------- #
class FakePlacementPlan:
    """RFC 04's ``PlacementPlan`` for the no-offload case: everything resident,
    no AllGather, no unmatched policy keys."""

    def __init__(self) -> None:
        self.bound_groups = None

    def collective_signature(self) -> Mapping[str, str]:
        return {}  # no component uses AllGather -> no collectives, no sharing

    def unmatched_policy_keys(self) -> frozenset[str]:
        return frozenset()

    def bind_groups(self, groups) -> None:
        self.bound_groups = groups


class FakeOffloadRuntime:
    """RFC 04's ``OffloadRuntime``: the post-commit owner of resources.

    Receives the transferred ledger at commit and registers one accelerator slot
    on top of it (RFC 01 fakes contract). ``close`` is idempotent, local, and
    terminal: it closes the ledger (releasing the slot and the device weights in
    ``ResourceKind`` order) and does nothing on a second call.
    """

    def __init__(self, ledger: ResourceLedger, log) -> None:
        self._ledger = ledger
        self._log = log
        self._closed = False
        # "runtime registers one accelerator slot" (RFC 01 Test plan).
        self._slot = object()
        self._ledger.register(
            self._slot,
            ResourceKind.EXECUTION,
            lambda: self._log("  release EXECUTION slot"),
        )

    def close(self) -> None:
        if self._closed:
            return  # idempotent
        self._closed = True
        self._log("runtime.close -> ledger.close():")
        self._ledger.close()

    @contextmanager
    def grad_mode(self):
        with torch.no_grad():
            yield

    @contextmanager
    def execution(self):
        self._log("  [execution scope enter]")
        try:
            yield
        finally:
            self._log("  [execution scope exit]")

    def before_sleep(self) -> None:
        self._log("runtime.before_sleep")

    def after_wake(self) -> None:
        self._log("runtime.after_wake")


class FakeLoadTransaction:
    """RFC 04's ``LoadTransaction``: sole owner of host storage + runtime
    resources between ``prepare_materialization`` and ``commit``.

    Holds a real ``ResourceLedger``. In the no-offload fake, ``target`` returns
    ``None`` for every weight (device-resident), so the Loader allocates on the
    execution device and registers each tensor here via ``register_device_weight``
    (in production the Offloader wires this; the fake exposes it directly so the
    ledger fills observably).
    """

    def __init__(self, model, plan, storage, memory_pools, log) -> None:
        self._model = model
        self._plan = plan
        self._storage = storage
        self._memory_pools = memory_pools
        self._log = log
        self.ledger = ResourceLedger()
        self._committed = False
        self._device_weights: dict[str, torch.Tensor] = {}

    def target(self, tensor_key: str) -> torch.Tensor | None:
        # No-offload: every tensor is permanently device-resident. The contract
        # says a None target instructs the Loader to allocate on the execution
        # device inside ctx.memory_pools.weights().
        return None

    def register_device_weight(self, tensor_key: str, tensor: torch.Tensor) -> None:
        """Study hook: record a Loader-allocated device weight in the ledger as a
        DEVICE resource so abort()/runtime.close() release it in order."""
        self._device_weights[tensor_key] = tensor
        self.ledger.register(
            tensor,
            ResourceKind.DEVICE,
            lambda key=tensor_key: self._log(f"  release DEVICE weight {key}"),
        )

    def commit(self, final_manifest) -> FakeOffloadRuntime:
        if self._committed:
            raise LifecycleError(
                "transaction already committed", where="FakeLoadTransaction.commit"
            )
        self._committed = True
        # Transfer ALL resources to the runtime as one operation, leaving an
        # empty transaction (RFC 01 LoadTransaction.commit contract).
        runtime_ledger = self.ledger.transfer()
        self._log("transaction.commit -> ledger transferred to runtime")
        return FakeOffloadRuntime(runtime_ledger, self._log)

    def abort(self) -> None:
        # Release all transaction resources in ledger order. Idempotent; no
        # action after commit (the ledger was transferred and is now empty).
        self._log("transaction.abort -> ledger.close():")
        self.ledger.close()


# --------------------------------------------------------------------------- #
# The three implementations, assembled by the factory.
# --------------------------------------------------------------------------- #
class FakeLoader:
    """RFC 02's ``Loader`` double."""

    def __init__(self, ctx, catalog, preflight, log) -> None:
        self._ctx = ctx
        self._catalog = catalog
        self._preflight = preflight
        self._log = log
        self._closed = False

    def build_structure(self):
        self._log("loader.build_structure (structure-only)")
        return _build_fake_pipeline(self._catalog), self._catalog

    def preflight(self, catalog, plan):
        self._log("loader.preflight (identities + expected_manifest, no write)")
        return self._preflight

    def write(self, model, storage, transaction) -> None:
        self._log("loader.write (materialise weights):")
        pool = (
            self._ctx.memory_pools.weights()
            if self._ctx.memory_pools is not None
            else nullcontext()
        )
        for key, spec in self._preflight.expected_manifest.items():
            target = transaction.target(key)
            if target is None:
                # device-resident -> allocate on execution device in weights()
                with pool:
                    tensor = torch.zeros(spec.shape, dtype=spec.dtype)
                transaction.register_device_weight(key, tensor)
                self._log(f"  wrote {key} -> device (resident)")
            else:  # pragma: no cover - no host-resident tensors in no-offload
                target.copy_(torch.zeros(spec.shape, dtype=spec.dtype))
                storage.note_written(key)

    def finalize(self, model, storage):
        storage.seal(final_manifest={"ok": True})
        self._log("loader.finalize (sealed; final manifest)")
        return {"final_manifest": True}

    def close(self) -> None:
        if self._closed:
            return  # idempotent
        self._closed = True
        self._log("loader.close (context, traps, readers, temporaries)")


class FakeHostWeightManager:
    """RFC 03's ``HostWeightManager`` double."""

    def __init__(self, ctx, layout, log) -> None:
        self._ctx = ctx
        self._layout = layout
        self._log = log

    def local_record(self):
        return FakeLocalRecord(self._ctx.rank)

    def deployment_id(self):
        # Configured, replica-scoped ID (32 hex chars). Only rank 0's is used.
        return ("a" * 32, True)

    def layout(self, plan, expected_manifest):
        self._log("hwm.layout (size host RAM from manifest -> empty here)")
        return self._layout

    def share_plan(self, topology, fingerprints):
        return FakeSharePlan()

    def elect(self, share_plan):
        # At world size 1, rank 0 is trivially the writer.
        is_writer = self._ctx.rank == 0
        self._log(f"hwm.elect (rank {self._ctx.rank} writer={is_writer})")
        return FakeElection(self._ctx.rank, is_writer, self._log)

    def admit(self, topology, share_plan, elections, layout) -> None:
        # Pure function reading only elections.records; empty host demand always
        # admits.
        self._log("hwm.admit (host capacity ok)")

    def allocate(self, layout, share_plan, elections):
        self._log("hwm.allocate (host storage; empty in no-offload)")
        return FakeHostStorage(self._log)


class FakeOffloader:
    """RFC 04's ``Offloader`` double."""

    def __init__(self, ctx, plan, log) -> None:
        self._ctx = ctx
        self._plan = plan
        self._log = log

    def plan(self, catalog, topology, policy):
        self._log("offloader.plan (no-offload: all resident)")
        return self._plan

    def prepare_materialization(self, model, plan, storage):
        self._log("offloader.prepare_materialization -> transaction owns ledger")
        return FakeLoadTransaction(
            model, plan, storage, self._ctx.memory_pools, self._log
        )

    def setup_compile(self, model, catalog, runtime) -> None:
        self._log("offloader.setup_compile")


class FakeStackFactory(StackFactory):
    """Injected in place of ``EntryPointFactory`` so the driver runs the full
    lifecycle against doubles (RFC 01 section 6.1; "Interfaces with other child
    RFCs": fakes replace other modules through factory injection).
    """

    def __init__(self, log: "list[str] | None" = None,
                 manifest: "Mapping[str, FakeTensorSpec] | None" = None) -> None:
        self._log = _sink(log)
        self._manifest = dict(manifest or _DEFAULT_MANIFEST)

    def create(self, ctx):
        catalog = FakeComponentCatalog(self._manifest)
        # content_identities: one digest per component (no AllGather -> the
        # collective map stays empty, so no sharing/grouping is attempted).
        identities = {comp: f"sha-{comp}" for comp in sorted(catalog.components())}
        preflight = FakePreflight(
            content_identities=identities, expected_manifest=self._manifest
        )
        layout = FakeHostLayout()
        plan = FakePlacementPlan()
        return (
            FakeLoader(ctx, catalog, preflight, self._log),
            FakeHostWeightManager(ctx, layout, self._log),
            FakeOffloader(ctx, plan, self._log),
        )


# --------------------------------------------------------------------------- #
# Context helpers. The driver reads ctx by attribute, so a SimpleNamespace is a
# faithful stand-in for the frozen StartupContext at world size 1.
# --------------------------------------------------------------------------- #
class FakeMemoryPools:
    """``MemoryPools`` double: weights()/slots() are no-op context managers.
    Without sleep mode the real pools are no-ops too (RFC 01 section 8.2)."""

    def weights(self):
        return nullcontext()

    def slots(self):
        return nullcontext()


def build_fake_context(*, rank: int = 0, world_size: int = 1, faults=None,
                        compile_enabled: bool = False) -> SimpleNamespace:
    """A StartupContext-shaped namespace for driving the fakes at world size 1."""
    return SimpleNamespace(
        od_config=SimpleNamespace(stage_id=0, diffusion_weight_management="unified"),
        vllm_config=None,
        device=torch.device("cpu"),
        rank=rank,
        local_rank=rank,
        world_size=world_size,
        load_format="default",
        custom_pipeline_name=None,
        memory_pools=FakeMemoryPools(),
        policy=None,
        compile_enabled=compile_enabled,
        faults=faults,
    )


# --------------------------------------------------------------------------- #
# Runnable study demo: drive the whole lifecycle and print every event, then
# exercise one forward and the terminal close so you can watch the ledger
# release in ResourceKind order. Run it (via your venv, never system python):
#     .venv/bin/python -m tests.diffusion.weight_management.fakes
# --------------------------------------------------------------------------- #
def _demo() -> None:
    from vllm_omni.diffusion.weight_management.interfaces import LoadState
    from vllm_omni.diffusion.weight_management.startup import run_unified_startup

    log: list[str] = []
    ctx = build_fake_context()

    print("=== STARTUP (13-step lifecycle, world size 1) ===\n")
    handle = run_unified_startup(ctx, FakeStackFactory(log))
    for line in log:
        print(line)
    print(f"\nstate after startup: {handle.state.name}  (expect COMMITTED)")
    assert handle.state is LoadState.COMMITTED

    print("\n=== SERVICE ===")
    handle.mark_in_service()
    print(f"state: {handle.state.name}  (expect IN_SERVICE)")
    with handle.runtime.execution():
        print("  (ran one forward inside execution scope)")

    print("\n=== SHUTDOWN (watch the ledger close in ResourceKind order) ===")
    before = len(log)
    handle.close()
    for line in log[before:]:  # only the lines runtime.close/ledger.close added
        print(line)
    print(f"\nstate: {handle.state.name}  (expect CLOSED)")
    handle.close()  # idempotent
    print("second close(): no-op (idempotent)")


if __name__ == "__main__":  # pragma: no cover
    _demo()
