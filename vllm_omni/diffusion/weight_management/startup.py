# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Startup-side resource bookkeeping for the unified weight management.

PR 01-1 of RFC 01 (child of umbrella RFC #9536).
    - Spec: RFC 01 section 5.2 "Resource ledger" (the ``ResourceKind`` table
      and the ``ResourceLedger`` method list are copied from there verbatim).
    - The ownership chain this ledger serves is RFC 01 section 5.1, and the
      state machine that drives it is RFC 01 section 4.

What this REPLACES (read these to see the problem it fixes):
    Today each offload backend tears its own resources down ad hoc inside its
    ``disable()`` method, with no shared ordering or exactly-once guarantee:
        - vllm_omni/diffusion/offloader/sequential_backend.py:281  disable()
        - vllm_omni/diffusion/offloader/layerwise_backend.py:401   disable()
        - vllm_omni/diffusion/offloader/base.py:90                 disable() (abstract)
    The unified stack instead registers every resource (streams, hooks,
    staging buffers, host mappings, ...) in ONE ledger that closes them in a
    fixed, safe order exactly once. This is umbrella RFC motivation #3
    ("implicit startup handoff") and #6 ("cleanup after termination").

Scope note: the full startup driver and state machine arrive in later PRs
(01-4, 01-5). This module currently provides only ``ResourceLedger``, which
the startup transaction and ``OffloadRuntime`` use for release.
"""
from __future__ import annotations

import enum
import gc
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

# RFC 01 section 1.2: at module scope, startup.py imports ONLY the standard
# library, torch, interfaces.py, and errors.py. Anything heavier (group.py,
# platform, torch.distributed, the engine) is imported INSIDE functions so that
# data.py / offloader / engine / executors can import this module freely without
# creating import cycles. That is why we use stdlib ``logging`` here rather than
# vllm's ``init_logger``.
from vllm_omni.diffusion.weight_management.errors import (
    AllocationError,
    CommitError,
    ConfigError,
    ConstructionError,
    FinalizeError,
    GroupConfirmationError,
    LifecycleError,
    PlanningError,
    PrepareError,
    PreflightError,
    StackUnavailableError,
    WeightManagementError,
    WriteError,
)

if TYPE_CHECKING:
    from vllm_omni.diffusion.weight_management.interfaces import (
        HostWeightManager,
        Loader,
        Offloader,
        StartupContext,
    )

logger = logging.getLogger(__name__)


class ResourceKind(enum.IntEnum):
    """Release order for held resources. Kinds close in ascending value."""

    EXECUTION = 1
    INFLIGHT = 2
    HOOKS = 3
    DEVICE = 4
    REGISTRATION = 5
    STAGING = 6
    PLACEHOLDER = 7
    HOST_STORAGE = 8


@dataclass
class _Entry:
    handle: object
    kind: ResourceKind
    close: Callable[[], None]


class ResourceLedger:
    """Tracks held resources and releases each exactly once, in order.

    Closes resources in ascending ``ResourceKind`` and, within one kind, in
    reverse registration order. The ledger performs no ``torch.distributed``
    operations.
    """

    def __init__(self) -> None:
        self._entries: list[_Entry] = []
        self._handle_ids: set[int] = set()
        self._sealed = False

    def register(
        self, handle: object, kind: ResourceKind, close: Callable[[], None]
    ) -> None:
        """Record a resource and its release callback.

        Raises:
            LifecycleError: If the ledger has been transferred or closed, or
                if ``handle`` is already registered.
        """
        if self._sealed:
            raise LifecycleError(
                "cannot register on a transferred or closed ledger",
                where="ResourceLedger.register",
            )
        if id(handle) in self._handle_ids:
            raise LifecycleError(
                f"resource handle {handle!r} already registered",
                where="ResourceLedger.register",
            )
        self._handle_ids.add(id(handle))
        self._entries.append(_Entry(handle, kind, close))

    def hand_over(self, handle: object) -> None:
        """Remove a handle so an external owner takes responsibility for it.

        Used at commit to pass dynamic LoRA overlay slots to the LoRA manager.

        Raises:
            LifecycleError: If ``handle`` is not registered.
        """
        for i, entry in enumerate(self._entries):
            if entry.handle is handle:
                del self._entries[i]
                self._handle_ids.discard(id(handle))
                return
        raise LifecycleError(
            f"resource handle {handle!r} not in ledger",
            where="ResourceLedger.hand_over",
        )

    def transfer(self) -> ResourceLedger:
        """Move all entries to a new ledger and seal this one.

        Commit uses this to hand resources to ``OffloadRuntime``. The original
        is emptied and rejects further registration.
        """
        new = ResourceLedger()
        new._entries = self._entries
        new._handle_ids = self._handle_ids
        self._entries = []
        self._handle_ids = set()
        self._sealed = True
        return new

    def close(self) -> None:
        """Release every resource once, in order. Idempotent.

        Runs all remaining callbacks even if some raise. Afterward, if any
        raised, re-raises ``LifecycleError`` with the first exception as
        ``__cause__`` and logs the rest.
        """
        entries, self._entries = self._entries, []
        self._handle_ids = set()
        self._sealed = True
        if not entries:
            return

        errors: list[BaseException] = []
        for kind in sorted(ResourceKind):
            for entry in reversed([e for e in entries if e.kind is kind]):
                try:
                    entry.close()
                except BaseException as exc:  # noqa: BLE001 - release must continue
                    errors.append(exc)

        if errors:
            for extra in errors[1:]:
                logger.error("ResourceLedger close callback failed: %r", extra)
            raise LifecycleError(
                "one or more resource close callbacks failed",
                where="ResourceLedger.close",
            ) from errors[0]


# =========================================================================== #
# PR 01-2: Implementation selector (RFC 01 section 1).
#
# One flag, ``--diffusion-weight-management {legacy,unified}``, chooses between the
# existing offloader and this unified stack. ``legacy`` is the default; there
# are no partial combinations. These functions are the authoritative readers of
# that choice. CONFIG-level code (arg parsing, engine, executors,
# offloader/config.py) reads it from a config object via ``is_unified(config)``;
# MODEL-level code (construction context, read traps, model files) cannot see
# config and instead reads process-level state via ``process_weight_management()``.
#
# NOTE: the CLI/config propagation half of PR 01-2 -- adding the
# ``diffusion_weight_management`` field to the five files in RFC 01 section 1.1
# (serve.py, arg_utils.py, stage_config.py, omni_config.py, data.py) -- is the
# integration wiring. It is intentionally NOT done here: those config classes
# differ from the RFC baseline in this checkout, so wiring them blind would be
# guesswork. This module provides the logic those sites will call.
# =========================================================================== #
class WeightManagement(str, enum.Enum):
    """Which weight implementation a worker runs. ``str`` mixin so the value
    round-trips through YAML/CLI as a plain string."""

    LEGACY = "legacy"
    UNIFIED = "unified"


def parse_weight_management(value: object) -> WeightManagement:
    """Coerce a config value to ``WeightManagement``.

    ``None`` means legacy (the field is optional everywhere upstream). A value
    that is neither ``legacy`` nor ``unified`` is a user error.

    Raises:
        ConfigError: If ``value`` is not None, a ``WeightManagement``, or one of the
            two valid strings.
    """
    if value is None:
        return WeightManagement.LEGACY
    if isinstance(value, WeightManagement):
        return value
    if isinstance(value, str):
        try:
            return WeightManagement(value)
        except ValueError:
            pass
    raise ConfigError(
        f"invalid diffusion_weight_management {value!r}",
        where="diffusion_weight_management",
        fix="use --diffusion-weight-management legacy (or unified)",
    )


def is_unified(config: object) -> bool:
    """True if ``config`` selects the unified stack. A missing attribute is
    treated as legacy."""
    return parse_weight_management(
        getattr(config, "diffusion_weight_management", None)
    ) is WeightManagement.UNIFIED


# ---- Process-level state (for model code that cannot see a config) --------- #
# A single OS process may host workers from several stages (one executor). They
# must all agree on the selector, so the state records the chosen value AND the
# set of worker owners that asserted it, and reverts to unset only after the
# last owner clears it.
_process_stack: WeightManagement | None = None
_process_owners: dict[int, object] = {}


def set_process_weight_management(value: WeightManagement, *, owner: object) -> None:
    """Assert the process selector on behalf of ``owner`` (a worker).

    Called by ``DiffusionWorker.__init__`` after ``init_device()``.

    Raises:
        ConfigError: If another live owner in this process already asserted a
            different value (two stages in one process disagree).
    """
    global _process_stack
    if _process_owners and _process_stack is not value:
        raise ConfigError(
            f"process already runs weight management {_process_stack.value!r}; "
            f"cannot also run {value.value!r} in the same process",
            where="set_process_weight_management",
            fix="give stages sharing a process the same --diffusion-weight-management",
        )
    _process_stack = value
    _process_owners[id(owner)] = owner


def clear_process_weight_management(*, owner: object) -> None:
    """Release ``owner``'s assertion. Reverts to unset after the last owner
    clears. Idempotent for an unknown owner. Called by both the exception path
    and normal ``DiffusionWorker.shutdown``."""
    global _process_stack
    _process_owners.pop(id(owner), None)
    if not _process_owners:
        _process_stack = None


def process_weight_management() -> WeightManagement:
    """The process selector, or ``LEGACY`` when none is set."""
    return _process_stack if _process_stack is not None else WeightManagement.LEGACY


# ---- Argument-time checks (RFC 01 section 1.3) ----------------------------- #
# Until RFC 05's policy parser lands, unified mode rejects any offload-related
# setting rather than silently ignoring it. Each check names the offending
# setting and recommends falling back to legacy.
def check_unified_offload_conflicts(config: object) -> None:
    """Raise ``ConfigError`` if unified mode is combined with offload settings.

    No-op in legacy mode. This is a temporary gate; RFC 05 replaces it with a
    real compatibility layer that translates legacy flags.

    Raises:
        ConfigError: If unified is selected and any offload setting is present.
    """
    if not is_unified(config):
        return

    fix = "use --diffusion-weight-management legacy"

    def reject(what: str, where: str) -> None:
        raise ConfigError(what, where=where, fix=fix)

    for flag in (
        "enable_cpu_offload",
        "enable_layerwise_offload",
        "enable_distributed_layerwise_offload",
    ):
        if getattr(config, flag, False):
            reject(f"{flag} is not supported in unified mode yet", flag)

    if getattr(config, "diffusion_offload_config", None) is not None:
        reject("diffusion_offload_config is not supported in unified mode yet",
               "diffusion_offload_config")

    if getattr(config, "dlo_host_registration_limit_gib", 0) > 0:
        reject("dlo_host_registration_limit_gib is not supported in unified mode "
               "yet", "dlo_host_registration_limit_gib")

    if getattr(config, "host_weight_runtime_mode", "disabled") != "disabled" or \
            getattr(config, "host_weight_runtime_root", None):
        reject("host_weight_runtime is not supported in unified mode yet",
               "host_weight_runtime_mode")

    for env in ("OMNI_DIFFUSION_PROMPT_EMBED_CACHE",
                "OMNI_DIFFUSION_PROMPT_EMBED_CACHE_SIZE"):
        if os.environ.get(env):
            reject(f"{env} is not read by unified workers", env)


# =========================================================================== #
# PR 01-3: Engine/executor execution facts (RFC 01 section 9.2).
#
# The engine and executors need a few offload-derived booleans at runtime (do
# we need weight collectives? is the text encoder all-gathered? do outputs move
# to CPU before postprocessing?). Instead of each site branching on the
# selector, they ALL call this one function, which returns ExecutionOffloadFacts.
#
# IMPORTANT (static test T1 enforces this): the legacy branch is the ONLY place
# in the whole weight_management package allowed to import legacy
# ``offloader.config``, and it must do so INSIDE the function. The unified
# branch imports ``weight_management.policy`` (RFC 05) the same way. That keeps
# module-scope import graphs clean and prevents import cycles.
#
# Until RFC 05's policy parser exists, unified mode rejects offload
# (check_unified_offload_conflicts), so all unified facts are False. RFC 05
# replaces the unified branch with real policy reads and this function keeps its
# signature. NOTE: the §9.1 rewrite of offloader/config.py's queries and the
# migration of the real engine/executor call sites are integration wiring
# against modules that differ in this checkout; not done here.
# =========================================================================== #
def execution_offload_facts(config: object):
    """Return the runtime offload facts for ``config`` (RFC 01 section 9.2).

    Returns:
        ExecutionOffloadFacts: all-False in unified mode until RFC 05 lands; in
        legacy mode, composed from the existing ``offloader.config`` queries.
    """
    from vllm_omni.diffusion.weight_management.interfaces import ExecutionOffloadFacts

    if is_unified(config):
        return ExecutionOffloadFacts(
            weight_collectives=False,
            text_encoder_allgather=False,
            whole_component_offload=False,
        )

    # Legacy: the one permitted function-local import of legacy offloader config
    # (T1 allows ONLY this module here, and ONLY inside the function). We compare
    # the strategy by value so we need not also import offloader.base.
    from vllm_omni.diffusion.offloader import config as offload_config

    strategy = offload_config.resolve_offload_strategy(config)
    return ExecutionOffloadFacts(
        weight_collectives=offload_config.any_selected_component_uses_allgather(config),
        text_encoder_allgather=offload_config.component_uses_allgather(
            config, "text_encoder"
        ),
        whole_component_offload=getattr(strategy, "value", strategy) == "model_level",
    )


# =========================================================================== #
# PR 01-5: Startup driver and state machine (RFC 01 sections 4 and 6).
#
# This is where everything converges. The driver runs the 13-step transactional
# lifecycle (RFC 01 §6.2), advancing LoadState (§4), moving resource ownership
# along the chain Election -> HostStorage -> LoadTransaction -> OffloadRuntime
# (§5.1), and pausing at four confirmation points (§7) so the whole replica
# agrees before anyone allocates, pins, or commits. On any failure it releases
# whatever the current owner holds, in ledger order, and enters FAILED.
#
# It talks to the Loader/HWM/Offloader ONLY through the Protocols in
# interfaces.py, so the concrete RFC 02/03/04 implementations can be swapped in
# (or replaced by test doubles) without touching this file.
# =========================================================================== #

# RFC 01 §3: wrap a foreign exception raised in a step as that step's error type.
_STEP_ERRORS: dict[str, type[WeightManagementError]] = {
    "construct_modules": StackUnavailableError,
    "build_structure": ConstructionError,
    "plan": PlanningError,
    "preflight": PreflightError,
    "point0_topology": GroupConfirmationError,
    "point1_fingerprints": GroupConfirmationError,
    "confirm": GroupConfirmationError,
    "point1_elections": GroupConfirmationError,
    "admit": GroupConfirmationError,
    "point2_ready": GroupConfirmationError,
    "point3_commit": GroupConfirmationError,
    "allocate": AllocationError,
    "connect_writer": AllocationError,
    "prepare": PrepareError,
    "write": WriteError,
    "finalize": FinalizeError,
    "commit": CommitError,
}


def _wrap(exc: Exception, phase: str, ctx: StartupContext) -> WeightManagementError:
    """Return a WeightManagementError for ``exc`` at ``phase`` with identity filled.

    An existing WeightManagementError is reused (identity fields set); anything else
    is wrapped in the step's type with the original kept as ``__cause__``.
    """
    if isinstance(exc, WeightManagementError):
        err = exc
    else:
        cls = _STEP_ERRORS.get(phase, WeightManagementError)
        err = cls(str(exc), where=phase)
        err.__cause__ = exc
    err.stage_id = getattr(ctx.od_config, "stage_id", None)
    err.rank = ctx.rank
    err.world_size = ctx.world_size
    return err


def _close_owner(owner: object | None) -> None:
    """Release the current ownership-chain holder (RFC 01 §5.1).

    A LoadTransaction releases via ``abort()`` (which frees host storage and all
    runtime resources in ledger order); an Election, HostStorage, or
    OffloadRuntime releases via ``close()``.
    """
    if owner is None:
        return
    if hasattr(owner, "abort"):
        owner.abort()
    else:
        owner.close()


class InjectedFault(RuntimeError):
    """Raised by FaultPlan at a named point (RFC 01 §10). The driver wraps it in
    the failing step's error type, so tests can force a failure anywhere."""


@dataclass(frozen=True)
class FaultPlan:
    """Test-only map of startup point -> (ranks, action) (RFC 01 §10).

    action is 'raise' (produce an InjectedFault) or 'hang' (block, for timeout
    testing). An omitted rank applies to all ranks.
    """

    points: dict[str, tuple[frozenset[int] | None, str]]

    @classmethod
    def from_environ(cls) -> FaultPlan | None:
        """Parse ``VLLM_OMNI_WEIGHT_STACK_FAULTS`` (comma-separated
        ``<point>[@<rank>][:raise|:hang]``). Absent -> None (checks disabled).
        """
        raw = os.environ.get("VLLM_OMNI_WEIGHT_STACK_FAULTS")
        if not raw:
            return None
        points: dict[str, tuple[frozenset[int] | None, str]] = {}
        for spec in raw.split(","):
            spec = spec.strip()
            if not spec:
                continue
            head, _, action = spec.partition(":")
            action = action or "raise"
            point, _, rank = head.partition("@")
            ranks = frozenset({int(rank)}) if rank else None
            points[point] = (ranks, action)
        return cls(points=points)

    def hit(self, point: str, rank: int) -> None:
        """Trigger the fault configured for ``point`` on ``rank``, if any."""
        entry = self.points.get(point)
        if entry is None:
            return
        ranks, action = entry
        if ranks is not None and rank not in ranks:
            return
        if action == "hang":  # real impl blocks; study build surfaces it instead
            raise InjectedFault(f"injected hang at {point} (rank {rank})")
        raise InjectedFault(f"injected fault at {point} (rank {rank})")


class StackFactory:
    """Protocol: create the three implementations for a StartupContext."""

    def create(self, ctx: StartupContext):  # -> (Loader, HostWeightManager, Offloader)
        raise NotImplementedError


class EntryPointFactory(StackFactory):
    """Default factory. Resolves the three implementations by dotted name with
    importlib, so startup.py never imports them at module scope (RFC 01 §6.1).

    A missing module or function means the unified stack is not installed, which
    is exactly RFC 01's coexistence state before RFC 02/03/04 land.
    """

    LOADER = "vllm_omni.diffusion.weight_management.loader:create_loader"
    HOST = "vllm_omni.diffusion.weight_management.host:create_host_weight_manager"
    OFFLOADER = "vllm_omni.diffusion.weight_management.offload:create_offloader"

    def create(self, ctx: StartupContext):
        import importlib

        resolved = []
        missing = []
        for name in (self.LOADER, self.HOST, self.OFFLOADER):
            module_path, _, attr = name.partition(":")
            try:
                module = importlib.import_module(module_path)
                resolved.append(getattr(module, attr))
            except (ImportError, AttributeError):
                missing.append(name)
        if missing:
            raise StackUnavailableError(
                f"unified weight management not available; missing: {missing}",
                where="construct_modules",
                fix="use --diffusion-weight-management legacy",
            )
        loader_fn, host_fn, offloader_fn = resolved
        return loader_fn(ctx), host_fn(ctx), offloader_fn(ctx)


class WeightManagementHandle:
    """What ``run_unified_startup`` returns on success (RFC 01 §6.1).

    Holds the committed model, its declarations, and the runtime. ``close()`` is
    idempotent and local: close the runtime, drop the model, enter CLOSED.
    """

    def __init__(self, state, model, catalog, runtime) -> None:
        self.state = state
        self.model = model
        self.catalog = catalog
        self.runtime = runtime

    def mark_in_service(self) -> None:
        from vllm_omni.diffusion.weight_management.interfaces import LoadState

        self.state = LoadState.IN_SERVICE

    def close(self) -> None:
        from vllm_omni.diffusion.weight_management.interfaces import LoadState

        if self.state is LoadState.CLOSED:
            return
        if self.runtime is not None:
            self.runtime.close()
        self.model = None
        self.state = LoadState.CLOSED


def run_unified_startup(ctx: StartupContext, factory: StackFactory | None = None):
    """Drive the unified weight-management startup to IN-SERVICE readiness.

    Runs the 13-step lifecycle (RFC 01 §6.2). On any failure, releases the
    current owner's resources in ledger order and re-raises.

    Returns:
        WeightManagementHandle: in COMMITTED state; the runner calls
        ``mark_in_service()`` after its own remaining steps.
    """
    s = _Startup(ctx, factory or EntryPointFactory())
    try:
        s.local("construct_modules", s.construct_modules)
        s.local("build_structure", s.build_structure)
        s.confirm_topology()
        s.local("plan", s.plan)
        s.local("preflight", s.preflight)
        s.confirm_admission()
        s.local("allocate", s.allocate)
        s.local("prepare", s.prepare)
        s.confirm_ready()
        s.local("connect_writer", s.connect_writer)
        s.local("write", s.write)
        s.local("finalize", s.finalize)
        s.local("commit", s.commit)
        s.confirm_commit()
        s.compile_setup()
        return s.handle()
    except BaseException:
        s.release()
        raise


class _Startup:
    """Mutable startup state. One per rank per startup (RFC 01 §6.2)."""

    def __init__(self, ctx: StartupContext, factory: StackFactory) -> None:
        self.ctx = ctx
        self.factory = factory
        self.owner: object | None = None  # current ownership-chain holder
        self.pending: WeightManagementError | None = None  # local error, not yet reported
        self.loader: Loader | None = None
        self.hwm: HostWeightManager | None = None
        self.offloader: Offloader | None = None
        self.model = None
        self.catalog = None
        self.runtime = None
        from vllm_omni.diffusion.weight_management.group import ReplicaGroup

        self.group = ReplicaGroup(ctx.world_size, ctx.rank)
        from vllm_omni.diffusion.weight_management.interfaces import LoadState

        self.state = None
        self._LoadState = LoadState

    def enter(self, state) -> None:
        self.state = state

    # ---- steps (RFC 01 §6.2; numbers are overall-design 4.2 steps) --------- #
    def construct_modules(self):  # step 1
        self.loader, self.hwm, self.offloader = self.factory.create(self.ctx)

    def build_structure(self):  # step 2
        self.model, self.catalog = self.loader.build_structure()
        self.enter(self._LoadState.STRUCTURE_READY)

    def confirm_topology(self):  # step 3 / point 0
        from vllm_omni.diffusion.weight_management.group import (
            Point0Payload,
            assemble_topology,
            read_rank_coordinates,
        )

        gathered = self.round("point0_topology", lambda: Point0Payload(
            record=self.hwm.local_record(),
            coordinates=read_rank_coordinates(),
            deployment=self.hwm.deployment_id(),
        ))
        if self.pending is not None:
            return
        stage_id = getattr(self.ctx.od_config, "stage_id", 0)
        self.topology = assemble_topology(gathered, False, stage_id)

    def plan(self):  # step 4
        self.plan_ = self.offloader.plan(self.catalog, self.topology, self.ctx.policy)
        self.enter(self._LoadState.PLANNED)

    def preflight(self):  # step 5
        from vllm_omni.diffusion.weight_management.interfaces import RankFingerprints

        self.preflight_ = self.loader.preflight(self.catalog, self.plan_)
        self.layout = self.hwm.layout(
            self.plan_, self.preflight_.expected_manifest
        )
        self.fingerprints = RankFingerprints(
            rank=self.ctx.rank,
            content_identities=self.preflight_.content_identities,
            plan_fingerprint=self.layout.plan_fingerprint,
            collective=self.plan_.collective_signature(),
        )
        self.enter(self._LoadState.SOURCES_PREPARED)

    def confirm_admission(self):  # step 6 / point 1 (two rounds)
        from vllm_omni.diffusion.weight_management.group import (
            Point1Payload,
            check_collective_parts,
            check_unmatched_policy_keys,
            create_allgather_groups,
            group_table_keys,
        )

        gathered = self.round("point1_fingerprints", lambda: Point1Payload(
            fingerprints=self.fingerprints,
            group_keys=group_table_keys(),
            unmatched_policy_keys=self.plan_.unmatched_policy_keys(),
        ))
        if self.pending is not None:
            return
        all_fps = tuple(p.fingerprints for p in gathered)
        check_unmatched_policy_keys(gathered)
        check_collective_parts(all_fps)
        groups = create_allgather_groups(all_fps, gathered)
        self.local("confirm", lambda: self.bind_and_elect(groups, all_fps))
        records = self.round("point1_elections", lambda: self.election.record())
        if self.pending is not None:
            return
        self.local("admit", lambda: self.admit(records))

    def bind_and_elect(self, groups, all_fps):
        self.layout.bind_groups(groups)
        self.plan_.bind_groups(groups)
        self.share_plan = self.hwm.share_plan(self.topology, all_fps)
        self.election = self.hwm.elect(self.share_plan)
        self.owner = self.election

    def admit(self, records):
        from vllm_omni.diffusion.weight_management.interfaces import Elections

        self.elections = Elections(local=self.election, records=records)
        self.hwm.admit(self.topology, self.share_plan, self.elections, self.layout)
        self.enter(self._LoadState.ADMITTED)

    def allocate(self):  # step 7 (host ownership -> storage)
        self.storage = self.hwm.allocate(
            self.layout, self.share_plan, self.elections
        )
        self.owner = self.storage

    def prepare(self):  # step 8 (host ownership -> transaction)
        self.transaction = self.offloader.prepare_materialization(
            self.model, self.plan_, self.storage
        )
        self.owner = self.transaction

    def confirm_ready(self):  # point 2
        self.round("point2_ready")
        if self.pending is None:
            self.enter(self._LoadState.TRANSACTION_READY)

    def connect_writer(self):
        self.storage.connect_writer()

    def write(self):  # step 9
        self.loader.write(self.model, self.storage, self.transaction)
        self.enter(self._LoadState.WEIGHTS_READY)

    def finalize(self):  # step 10 (closes Loader)
        self.final_manifest = self.loader.finalize(self.model, self.storage)
        self.loader.close()
        self.enter(self._LoadState.FINALIZED)

    def commit(self):  # step 11 (ownership -> runtime)
        self.runtime = self.transaction.commit(self.final_manifest)
        self.owner = self.runtime
        self.enter(self._LoadState.COMMITTED)

    def confirm_commit(self):  # point 3
        self.round("point3_commit")

    def compile_setup(self):  # step 12
        if self.ctx.compile_enabled:
            try:
                if self.ctx.faults is not None:
                    self.ctx.faults.hit("compile_setup", self.ctx.rank)
                self.offloader.setup_compile(self.model, self.catalog, self.runtime)
            except Exception as exc:  # noqa: BLE001 - compile is best-effort
                logger.warning(
                    "setup_compile failed (%s); running without compile", exc
                )

    def handle(self) -> WeightManagementHandle:
        return WeightManagementHandle(self.state, self.model, self.catalog, self.runtime)

    # ---- error capture and release (RFC 01 §6.2) --------------------------- #
    def local(self, phase: str, fn: Callable[[], None]) -> None:
        if self.pending is not None:
            return
        try:
            if self.ctx.faults is not None:
                self.ctx.faults.hit(phase, self.ctx.rank)
            fn()
        except Exception as exc:  # noqa: BLE001 - captured and reported at a point
            self.fail(exc, phase)

    def round(self, point: str, compute: Callable[[], object] | None = None):
        from vllm_omni.diffusion.weight_management.group import (
            RoundMessage,
            raise_if_any_failed,
        )

        payload = None
        if self.pending is None:
            try:
                if self.ctx.faults is not None:
                    self.ctx.faults.hit(point, self.ctx.rank)
                payload = compute() if compute is not None else None
            except Exception as exc:  # noqa: BLE001
                self.fail(exc, point)
        message = RoundMessage.of(point, self.ctx.rank, self.pending, payload)
        gathered = self.group.exchange(point, message)
        raise_if_any_failed(gathered, self.pending)
        return tuple(m.payload for m in gathered)

    def fail(self, exc: Exception, phase: str) -> None:
        self.pending = _wrap(exc, phase, self.ctx)
        self.release()

    def release(self) -> None:
        _close_owner(self.owner)
        self.owner = None
        if self.loader is not None:
            try:
                self.loader.close()
            except Exception:  # noqa: BLE001 - cleanup is best-effort
                pass
        self.model = self.catalog = None
        gc.collect()
        try:
            from vllm_omni.platforms import current_omni_platform

            current_omni_platform.empty_cache()
        except Exception:  # noqa: BLE001 - no accelerator in study/test env
            pass
        self.state = self._LoadState.FAILED
