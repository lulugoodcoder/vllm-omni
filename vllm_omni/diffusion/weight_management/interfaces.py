# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Interfaces and data types for the unified weight management.

PR 01-1 of RFC 01 (child of umbrella RFC #9536).
    - Spec: RFC 01 section 2 "Interfaces". The LoadState enum, the frozen
      dataclasses, and the six Protocols below are copied from there; the
      per-method contracts are RFC 01 section 2's contract table.

Why this file matters: these Protocols are the CONTRACT every other child RFC
codes against. 01-1 ships only the *shapes*; the concrete classes arrive later:
    Loader, LoadTransaction          -> RFC 02 (vllm-omni #9875)
    HostWeightManager, HostStorage   -> RFC 03 (vllm-omni #9876)
    Offloader, OffloadRuntime        -> RFC 04 (vllm-omni #9877)
    the driver that composes them    -> RFC 01 (this stack, later PRs 01-4/01-5)

Dependency rule (RFC 01 section 2): this module depends ONLY on the standard
library and torch. Types owned by other child RFCs are referenced under
``if TYPE_CHECKING:`` with no runtime import, so nothing here forces RFC 02-05
code to exist yet. ``from __future__ import annotations`` makes every
annotation a string, so these forward references never raise at runtime.

Legacy contrast: the existing offloader has no shared contract -- each backend
(``offloader/sequential_backend.py``, ``layerwise_backend.py``) exposes its own
``enable()``/``disable()`` surface (``offloader/base.py:69-99``). This file is
the single typed boundary that replaces those ad-hoc surfaces.
"""
from __future__ import annotations

import enum
from collections.abc import Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import torch
from torch import nn

if TYPE_CHECKING:
    # Not imported at runtime (TYPE_CHECKING is False); these are owned by the
    # other child RFCs and only need to exist for type checkers.
    from torch.distributed import ProcessGroup
    from vllm.config import VllmConfig

    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.weight_management.components import ComponentCatalog  # RFC 02
    from vllm_omni.diffusion.weight_management.host import (  # RFC 03
        Election,
        ElectionRecord,
        HostLayout,
        LocalRecord,
        MemoryRange,
        SharePlan,
    )
    from vllm_omni.diffusion.weight_management.loader import (  # RFC 02
        ExpectedManifest,
        FinalManifest,
        Preflight,
    )
    from vllm_omni.diffusion.weight_management.offload import PlacementPlan, Unit  # RFC 04
    from vllm_omni.diffusion.weight_management.policy import OffloadPolicy  # RFC 05
    from vllm_omni.diffusion.weight_management.startup import FaultPlan  # RFC 01


class LoadState(enum.Enum):
    """Startup state-machine states (RFC 01 section 4).

    The driver advances strictly in this order or jumps to FAILED; any other
    transition raises LifecycleError. FAILED and CLOSED are terminal: startup
    cleanup and normal shutdown, respectively.
    """

    STRUCTURE_READY = "structure_ready"
    PLANNED = "planned"
    SOURCES_PREPARED = "sources_prepared"
    ADMITTED = "admitted"
    TRANSACTION_READY = "transaction_ready"
    WEIGHTS_READY = "weights_ready"
    FINALIZED = "finalized"
    COMMITTED = "committed"
    IN_SERVICE = "in_service"
    FAILED = "failed"
    CLOSED = "closed"


@dataclass(frozen=True)
class ParallelCoordinate:
    size: int
    rank: int


@dataclass(frozen=True)
class RankCoordinates:
    """One rank's position in every parallelism axis. ep/fs are None when EP or
    HSDP is disabled."""

    tp: ParallelCoordinate
    sp: ParallelCoordinate
    ulysses: ParallelCoordinate
    ring: ParallelCoordinate
    cfg: ParallelCoordinate
    dp: ParallelCoordinate
    pp: ParallelCoordinate
    ep: ParallelCoordinate | None
    fs: ParallelCoordinate | None


@dataclass(frozen=True)
class Topology:
    """The replica's shape, produced at coordination point 0.

    deployment_id has 32 hex characters. deployment_scoped=False means no ID was
    configured and rank 0 generated one for this replica only. records and
    coordinates are rank-ordered and have length world_size.
    """

    deployment_id: str
    deployment_scoped: bool
    stage_id: int
    world_size: int
    records: tuple[LocalRecord, ...]
    coordinates: tuple[RankCoordinates, ...]


@dataclass(frozen=True)
class RankFingerprints:
    """One rank's content identities, collected at point 1 to decide sharing.

    content_identities maps component name -> identity digest (locally declared
    components only). plan_fingerprint describes the local-backend part of the
    HostLayout. collective maps components with AllGather Units to digests.
    """

    rank: int
    content_identities: Mapping[str, str]
    plan_fingerprint: str
    collective: Mapping[str, str]


Fingerprints = tuple[RankFingerprints, ...]


@dataclass(frozen=True)
class AllGatherGroup:
    """One AllGather group for a component. ranks is ascending with >=2 members;
    group is None on nonmembers."""

    component: str
    ranks: tuple[int, ...]
    group: ProcessGroup | None


AllGatherGroups = Mapping[str, tuple[AllGatherGroup, ...]]


@dataclass(frozen=True)
class Elections:
    """Election outcome: local owns this rank's election resources; records is
    rank-ordered and serializable (collected in point 1's second round)."""

    local: Election
    records: tuple[ElectionRecord, ...]


class MemoryPools(Protocol):
    """Two accelerator pools: weights() is backed up during sleep; slots() is
    released without backup."""

    def weights(self) -> AbstractContextManager[None]: ...
    def slots(self) -> AbstractContextManager[None]: ...


@dataclass(frozen=True)
class StartupContext:
    """Everything the startup driver needs to construct and drive the stack.

    load_format is 'default', 'diffusers', or 'custom_pipeline'. policy=None
    disables offload. compile_enabled requires enforce_eager false and inductor
    support. faults is used only by fault-injection tests.
    """

    od_config: OmniDiffusionConfig
    vllm_config: VllmConfig
    device: torch.device
    rank: int
    local_rank: int
    world_size: int
    load_format: str
    custom_pipeline_name: str | None
    memory_pools: MemoryPools
    policy: OffloadPolicy | None
    compile_enabled: bool
    faults: FaultPlan | None


@dataclass(frozen=True)
class ExecutionOffloadFacts:
    """The offload-derived facts the engine and executors need at runtime
    (RFC 01 section 9.2). Produced by ``startup.execution_offload_facts`` so
    call sites do not branch on the selector themselves.

    weight_collectives: any selector uses AllGather, so DP ranks must execute
        blocks in the same order.
    text_encoder_allgather: the text-encoder role (or a component-name selector)
        uses AllGather.
    whole_component_offload: module-mode offload, so outputs move to CPU before
        postprocessing.
    """

    weight_collectives: bool
    text_encoder_allgather: bool
    whole_component_offload: bool


# --------------------------------------------------------------------------- #
# The six Protocols. Concrete classes live in the child RFCs noted above; the
# contracts are RFC 01 section 2's table.
# --------------------------------------------------------------------------- #
class Loader(Protocol):
    """Owns checkpoint and model semantics (RFC 02)."""

    def build_structure(self) -> tuple[nn.Module, ComponentCatalog]:
        """Construct the pipeline structure-only; declarations stay queryable
        after close()."""
        ...

    def preflight(self, catalog: ComponentCatalog, plan: PlacementPlan) -> Preflight:
        """Return per-component identities and the expected manifest without
        writing the model."""
        ...

    def write(
        self, model: nn.Module, storage: HostStorage, transaction: LoadTransaction
    ) -> None:
        """Obtain every target via transaction.target(key); a None target means
        allocate on the execution device in ctx.memory_pools.weights()."""
        ...

    def finalize(self, model: nn.Module, storage: HostStorage) -> FinalManifest:
        """Run finalization, call storage.seal(), return the final manifest."""
        ...

    def close(self) -> None:
        """Close construction context, traps, readers, temporaries. Idempotent."""
        ...


class Offloader(Protocol):
    """Owns runtime device transfers and resources (RFC 04)."""

    def plan(
        self,
        catalog: ComponentCatalog,
        topology: Topology,
        policy: OffloadPolicy | None,
    ) -> PlacementPlan:
        """Pure computation: no large allocations, model mutations, or hooks."""
        ...

    def prepare_materialization(
        self, model: nn.Module, plan: PlacementPlan, storage: HostStorage
    ) -> LoadTransaction:
        """On success the returned transaction owns storage and all runtime
        resources; on failure it releases what it allocated (storage stays with
        the caller)."""
        ...

    def setup_compile(
        self, model: nn.Module, catalog: ComponentCatalog, runtime: OffloadRuntime
    ) -> None:
        """Run after commit; select compile targets from declarations."""
        ...


class LoadTransaction(Protocol):
    """Sole owner of host storage + runtime resources between
    prepare_materialization and commit. Backed by a ResourceLedger (startup.py)."""

    def target(self, tensor_key: str) -> torch.Tensor | None:
        """storage.view(key) for host-resident tensors; None for permanently
        device-resident and HSDP-managed tensors."""
        ...

    def commit(self, final_manifest: FinalManifest) -> OffloadRuntime:
        """Transfer resources as one operation to the runtime (ledger.transfer),
        leaving an empty transaction. On failure, keep them for abort()."""
        ...

    def abort(self) -> None:
        """Release all transaction resources in ledger order. Idempotent; no
        action after commit."""
        ...


class OffloadRuntime(Protocol):
    """Owns device transfers in service; the post-commit owner of resources."""

    def close(self) -> None:
        """Idempotent, local, terminal; no CPU restoration or new collectives."""
        ...

    def grad_mode(self) -> AbstractContextManager[None]: ...
    def execution(self) -> AbstractContextManager[None]:
        """Enclose one forward; reject concurrent or reentrant use."""
        ...

    def before_sleep(self) -> None: ...
    def after_wake(self) -> None: ...


class HostWeightManager(Protocol):
    """Owns host memory and cross-worker sharing (RFC 03)."""

    def local_record(self) -> LocalRecord: ...
    def deployment_id(self) -> tuple[str, bool]:
        """Return (configured_id, True) or (secrets.token_hex(16), False). Only
        rank 0's result is used."""
        ...

    def layout(
        self, plan: PlacementPlan, expected_manifest: ExpectedManifest
    ) -> HostLayout: ...
    def share_plan(
        self, topology: Topology, fingerprints: Fingerprints
    ) -> SharePlan: ...
    def elect(self, share_plan: SharePlan) -> Election:
        """Called on every rank; nonrepresentatives return an empty election."""
        ...

    def admit(
        self,
        topology: Topology,
        share_plan: SharePlan,
        elections: Elections,
        layout: HostLayout,
    ) -> None:
        """Pure function reading only elections.records."""
        ...

    def allocate(
        self, layout: HostLayout, share_plan: SharePlan, elections: Elections
    ) -> HostStorage:
        """On success take ownership of elections.local's listener socket and
        thread; on failure election ownership stays with the caller."""
        ...


class HostStorage(Protocol):
    """The host memory holding final weight bytes (RFC 03)."""

    def view(self, tensor_key: str) -> torch.Tensor: ...
    def ranges(self, unit: Unit) -> list[MemoryRange]: ...
    def seal(self, final_manifest: FinalManifest) -> None: ...
    def connect_writer(self) -> None:
        """After point 2, connect waiters to writers; no action on nonwaiters."""
        ...

    def note_written(self, tensor_key: str) -> None:
        """Loader calls once per completed host-tensor write, synchronously, no
        collectives. Calls after sealing raise LifecycleError."""
        ...

    def close(self) -> None:
        """Idempotently release only this process's mappings."""
        ...
