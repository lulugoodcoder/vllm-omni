# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unified weight management: Loader, Host Weight Manager, Offloader, and startup.

Umbrella RFC #9536; this package is built by child RFCs 01-05. The four
responsibilities map to umbrella RFC section "Architecture":
    Loader    -> weight semantics / manifest   (RFC 02)
    HWM       -> host memory + sharing          (RFC 03)
    Offloader -> device transfers               (RFC 04)
    startup   -> the driver that composes them  (RFC 01, this PR 01-1)

Selection and coexistence (RFC 01 section 1): the unified stack is chosen by
``--diffusion-weight-management unified``. Until the release gates flip the default,
it runs ALONGSIDE the legacy offloader, whose entry points are:
    - vllm_omni/diffusion/offloader/__init__.py:32  get_offload_backend()
    - vllm_omni/diffusion/offloader/base.py:42-60   the boolean selector
      (``enable_cpu_offload`` / ``enable_layerwise_offload``) this replaces.
"""
from vllm_omni.diffusion.weight_management.errors import (
    AllocationError,
    CommitError,
    ConfigError,
    ConfirmationTimeoutError,
    ConstructionError,
    DeviceCapacityError,
    ErrorSummary,
    FinalizeError,
    GroupConfirmationError,
    HostCapacityError,
    LifecycleError,
    PeerFailureError,
    PlanningError,
    PrepareError,
    PreflightError,
    RuntimeFault,
    StackUnavailableError,
    WeightManagementError,
    WriteError,
)
from vllm_omni.diffusion.weight_management.interfaces import (
    AllGatherGroup,
    AllGatherGroups,
    Elections,
    ExecutionOffloadFacts,
    Fingerprints,
    HostStorage,
    HostWeightManager,
    Loader,
    LoadState,
    LoadTransaction,
    MemoryPools,
    Offloader,
    OffloadRuntime,
    ParallelCoordinate,
    RankCoordinates,
    RankFingerprints,
    StartupContext,
    Topology,
)
from vllm_omni.diffusion.weight_management.startup import (
    EntryPointFactory,
    FaultPlan,
    InjectedFault,
    ResourceKind,
    ResourceLedger,
    StackFactory,
    WeightManagement,
    WeightManagementHandle,
    check_unified_offload_conflicts,
    clear_process_weight_management,
    execution_offload_facts,
    is_unified,
    parse_weight_management,
    process_weight_management,
    run_unified_startup,
    set_process_weight_management,
)

__all__ = [
    "AllGatherGroup",
    "AllGatherGroups",
    "Elections",
    "EntryPointFactory",
    "ExecutionOffloadFacts",
    "FaultPlan",
    "InjectedFault",
    "Fingerprints",
    "HostStorage",
    "HostWeightManager",
    "LoadState",
    "LoadTransaction",
    "Loader",
    "MemoryPools",
    "OffloadRuntime",
    "Offloader",
    "ParallelCoordinate",
    "RankCoordinates",
    "RankFingerprints",
    "StartupContext",
    "Topology",
    "AllocationError",
    "CommitError",
    "ConfigError",
    "ConfirmationTimeoutError",
    "ConstructionError",
    "DeviceCapacityError",
    "ErrorSummary",
    "FinalizeError",
    "GroupConfirmationError",
    "HostCapacityError",
    "LifecycleError",
    "PeerFailureError",
    "PlanningError",
    "PrepareError",
    "PreflightError",
    "ResourceKind",
    "ResourceLedger",
    "RuntimeFault",
    "StackFactory",
    "WeightManagement",
    "WeightManagementHandle",
    "check_unified_offload_conflicts",
    "clear_process_weight_management",
    "execution_offload_facts",
    "is_unified",
    "parse_weight_management",
    "process_weight_management",
    "run_unified_startup",
    "set_process_weight_management",
    "StackUnavailableError",
    "WeightManagementError",
    "WriteError",
]
