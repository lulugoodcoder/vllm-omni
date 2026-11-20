# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker and model-runner dispatch (study module).

PR 01-6 of RFC 01 (child of umbrella RFC #9536).
    - Spec: RFC 01 section 8 (loading §8.1, sleep pools §8.2, shutdown §8.3,
      sleep/wake §8.4, custom-pipeline reinit §8.5).

STUDY NOTE: the RFC inlines this logic directly into the real
``DiffusionWorker`` and ``DiffusionModelRunner`` (which differ from the RFC
baseline in this checkout). This module models the SAME dispatch as small,
testable functions over a duck-typed ``runner``/``worker`` so the legacy/unified
branch points and the 11-step unified load sequence can be studied in isolation.
The real PR adds branches to existing methods; it does not add this file.
"""
from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING

from vllm_omni.diffusion.weight_management.errors import LifecycleError
from vllm_omni.diffusion.weight_management.startup import (
    FaultPlan,
    StackFactory,
    is_unified,
    run_unified_startup,
)

if TYPE_CHECKING:
    from vllm_omni.diffusion.weight_management.interfaces import MemoryPools


class SleepMemoryPools:
    """The two accelerator pools the worker hands the runner (RFC 01 §8.2).

    weights() is backed up during sleep; slots() is released without backup.
    Without sleep mode both are no-ops. With sleep mode they enter the CuMem
    allocator's tagged pools ("weights" / "weight_slots").
    """

    def __init__(self, allocator: object | None = None, sleep_enabled: bool = False):
        self._allocator = allocator
        self._sleep_enabled = sleep_enabled

    def weights(self):
        if not self._sleep_enabled or self._allocator is None:
            return nullcontext()
        return self._allocator.use_memory_pool(tag="weights")

    def slots(self):
        if not self._sleep_enabled or self._allocator is None:
            return nullcontext()
        return self._allocator.use_memory_pool(tag="weight_slots")


def _build_startup_context(runner, memory_pools, load_format, custom_pipeline_name):
    """Assemble StartupContext from the runner (RFC 01 §8.1 step 5).

    policy is None before RFC 05-3 lands; faults come from the environment.
    """
    from vllm_omni.diffusion.weight_management.interfaces import StartupContext

    od = runner.od_config
    return StartupContext(
        od_config=od,
        vllm_config=getattr(runner, "vllm_config", None),
        device=getattr(runner, "device", None),
        rank=getattr(runner, "rank", 0),
        local_rank=getattr(runner, "local_rank", 0),
        world_size=getattr(runner, "world_size", 1),
        load_format=load_format,
        custom_pipeline_name=custom_pipeline_name,
        memory_pools=memory_pools,
        policy=None,  # RFC 05-3 supplies this later
        compile_enabled=getattr(runner, "compile_enabled", False),
        faults=FaultPlan.from_environ(),
    )


def load_model_unified(
    runner,
    memory_pools: MemoryPools,
    load_format: str,
    custom_pipeline_name: str | None = None,
    factory: StackFactory | None = None,
) -> None:
    """The unified branch of ``DiffusionModelRunner.load_model`` (RFC 01 §8.1).

    Mirrors the 11-step sequence. ``factory`` is a study hook: in the real code
    ``run_unified_startup(ctx)`` uses the default EntryPointFactory.

    Raises:
        LifecycleError: If a runtime is already installed (step 1).
    """
    # 1. The old runtime must be closed first.
    if getattr(runner, "weight_management", None) is not None:
        raise LifecycleError(
            "a weight management is already installed; close it first",
            where="load_model",
        )
    # 2. Read prompt-embed-cache config.
    enable_pec = getattr(runner.od_config, "enable_prompt_embed_cache", False)
    pec_size = getattr(runner.od_config, "prompt_embed_cache_size", 0)
    # 3-4. platform runtime init + observability.bind_context: real code only.
    # 5. Build the startup context.
    ctx = _build_startup_context(
        runner, memory_pools, load_format, custom_pipeline_name
    )
    # 6. Run the driver (real code wraps this in DeviceMemoryProfiler).
    handle = run_unified_startup(ctx, factory)
    # 7. Wire the handle onto the runner.
    runner.weight_management = handle
    runner.pipeline = handle.model
    runner.offload_backend = None
    # 8. (step-execution / streaming-output checks: real code only)
    # 9. Install shared runtime services (cache, prompt-embed cache, coordinator).
    if hasattr(runner, "_install_runtime_services"):
        runner._install_runtime_services(enable_pec, pec_size)
    # 10. observability.log_startup_summary: real code only.
    # 11. Enter service.
    handle.mark_in_service()


def close_weight_management(runner) -> None:
    """Take and clear the runner's handle, close it, drop the pipeline (§8.3).

    Idempotent: safe to call when nothing is installed.
    """
    handle = getattr(runner, "weight_management", None)
    runner.weight_management = None
    if handle is not None:
        handle.close()
    runner.pipeline = None


def worker_shutdown_unified(worker) -> None:
    """The unified-specific DiffusionWorker.shutdown steps (RFC 01 §8.3).

    Order within the worker's full shutdown: step 2 (close_weight_management),
    step 4 (destroy point-1 groups), step 6 (clear process selector state).
    Each wrapped so a failure still lets the rest of cleanup run.
    """
    from vllm_omni.diffusion.weight_management.group import destroy_allgather_groups
    from vllm_omni.diffusion.weight_management.startup import clear_process_weight_management

    try:  # step 2
        close_weight_management(worker.model_runner)
    except Exception:  # noqa: BLE001 - log-and-continue, like existing steps
        pass
    try:  # step 4
        destroy_allgather_groups()
    except Exception:  # noqa: BLE001
        pass
    # step 6 (after destroy_distributed_env in the real order)
    clear_process_weight_management(owner=worker)


def sleep_unified(runtime) -> None:
    """Call runtime.before_sleep() before the allocator sleeps (RFC 01 §8.4)."""
    if runtime is not None:
        runtime.before_sleep()


def wake_unified(runtime) -> None:
    """Call runtime.after_wake() after wake statements (RFC 01 §8.4)."""
    if runtime is not None:
        runtime.after_wake()
