# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for worker/runner dispatch (RFC 01 section 8).

Isolates 01-6's dispatch logic from the driver by stubbing run_unified_startup,
so these pin the WIRING (handle -> runner), the one-startup guard, idempotent
close, shutdown ordering, and the sleep/wake calls.
"""
from types import SimpleNamespace

import pytest

import vllm_omni.diffusion.weight_management.dispatch as dispatch
from vllm_omni.diffusion.weight_management.dispatch import (
    SleepMemoryPools,
    close_weight_management,
    sleep_unified,
    wake_unified,
    worker_shutdown_unified,
)
from vllm_omni.diffusion.weight_management.errors import LifecycleError
from vllm_omni.diffusion.weight_management.startup import (
    WeightManagement,
    process_weight_management,
    set_process_weight_management,
)


class _Handle:
    def __init__(self):
        self.model = object()
        self.closed = False
        self.in_service = False

    def mark_in_service(self):
        self.in_service = True

    def close(self):
        self.closed = True


def _runner():
    return SimpleNamespace(
        od_config=SimpleNamespace(
            enable_prompt_embed_cache=False, prompt_embed_cache_size=0
        ),
        weight_management=None,
        pipeline=None,
        offload_backend="stale",
        installed=None,
        _install_runtime_services=lambda e, s: None,
    )


def test_load_model_unified_wires_handle_onto_runner():
    runner = _runner()
    handle = _Handle()
    installed = []
    runner._install_runtime_services = lambda e, s: installed.append((e, s))

    saved = dispatch.run_unified_startup
    dispatch.run_unified_startup = lambda ctx, factory=None: handle
    try:
        dispatch.load_model_unified(runner, SleepMemoryPools(), "default")
    finally:
        dispatch.run_unified_startup = saved

    assert runner.weight_management is handle
    assert runner.pipeline is handle.model
    assert runner.offload_backend is None
    assert installed == [(False, 0)]
    assert handle.in_service is True


def test_load_model_unified_rejects_an_existing_stack():
    runner = _runner()
    runner.weight_management = _Handle()
    with pytest.raises(LifecycleError):
        dispatch.load_model_unified(runner, SleepMemoryPools(), "default")


def test_close_weight_management_is_idempotent():
    runner = _runner()
    handle = _Handle()
    runner.weight_management = handle

    close_weight_management(runner)
    assert handle.closed is True
    assert runner.weight_management is None
    assert runner.pipeline is None

    close_weight_management(runner)  # no error when nothing is installed


def test_worker_shutdown_clears_process_selector_state():
    worker = SimpleNamespace(model_runner=_runner())
    set_process_weight_management(WeightManagement.UNIFIED, owner=worker)
    assert process_weight_management() is WeightManagement.UNIFIED

    worker_shutdown_unified(worker)
    assert process_weight_management() is WeightManagement.LEGACY


def test_sleep_and_wake_call_runtime_hooks():
    events = []
    runtime = SimpleNamespace(
        before_sleep=lambda: events.append("before_sleep"),
        after_wake=lambda: events.append("after_wake"),
    )
    sleep_unified(runtime)
    wake_unified(runtime)
    assert events == ["before_sleep", "after_wake"]
    sleep_unified(None)  # no runtime (dummy load): no error


def test_sleep_pools_are_noops_without_sleep_mode():
    pools = SleepMemoryPools(sleep_enabled=False)
    with pools.weights():
        pass
    with pools.slots():
        pass
