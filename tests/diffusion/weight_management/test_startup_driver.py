# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the startup driver (RFC 01 sections 4, 6, 7).

Exercises the full 13-step lifecycle at world_size=1 (which RFC 01 §7.1 runs
locally with no torch.distributed) using test doubles for Loader / HWM /
Offloader. Covers the happy path to COMMITTED, handle close, and two failure
paths -- one before any owner exists, one after the transaction owns resources
(proving release() aborts the transaction).
"""
from types import SimpleNamespace

import pytest

from vllm_omni.diffusion.weight_management.errors import (
    ConstructionError,
    PlanningError,
    WriteError,
)
from vllm_omni.diffusion.weight_management.interfaces import LoadState
from vllm_omni.diffusion.weight_management.startup import (
    FaultPlan,
    StackFactory,
    run_unified_startup,
)


# --------------------------------------------------------------------------- #
# Minimal doubles: only the methods the driver actually calls.
# --------------------------------------------------------------------------- #
class _Plan:
    def collective_signature(self):
        return {}

    def unmatched_policy_keys(self):
        return frozenset()

    def bind_groups(self, groups):
        pass


class _Layout:
    plan_fingerprint = "fp"

    def bind_groups(self, groups):
        pass


class _Election:
    def record(self):
        return "election-record"

    def close(self):
        pass


class _Storage:
    def __init__(self, calls):
        self._calls = calls

    def connect_writer(self):
        self._calls.append("connect_writer")

    def close(self):
        self._calls.append("storage.close")


class _Runtime:
    def __init__(self, calls):
        self._calls = calls

    def close(self):
        self._calls.append("runtime.close")


class _Transaction:
    def __init__(self, calls):
        self._calls = calls

    def commit(self, manifest):
        return _Runtime(self._calls)

    def abort(self):
        self._calls.append("transaction.abort")


class FakeLoader:
    def __init__(self, calls, fail_build=False, fail_write=False):
        self._calls = calls
        self._fail_build = fail_build
        self._fail_write = fail_write

    def build_structure(self):
        if self._fail_build:
            raise RuntimeError("boom in build_structure")
        return object(), object()  # model, catalog

    def preflight(self, catalog, plan):
        return SimpleNamespace(content_identities={}, expected_manifest=object())

    def write(self, model, storage, transaction):
        if self._fail_write:
            raise RuntimeError("boom in write")

    def finalize(self, model, storage):
        return object()

    def close(self):
        self._calls.append("loader.close")


class FakeHWM:
    def __init__(self, calls):
        self._calls = calls

    def local_record(self):
        return object()

    def deployment_id(self):
        return ("a" * 32, True)

    def layout(self, plan, expected_manifest):
        return _Layout()

    def share_plan(self, topology, fingerprints):
        return object()

    def elect(self, share_plan):
        return _Election()

    def admit(self, topology, share_plan, elections, layout):
        pass

    def allocate(self, layout, share_plan, elections):
        return _Storage(self._calls)


class FakeOffloader:
    def __init__(self, calls):
        self._calls = calls

    def plan(self, catalog, topology, policy):
        return _Plan()

    def prepare_materialization(self, model, plan, storage):
        return _Transaction(self._calls)

    def setup_compile(self, model, catalog, runtime):
        pass


class FakeFactory(StackFactory):
    def __init__(self, calls, fail_build=False, fail_write=False):
        self._calls = calls
        self._fail_build = fail_build
        self._fail_write = fail_write

    def create(self, ctx):
        return (
            FakeLoader(self._calls, self._fail_build, self._fail_write),
            FakeHWM(self._calls),
            FakeOffloader(self._calls),
        )


def _ctx():
    return SimpleNamespace(
        od_config=SimpleNamespace(stage_id=0),
        vllm_config=None,
        device=None,
        rank=0,
        local_rank=0,
        world_size=1,
        load_format="default",
        custom_pipeline_name=None,
        memory_pools=None,
        policy=None,
        compile_enabled=False,
        faults=None,
    )


def test_happy_path_reaches_committed_then_in_service_then_closed():
    calls = []
    handle = run_unified_startup(_ctx(), FakeFactory(calls))

    assert handle.state is LoadState.COMMITTED
    assert handle.model is not None
    assert "connect_writer" in calls
    assert "loader.close" in calls  # finalize() closes the loader

    handle.mark_in_service()
    assert handle.state is LoadState.IN_SERVICE

    handle.close()
    assert handle.state is LoadState.CLOSED
    assert "runtime.close" in calls
    handle.close()  # idempotent


def test_failure_in_build_structure_raises_construction_error():
    calls = []
    with pytest.raises(ConstructionError):
        run_unified_startup(_ctx(), FakeFactory(calls, fail_build=True))
    # No owner yet, but the loader must be closed during release.
    assert "loader.close" in calls


def test_failure_in_write_aborts_the_owning_transaction():
    calls = []
    with pytest.raises(WriteError):
        run_unified_startup(_ctx(), FakeFactory(calls, fail_write=True))
    # By the write step the transaction owns resources, so release() aborts it.
    assert "transaction.abort" in calls
    assert "loader.close" in calls


def test_fault_injection_forces_a_step_to_fail():
    # FaultPlan (RFC 01 §10) lets a test force a failure at any named point;
    # the driver wraps the injected fault in that step's error type.
    ctx = _ctx()
    ctx.faults = FaultPlan(points={"plan": (None, "raise")})
    with pytest.raises(PlanningError):
        run_unified_startup(ctx, FakeFactory([]))
