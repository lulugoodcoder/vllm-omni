# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fault-injection and cleanup tests for the startup driver (RFC 01 §6/§10, PR 01-5).

Complements ``test_startup_driver.py`` (happy path + a couple of failures) with
the RFC 01 "Startup driver" test-plan matrix, driven through the realistic
``fakes.py`` doubles:

    * every named fault point raises the phase-specific error type (RFC 01 §3
      wrap table),
    * cleanup follows the current owner (RFC 01 §5.1): a transaction aborts, a
      host storage / election closes,
    * no bytes are written when a step before write (step 9) fails
      ("predictable errors precede first model write", invariant 1),
    * a missing factory reports ``StackUnavailableError`` with legacy guidance,
    * a compilation fault only warns and still returns COMMITTED (RFC 01 §6.2).

Runs under pytest; the ``__main__`` block self-runs through the study bypass
harness while this checkout's ``vllm_omni`` import chain is broken.
"""
from __future__ import annotations

import pytest

from tests.diffusion.weight_management.fakes import FakeStackFactory, build_fake_context
from vllm_omni.diffusion.weight_management.errors import (
    AllocationError,
    CommitError,
    ConstructionError,
    FinalizeError,
    GroupConfirmationError,
    PlanningError,
    PreflightError,
    PrepareError,
    StackUnavailableError,
    WriteError,
)
from vllm_omni.diffusion.weight_management.interfaces import LoadState
from vllm_omni.diffusion.weight_management.startup import FaultPlan, run_unified_startup


def _run(point, *, compile_enabled=False):
    """Drive a startup with a 'raise' fault at ``point``; return the log."""
    log: list[str] = []
    ctx = build_fake_context(
        faults=FaultPlan(points={point: (None, "raise")}),
        compile_enabled=compile_enabled,
    )
    return log, ctx, FakeStackFactory(log)


# --------------------------------------------------------------------------- #
# Every named fault point -> its phase-specific wrapped error (RFC 01 §3).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "point, error_type",
    [
        ("build_structure", ConstructionError),
        ("plan", PlanningError),
        ("preflight", PreflightError),
        ("point0_topology", GroupConfirmationError),
        ("allocate", AllocationError),
        ("prepare", PrepareError),
        ("point2_ready", GroupConfirmationError),
        ("connect_writer", AllocationError),
        ("write", WriteError),
        ("finalize", FinalizeError),
        ("commit", CommitError),
        ("point3_commit", GroupConfirmationError),
    ],
)
def test_fault_point_raises_phase_specific_error(point, error_type):
    log, ctx, factory = _run(point)
    with pytest.raises(error_type):
        run_unified_startup(ctx, factory)


# --------------------------------------------------------------------------- #
# Cleanup follows the current owner (RFC 01 §5.1).
# --------------------------------------------------------------------------- #
def test_write_fault_aborts_the_owning_transaction():
    log, ctx, factory = _run("write")
    with pytest.raises(WriteError):
        run_unified_startup(ctx, factory)
    assert any("transaction.abort" in line for line in log)
    assert any("loader.close" in line for line in log)


def test_prepare_fault_closes_host_storage():
    # By prepare, the owner is the host storage (allocate set it).
    log, ctx, factory = _run("prepare")
    with pytest.raises(PrepareError):
        run_unified_startup(ctx, factory)
    assert any("storage.close" in line for line in log)


def test_allocate_fault_closes_the_election():
    # allocate faults before it runs, so the owner is still the election.
    log, ctx, factory = _run("allocate")
    with pytest.raises(AllocationError):
        run_unified_startup(ctx, factory)
    assert any("election.close" in line for line in log)


# --------------------------------------------------------------------------- #
# Invariant 1: no bytes written when a pre-write step fails.
# --------------------------------------------------------------------------- #
def test_no_writes_when_a_pre_write_step_fails():
    log, ctx, factory = _run("plan")  # step 4, well before write (step 9)
    with pytest.raises(PlanningError):
        run_unified_startup(ctx, factory)
    assert not any("wrote " in line for line in log)  # zero materialised weights
    # A pre-transaction failure must NOT abort a transaction (there is none).
    assert not any("transaction.abort" in line for line in log)
    assert any("loader.close" in line for line in log)  # loader still released


# --------------------------------------------------------------------------- #
# Missing factory -> StackUnavailableError with legacy guidance (RFC 01 §13).
# --------------------------------------------------------------------------- #
def test_missing_factory_reports_stack_unavailable():
    ctx = build_fake_context()
    # No factory argument -> the default EntryPointFactory tries to importlib the
    # (nonexistent) RFC 02/03/04 modules and fails closed.
    with pytest.raises(StackUnavailableError) as exc:
        run_unified_startup(ctx)
    assert "legacy" in str(exc.value)  # fix: use --diffusion-weight-management legacy


# --------------------------------------------------------------------------- #
# Compilation fault only warns; startup still returns COMMITTED (RFC 01 §6.2).
# --------------------------------------------------------------------------- #
def test_compile_fault_warns_and_still_commits():
    log, ctx, factory = _run("compile_setup", compile_enabled=True)
    handle = run_unified_startup(ctx, factory)  # does NOT raise
    assert handle.state is LoadState.COMMITTED


if __name__ == "__main__":  # pragma: no cover - study bypass for the broken env
    import sys
    import traceback

    passed = failed = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_"):
            continue
        # Expand @parametrize manually for the standalone runner.
        marks = getattr(fn, "pytestmark", [])
        cases = []
        for m in marks:
            if m.name == "parametrize":
                argnames, argvalues = m.args
                names = [a.strip() for a in argnames.split(",")]
                cases = [dict(zip(names, v)) for v in argvalues]
        runs = cases or [{}]
        for kw in runs:
            label = f"{name}{tuple(kw.values()) if kw else ''}"
            try:
                fn(**kw)
                passed += 1
                print(f"PASS {label}")
            except Exception:  # noqa: BLE001
                failed += 1
                print(f"FAIL {label}")
                traceback.print_exc()
    print(f"\n{passed}/{passed + failed} passed")
    sys.exit(1 if failed else 0)
