# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Contract tests for ResourceLedger (RFC 01 section 5.2).

These assert the ownership guarantees the unified startup depends on:
exactly-once release, ascending-kind / reverse-registration order,
double-register rejection, hand-over, transfer, and close-time exception
aggregation.
"""
import pytest

from vllm_omni.diffusion.weight_management.errors import LifecycleError
from vllm_omni.diffusion.weight_management.startup import ResourceKind, ResourceLedger


def _recorder(log: list, tag: str):
    def close() -> None:
        log.append(tag)

    return close


def test_close_orders_by_ascending_kind_then_reverse_registration():
    log: list[str] = []
    ledger = ResourceLedger()
    # Register out of kind order and with two entries sharing one kind.
    ledger.register(object(), ResourceKind.HOST_STORAGE, _recorder(log, "host"))
    ledger.register(object(), ResourceKind.EXECUTION, _recorder(log, "exec"))
    ledger.register(object(), ResourceKind.DEVICE, _recorder(log, "device_a"))
    ledger.register(object(), ResourceKind.DEVICE, _recorder(log, "device_b"))

    ledger.close()

    # Ascending kind: EXECUTION(1) < DEVICE(4) < HOST_STORAGE(8).
    # Within DEVICE, reverse registration order: device_b before device_a.
    assert log == ["exec", "device_b", "device_a", "host"]


def test_close_is_idempotent_and_releases_once():
    log: list[str] = []
    ledger = ResourceLedger()
    ledger.register(object(), ResourceKind.DEVICE, _recorder(log, "x"))

    ledger.close()
    ledger.close()

    assert log == ["x"]


def test_register_same_handle_twice_raises():
    ledger = ResourceLedger()
    handle = object()
    ledger.register(handle, ResourceKind.DEVICE, lambda: None)
    with pytest.raises(LifecycleError):
        ledger.register(handle, ResourceKind.STAGING, lambda: None)


def test_hand_over_removes_entry_so_close_skips_it():
    log: list[str] = []
    ledger = ResourceLedger()
    handle = object()
    ledger.register(handle, ResourceKind.PLACEHOLDER, _recorder(log, "overlay"))
    ledger.register(object(), ResourceKind.DEVICE, _recorder(log, "device"))

    ledger.hand_over(handle)
    ledger.close()

    assert log == ["device"]


def test_hand_over_unknown_handle_raises():
    ledger = ResourceLedger()
    with pytest.raises(LifecycleError):
        ledger.hand_over(object())


def test_transfer_moves_entries_and_seals_original():
    log: list[str] = []
    ledger = ResourceLedger()
    ledger.register(object(), ResourceKind.DEVICE, _recorder(log, "device"))

    new = ledger.transfer()

    # Original is emptied and rejects further registration.
    with pytest.raises(LifecycleError):
        ledger.register(object(), ResourceKind.STAGING, lambda: None)
    ledger.close()  # no-op, nothing left to release
    assert log == []

    # The new ledger owns and releases the resource.
    new.close()
    assert log == ["device"]


def test_close_runs_all_callbacks_then_raises_with_first_cause():
    log: list[str] = []
    ledger = ResourceLedger()
    first = RuntimeError("boom-device")

    def raise_device() -> None:
        raise first

    ledger.register(object(), ResourceKind.EXECUTION, _recorder(log, "exec"))
    ledger.register(object(), ResourceKind.DEVICE, raise_device)
    ledger.register(object(), ResourceKind.HOST_STORAGE, _recorder(log, "host"))

    with pytest.raises(LifecycleError) as excinfo:
        ledger.close()

    # Every remaining callback still ran despite the failure in the middle.
    assert log == ["exec", "host"]
    assert excinfo.value.__cause__ is first
