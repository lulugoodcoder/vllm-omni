# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the weight-management selector (RFC 01 section 1).

Covers parsing, the config vs process-level readers, the multi-owner process
state (two stages in one process must agree), and the temporary unified-mode
offload rejection in section 1.3.
"""
from dataclasses import dataclass

import pytest

from vllm_omni.diffusion.weight_management.errors import ConfigError
from vllm_omni.diffusion.weight_management.startup import (
    WeightManagement,
    check_unified_offload_conflicts,
    clear_process_weight_management,
    is_unified,
    parse_weight_management,
    process_weight_management,
    set_process_weight_management,
)


@dataclass
class Cfg:
    diffusion_weight_management: object = None
    enable_cpu_offload: bool = False
    diffusion_offload_config: object = None


@pytest.fixture(autouse=True)
def _reset_process_state():
    # Process state is a module global; isolate every test from the next.
    yield
    import vllm_omni.diffusion.weight_management.startup as s
    s._process_stack = None
    s._process_owners.clear()


def test_parse_none_is_legacy():
    assert parse_weight_management(None) is WeightManagement.LEGACY


def test_parse_valid_strings():
    assert parse_weight_management("legacy") is WeightManagement.LEGACY
    assert parse_weight_management("unified") is WeightManagement.UNIFIED


def test_parse_invalid_raises_config_error():
    with pytest.raises(ConfigError):
        parse_weight_management("turbo")


def test_is_unified_reads_config_and_defaults_legacy():
    assert is_unified(Cfg(diffusion_weight_management="unified")) is True
    assert is_unified(Cfg(diffusion_weight_management="legacy")) is False
    assert is_unified(object()) is False  # missing attribute -> legacy


def test_process_state_defaults_to_legacy_when_unset():
    assert process_weight_management() is WeightManagement.LEGACY


def test_process_state_set_and_clear_by_owner():
    owner = object()
    set_process_weight_management(WeightManagement.UNIFIED, owner=owner)
    assert process_weight_management() is WeightManagement.UNIFIED
    clear_process_weight_management(owner=owner)
    assert process_weight_management() is WeightManagement.LEGACY


def test_process_state_reverts_only_after_last_owner_clears():
    a, b = object(), object()
    set_process_weight_management(WeightManagement.UNIFIED, owner=a)
    set_process_weight_management(WeightManagement.UNIFIED, owner=b)  # same value: fine
    clear_process_weight_management(owner=a)
    assert process_weight_management() is WeightManagement.UNIFIED  # b still holds it
    clear_process_weight_management(owner=b)
    assert process_weight_management() is WeightManagement.LEGACY


def test_conflicting_owners_in_one_process_raise():
    set_process_weight_management(WeightManagement.UNIFIED, owner=object())
    with pytest.raises(ConfigError):
        set_process_weight_management(WeightManagement.LEGACY, owner=object())


def test_offload_check_noop_in_legacy_mode():
    # No raise even though offload flags are on, because mode is legacy.
    check_unified_offload_conflicts(Cfg(enable_cpu_offload=True))


def test_offload_check_rejects_flag_in_unified_mode():
    cfg = Cfg(diffusion_weight_management="unified", enable_cpu_offload=True)
    with pytest.raises(ConfigError):
        check_unified_offload_conflicts(cfg)


def test_offload_check_rejects_offload_config_in_unified_mode():
    cfg = Cfg(diffusion_weight_management="unified", diffusion_offload_config={"x": 1})
    with pytest.raises(ConfigError):
        check_unified_offload_conflicts(cfg)
