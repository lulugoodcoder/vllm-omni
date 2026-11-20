# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for execution_offload_facts (RFC 01 section 9.2).

The unified branch is fully specified (all facts False until RFC 05). The legacy
branch is verified against stubbed offloader.config queries injected into
sys.modules, which also proves the function-local-import coexistence shape.
"""
import sys
import types
from dataclasses import dataclass

from vllm_omni.diffusion.weight_management.interfaces import ExecutionOffloadFacts
from vllm_omni.diffusion.weight_management.startup import execution_offload_facts


@dataclass
class Cfg:
    diffusion_weight_management: object = None


def test_unified_facts_are_all_false():
    facts = execution_offload_facts(Cfg(diffusion_weight_management="unified"))
    assert facts == ExecutionOffloadFacts(False, False, False)


def test_legacy_facts_map_the_offloader_config_queries():
    # Inject fake legacy modules at the exact dotted paths the function imports.
    base = types.ModuleType("vllm_omni.diffusion.offloader.base")

    class OffloadStrategy:
        NONE = "none"
        MODEL_LEVEL = "model_level"

    base.OffloadStrategy = OffloadStrategy

    cfg_mod = types.ModuleType("vllm_omni.diffusion.offloader.config")
    cfg_mod.any_selected_component_uses_allgather = lambda c: True
    cfg_mod.component_uses_allgather = lambda c, name: name == "text_encoder"
    cfg_mod.resolve_offload_strategy = lambda c: OffloadStrategy.MODEL_LEVEL

    # The parent package must exist for `from ...offloader import config` to
    # resolve, so inject it too and hang the submodules off it.
    offloader_pkg = types.ModuleType("vllm_omni.diffusion.offloader")
    offloader_pkg.__path__ = []
    offloader_pkg.config = cfg_mod
    offloader_pkg.base = base

    keys = ("vllm_omni.diffusion.offloader",
            "vllm_omni.diffusion.offloader.base",
            "vllm_omni.diffusion.offloader.config")
    saved = {k: sys.modules.get(k) for k in keys}
    sys.modules["vllm_omni.diffusion.offloader"] = offloader_pkg
    sys.modules["vllm_omni.diffusion.offloader.base"] = base
    sys.modules["vllm_omni.diffusion.offloader.config"] = cfg_mod
    try:
        facts = execution_offload_facts(Cfg(diffusion_weight_management="legacy"))
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    assert facts == ExecutionOffloadFacts(
        weight_collectives=True,
        text_encoder_allgather=True,
        whole_component_offload=True,
    )
