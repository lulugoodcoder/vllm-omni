# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The unified Loader package (RFC 02 "Unified Loader", #9875).

STUDY REFERENCE. Not for upstream merge.

Re-exports the names RFC 01's ``interfaces.py`` imports from this package
(``Preflight``, ``ExpectedManifest``, ``FinalManifest``) and the
``create_loader(ctx)`` entry point that 01's ``EntryPointFactory`` resolves.

Reading order for the slice (each file is independently debuggable):
    context.py   -- §3  structure-only construction (meta parameters)
    trap.py      -- §4  read traps (no checkpoint reads during construction)
    catalog.py   -- §2  tensors assigned to components by path
    manifest.py  -- §10 content identity + manifests (the sharing decision)
    write.py     -- §7  swap_tensors binding + write tracking
    loader.py    -- §1  the UnifiedLoader lifecycle that composes them all
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from vllm_omni.diffusion.weight_management.loader.loader import Preflight, UnifiedLoader
from vllm_omni.diffusion.weight_management.loader.manifest import (
    ExpectedManifest,
    FinalManifest,
)

if TYPE_CHECKING:
    from vllm_omni.diffusion.weight_management.interfaces import Loader, StartupContext

__all__ = [
    "Preflight",
    "ExpectedManifest",
    "FinalManifest",
    "UnifiedLoader",
    "create_loader",
]


def create_loader(ctx: "StartupContext") -> "Loader":
    """Resolve the unified Loader for a startup (RFC 02 §1).

    In the real repo this reads ``ctx`` (od_config/load_format/custom_pipeline)
    and selects the diffusers/custom/registered pipeline factory, wiring the real
    checkpoint reader. This study build raises if called without those wired --
    use ``UnifiedLoader(...)`` directly with injected fakes, as the RFC-02
    ``debug_flow`` does.
    """
    raise NotImplementedError(
        "create_loader needs the real pipeline registry + checkpoint reader; "
        "the study slice constructs UnifiedLoader(...) directly with injected "
        "fakes (see the RFC-02 debug flow)."
    )
