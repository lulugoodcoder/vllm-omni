# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Coexistence registry and frozen allowlists (RFC 01 §9.4, §11).

PR 01-7 establishes ``ROWS`` and the four allowlists; later PRs add
``sites/<row>_<PR>.py`` files WITHOUT editing this module. The allowlists are
frozen by SHA256 in ``test_static_dependencies.py`` (T8): any change fails the
test, so loosening a coexistence boundary is a deliberate, reviewed act.

STUDY NOTE: the real 01-7 lists all C01-C19/D01-D05 rows and the full 28-file
offloader-import allowlist. This is a representative subset carrying the exact
entries the RFC names, enough to study the mechanism.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class CoexistenceRow:
    """One legacy/unified dispatch point to be removed in Job 6 (RFC 01 §9.4).

    row_id: C01-C19 (RFC 02/04), D01-D06 (selector dispatch points).
    carrier: which RFC (01-05) owns the change.
    legacy_removal: the statements deleted when legacy is retired.
    """

    row_id: str
    title: str
    carrier: str
    legacy_removal: str


# RFC 01's own selector-dispatch rows (D01-D05). Each gets a site file in a
# later PR recording exactly where the selector is read.
ROWS: tuple[CoexistenceRow, ...] = (
    CoexistenceRow("D01", "Selector propagation and process state", "01",
                   "legacy diffusion_weight_management default path"),
    CoexistenceRow("D02", "Coexisting offload query functions", "01",
                   "legacy offloader.config query bodies"),
    CoexistenceRow("D03", "Worker load_model dispatch", "01",
                   "legacy DiffusionModelRunner.load_model body"),
    CoexistenceRow("D04", "Worker shutdown dispatch", "01",
                   "legacy offload_backend.shutdown call"),
    CoexistenceRow("D05", "Sleep / wake dispatch", "01",
                   "legacy sleep/wake offload statements"),
)

# T3: non-weight-management code allowed to import the legacy Offloader. Real list is
# 28 model files + lora/manager.py; the carrier entry below is the one the RFC
# names explicitly. Paths are relative to vllm_omni/diffusion/.
OFFLOADER_IMPORT_ALLOWLIST: tuple[str, ...] = (
    "lora/manager.py",
)

# T4: files allowed to use offload hook-name string literals.
HOOK_NAME_ALLOWLIST: tuple[str, ...] = (
    "cache/seacache/hook.py",
)

# T5: files allowed to read offload/host/selector config fields directly.
# No allowlist entry ever permits reading `diffusion_weight_management`.
CONFIG_FIELD_ALLOWLIST: tuple[str, ...] = (
    "models/diffusers_adapter/pipeline_diffusers_adapter.py",
    "models/joy_image/pipeline_joy_image_edit.py",
    "models/magi2/pipeline_magi2.py",
    "models/minimax_h3/pipeline_minimax_h3.py",
)

# T7: the only non-model callers of the §9.1 offload queries. Frozen after 01-3.
QUERY_CALLER_ALLOWLIST: tuple[str, ...] = (
    "OmniDiffusionConfig.__post_init__",
    "DiffusionModelRunner.load_model",
    "DiffusionModelRunner._execute_request_list",
    "DiffusersPipelineLoader.load_model",
    "startup.execution_offload_facts",
)

ALLOWLISTS: dict[str, tuple[str, ...]] = {
    "OFFLOADER_IMPORT_ALLOWLIST": OFFLOADER_IMPORT_ALLOWLIST,
    "HOOK_NAME_ALLOWLIST": HOOK_NAME_ALLOWLIST,
    "CONFIG_FIELD_ALLOWLIST": CONFIG_FIELD_ALLOWLIST,
    "QUERY_CALLER_ALLOWLIST": QUERY_CALLER_ALLOWLIST,
}
