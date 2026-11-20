# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Static dependency tests (RFC 01 §11): T1 import direction + T8 frozen lists.

These are the capstone of coexistence: they MACHINE-ENFORCE the layering the
rest of the design relies on, by AST-scanning the weight_management package rather
than trusting reviewers. T1 is the keystone (import direction); T8 freezes the
allowlists by SHA256 so loosening a boundary fails the test.

STUDY NOTE: the real 01-7 implements T1-T8. This file implements the two that
are fully checkable against this study package: T1 and T8.
"""
import ast
import hashlib
from pathlib import Path

import pytest

from tests.diffusion.weight_management.coexistence import registry

PKG_DIR = Path(__file__).resolve().parents[3] / (
    "vllm_omni/diffusion/weight_management"
)


def _is_type_checking(test: ast.expr) -> bool:
    return isinstance(test, ast.Name) and test.id == "TYPE_CHECKING"


def _dotted(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Import):
        return [a.name for a in node.names]
    if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
        return [node.module]
    return []


def _analyze(path: Path):
    """Return (module_scope, all_imports, func_local) dotted module names.

    module_scope excludes imports under ``if TYPE_CHECKING:`` and inside
    functions; func_local is everything inside any function body.
    """
    tree = ast.parse(path.read_text())
    module_scope: list[str] = []
    for node in tree.body:  # top-level statements only
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            module_scope.extend(_dotted(node))
        # `if TYPE_CHECKING:` blocks are deliberately skipped (not runtime).

    all_imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            all_imports.extend(_dotted(node))

    func_local: list[str] = []
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for n in ast.walk(fn):
                if isinstance(n, (ast.Import, ast.ImportFrom)):
                    func_local.extend(_dotted(n))
    return module_scope, all_imports, func_local


def _modules():
    return {p.name: _analyze(p) for p in PKG_DIR.glob("*.py")}


# --------------------------------------------------------------------------- #
# T1: internal import direction (RFC 01 §11).
# --------------------------------------------------------------------------- #
def test_t1_no_module_imports_host_weight_runtime():
    for name, (_, all_imports, _) in _modules().items():
        offenders = [m for m in all_imports if m.startswith(
            "vllm_omni.host_weight_runtime")]
        assert not offenders, f"{name} imports host_weight_runtime: {offenders}"


def test_t1_errors_is_stdlib_only_at_module_scope():
    module_scope, _, _ = _analyze(PKG_DIR / "errors.py")
    heavy = [m for m in module_scope if m.startswith(("vllm", "torch"))]
    assert not heavy, f"errors.py must be stdlib-only; found {heavy}"


def test_t1_interfaces_module_scope_is_stdlib_torch_or_ws_errors():
    module_scope, _, _ = _analyze(PKG_DIR / "interfaces.py")
    for m in module_scope:
        if m.startswith("vllm"):
            assert m.startswith("vllm_omni.diffusion.weight_management.errors"), (
                f"interfaces.py may not import {m} at module scope")


def test_t1_startup_and_group_have_no_concrete_impl_at_module_scope():
    concrete = (
        "vllm_omni.diffusion.weight_management.loader",
        "vllm_omni.diffusion.weight_management.host",
        "vllm_omni.diffusion.weight_management.offload",
        "vllm_omni.diffusion.weight_management.policy",
    )
    for fname in ("startup.py", "group.py"):
        module_scope, _, _ = _analyze(PKG_DIR / fname)
        bad = [m for m in module_scope if m.startswith(concrete)]
        assert not bad, f"{fname} imports a concrete impl at module scope: {bad}"


def test_t1_offloader_config_only_in_startup_and_function_local():
    target = "vllm_omni.diffusion.offloader.config"
    for name, (module_scope, all_imports, func_local) in _modules().items():
        if target in all_imports:
            assert name == "startup.py", (
                f"only startup.py may import {target}; found in {name}")
            assert target not in module_scope, (
                f"{target} must be function-local, not module scope")
            assert target in func_local


# --------------------------------------------------------------------------- #
# T8: frozen allowlists (RFC 01 §11). Changing a list changes its SHA256.
# --------------------------------------------------------------------------- #
_FROZEN = {
    "OFFLOADER_IMPORT_ALLOWLIST":
        "0b95a2eec1d7ac3c3ecf47df82f6533c2a16f3cde9c5b2a4e575a7e67199a974",
    "HOOK_NAME_ALLOWLIST":
        "f7c02fce26542bd46b821a7099e9c7da3e36fd34035806296e93b1fc90f67b90",
    "CONFIG_FIELD_ALLOWLIST":
        "9e0062f6f5ff6416f6c1f1a5d726f69d967cddc890ae98827375c52212ba56e3",
    "QUERY_CALLER_ALLOWLIST":
        "74a0157cc6b9e8ce0b6ebdd60d27518f24432e3a6783a3f06e0b635427de6223",
}


@pytest.mark.parametrize("name,expected", sorted(_FROZEN.items()))
def test_t8_allowlists_are_frozen(name, expected):
    items = registry.ALLOWLISTS[name]
    digest = hashlib.sha256("\n".join(sorted(items)).encode()).hexdigest()
    assert digest == expected, (
        f"{name} changed; update only for a deliberate Job 6 removal")
