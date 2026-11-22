# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The component catalog: tensors assigned to components (RFC 02 §2.7).

STUDY REFERENCE for RFC 02 "Unified Loader" (#9875). Not for upstream merge.

This is the backbone that ties the two earlier files together. ``components.py``
gave the *vocabulary* a model author writes (``ComponentDecl``); ``context.py``
gave a *meta model* (structure, no weight bytes). This file does the assembly
the RFC puts in ``loader/catalog.py`` (§2.7 "types live in components; assembly
lives in loader/catalog.py"): walk every parameter / persistent buffer of the
meta model and decide **which component owns it**, by module path.

Why ownership matters: every later decision in the stack is *per component* --
how a component is sharded, whether its weights are shared across ranks, which
rank writes it, where it lives (device/host/offloaded). None of that can happen
until each tensor has exactly one owner. So the catalog is the hinge between
"a model exists" and "the loader/offloader can plan".

The owner rule (RFC 02 §2.3): a tensor ``t`` belongs to the component whose
declared root is the **longest** prefix of ``t``. Nested roots carve out of
their encloser -- Cosmos3's ``transformer.language_model`` (a text encoder) is
carved out of ``transformer`` (the DiT), so ``transformer.language_model.*``
tensors go to the text encoder, not the DiT, purely because its root is longer.
A tensor under *no* declared root is a **missing-ownership** violation; the
catalog records it rather than raising, and ``require_complete()`` (what the
Offloader calls first, §2.4) turns the whole violation list into one
``PlanningError``.

The catalog is *mostly frozen*. RFC 02 §2.7 allows exactly three post-build
writes, each once: ``set_discovered_blocks`` (04 planning), ``bind_overlay_slot``
(04 commit), ``set_fully_resident`` (commit). A second write of any of them is a
``LifecycleError`` -- same single-writer discipline as RFC 01's selector.

Where the class lives: the real repo keeps the ``ComponentCatalog`` *class* (and
the ``TensorDecl``/``CatalogEntry`` types) in ``components.py`` -- 01's
``interfaces.py`` imports ``ComponentCatalog`` from there -- and puts only the
*builder* in ``loader/catalog.py``. This study slice co-locates the class with
its builder here, so you read "the thing" and "how it's assembled" together; at
runtime it makes no difference (every cross-file reference is a TYPE_CHECKING
string annotation).

Deferred from the real §2.7 (noted): tied-tensor alias dedup (§2.3), quant /
fixed-overlay tensor kinds and ``dtype_violations`` (§2.6/§10), rule-table names,
``shapes_final`` / ``finalize_replaced`` bookkeeping (§3.5 step 8), dynamic-LoRA
slots (§11). This slice keeps the load-bearing idea -- ownership by path +
completeness + the three-writes rule -- as real, debuggable code.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import torch
    import torch.nn as nn

    from vllm_omni.diffusion.weight_management.components import ComponentDecl


# --------------------------------------------------------------------------- #
# §2.7 Catalog data types.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TensorDecl:
    """One tensor the catalog knows about (RFC 02 §2.7).

    ``name`` is the canonical pre-wrapping name (e.g.
    ``transformer.blocks.0.attn.to_q.weight``). ``module`` is its owning module
    path. ``kind`` distinguishes a parameter from a persistent buffer (the real
    RFC also has ``data``/``quant_output``/``fixed_overlay``, deferred here).
    ``required`` marks a source-required write.
    """

    name: str
    module: str
    kind: Literal["parameter", "buffer"]
    shape: tuple[int, ...]
    dtype: "torch.dtype"
    required: bool = True
    aliases: tuple[str, ...] = ()
    source: str | None = None


@dataclass
class CatalogEntry:
    """Everything the catalog knows about one component (RFC 02 §2.7).

    Mutable because planning/commit fill three fields later (discovered blocks,
    residency); the rest is frozen at build time.
    """

    decl: "ComponentDecl"
    roots: tuple[str, ...]
    carved: tuple[str, ...]  # inner roots removed from this component (§2.3)
    tensors: tuple[TensorDecl, ...]
    # Filled once by set_discovered_blocks (04 planning):
    discovered_blocks: tuple[str, ...] | None = None
    # Filled once by set_fully_resident (commit); None before commit (§2.7):
    fully_resident: bool | None = None


@dataclass(frozen=True)
class OverlaySlotDecl:
    """A dynamic-overlay slot, owned by no component (RFC 02 §2.7/§11)."""

    name: str
    layer: str
    shape: tuple[int, ...]
    dtype: "torch.dtype"


# --------------------------------------------------------------------------- #
# §2.7 ComponentCatalog: the queryable result the Offloader reads.
# --------------------------------------------------------------------------- #
class ComponentCatalog:
    """Tensors grouped by owning component, plus the three allowed writes.

    Built by ``build_catalog`` below; read by the Offloader (which imports this,
    not the Loader) and by the query functions in ``components.py``. Ownership is
    frozen at build time; everything mutable goes through the three guarded
    setters so the single-writer rule (RFC 02 §2.7) holds.
    """

    def __init__(
        self,
        entries: dict[str, CatalogEntry],
        declaration_order: tuple[str, ...],
        ownership_violations: tuple[str, ...],
    ) -> None:
        self.entries = entries
        self.declaration_order = declaration_order
        self.ownership_violations = ownership_violations
        self.dtype_violations: tuple[str, ...] = ()  # (deferred; see module docstring)
        self.overlay_slots: tuple[OverlaySlotDecl, ...] = ()
        self._overlay_bindings: dict[str, object] = {}
        self._lock = threading.Lock()
        # tensor name -> owning component, precomputed for O(1) owner lookup.
        self._owner_of: dict[str, str] = {}
        for comp_name, entry in entries.items():
            for t in entry.tensors:
                self._owner_of[t.name] = comp_name

    # -- completeness (RFC 02 §2.4) ---------------------------------------- #
    def require_complete(self) -> None:
        """Raise if any ownership (or dtype) violation was recorded (§2.4).

        The Offloader calls this *first*, before planning. One raise lists every
        offending path, so a model author fixes all gaps in one pass instead of
        whack-a-mole.
        """
        from vllm_omni.diffusion.weight_management.errors import PlanningError

        problems = self.ownership_violations + self.dtype_violations
        if problems:
            raise PlanningError(
                f"catalog is incomplete: {len(problems)} path(s) without exactly "
                f"one owner / with a bad dtype",
                where="ComponentCatalog.require_complete",
                fix="declare a component root covering each listed path:\n  "
                + "\n  ".join(problems),
            )

    # -- lookups (RFC 02 §2.7) --------------------------------------------- #
    def owner(self, tensor_name: str) -> str:
        """The component that owns ``tensor_name`` (RFC 02 §2.7)."""
        from vllm_omni.diffusion.weight_management.errors import LifecycleError

        comp = self._owner_of.get(tensor_name)
        if comp is None:
            raise LifecycleError(
                f"no component owns tensor {tensor_name!r}",
                where="ComponentCatalog.owner",
            )
        return comp

    def module_owner(self, module_path: str) -> str:
        """The component owning a module path, by longest-root match (§2.7)."""
        from vllm_omni.diffusion.weight_management.errors import LifecycleError

        best: tuple[int, str] | None = None
        for comp_name, entry in self.entries.items():
            for root in entry.roots:
                if _covers(root, module_path) and (best is None or len(root) > best[0]):
                    best = (len(root), comp_name)
        if best is None:
            raise LifecycleError(
                f"no component owns module {module_path!r}",
                where="ComponentCatalog.module_owner",
            )
        return best[1]

    # -- the three allowed writes (RFC 02 §2.7) ---------------------------- #
    def set_discovered_blocks(self, component: str, paths: tuple[str, ...]) -> None:
        """04 planning records a component's repeated blocks, once (§2.7)."""
        from vllm_omni.diffusion.weight_management.errors import LifecycleError

        with self._lock:
            entry = self._entry(component)
            if entry.discovered_blocks is not None:
                raise LifecycleError(
                    f"discovered blocks already set for {component!r}",
                    where="ComponentCatalog.set_discovered_blocks",
                )
            entry.discovered_blocks = paths

    def bind_overlay_slot(self, name: str, tensor: "torch.Tensor") -> None:
        """04 commit binds an overlay slot to a real tensor, once (§2.7)."""
        from vllm_omni.diffusion.weight_management.errors import LifecycleError

        with self._lock:
            if name in self._overlay_bindings:
                raise LifecycleError(
                    f"overlay slot {name!r} already bound",
                    where="ComponentCatalog.bind_overlay_slot",
                )
            self._overlay_bindings[name] = tensor

    def set_fully_resident(self, component: str, value: bool) -> None:
        """commit sets a component's residency, once; then read-only (§2.7)."""
        from vllm_omni.diffusion.weight_management.errors import LifecycleError

        with self._lock:
            entry = self._entry(component)
            if entry.fully_resident is not None:
                raise LifecycleError(
                    f"residency already set for {component!r}",
                    where="ComponentCatalog.set_fully_resident",
                )
            entry.fully_resident = value

    def _entry(self, component: str) -> CatalogEntry:
        from vllm_omni.diffusion.weight_management.errors import LifecycleError

        entry = self.entries.get(component)
        if entry is None:
            raise LifecycleError(
                f"unknown component {component!r}",
                where="ComponentCatalog._entry",
            )
        return entry


# --------------------------------------------------------------------------- #
# §2.3/§2.7 Assembly: walk the meta model, assign each tensor to a component.
# --------------------------------------------------------------------------- #
def _covers(root: str, name: str) -> bool:
    """True if module path ``name`` lies at or under ``root`` (a path prefix)."""
    return name == root or name.startswith(root + ".")


def _module_path_of(tensor_name: str) -> str:
    """The owning module path of a tensor (drop the final ``.weight`` etc.)."""
    return tensor_name.rsplit(".", 1)[0] if "." in tensor_name else ""


def build_catalog(
    pipeline: "nn.Module", decls: "tuple[ComponentDecl, ...]"
) -> ComponentCatalog:
    """Assemble a ``ComponentCatalog`` from a meta model + its declarations.

    This is the teachable core of RFC 02 §2.7. Steps:

    1. Collect every component root, and compute which roots are *carved* -- a
       root strictly under another component's root (nested ownership, §2.3).
    2. Walk the model's parameters and persistent buffers. Assign each tensor to
       the component whose root is its **longest** matching prefix.
    3. A tensor under no root is a missing-ownership violation (recorded, not
       raised -- ``require_complete()`` raises later, §2.4).

    Works on a meta model (no weight bytes): shapes/dtypes come from the meta
    tensors, which is exactly what construction produced in ``context.py``.
    """
    from vllm_omni.diffusion.weight_management.components import validate_decl

    # 1. roots per component, and carved (nested) roots (§2.3).
    roots_of: dict[str, tuple[str, ...]] = {}
    all_roots: list[tuple[str, str]] = []  # (root, component)
    for d in decls:
        validate_decl(d)
        rs = d.roots()
        roots_of[d.name] = rs
        for r in rs:
            all_roots.append((r, d.name))

    carved_of: dict[str, list[str]] = {d.name: [] for d in decls}
    for inner, inner_comp in all_roots:
        for outer, outer_comp in all_roots:
            if inner_comp != outer_comp and _covers(outer, inner) and inner != outer:
                # inner root sits under outer's root -> carve it out of outer.
                carved_of[outer_comp].append(inner)

    # 2. assign each tensor to its longest-matching root's component.
    tensors_of: dict[str, list[TensorDecl]] = {d.name: [] for d in decls}
    violations: list[str] = []

    def assign(name: str, kind: Literal["parameter", "buffer"], tensor) -> None:
        module_path = _module_path_of(name)
        best: tuple[int, str] | None = None
        for root, comp in all_roots:
            if _covers(root, module_path) and (best is None or len(root) > best[0]):
                best = (len(root), comp)
        if best is None:
            violations.append(name)
            return
        tensors_of[best[1]].append(
            TensorDecl(
                name=name,
                module=module_path,
                kind=kind,
                shape=tuple(tensor.shape),
                dtype=tensor.dtype,
            )
        )

    for name, p in pipeline.named_parameters():
        assign(name, "parameter", p)
    # Persistent buffers only (a buffer in _non_persistent_buffers_set is not a
    # weight the loader writes -- e.g. a config-derived RoPE table, §2.7).
    persistent = _persistent_buffer_names(pipeline)
    for name, b in pipeline.named_buffers():
        if name in persistent:
            assign(name, "buffer", b)

    # 3. build entries in declaration order.
    entries: dict[str, CatalogEntry] = {}
    for d in decls:
        entries[d.name] = CatalogEntry(
            decl=d,
            roots=roots_of[d.name],
            carved=tuple(carved_of[d.name]),
            tensors=tuple(tensors_of[d.name]),
        )
    return ComponentCatalog(
        entries=entries,
        declaration_order=tuple(d.name for d in decls),
        ownership_violations=tuple(violations),
    )


def _persistent_buffer_names(pipeline: "nn.Module") -> set[str]:
    """Fully-qualified names of persistent buffers (exclude non-persistent)."""
    names: set[str] = set()
    for mod_path, module in pipeline.named_modules():
        nonpersist = getattr(module, "_non_persistent_buffers_set", set())
        for bname, buf in module._buffers.items():
            if buf is None or bname in nonpersist:
                continue
            names.add(f"{mod_path}.{bname}" if mod_path else bname)
    return names
