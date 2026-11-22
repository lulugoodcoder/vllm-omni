# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-facing component declarations and the catalog query table (RFC 02 §2).

STUDY REFERENCE for child RFC 02 "Unified Loader" (issue #9875, umbrella #9536).
Not for upstream merge; built to be read and debugged against the RFC.

This module is the *model-facing* half of RFC 02: the small, dependency-light
types a pipeline uses to describe its own structure, plus the process-level
query table the rest of the system reads. Per RFC 02 §2.1 it must be importable
by models / LoRA / caches **without** importing the concrete Loader, HWM, or
Offloader -- so it holds only plain data and a weak-ref registry, nothing that
touches construction or storage.

What a pipeline does with it (RFC 02 §2.1):

    class MyPipeline(nn.Module):
        def __init__(self, od_config):
            ...                                   # build the module tree
            declare_components(self, [
                ComponentDecl(name="transformer", role="dit", path="transformer"),
                ComponentDecl(name="text_encoder", role="text_encoder",
                              path="text_encoder"),
            ])

The Loader later reads these declarations (or *infers* them, §2.2) and assembles
the full ``ComponentCatalog`` in ``loader/catalog.py``. The split matters: this
file is the vocabulary a model author writes; the catalog is what the loader
builds from it.

Deferred vs. the real RFC (kept out of this study slice, noted where relevant):
  - §2.2 default inference from discovery attributes (``_dit_modules`` etc.),
  - the legacy closures threaded through the query functions,
  - adapter/overlay wiring beyond the declaration types.
"""
from __future__ import annotations

import threading
import weakref
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Literal, Mapping, Sequence

if TYPE_CHECKING:  # torch only needed for type hints here; keep import light
    import torch
    import torch.nn as nn

    from vllm_omni.diffusion.weight_management.loader.catalog import ComponentCatalog


# --------------------------------------------------------------------------- #
# §2.1 Model-facing declaration types.
# --------------------------------------------------------------------------- #
# A diffusion pipeline is several sub-models (a DiT/transformer, text encoders,
# a VAE, ...). ``Role`` names what kind each component is; placement and sharing
# decisions are made per component, so the role guides them (RFC 02 §2.1).
Role = Literal["dit", "text_encoder", "image_encoder", "audio_encoder", "vae", "other"]


@dataclass(frozen=True)
class ShardGroupSpec:
    """A component's membership in a cross-rank shard group (RFC 02 §2.1/§2.6).

    ``name`` is stable across ranks (e.g. ``minimax_h3_text_encoder``); equal
    names must carry equal ``size``/``rank`` within one rank's declarations.
    ``rank`` is this rank's local coordinate in the group. Feeds content
    identity and AllGather grouping (the 01 point-1 machinery).
    """

    name: str
    size: int
    rank: int


@dataclass(frozen=True)
class AdapterLocation:
    """Where an adapter's weights live: a local path or a Hub repo (RFC 02 §2.1)."""

    location: str
    filename: str | None
    revision: str | None
    format: str


@dataclass(frozen=True)
class FixedAdapter:
    """A baked-in adapter applied at load time (RFC 02 §2.1/§8.5).

    ``apply="fuse"`` folds it into the base weights (it disappears into the
    tensors); ``apply="overlay"`` keeps it as an always-on A/B overlay. Names
    are unique within a component; ``targets`` are root-relative, wildcards ok.
    """

    name: str
    source: AdapterLocation
    targets: tuple[str, ...]
    apply: Literal["fuse", "overlay"]


@dataclass(frozen=True)
class ComponentDecl:
    """One component a pipeline declares about itself (RFC 02 §2.1).

    ``path`` is one module path or several (several roots combine subtrees into
    one component, e.g. MiniMax H3's split video-VAE halves). The typed fields
    (``repeated_blocks``/``unit_groups``/``shard_group``/``fixed_adapters``) are
    a *closed* set (§2.6) -- construction validates each and raises
    ``ConstructionError`` on a bad value.
    """

    name: str
    role: Role
    path: str | tuple[str, ...]
    repeated_blocks: tuple[str, ...] | None = None
    unit_groups: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    shard_group: ShardGroupSpec | None = None
    fixed_adapters: tuple[FixedAdapter, ...] = ()

    def roots(self) -> tuple[str, ...]:
        """Normalize ``path`` (one or several) to a tuple of root paths."""
        return (self.path,) if isinstance(self.path, str) else tuple(self.path)


# --------------------------------------------------------------------------- #
# §2.6 Typed-field validation (closed set). Raises ConstructionError.
# --------------------------------------------------------------------------- #
def validate_decl(decl: ComponentDecl) -> None:
    """Check one declaration's typed fields (RFC 02 §2.6 table).

    Construction-time syntax checks only; planning does the match-time checks
    (which paths actually resolve). Raises ``ConstructionError`` -- the phase
    error type RFC 01 §3 maps construction failures to.
    """
    from vllm_omni.diffusion.weight_management.errors import ConstructionError

    where = f"ComponentDecl(name={decl.name!r})"
    if not decl.name:
        raise ConstructionError("component name is empty", where=where)

    # shard_group: size >= 1 and a valid rank within it.
    sg = decl.shard_group
    if sg is not None and not (sg.size >= 1 and 0 <= sg.rank < sg.size):
        raise ConstructionError(
            f"invalid shard_group size/rank: size={sg.size} rank={sg.rank}",
            where=where,
            fix="size must be >= 1 and 0 <= rank < size",
        )

    # unit_groups: names are identifiers; '*' matches exactly one path segment.
    for gname, patterns in decl.unit_groups.items():
        if not gname.isidentifier():
            raise ConstructionError(
                f"unit_group name {gname!r} is not an identifier", where=where
            )
        for pat in patterns:
            # A '**' would match more than one segment -> rejected (syntax only).
            if "**" in pat:
                raise ConstructionError(
                    f"unit_group pattern {pat!r} uses '**'; '*' matches exactly "
                    "one path segment",
                    where=where,
                )

    # fixed_adapters: component-locally-unique names.
    seen: set[str] = set()
    for fa in decl.fixed_adapters:
        if fa.name in seen:
            raise ConstructionError(
                f"duplicate fixed_adapter name {fa.name!r} in component "
                f"{decl.name!r}",
                where=where,
            )
        seen.add(fa.name)


# --------------------------------------------------------------------------- #
# §2.1 declare_components: a pipeline declares its structure, once.
# --------------------------------------------------------------------------- #
def declare_components(pipeline: "nn.Module", decls: Sequence[ComponentDecl]) -> None:
    """Record a pipeline's component declarations (RFC 02 §2.1).

    Called once in the pipeline constructor after config parsing. Repetition
    fails (``ConstructionError``). In legacy mode this merely stores the
    declarations -- there are no consumers; only the unified Loader reads them.

    The declarations live in a process-level weak table keyed by the pipeline
    object (RFC 02 §2.8), *not* as an attribute on the pipeline, so startup
    state never leaks onto the model and disappears when the pipeline is
    collected.
    """
    from vllm_omni.diffusion.weight_management.errors import ConstructionError

    if _decls_table.get(pipeline) is not None:
        raise ConstructionError(
            "declare_components called twice on one pipeline",
            where=type(pipeline).__name__,
            fix="declare all components in a single call in the constructor",
        )
    for d in decls:
        validate_decl(d)
    # Freeze to a tuple so later reads cannot mutate the recorded declarations.
    _decls_table[pipeline] = tuple(decls)


# --------------------------------------------------------------------------- #
# §2.8 Process-level weak-reference tables, keyed by pipeline.
# --------------------------------------------------------------------------- #
# Everything startup/catalog-related about a pipeline is kept *off* the pipeline
# object, in weak tables here. Entries vanish when the pipeline is garbage
# collected; reinitialization registers a fresh pipeline/catalog pair. This is
# the §2.8 "do not attach startup state to the pipeline" rule.
_decls_table: "weakref.WeakKeyDictionary[nn.Module, tuple[ComponentDecl, ...]]" = (
    weakref.WeakKeyDictionary()
)
_catalog_table: "weakref.WeakKeyDictionary[nn.Module, ComponentCatalog]" = (
    weakref.WeakKeyDictionary()
)
_generation_table: "weakref.WeakKeyDictionary[nn.Module, int]" = (
    weakref.WeakKeyDictionary()
)
_table_lock = threading.Lock()


def declarations_of(pipeline: "nn.Module") -> tuple[ComponentDecl, ...] | None:
    """The declarations a pipeline recorded, or ``None`` if it inferred them."""
    return _decls_table.get(pipeline)


def register_catalog(pipeline: "nn.Module", catalog: "ComponentCatalog") -> None:
    """Bind a built catalog to its pipeline (RFC 02 §2.8). Loader calls this."""
    with _table_lock:
        _catalog_table[pipeline] = catalog
        _generation_table.setdefault(pipeline, 0)


def catalog(pipeline: "nn.Module") -> "ComponentCatalog":
    """The pipeline's catalog (RFC 02 §2.8). Rejects legacy / unregistered.

    ``catalog()`` rejects legacy calls -- there is no catalog in legacy mode.
    """
    from vllm_omni.diffusion.weight_management.errors import LifecycleError
    from vllm_omni.diffusion.weight_management.startup import process_weight_management

    if not is_unified_stack():
        raise LifecycleError(
            "catalog() is unified-only; no catalog exists in legacy mode",
            where="components.catalog",
            fix="only call under --diffusion-weight-management unified",
        )
    cat = _catalog_table.get(pipeline)
    if cat is None:
        raise LifecycleError(
            "no catalog registered for this pipeline",
            where="components.catalog",
            fix="the Loader registers the catalog during build_structure",
        )
    _ = process_weight_management  # documents the selector dependency (§2.8)
    return cat


def pipeline_of(catalog_obj: "ComponentCatalog") -> "nn.Module":
    """The pipeline a catalog belongs to (RFC 02 §2.8).

    Uses the same weak table in reverse. Raises ``LifecycleError`` if the
    pipeline is already gone. Only 04 planning needs this (for module types /
    child order); 02 provides it.
    """
    from vllm_omni.diffusion.weight_management.errors import LifecycleError

    for pipe, cat in list(_catalog_table.items()):
        if cat is catalog_obj:
            return pipe
    raise LifecycleError(
        "pipeline for this catalog is no longer registered",
        where="components.pipeline_of",
    )


def is_unified_stack() -> bool:
    """True when this process runs the unified weight stack (RFC 02 §2.8).

    Reads RFC 01's process selector. In the real repo this call site is on the
    T6 coexistence allowlist (only registered sites may read the selector).
    """
    from vllm_omni.diffusion.weight_management.startup import (
        WeightManagement,
        process_weight_management,
    )

    return process_weight_management() is WeightManagement.UNIFIED


def weight_generation(pipeline: "nn.Module") -> int:
    """The pipeline's weight generation (RFC 02 §2.8).

    Supports weight-keyed cache signatures: base weights are fixed after commit,
    so this only changes when the dynamic LoRA manager writes a slot.
    """
    return _generation_table.get(pipeline, 0)


def bump_weight_generation(pipeline: "nn.Module") -> None:
    """Advance the weight generation (RFC 02 §2.8). The LoRA manager calls this."""
    with _table_lock:
        _generation_table[pipeline] = _generation_table.get(pipeline, 0) + 1


def execution_device() -> "torch.device":
    """The local execution device (RFC 02 §2.8).

    In the real repo this matches ``distributed.utils.get_local_device()``. In
    this study slice we keep it dependency-light: CUDA if available, else CPU.
    """
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _reset_component_tables_for_tests() -> None:
    """Clear the process-level tables (test isolation, mirrors 01's selector reset)."""
    with _table_lock:
        _decls_table.clear()
        _catalog_table.clear()
        _generation_table.clear()
