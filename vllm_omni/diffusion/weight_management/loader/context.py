# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Structure-only construction context (RFC 02 §3).

STUDY REFERENCE for RFC 02 "Unified Loader" (#9875). Not for upstream merge.

This is the single most important idea in RFC 02, and the one you asked about
long ago ("device=meta? structure-only without writing the model?"). The Loader
must build the *shape* of a model -- every module, every parameter's
shape/dtype/subclass -- **without allocating a single weight byte and without
reading any checkpoint**. Why: placement (sharding/offload) is decided from the
structure, host storage is sized from it, and ranks must agree on it, all before
anybody writes real tensors (the 01 lifecycle: build_structure -> plan ->
preflight -> ... -> write).

How it works (RFC 02 §3.2/§3.3): while the context is open it **patches torch**
so that ordinary constructor code produces a meta model:

  * ``nn.Module.register_parameter`` -> every real parameter is converted to an
    equivalent ``meta`` parameter (same shape/dtype/subclass/requires_grad,
    ``__dict__`` preserved so vLLM's ``weight_loader`` survives). So a model that
    does ``self.w = nn.Parameter(torch.empty(1024, 1024))`` ends up with a
    *meta* ``w`` -- no 4 MB allocation.
  * ``Module.to/.cuda/.cpu/.xpu`` -> device request discarded, dtype kept. A
    constructor that calls ``.to("cuda")`` doesn't move anything off meta.
  * ``Module.to_empty`` -> raises ``ConstructionError`` (it reallocates).

Phases (RFC 02 §3.1):

    CONSTRUCT  (build_structure runs the constructor; traps + meta on)
       |  seal()
    SEALED     (factories/declares disabled; file-read traps stay via ReadTrap)
       |  close()
    (patches removed, final violation check)

It's a **process-wide singleton**: opening a second context before the first
closes raises ``LifecycleError`` (two structure builds at once would corrupt the
shared patches). RFC 01's single-process executor runs startups sequentially, so
this is safe.

Deferred from the real §3 (noted): the full post-construction pipeline (§3.5
steps 2-8: inference, rule-table dtype overrides, swap_tensors final dtype,
quant output declaration, the meta/violation scans) lives with the catalog and
rule-table work; this file is the context + meta/device patching core, which is
what makes the whole thing debuggable on its own.
"""
from __future__ import annotations

import enum
import threading
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    pass


class Phase(enum.Enum):
    """The structure context's phase (RFC 02 §3.1)."""

    CONSTRUCT = "construct"  # running the model constructor; meta + traps on
    SEALED = "sealed"  # structure built; declarations disabled, traps remain


# Process-wide singleton: only one structure build at a time (RFC 02 §3.1).
_active: "StructureContext | None" = None
_lock = threading.Lock()


def is_building_structure() -> bool:
    """True only during ``CONSTRUCT`` (RFC 02 §3.1).

    ``build_component`` / ``declare_weights`` / HF dispatch read this to decide
    whether they are inside a live structure build. In the real repo this call
    site is on the T6 coexistence allowlist.
    """
    return _active is not None and _active.phase is Phase.CONSTRUCT


def active_context() -> "StructureContext | None":
    """The current structure context, or ``None``. For dispatch helpers."""
    return _active


class StructureContext:
    """Patches torch so a constructor yields a meta-only model (RFC 02 §3).

    Use as a context manager::

        with StructureContext() as ctx:
            model = MyPipeline(od_config)   # built entirely on meta
            ctx.seal()                      # structure frozen
            ...                             # (write happens later)
        # patches removed here; final violation check runs

    Or drive the phases manually (as ``UnifiedLoader`` does across the 01
    lifecycle, where construction and writing are separate steps).
    """

    def __init__(self) -> None:
        self.phase: Phase | None = None
        self._saved: dict[str, object] = {}
        # Violations recorded by patched ops / read traps, raised at close
        # (RFC 02 §3.5 step 7 / §4). Kept as plain tuples here.
        self.violations: list[tuple[str, str]] = []

    # -- lifecycle ---------------------------------------------------------- #
    def __enter__(self) -> "StructureContext":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def open(self) -> None:
        """Enter ``CONSTRUCT``: install the meta + device-movement patches."""
        global _active
        from vllm_omni.diffusion.weight_management.errors import LifecycleError

        with _lock:
            if _active is not None:
                raise LifecycleError(
                    "a structure construction context is already open",
                    where="StructureContext.open",
                    fix="unified startups in one process run sequentially; "
                    "close the first context before opening another",
                )
            _active = self
        self.phase = Phase.CONSTRUCT
        self._install_patches()

    def seal(self) -> None:
        """Move to ``SEALED``: structure is built; declarations now disabled.

        The meta/device patches stay on (write happens under SEALED), but
        ``is_building_structure()`` goes False so ``declare_weights`` / factories
        raise ``LifecycleError`` if called after construction (RFC 02 §3.1).
        """
        from vllm_omni.diffusion.weight_management.errors import LifecycleError

        if self.phase is not Phase.CONSTRUCT:
            raise LifecycleError(
                f"cannot seal from phase {self.phase}", where="StructureContext.seal"
            )
        self.phase = Phase.SEALED

    def close(self) -> None:
        """Remove every patch and clear the singleton (RFC 02 §3.1).

        Idempotent and called on failure paths too, so a broken construction
        never leaves torch globally patched.
        """
        global _active
        if not self._saved and _active is not self:
            return  # already closed / never opened
        self._restore_patches()
        self.phase = None
        with _lock:
            if _active is self:
                _active = None

    # -- the patches -------------------------------------------------------- #
    def _install_patches(self) -> None:
        """Patch register_parameter + device-movement methods (RFC 02 §3.2/§3.3)."""
        self._saved["register_parameter"] = nn.Module.register_parameter
        self._saved["to"] = nn.Module.to
        self._saved["cuda"] = nn.Module.cuda
        self._saved["cpu"] = nn.Module.cpu
        self._saved["to_empty"] = nn.Module.to_empty

        orig_register = nn.Module.register_parameter

        def patched_register_parameter(module, name, param):
            # §3.2: convert a real parameter to an equivalent meta parameter;
            # already-meta params (and None) register unchanged, preserving tied
            # object identities.
            if param is not None and param.device.type != "meta":
                param = _to_meta_parameter(param)
            return orig_register(module, name, param)

        orig_to = nn.Module.to

        def patched_to(module, *args, **kwargs):
            # §3.3: keep dtype/memory_format, discard the device request.
            device, dtype, non_blocking, memory_format = torch._C._nn._parse_to(
                *args, **kwargs
            )
            if dtype is None and memory_format is None:
                return module  # device-only move -> no-op (stay on meta)
            return orig_to(
                module,
                dtype=dtype,
                non_blocking=non_blocking,
                memory_format=memory_format,
            )

        def patched_device_only(module, *args, **kwargs):
            return module  # .cuda()/.cpu()/.xpu() -> no-op

        def patched_to_empty(module, *args, **kwargs):
            from vllm_omni.diffusion.weight_management.errors import ConstructionError

            raise ConstructionError(
                "to_empty() is not allowed during structure construction; it "
                "reallocates buffers",
                where="StructureContext (to_empty)",
                fix="let the Loader allocate at write time via transaction.target",
            )

        nn.Module.register_parameter = patched_register_parameter
        nn.Module.to = patched_to
        nn.Module.cuda = patched_device_only
        nn.Module.cpu = patched_device_only
        nn.Module.to_empty = patched_to_empty

    def _restore_patches(self) -> None:
        if "register_parameter" in self._saved:
            nn.Module.register_parameter = self._saved["register_parameter"]  # type: ignore[assignment]
            nn.Module.to = self._saved["to"]  # type: ignore[assignment]
            nn.Module.cuda = self._saved["cuda"]  # type: ignore[assignment]
            nn.Module.cpu = self._saved["cpu"]  # type: ignore[assignment]
            nn.Module.to_empty = self._saved["to_empty"]  # type: ignore[assignment]
        self._saved.clear()


def _to_meta_parameter(param: nn.Parameter) -> nn.Parameter:
    """Build a meta parameter equivalent to ``param`` (RFC 02 §3.2).

    Mirrors the real ``release_module_parameters_to_meta`` pattern: same
    shape/dtype/requires_grad, ``__dict__`` copied so a vLLM ``weight_loader``
    (and other per-parameter metadata) survives. The real helper also preserves
    the exact ``nn.Parameter`` *subclass*; this study version keeps the common
    case faithful and copies ``__dict__`` for the rest.
    """
    meta_data = torch.empty_strided(
        tuple(param.shape), tuple(param.stride()), dtype=param.dtype, device="meta"
    )
    new = nn.Parameter(meta_data, requires_grad=param.requires_grad)
    # Preserve per-parameter metadata (e.g. weight_loader, TP info).
    new.__dict__.update(param.__dict__)
    return new
