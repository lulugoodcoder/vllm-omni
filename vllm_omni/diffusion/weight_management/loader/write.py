# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Storage binding and write tracking (RFC 02 §7.4/§7.5).

STUDY REFERENCE for RFC 02 "Unified Loader" (#9875). Not for upstream merge.

This file is the exact inverse of ``context.py``. There, construction turned
every real parameter into a ``meta`` one -- a shape with no storage. Here, write
time binds **real storage back into those same Parameter objects**, and then
copies checkpoint bytes into them. After this, the model is no longer structure;
it is weights.

Two ideas, both runnable:

**1. Binding with ``torch.utils.swap_tensors`` (RFC 02 §7.4).** The obvious way
to give a meta param real data -- ``module.weight = nn.Parameter(real)`` --
creates a *new* object, breaking every reference the model/offloader/LoRA
already holds, and dropping the ``weight_loader`` attribute vLLM set on it.
Instead we allocate a real-storage parameter and ``swap_tensors(meta_param,
real_param)``, which swaps their guts **in place**: the original Parameter object
survives with its identity, its ``__dict__`` (``weight_loader`` and TP info), but
now points at real storage. The target storage comes from
``transaction.target(name)`` (host memory sized by the manifest); a ``None``
target means "allocate on the execution device" (§7.4).

**2. Write tracking with ``TorchDispatchMode`` (RFC 02 §7.5).** Model
``load_weights`` code does ``param.copy_(checkpoint_tensor)`` -- but it might do
it twice, or partially, or not at all, and the loader must *know*, because
"every required tensor written exactly once" is the completeness guarantee.
``WriteTracker`` is a dispatch mode that sees every in-place op, matches the
destination's storage to a catalog tensor, and counts elements written. Two
signals fall out: writing more elements than a tensor has raises ``WriteError``
(a double write), and on the first *complete* write it calls
``storage.note_written(name)`` once -- the hook RFC 03 uses to release host
memory as soon as a tensor is done.

Deferred from the real §7.4/§7.5 (noted): matching by storage ``_cdata`` +
offset so tensor *views* bind correctly (we match by ``data_ptr``, fine for whole
tensors); the full in-place op set and per-quant-layer completion; wrapping each
parameter's ``weight_loader``; aliases sharing one object. The two load-bearing
mechanisms -- ``swap_tensors`` identity-preserving binding and dispatch-mode
write counting -- are real and debuggable.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.utils._python_dispatch import TorchDispatchMode

if TYPE_CHECKING:
    from vllm_omni.diffusion.weight_management.loader.catalog import ComponentCatalog


# In-place ops whose first argument is the tensor being written (RFC 02 §7.5
# lists copy_, index_copy_, index_put_, fill_, zero_, masked_fill_, scatter_).
_WRITE_OPS = {
    "copy_",
    "fill_",
    "zero_",
    "index_copy_",
    "index_put_",
    "masked_fill_",
    "scatter_",
    "add_",
    "mul_",
}


def _module_and_attr(model: nn.Module, qualified: str) -> tuple[nn.Module, str]:
    """Resolve ``a.b.weight`` to (module ``a.b``, attr ``weight``)."""
    *parents, leaf = qualified.split(".")
    mod = model
    for p in parents:
        mod = getattr(mod, p)
    return mod, leaf


def bind_target(
    model: nn.Module, qualified_name: str, target: "torch.Tensor | None"
) -> torch.Tensor:
    """Bind real storage into a meta parameter, preserving identity (§7.4).

    ``target`` is ``transaction.target(name)`` -- host storage to share, or
    ``None`` to allocate on the meta param's own device-equivalent. Returns the
    now-real tensor (the *same* Parameter object the model already references).
    """
    from vllm_omni.diffusion.weight_management.errors import WriteError

    mod, attr = _module_and_attr(model, qualified_name)
    old = getattr(mod, attr)
    if old.device.type != "meta":
        raise WriteError(
            f"{qualified_name} is already bound to real storage",
            where="bind_target",
            fix="each catalog tensor is bound exactly once",
        )

    if target is None:
        # No host target -> allocate real storage matching the meta spec.
        real = torch.empty_strided(
            tuple(old.shape), tuple(old.stride()), dtype=old.dtype, device="cpu"
        )
    else:
        if (
            tuple(target.shape) != tuple(old.shape)
            or target.dtype != old.dtype
        ):
            raise WriteError(
                f"target for {qualified_name} has shape/dtype "
                f"{tuple(target.shape)}/{target.dtype}, expected "
                f"{tuple(old.shape)}/{old.dtype}",
                where="bind_target",
            )
        real = target

    if isinstance(old, nn.Parameter):
        # Same-subclass parameter sharing `real`'s storage; swap in place so the
        # original object identity, __dict__ (weight_loader), survive (§7.4).
        new_param = nn.Parameter(real, requires_grad=old.requires_grad)
        new_param.__dict__.update(old.__dict__)
        torch.utils.swap_tensors(old, new_param)
        # `old` now carries real storage and is still the module's parameter.
        return old
    else:
        # A buffer: set directly in module._buffers (§7.4).
        mod._buffers[attr] = real
        return real


class WriteTracker(TorchDispatchMode):
    """Count in-place writes into bound tensors; enforce one write (§7.5).

    Enter it around the model's ``load_weights``; it watches every tensor op in
    the process, and whenever an in-place write lands on a tracked tensor it
    accumulates the element count. Over-count -> ``WriteError``; first complete
    count -> one ``storage.note_written(name)``.
    """

    def __init__(
        self,
        tracked: dict[int, tuple[str, int]],  # data_ptr -> (name, numel)
        on_complete=None,  # called once per tensor: on_complete(name)
    ) -> None:
        super().__init__()
        self._tracked = tracked
        self._on_complete = on_complete
        self.written: dict[str, int] = {}  # name -> elements written so far
        self.completed: list[str] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)

        name = func._schema.name.split("::")[-1]  # e.g. "copy_"
        if name in _WRITE_OPS and args:
            dest = args[0]
            if isinstance(dest, torch.Tensor) and dest.device.type != "meta":
                entry = self._tracked.get(dest.data_ptr())
                if entry is not None:
                    self._count(*entry, dest.numel())
        return out

    def _count(self, name: str, numel: int, written_now: int) -> None:
        from vllm_omni.diffusion.weight_management.errors import WriteError

        total = self.written.get(name, 0) + written_now
        if total > numel:
            raise WriteError(
                f"{name} written {total} elements but has only {numel}; a "
                f"tensor must be written exactly once",
                where="WriteTracker",
                fix="one semantic write per tensor; check for a double load",
            )
        self.written[name] = total
        if total == numel and name not in self.completed:
            self.completed.append(name)
            if self._on_complete is not None:
                self._on_complete(name)  # storage.note_written(name), §7.5


def tracked_table(
    model: nn.Module, catalog: "ComponentCatalog"
) -> dict[int, tuple[str, int]]:
    """Map each bound catalog tensor's ``data_ptr`` -> (name, numel) (§7.5)."""
    table: dict[int, tuple[str, int]] = {}
    for entry in catalog.entries.values():
        for t in entry.tensors:
            mod, attr = _module_and_attr(model, t.name)
            obj = getattr(mod, attr, None)
            if isinstance(obj, torch.Tensor) and obj.device.type != "meta":
                n = 1
                for s in obj.shape:
                    n *= s
                table[obj.data_ptr()] = (t.name, n)
    return table
