# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read traps: reject checkpoint reads during construction (RFC 02 §4).

STUDY REFERENCE for RFC 02 "Unified Loader" (#9875). Not for upstream merge.

``context.py`` is half of "structure-only": it makes weight *allocation* produce
meta tensors (no bytes). This file is the other half: it makes weight *reading*
fail. Together they guarantee the constructor builds pure structure -- it cannot
sneak real weights in by allocating them (context stops that) and it cannot pull
them from a checkpoint either (this stops that).

Why reading must be trapped, not just allocation: a model author used to write
``sd = torch.load("model.pt"); self.w = nn.Parameter(sd["w"])`` right in the
constructor. The meta patch would catch the ``nn.Parameter`` -- but the
``torch.load`` already read 4 GB off disk into host RAM. The RFC moves *all*
reading into the Loader's own reader (which runs later, into sized storage), so
any read from inside construction is a bug the trap must surface with a location
(RFC 02 §4), not silently allow.

The escape hatch (RFC 02 §4): the Loader's reader *does* read checkpoints -- but
it calls the originals inside ``trusted()``, a thread-local marker. So the rule
is precisely "reading is rejected *unless* you are the Loader's reader". Everyone
else (the model constructor, stray library code) is rejected.

What's intercepted here (the core of the §4 table):

    torch.load                         -> reject
    safetensors.safe_open  get_tensor  -> reject   (keys()/metadata() allowed)
    safetensors.torch.load_file / load -> reject

Before raising, a ``Violation(kind, call, where)`` is recorded. ``where`` is the
first stack frame *outside* torch/safetensors/weight-management -- i.e. the
model author's own line, which is who needs to fix it. Even if someone catches
the exception, the recorded violation still fails construction's final check
(RFC 02 §3.5 step 7 / §4 "swallowed exceptions still fail").

Deferred from the real §4 (noted): rescanning ``sys.modules`` for functions
re-bound into other namespaces (``from safetensors.torch import load_file`` as a
module global) plus the ``sys.meta_path`` finder for modules imported later;
``from_single_file`` / ``from_pretrained`` class-method traps; the
``load_state_dict``-with-real-tensors trap; the CONSTRUCT-vs-SEALED error split
beyond ConstructionError/WriteError. This slice patches the canonical functions
(``torch.load`` and the ``safetensors`` entry points), which is enough to see
the mechanism run and debug it.
"""
from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Iterator

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class Violation:
    """A trapped read (RFC 02 §4). ``where`` points at the offending line."""

    kind: str  # e.g. "torch.load", "safetensors.get_tensor"
    call: str  # a short description of the attempted call
    where: str  # "file:lineno in function" -- the first non-framework frame


# Process-wide singleton (one structure build at a time, like StructureContext).
_trap: "ReadTrap | None" = None
_trap_lock = threading.Lock()
# Thread-local "I am the Loader's trusted reader" marker (RFC 02 §4).
_tls = threading.local()


def is_trusted() -> bool:
    """True inside a ``trusted()`` scope on this thread (RFC 02 §4)."""
    return getattr(_tls, "trusted", False)


@contextmanager
def trusted() -> Iterator[None]:
    """Mark this thread as the Loader's reader: reads pass through (§4).

    The reader pool enters this per worker thread. It is thread-local, so a
    trusted read on the reader thread never lets the *model constructor* thread
    read -- the trap stays armed everywhere else.
    """
    prev = getattr(_tls, "trusted", False)
    _tls.trusted = True
    try:
        yield
    finally:
        _tls.trusted = prev


def active_trap() -> "ReadTrap | None":
    return _trap


# --------------------------------------------------------------------------- #
# Finding the offending frame (RFC 02 §4: "first frame outside the framework").
# --------------------------------------------------------------------------- #
_FRAMEWORK_MARKERS = (
    "/torch/",
    "/safetensors/",
    "/weight_management/",
    "site-packages/transformers/",
    "site-packages/diffusers/",
)


def _offending_frame() -> str:
    """The first stack frame outside torch/safetensors/weight-management."""
    import traceback

    for frame in reversed(traceback.extract_stack()[:-2]):
        if not any(m in frame.filename for m in _FRAMEWORK_MARKERS):
            return f"{frame.filename}:{frame.lineno} in {frame.name}"
    return "<unknown>"


class ReadTrap:
    """Patches checkpoint-read entry points to reject untrusted reads (§4).

    Active from structure construction through finalization. Use as a context
    manager or drive ``open()``/``close()`` across the loader lifecycle.
    """

    def __init__(self) -> None:
        self.violations: list[Violation] = []
        self._saved: dict[str, object] = {}

    def __enter__(self) -> "ReadTrap":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # -- lifecycle --------------------------------------------------------- #
    def open(self) -> None:
        global _trap
        from vllm_omni.diffusion.weight_management.errors import LifecycleError

        with _trap_lock:
            if _trap is not None:
                raise LifecycleError(
                    "a read trap is already open",
                    where="ReadTrap.open",
                )
            _trap = self
        self._install_patches()

    def close(self) -> None:
        global _trap
        if not self._saved and _trap is not self:
            return
        self._restore_patches()
        with _trap_lock:
            if _trap is self:
                _trap = None

    # -- violation recording + raising ------------------------------------ #
    def _reject(self, kind: str, call: str) -> None:
        """Record a violation and raise the phase-appropriate error (§4)."""
        v = Violation(kind=kind, call=call, where=_offending_frame())
        self.violations.append(v)
        raise self._error_type()(
            f"checkpoint read {kind!r} is not allowed during structure "
            f"construction ({call})",
            where=v.where,
            fix="move all checkpoint reading into the Loader (declare_weights / "
            "hf_component / a WeightSource); constructors build structure only",
        )

    def _error_type(self):
        """ConstructionError in CONSTRUCT, WriteError once SEALED (§4)."""
        from vllm_omni.diffusion.weight_management.errors import (
            ConstructionError,
            WriteError,
        )

        try:
            from vllm_omni.diffusion.weight_management.loader.context import (
                Phase,
                active_context,
            )

            ctx = active_context()
            if ctx is not None and ctx.phase is Phase.SEALED:
                return WriteError
        except Exception:
            pass
        return ConstructionError

    def require_clean(self) -> None:
        """Raise if any violation was recorded, even a swallowed one (§4/§3.5.7)."""
        if not self.violations:
            return
        from vllm_omni.diffusion.weight_management.errors import ConstructionError

        lines = [f"  {v.kind} at {v.where} ({v.call})" for v in self.violations]
        raise ConstructionError(
            f"{len(self.violations)} checkpoint read(s) happened during "
            f"construction:\n" + "\n".join(lines),
            where="ReadTrap.require_clean",
            fix="each listed line must stop reading weights during construction",
        )

    # -- the patches ------------------------------------------------------- #
    def _install_patches(self) -> None:
        import torch

        self._saved["torch.load"] = torch.load

        orig_load = torch.load

        def patched_torch_load(*args, **kwargs):
            if is_trusted():
                return orig_load(*args, **kwargs)
            target = args[0] if args else kwargs.get("f", "<stream>")
            self._reject("torch.load", f"torch.load({target!r})")

        torch.load = patched_torch_load

        # safetensors is optional in this study env; patch it only if present.
        try:
            import safetensors
            import safetensors.torch as st_torch
        except Exception:
            self._saved["_safetensors"] = None
            return

        self._saved["_safetensors"] = True
        self._saved["st.safe_open"] = safetensors.safe_open
        self._saved["st_torch.load_file"] = st_torch.load_file
        self._saved["st_torch.load"] = st_torch.load
        trap = self

        orig_safe_open = safetensors.safe_open

        class _TrappedHandle:
            """Allows keys()/metadata(); rejects get_tensor()/get_slice() (§4)."""

            def __init__(self, inner):
                self._inner = inner

            def keys(self):
                return self._inner.keys()

            def metadata(self):
                return self._inner.metadata()

            def get_tensor(self, name):
                if is_trusted():
                    return self._inner.get_tensor(name)
                trap._reject("safetensors.get_tensor", f"get_tensor({name!r})")

            def get_slice(self, name):
                if is_trusted():
                    return self._inner.get_slice(name)
                trap._reject("safetensors.get_slice", f"get_slice({name!r})")

            def __enter__(self):
                self._inner.__enter__()
                return self

            def __exit__(self, *a):
                return self._inner.__exit__(*a)

        def patched_safe_open(*args, **kwargs):
            inner = orig_safe_open(*args, **kwargs)
            # keys()/metadata() stay open even untrusted; tensor reads are trapped.
            return _TrappedHandle(inner)

        orig_load_file = st_torch.load_file
        orig_st_load = st_torch.load

        def patched_load_file(*args, **kwargs):
            if is_trusted():
                return orig_load_file(*args, **kwargs)
            target = args[0] if args else kwargs.get("filename", "<file>")
            self._reject("safetensors.load_file", f"load_file({target!r})")

        def patched_st_load(*args, **kwargs):
            if is_trusted():
                return orig_st_load(*args, **kwargs)
            self._reject("safetensors.load", "safetensors.torch.load(...)")

        safetensors.safe_open = patched_safe_open
        st_torch.load_file = patched_load_file
        st_torch.load = patched_st_load

    def _restore_patches(self) -> None:
        import torch

        if "torch.load" in self._saved:
            torch.load = self._saved["torch.load"]  # type: ignore[assignment]
        if self._saved.get("_safetensors"):
            import safetensors
            import safetensors.torch as st_torch

            safetensors.safe_open = self._saved["st.safe_open"]  # type: ignore[assignment]
            st_torch.load_file = self._saved["st_torch.load_file"]  # type: ignore[assignment]
            st_torch.load = self._saved["st_torch.load"]  # type: ignore[assignment]
        self._saved.clear()
