# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The UnifiedLoader: 01's Loader protocol, tying §2-§10 together (RFC 02 §1).

STUDY REFERENCE for RFC 02 "Unified Loader" (#9875). Not for upstream merge.

This is the file the other six exist to serve. ``UnifiedLoader`` implements the
``Loader`` Protocol that RFC 01 defined (``interfaces.py``: build_structure,
preflight, write, finalize, close) and that the 01 startup driver calls at steps
2/5/9/10. Reading it top to bottom is the whole RFC 02 story in order:

    build_structure()  §3+§4+§2  open context (meta) + trap (no reads), run the
                                 constructor, seal, assemble the catalog.
    preflight()        §10       per-component content identities + the expected
                                 manifest (what we will write; HWM sizes storage).
    write()            §7        for each tensor: target = transaction.target();
                                 bind real storage (swap_tensors); read the
                                 checkpoint inside trusted(); copy under the
                                 WriteTracker (one write each).
    finalize()         §9+§10    check every param is real + no swallowed reads;
                                 build the final manifest; seal storage.
    close()            §1        remove context + trap patches; idempotent, runs
                                 on failure paths too.

The point of the split across files is that each phase is independently
debuggable (which is how you've been studying) -- but this file is where you see
them compose into the real contract the driver drives.

How it meets the 01 driver: ``create_loader(ctx)`` (in ``__init__.py``) is what
01's ``EntryPointFactory`` resolves; the driver then calls the five methods in
order, passing the ``HostStorage`` (RFC 03) and ``LoadTransaction`` (RFC 04) it
owns. This study ``UnifiedLoader`` takes those as constructor injections so it
runs against fakes -- the same dependency-injection seam RFC 01 uses throughout.

Deferred from the real §1 (noted): rule-table dtype application (§6), lazy
tensors / adapter fusion / quantization in the read pipeline (§7.3/§7.6/§8), the
finalization step-2 byte-changing work and eval (§9 steps 2/7), HSDP (§13), and
the 19-family migration (§16). This slice keeps the lifecycle *shape* and the
real mechanisms from the six modules, so the contract is runnable end to end.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Iterable, Mapping

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from vllm_omni.diffusion.weight_management.components import ComponentDecl
    from vllm_omni.diffusion.weight_management.loader.catalog import ComponentCatalog
    from vllm_omni.diffusion.weight_management.loader.manifest import (
        ContentIdentity,
        ExpectedManifest,
        FinalManifest,
    )


@dataclass(frozen=True)
class Preflight:
    """Preflight result (RFC 02 §1): identities + the expected manifest.

    ``content_identities`` (name -> digest) feeds RFC 01's rank fingerprints so
    ranks decide sharing; ``identities`` keeps the full objects for logs/tests;
    ``expected_manifest`` is what RFC 03 sizes host storage from.
    """

    content_identities: Mapping[str, str]
    identities: Mapping[str, "ContentIdentity"]
    expected_manifest: "ExpectedManifest"


# A reader yields (canonical_name, tensor) pairs -- the stand-in for §7's
# checkpoint read pipeline. The loader calls it inside trap.trusted().
ReadStream = Callable[[Iterable[str]], Iterable[tuple[str, torch.Tensor]]]


class UnifiedLoader:
    """Implements 01's ``Loader`` Protocol (RFC 02 §1).

    Constructor injections (the RFC 01 test seam): ``build_pipeline`` constructs
    the model (and calls ``declare_components`` itself, as a real pipeline does);
    ``read_stream`` is the checkpoint reader; ``model_id`` / ``tp`` feed content
    identity.
    """

    def __init__(
        self,
        *,
        build_pipeline: Callable[[], nn.Module],
        decls: "tuple[ComponentDecl, ...]",
        read_stream: ReadStream,
        model_id: str = "study/model",
        stage_dtype: str = "float32",
        tp: tuple[int, int] = (1, 0),
    ) -> None:
        self._build_pipeline = build_pipeline
        self._decls = decls
        self._read_stream = read_stream
        self._model_id = model_id
        self._stage_dtype = stage_dtype
        self._tp = tp
        # lifecycle state
        self._context = None
        self._trap = None
        self._model: nn.Module | None = None
        self._catalog: "ComponentCatalog | None" = None

    # -- step 2: build structure only (§3 + §4 + §2) ----------------------- #
    def build_structure(self) -> tuple[nn.Module, "ComponentCatalog"]:
        from vllm_omni.diffusion.weight_management.components import register_catalog
        from vllm_omni.diffusion.weight_management.loader.catalog import build_catalog
        from vllm_omni.diffusion.weight_management.loader.context import StructureContext
        from vllm_omni.diffusion.weight_management.loader.trap import ReadTrap

        # Open BOTH guards: meta allocation (§3) + read rejection (§4).
        self._context = StructureContext()
        self._trap = ReadTrap()
        self._context.open()
        self._trap.open()

        # Run the real constructor -- produces a meta model, no bytes read.
        self._model = self._build_pipeline()

        # Structure is built: seal (declarations now frozen; traps remain, §3.1).
        self._context.seal()

        # Assemble the catalog and register it off-model in the weak table (§2.8).
        self._catalog = build_catalog(self._model, self._decls)
        register_catalog(self._model, self._catalog)
        return self._model, self._catalog

    # -- step 5: preflight (§10) ------------------------------------------- #
    def preflight(self, catalog: "ComponentCatalog", plan: object = None) -> Preflight:
        from vllm_omni.diffusion.weight_management.loader.manifest import (
            build_expected_manifest,
            content_identity,
        )

        identities: dict[str, "ContentIdentity"] = {}
        digests: dict[str, str] = {}
        for name, entry in catalog.entries.items():
            ident = content_identity(
                entry,
                model=self._model_id,
                stage_dtype=self._stage_dtype,
                tp=self._tp,
            )
            identities[name] = ident
            digests[name] = ident.digest()
        return Preflight(
            content_identities=digests,
            identities=identities,
            expected_manifest=build_expected_manifest(catalog),
        )

    # -- step 9: write (§7) ------------------------------------------------- #
    def write(self, model: nn.Module, storage: object, transaction: object) -> None:
        from vllm_omni.diffusion.weight_management.errors import WriteError
        from vllm_omni.diffusion.weight_management.loader.trap import trusted
        from vllm_omni.diffusion.weight_management.loader.write import (
            WriteTracker,
            bind_target,
            tracked_table,
        )

        assert self._catalog is not None
        names = [t.name for e in self._catalog.entries.values() for t in e.tensors]

        # Bind storage for every tensor: a real target from the transaction, or
        # None -> allocate on the execution device (§7.4).
        for name in names:
            target = transaction.target(name)  # RFC 04 hands out host views
            bind_target(model, name, target)

        # Track writes; note each tensor to storage as it completes (§7.5).
        table = tracked_table(model, self._catalog)
        on_done = getattr(storage, "note_written", None)
        tracker = WriteTracker(table, on_complete=on_done)

        # Read the checkpoint INSIDE trusted() so the trap lets the reader in,
        # and copy into bound targets under the tracker + no_grad (§7.2/§7.5).
        with torch.no_grad(), tracker:
            with trusted():
                stream = list(self._read_stream(names))
            for name, src in stream:
                dst = _get_tensor(model, name)
                if tuple(src.shape) != tuple(dst.shape):
                    raise WriteError(
                        f"source for {name} is {tuple(src.shape)}, target is "
                        f"{tuple(dst.shape)}",
                        where="UnifiedLoader.write",
                    )
                dst.copy_(src)

        missing = [n for n in names if n not in tracker.completed]
        if missing:
            raise WriteError(
                f"{len(missing)} tensor(s) never written: {missing}",
                where="UnifiedLoader.write",
                fix="every catalog tensor needs exactly one write",
            )

    # -- step 10: finalize (§9 + §10) -------------------------------------- #
    def finalize(self, model: nn.Module, storage: object) -> "FinalManifest":
        from vllm_omni.diffusion.weight_management.errors import FinalizeError
        from vllm_omni.diffusion.weight_management.loader.manifest import (
            FinalEntry,
            FinalManifest,
            build_expected_manifest,
        )

        assert self._catalog is not None

        # §9 step 4: every param must be real now (nothing left on meta).
        stragglers = [n for n, p in model.named_parameters() if p.device.type == "meta"]
        if stragglers:
            raise FinalizeError(
                f"{len(stragglers)} parameter(s) still meta after write: "
                f"{stragglers}",
                where="UnifiedLoader.finalize",
            )
        # §4: no swallowed checkpoint read slipped through construction/write.
        if self._trap is not None:
            self._trap.require_clean()

        em = build_expected_manifest(self._catalog)
        entries = []
        for comp_name, entry in self._catalog.entries.items():
            for t in entry.tensors:
                entries.append(
                    FinalEntry(
                        name=t.name,
                        component=comp_name,
                        shape=t.shape,
                        dtype=t.dtype,
                        residency="host",
                        storage_key=t.name,
                        written="self",
                    )
                )
        entries.sort(key=lambda e: e.name)
        manifest = FinalManifest(
            version=1,
            expected_digest=em.digest(),
            entries=tuple(entries),
        )
        seal = getattr(storage, "seal", None)
        if seal is not None:
            seal(manifest)  # §9 step 6
        return manifest

    # -- failure/close (§1) ------------------------------------------------- #
    def close(self) -> None:
        """Remove context + trap patches. Idempotent; safe on failure paths."""
        if self._trap is not None:
            self._trap.close()
            self._trap = None
        if self._context is not None:
            self._context.close()
            self._context = None


def _get_tensor(model: nn.Module, qualified: str) -> torch.Tensor:
    *parents, leaf = qualified.split(".")
    mod = model
    for p in parents:
        mod = getattr(mod, p)
    return getattr(mod, leaf)
