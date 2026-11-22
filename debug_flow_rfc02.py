"""Local debugger entry point for the RFC 02 unified-loader study build.

Companion to debug_flow.py (RFC 01). Same trick: vllm_omni/__init__.py dies on
the installed-vLLM mismatch, so we stub the parent packages as empty namespace
packages and load the real weight_management/loader submodules directly (they
need only stdlib + torch + each other).

This drives the UnifiedLoader through 01's Loader lifecycle on a toy pipeline,
with fakes standing in for RFC 03 (HostStorage) and RFC 04 (LoadTransaction) --
the same dependency-injection seam RFC 01 uses:

    build_structure()  meta model + read trap + catalog
    preflight()        content identities (digests) + expected manifest
    write()            size storage from manifest, bind via swap_tensors,
                       read a REAL checkpoint inside trusted(), copy + track
    finalize()         all-params-real check + final manifest + seal
    close()            remove context + trap patches

HOW TO USE (run in YOUR terminal, not pytest):

    .venv/bin/python debug_flow_rfc02.py

Set a breakpoint anywhere in the real source to step line by line, e.g.:
  * loader/loader.py           UnifiedLoader.build_structure / write / finalize
  * loader/context.py          patched_register_parameter  (watch meta happen)
  * loader/write.py            bind_target                 (watch swap_tensors)
  * loader/trap.py             ReadTrap._reject            (watch a read blocked)

    breakpoint()   # <- add where you want to stop, then run the command above
    n next | s step-in | c continue | p expr | bt stack | l list | q quit
"""
import importlib.util
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path("/home/lulud/vllm-omni")
OMNI = ROOT / "vllm_omni"
WMDIR = OMNI / "diffusion" / "weight_management"


def _stub(name: str, path: Path) -> None:
    m = types.ModuleType(name)
    m.__path__ = [str(path)]
    m.__package__ = name
    sys.modules[name] = m


for _name, _path in [
    ("vllm_omni", OMNI),
    ("vllm_omni.diffusion", OMNI / "diffusion"),
    ("vllm_omni.diffusion.weight_management", WMDIR),
    ("vllm_omni.diffusion.weight_management.loader", WMDIR / "loader"),
]:
    _stub(_name, _path)

_plat = types.ModuleType("vllm_omni.platforms")
_plat.current_omni_platform = types.SimpleNamespace(
    empty_cache=lambda: None, dist_backend="gloo"
)
sys.modules["vllm_omni.platforms"] = _plat


def _load(mod_name: str, rel: str):
    spec = importlib.util.spec_from_file_location(mod_name, WMDIR / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


WM = "vllm_omni.diffusion.weight_management."
_load(WM + "errors", "errors.py")
_load(WM + "components", "components.py")
_load(WM + "loader.context", "loader/context.py")
_load(WM + "loader.trap", "loader/trap.py")
_load(WM + "loader.catalog", "loader/catalog.py")
_load(WM + "loader.manifest", "loader/manifest.py")
_load(WM + "loader.write", "loader/write.py")
_load(WM + "loader.loader", "loader/loader.py")

import torch
import torch.nn as nn

from vllm_omni.diffusion.weight_management.components import (
    ComponentDecl,
    declare_components,
)
from vllm_omni.diffusion.weight_management.loader.loader import UnifiedLoader


# --------------------------------------------------------------------------- #
# A toy pipeline: a DiT with a nested text encoder (the carve-out case) + a VAE.
# --------------------------------------------------------------------------- #
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Linear(16, 16, bias=False)
        self.mlp = nn.Linear(16, 16, bias=False)


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([Block() for _ in range(2)])
        self.language_model = nn.Linear(16, 8, bias=False)  # carved -> text_encoder


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Linear(8, 8, bias=False)


class ToyPipeline(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = Transformer()
        self.vae = VAE()


DECLS = (
    ComponentDecl(name="transformer", role="dit", path="transformer"),
    ComponentDecl(name="text_encoder", role="text_encoder",
                  path="transformer.language_model"),
    ComponentDecl(name="vae", role="vae", path="vae"),
)


def build_pipeline() -> nn.Module:
    # A real pipeline declares its own components in __init__; so do we.
    pipe = ToyPipeline()
    declare_components(pipe, DECLS)
    return pipe


# --------------------------------------------------------------------------- #
# Fakes for RFC 03 (HostStorage) and RFC 04 (LoadTransaction).
# --------------------------------------------------------------------------- #
class FakeHostStorage:
    """Host storage sized from the expected manifest (what HWM does, §10.2)."""

    def __init__(self, expected_manifest):
        self.views = {
            e.name: torch.empty(e.shape, dtype=e.dtype) for e in expected_manifest.entries
        }
        self.noted: list[str] = []
        self.sealed = None

    def view(self, name):
        return self.views[name]

    def note_written(self, name):  # §7.5 hook
        self.noted.append(name)

    def seal(self, final_manifest):  # §9 step 6
        self.sealed = final_manifest


class FakeTransaction:
    """Hands out host targets (RFC 04 §7.4). target() -> a host view."""

    def __init__(self, storage: FakeHostStorage):
        self._storage = storage

    def target(self, name):
        return self._storage.view(name)  # not None -> share host storage


def make_reader(ckpt_path: Path):
    """A reader that reads a REAL checkpoint; the loader calls it in trusted()."""

    def read_stream(names):
        sd = torch.load(ckpt_path)  # trapped for everyone else; allowed in trusted()
        for n in names:
            yield n, sd[n]

    return read_stream


def main():
    print(">>> RFC 02 unified-loader flow (fakes for RFC 03/04)\n")

    # Make a real checkpoint on disk: random weights for every declared tensor.
    with torch.no_grad():
        proto = build_pipeline()
    sd = {name: torch.randn(p.shape) for name, p in proto.named_parameters()}
    ckpt = Path(tempfile.mkdtemp()) / "toy.pt"
    torch.save(sd, ckpt)
    print(f"[0] wrote a real checkpoint with {len(sd)} tensors -> {ckpt.name}")

    loader = UnifiedLoader(
        build_pipeline=build_pipeline,
        decls=DECLS,
        read_stream=make_reader(ckpt),
        model_id="study/toy-omni",
        stage_dtype="float32",
        tp=(1, 0),
    )

    # step 2 -------------------------------------------------------------
    model, catalog = loader.build_structure()
    metas = sum(1 for _, p in model.named_parameters() if p.device.type == "meta")
    print(f"[2] build_structure: {metas} params, ALL on meta; "
          f"catalog components = {catalog.declaration_order}")
    catalog.require_complete()
    print("    catalog.require_complete() OK (every tensor owned)")

    # step 5 -------------------------------------------------------------
    pf = loader.preflight(catalog)
    for comp, dg in pf.content_identities.items():
        print(f"[5] preflight identity  {comp:<13} {dg[:16]}...")
    storage = FakeHostStorage(pf.expected_manifest)
    print(f"    host storage sized from manifest: {len(storage.views)} tensors")

    # step 9 -------------------------------------------------------------
    txn = FakeTransaction(storage)
    loader.write(model, storage, txn)
    real = sum(1 for _, p in model.named_parameters() if p.device.type != "meta")
    print(f"[9] write: {real} params now REAL; storage.note_written fired "
          f"{len(storage.noted)} times")
    # spot-check bytes landed:
    w = model.transformer.blocks[0].attn.weight
    assert torch.equal(w.detach(), sd["transformer.blocks.0.attn.weight"])
    print("    verified: transformer.blocks.0.attn.weight bytes match the checkpoint")

    # step 10 ------------------------------------------------------------
    manifest = loader.finalize(model, storage)
    print(f"[10] finalize: final manifest {len(manifest.entries)} entries, "
          f"storage sealed = {storage.sealed is not None}")

    # close --------------------------------------------------------------
    loader.close()
    # trap + context patches gone: torch.load works normally again.
    assert torch.load(ckpt)  # no ConstructionError now
    print("[x] close: context + trap patches removed; torch.load normal again")

    print("\n>>> flow complete: meta -> catalog -> identities -> real weights -> sealed.")


if __name__ == "__main__":
    main()
