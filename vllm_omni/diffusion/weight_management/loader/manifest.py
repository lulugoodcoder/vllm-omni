# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Content identity and manifests (RFC 02 §10).

STUDY REFERENCE for RFC 02 "Unified Loader" (#9875). Not for upstream merge.

This file answers one question: **do two ranks hold the same weights, byte for
byte?** If yes, they can share a single host copy instead of each loading their
own (the umbrella's whole-deployment memory saving). The answer must be decided
*before* any bytes are read -- from structure alone -- so it is a hash.

``ContentIdentity.digest()`` is a SHA256 over a canonical JSON description of a
component. The art is entirely in **what goes in vs. what stays out** (RFC 02
§10.1), because that choice *is* the sharing policy:

  IN  (change it -> different weights -> different digest -> cannot share):
      the tensor digest (final names/shapes/dtypes/strides/kinds), roots/carved
      boundaries, stage dtype, quantization, rule tables, and the **TP/EP
      coordinates and shard_group** -- because different TP ranks hold different
      *shards*, which are genuinely different bytes.

  OUT (differs between ranks that nonetheless hold identical weights -> must be
      excluded or they'd never share):
      **DP/SP/CFG rank**, device addresses, resolved file/cache paths, the
      deployment ID, and the checkpoint file bytes themselves.

So two data-parallel replicas -- same structure, same TP, different DP rank, on
different GPUs in different processes -- produce the *identical* digest purely
because DP rank and device are not inputs. That identical digest is what lets
RFC 03 elect one writer and have the others share its host memory. Two ranks
with a different dtype, or a different TP shard, get different digests and each
keep their own. The exclusion list is not an implementation detail; it is the
feature.

The two manifests (§10.2/§10.3) are the name->tensor contracts built from the
same catalog: ``ExpectedManifest`` (preflight: what we *will* write, used by HWM
to size host storage) and ``FinalManifest`` (finalization: what we *did* write,
where each tensor ended up). Their digests let a writer and a waiter confirm
they agree on the shared part before one trusts the other's memory.

Deferred from the real §10 (noted): adapter identities, the full quantization
config digest, loader-rules version threading, EP coordinates, and
``FinalManifest.live()`` alias rebinding (we keep a simple ``live`` map). The
digest construction -- canonical JSON, sorted keys, the exact include/exclude
split -- is faithful and runnable.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Literal

if TYPE_CHECKING:
    import torch

    from vllm_omni.diffusion.weight_management.loader.catalog import (
        CatalogEntry,
        ComponentCatalog,
        TensorDecl,
    )


def _sha(obj: object) -> str:
    """SHA256 of canonical, sorted-key JSON (RFC 02 §10.1)."""
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


# --------------------------------------------------------------------------- #
# §10.1 Content identity.
# --------------------------------------------------------------------------- #
def _tensors_digest(tensors: "tuple[TensorDecl, ...]") -> str:
    """Digest of a component's tensors: names/shapes/dtypes/strides/kinds (§10.1).

    Sorted by name so iteration order can't change the digest. These are the
    *final* tensor facts, so any real structural difference (a shape, a dtype)
    flows through to the component identity.
    """
    rows = sorted(
        [t.name, list(t.shape), str(t.dtype), t.kind] for t in tensors
    )
    return _sha(rows)


@dataclass(frozen=True)
class ContentIdentity:
    """A component's structural fingerprint (RFC 02 §10.1).

    Every field here is *included* in the digest. Things that differ between
    ranks holding identical weights (DP rank, device, file paths, deployment ID)
    are simply **not fields** -- that's how they're excluded.
    """

    schema: int
    component: str
    model: str
    revision: str | None
    roots: tuple[str, ...]
    carved: tuple[str, ...]
    tensors_digest: str
    stage_dtype: str
    quantization: str | None
    rule_tables: tuple[tuple[str, int], ...]
    tp: tuple[int, int]  # (size, rank) -- IN: different TP rank = different shard
    shard_group: tuple[str, int, int] | None  # (name, size, rank)

    def digest(self) -> str:
        """SHA256 of this identity's canonical JSON (§10.1)."""
        return _sha(
            {
                "schema": self.schema,
                "component": self.component,
                "model": self.model,
                "revision": self.revision,
                "roots": list(self.roots),
                "carved": list(self.carved),
                "tensors_digest": self.tensors_digest,
                "stage_dtype": self.stage_dtype,
                "quantization": self.quantization,
                "rule_tables": [list(rt) for rt in self.rule_tables],
                "tp": list(self.tp),
                "shard_group": list(self.shard_group) if self.shard_group else None,
            }
        )


def content_identity(
    entry: "CatalogEntry",
    *,
    model: str,
    stage_dtype: str,
    tp: tuple[int, int],
    revision: str | None = None,
    quantization: str | None = None,
    rule_tables: tuple[tuple[str, int], ...] = (),
    shard_group: tuple[str, int, int] | None = None,
) -> ContentIdentity:
    """Build a component's content identity from its catalog entry (§10.1).

    Note what is *not* a parameter: DP/SP/CFG rank, device, file paths,
    deployment ID. They cannot affect the digest because they never enter it.
    """
    return ContentIdentity(
        schema=1,
        component=entry.decl.name,
        model=model,
        revision=revision,
        roots=entry.roots,
        carved=entry.carved,
        tensors_digest=_tensors_digest(entry.tensors),
        stage_dtype=stage_dtype,
        quantization=quantization,
        rule_tables=rule_tables,
        tp=tp,
        shard_group=shard_group,
    )


# --------------------------------------------------------------------------- #
# §10.2 Expected manifest (preflight: what we WILL write; HWM sizes storage).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ManifestEntry:
    name: str
    component: str
    kind: str
    shape: tuple[int, ...]
    dtype: "torch.dtype"
    nbytes: int
    aliases: tuple[str, ...] = ()
    replaced_at_finalize: bool = False


@dataclass(frozen=True)
class ExpectedManifest:
    """The name-sorted projection of the final catalog (RFC 02 §10.2)."""

    version: int
    entries: tuple[ManifestEntry, ...]

    def component(self, name: str) -> tuple[ManifestEntry, ...]:
        return tuple(e for e in self.entries if e.component == name)

    def digest(self, names: "Iterable[str] | None" = None) -> str:
        """Digest of all entries, or just ``names`` (handoff checks, §10.2)."""
        wanted = set(names) if names is not None else None
        rows = sorted(
            [e.name, e.component, e.kind, list(e.shape), str(e.dtype), e.nbytes]
            for e in self.entries
            if wanted is None or e.name in wanted
        )
        return _sha(rows)


def _nbytes(shape: tuple[int, ...], dtype: "torch.dtype") -> int:
    import torch

    n = 1
    for s in shape:
        n *= s
    return n * torch.empty((), dtype=dtype).element_size()


def build_expected_manifest(catalog: "ComponentCatalog") -> ExpectedManifest:
    """Project a catalog into a name-sorted expected manifest (§10.2).

    HWM reads this to compute host-storage layout -- it needs every tensor's
    name, component, shape, dtype and byte size, nothing about where the bytes
    come from.
    """
    entries: list[ManifestEntry] = []
    for comp_name, entry in catalog.entries.items():
        for t in entry.tensors:
            entries.append(
                ManifestEntry(
                    name=t.name,
                    component=comp_name,
                    kind=t.kind,
                    shape=t.shape,
                    dtype=t.dtype,
                    nbytes=_nbytes(t.shape, t.dtype),
                    aliases=t.aliases,
                )
            )
    entries.sort(key=lambda e: e.name)
    return ExpectedManifest(version=1, entries=tuple(entries))


# --------------------------------------------------------------------------- #
# §10.3 Final manifest (finalization: what we DID write, and where).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FinalEntry:
    name: str
    component: str
    shape: tuple[int, ...]
    dtype: "torch.dtype"
    residency: Literal["host", "device", "hsdp"]
    storage_key: str | None
    aliases: tuple[str, ...] = ()
    # Excluded from the digest (§10.3): how this rank obtained the tensor.
    written: Literal["self", "shared", "out_of_range"] = "self"


@dataclass(frozen=True)
class FinalManifest:
    """The sole name-binding authority for shared storage (RFC 02 §10.3)."""

    version: int
    expected_digest: str
    entries: tuple[FinalEntry, ...]

    def digest(self, names: "Iterable[str] | None" = None) -> str:
        """Digest excluding ``written`` source and live objects (§10.3).

        A writer and a waiter independently build their own final manifests; the
        *shared part* of these digests must be equal, which is how the handoff is
        validated. ``written`` ("self"/"shared"/"out_of_range") differs between
        them by design, so it is excluded.
        """
        wanted = set(names) if names is not None else None
        rows = sorted(
            [
                e.name,
                e.component,
                list(e.shape),
                str(e.dtype),
                e.residency,
                e.storage_key,
            ]
            for e in self.entries
            if wanted is None or e.name in wanted
        )
        return _sha(rows)
