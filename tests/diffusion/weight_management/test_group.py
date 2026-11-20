# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for replica-wide confirmation (RFC 01 section 7 / PR 01-4).

Covers everything in ``group.py`` that is exercisable at world size 1 -- which
RFC 01 section 7.1 runs purely locally, with no ``torch.distributed`` calls:

    * the round message / fail-together protocol (``RoundMessage``,
      ``raise_if_any_failed``),
    * single-rank ``exchange`` (no distributed call),
    * ``read_rank_coordinates`` single-rank fallback,
    * ``assemble_topology`` deployment-ID adoption and mismatch rejection,
    * ``create_allgather_groups`` identity partitioning and singleton rejection
      (so world size 1 rejects any AllGather component),
    * ``check_unmatched_policy_keys`` intersection rule,
    * ``check_collective_parts`` collective-consistency rule.

The genuine multi-rank path (Gloo processes, real ``new_group``) is covered by
``test_group_multiproc.py`` in the real RFC; this file is the world-size-1 half.

Runs under pytest normally. Because this checkout's ``vllm_omni`` import chain is
currently broken by a vLLM version mismatch, the ``__main__`` block also lets the
file self-run through the study bypass harness.
"""
from __future__ import annotations

import pytest

from vllm_omni.diffusion.weight_management.errors import (
    ConfigError,
    GroupConfirmationError,
    PeerFailureError,
    PlanningError,
)
from vllm_omni.diffusion.weight_management.group import (
    Point0Payload,
    Point1Payload,
    ReplicaGroup,
    RoundMessage,
    assemble_topology,
    check_collective_parts,
    check_unmatched_policy_keys,
    create_allgather_groups,
    destroy_allgather_groups,
    group_table_keys,
    raise_if_any_failed,
    read_rank_coordinates,
)
from vllm_omni.diffusion.weight_management.interfaces import RankFingerprints


# --------------------------------------------------------------------------- #
# Small builders.
# --------------------------------------------------------------------------- #
def _fp(rank, content_identities, collective=None, plan_fingerprint="fp"):
    return RankFingerprints(
        rank=rank,
        content_identities=dict(content_identities),
        plan_fingerprint=plan_fingerprint,
        collective=dict(collective or {}),
    )


def _p0(rank, deployment):
    return Point0Payload(
        record=object(),
        coordinates=read_rank_coordinates(),  # single-rank fallback
        deployment=deployment,
    )


def _p1(fp, unmatched=frozenset(), group_keys=frozenset()):
    return Point1Payload(
        fingerprints=fp, group_keys=group_keys, unmatched_policy_keys=unmatched
    )


# --------------------------------------------------------------------------- #
# RoundMessage + raise_if_any_failed: the fail-together core (RFC 01 §7.1).
# --------------------------------------------------------------------------- #
def test_round_message_of_ok_carries_payload():
    m = RoundMessage.of("point0_topology", 0, pending=None, payload={"x": 1})
    assert m.ok is True
    assert m.payload == {"x": 1}
    assert m.error is None


def test_round_message_of_failed_carries_summary_not_payload():
    err = ConfigError("bad selector", where="cli")
    err.rank = 0
    m = RoundMessage.of("point0_topology", 0, pending=err, payload={"x": 1})
    assert m.ok is False
    assert m.payload is None
    assert m.error is not None
    assert m.error.error_type == "ConfigError"


def test_raise_if_any_failed_all_ok_returns():
    gathered = (
        RoundMessage.of("p", 0, None, "a"),
        RoundMessage.of("p", 1, None, "b"),
    )
    raise_if_any_failed(gathered, pending=None)  # no raise


def test_raise_if_any_failed_mixed_points_raises_group_error():
    gathered = (
        RoundMessage.of("point2_ready", 0, None, None),
        RoundMessage.of("point3_commit", 1, None, None),
    )
    with pytest.raises(GroupConfirmationError):
        raise_if_any_failed(gathered, pending=None)


def test_raise_if_any_failed_peer_failed_raises_peer_error():
    err = ConfigError("boom on rank 1", where="x")
    err.rank = 1
    gathered = (
        RoundMessage.of("p", 0, None, None),       # this rank ok
        RoundMessage.of("p", 1, err, None),        # peer failed
    )
    with pytest.raises(PeerFailureError):
        raise_if_any_failed(gathered, pending=None)


def test_raise_if_any_failed_local_pending_raises_own_error():
    my = ConfigError("my own failure", where="x")
    my.rank = 0
    gathered = (RoundMessage.of("p", 0, my, None),)
    # When THIS rank has a pending error, it raises its own, not PeerFailureError.
    with pytest.raises(ConfigError):
        raise_if_any_failed(gathered, pending=my)


# --------------------------------------------------------------------------- #
# exchange at world size 1 is purely local (RFC 01 §7.1).
# --------------------------------------------------------------------------- #
def test_exchange_world_size_1_is_local():
    group = ReplicaGroup(world_size=1, rank=0)
    msg = RoundMessage.of("p", 0, None, "payload")
    gathered = group.exchange("p", msg)
    assert gathered == (msg,)  # no torch.distributed call


# --------------------------------------------------------------------------- #
# read_rank_coordinates single-rank fallback (RFC 01 §7.2).
# --------------------------------------------------------------------------- #
def test_read_rank_coordinates_single_rank_fallback():
    c = read_rank_coordinates()
    assert c.tp.size == 1 and c.tp.rank == 0
    assert c.dp.size == 1
    # Disabled EP/HSDP are None (not a size-1 coordinate).
    assert c.ep is None
    assert c.fs is None


# --------------------------------------------------------------------------- #
# assemble_topology: adopt rank 0; reject mismatched deployment IDs (RFC 01 §7.2).
# --------------------------------------------------------------------------- #
def test_assemble_topology_adopts_rank0_deployment():
    dep = ("a" * 32, True)
    topo = assemble_topology((_p0(0, dep), _p0(1, dep)), False, stage_id=3)
    assert topo.deployment_id == "a" * 32
    assert topo.deployment_scoped is True
    assert topo.stage_id == 3
    assert topo.world_size == 2
    assert len(topo.coordinates) == 2


def test_assemble_topology_rejects_mismatched_ids():
    gathered = (_p0(0, ("a" * 32, True)), _p0(1, ("b" * 32, True)))
    with pytest.raises(GroupConfirmationError):
        assemble_topology(gathered, False, stage_id=0)


# --------------------------------------------------------------------------- #
# create_allgather_groups: identity partitioning + singleton rejection.
# --------------------------------------------------------------------------- #
def test_create_allgather_groups_rejects_singleton():
    # One rank declaring an AllGather component -> singleton -> rejected.
    fp = _fp(0, {"transformer": "idA"}, collective={"transformer": "dig"})
    with pytest.raises(GroupConfirmationError):
        create_allgather_groups((fp,), (_p1(fp),))


def test_world_size_1_rejects_any_allgather_component():
    # The world-size-1 consequence of the singleton rule (RFC 01 §7.2).
    fp = _fp(0, {"dit": "x"}, collective={"dit": "d"})
    with pytest.raises(GroupConfirmationError):
        create_allgather_groups((fp,), (_p1(fp),))


def test_create_allgather_groups_partitions_by_identity():
    # 4 ranks: {0,2} share identity idA, {1,3} share idB -> two groups of two.
    fps = (
        _fp(0, {"transformer": "idA"}, {"transformer": "d"}),
        _fp(1, {"transformer": "idB"}, {"transformer": "d"}),
        _fp(2, {"transformer": "idA"}, {"transformer": "d"}),
        _fp(3, {"transformer": "idB"}, {"transformer": "d"}),
    )
    groups = create_allgather_groups(fps, tuple(_p1(f) for f in fps))
    comp = groups["transformer"]
    assert len(comp) == 2
    members = sorted(g.ranks for g in comp)
    assert members == [(0, 2), (1, 3)]
    # groups ordered by smallest member rank
    assert comp[0].ranks == (0, 2)


def test_create_allgather_groups_empty_when_no_collective():
    # No component in `collective` -> no AllGather groups at all.
    fp = _fp(0, {"transformer": "idA"}, collective={})
    assert create_allgather_groups((fp,), (_p1(fp),)) == {}


# --------------------------------------------------------------------------- #
# check_unmatched_policy_keys: fail only on the intersection (RFC 01 §7.2).
# --------------------------------------------------------------------------- #
def test_unmatched_policy_keys_common_key_raises():
    g = (
        _p1(_fp(0, {}), unmatched=frozenset({"k1", "k2"})),
        _p1(_fp(1, {}), unmatched=frozenset({"k1"})),  # intersection = {k1}
    )
    with pytest.raises(PlanningError):
        check_unmatched_policy_keys(g)


def test_unmatched_policy_keys_disjoint_passes():
    g = (
        _p1(_fp(0, {}), unmatched=frozenset({"k1"})),
        _p1(_fp(1, {}), unmatched=frozenset({"k2"})),  # empty intersection
    )
    check_unmatched_policy_keys(g)  # no raise


# --------------------------------------------------------------------------- #
# check_collective_parts: collective consistency across ranks (RFC 01 §7.2).
# --------------------------------------------------------------------------- #
def test_check_collective_parts_inconsistent_raises():
    # rank 1 declares 'transformer' but omits it from its collective set.
    fps = (
        _fp(0, {"transformer": "id"}, collective={"transformer": "d"}),
        _fp(1, {"transformer": "id"}, collective={}),
    )
    with pytest.raises(GroupConfirmationError):
        check_collective_parts(fps)


def test_check_collective_parts_consistent_passes():
    fps = (
        _fp(0, {"transformer": "id"}, collective={"transformer": "d"}),
        _fp(1, {"transformer": "id"}, collective={"transformer": "d"}),
    )
    check_collective_parts(fps)  # no raise


# --------------------------------------------------------------------------- #
# Process-table helpers.
# --------------------------------------------------------------------------- #
def test_group_table_starts_empty_and_destroy_is_noop():
    assert group_table_keys() == frozenset()
    destroy_allgather_groups()  # empty table -> no-op, no distributed call


if __name__ == "__main__":  # pragma: no cover - study bypass for the broken env
    import sys

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
