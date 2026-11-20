# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replica-wide confirmation and AllGather groups.

PR 01-4 of RFC 01 (child of umbrella RFC #9536).
    - Spec: RFC 01 section 7 "Replica-wide confirmation" (round protocol §7.1,
      confirmation points §7.2) and the Topology assembled at point 0 (§6.2).
    - The AllGather process-group table is the one described in RFC 01 §5.1
      ("A process-level table in group.py").

Why a dedicated round protocol: startup must be transactional ACROSS ranks, not
just locally. Every rank has to agree, at a few checkpoints ("points"), that all
peers are healthy before anyone allocates host memory, pins buffers, or commits.
A single rank failing must fail the whole replica cleanly -- never leave peers
blocked in a collective. That is what ``ReplicaGroup.exchange`` guarantees: a
monitored barrier (so a dead peer is detected) followed by an all-gather of one
message per rank, then a check that every rank reported ok.

Module-scope imports stay stdlib + torch + interfaces + errors (RFC 01 §1.2).
``torch.distributed`` and ``parallel_state`` are imported INSIDE functions.

Scope note (study build): at ``world_size == 1`` the round protocol runs locally
with no ``torch.distributed`` calls (RFC 01 §7.1), which is what the driver tests
exercise. The real cross-rank path and ``new_group`` creation are kept
function-local and are not exercised without a distributed launch.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from vllm_omni.diffusion.weight_management.errors import (
    ConfirmationTimeoutError,
    GroupConfirmationError,
    PeerFailureError,
)
from vllm_omni.diffusion.weight_management.interfaces import (
    AllGatherGroup,
    RankCoordinates,
    RankFingerprints,
    Topology,
)

if TYPE_CHECKING:
    from vllm_omni.diffusion.weight_management.errors import ErrorSummary

_DEFAULT_TIMEOUT_S = 1800


@dataclass(frozen=True)
class Point0Payload:
    """Gathered at point 0 to build the Topology (RFC 01 §7.2)."""

    record: object  # LocalRecord (RFC 03)
    coordinates: RankCoordinates
    deployment: tuple[str, bool]  # (deployment_id, scoped)


@dataclass(frozen=True)
class Point1Payload:
    """Gathered at point 1's first round to decide sharing/groups (RFC 01 §7.2)."""

    fingerprints: RankFingerprints
    group_keys: frozenset  # existing process-table (members, backend) entries
    unmatched_policy_keys: frozenset


@dataclass(frozen=True)
class RoundMessage:
    """One rank's message in a round (RFC 01 §7.1).

    ok=False carries an ErrorSummary; ok=True carries a picklable payload.
    """

    point: str
    rank: int
    ok: bool
    error: ErrorSummary | None
    payload: object | None

    @classmethod
    def of(
        cls,
        point: str,
        rank: int,
        pending: object | None,
        payload: object | None,
    ) -> RoundMessage:
        """Build a message, marking it failed when a local error is pending."""
        if pending is not None:
            return cls(point, rank, ok=False, error=pending.summary(), payload=None)
        return cls(point, rank, ok=True, error=None, payload=payload)


def raise_if_any_failed(
    gathered: tuple[RoundMessage, ...], pending: object | None
) -> None:
    """Fail every rank if any message failed, or points disagree (RFC 01 §7.1).

    Raises:
        GroupConfirmationError: If ranks report different coordination points.
        PeerFailureError: If a peer failed and this rank did not.
    """
    points = {m.point for m in gathered}
    if len(points) > 1:
        raise GroupConfirmationError(
            f"ranks reached different coordination points: {sorted(points)}",
            where="ReplicaGroup.exchange",
        )
    failed = [m for m in gathered if not m.ok]
    if not failed:
        return
    if pending is not None:
        # This rank has its own error; let the driver raise that one.
        raise pending
    summaries = "; ".join(
        f"rank {m.rank}: {m.error.message if m.error else '?'}" for m in failed
    )
    raise PeerFailureError(
        f"peer rank(s) failed at {gathered[0].point}: {summaries}",
        where=gathered[0].point,
    )


class ReplicaGroup:
    """The world-group CPU (Gloo) channel for one replica (RFC 01 §7.1).

    Replicas do not talk to each other over torch.distributed; this covers only
    the ranks of one replica.
    """

    def __init__(
        self,
        world_size: int,
        rank: int,
        timeout_s: int = _DEFAULT_TIMEOUT_S,
        cpu_group: object | None = None,
    ) -> None:
        self.world_size = world_size
        self.rank = rank
        self.timeout_s = timeout_s
        self.cpu_group = cpu_group

    def exchange(
        self, point: str, message: RoundMessage
    ) -> tuple[RoundMessage, ...]:
        """One round: barrier, then all-gather one message per rank.

        At world size 1 this is purely local (RFC 01 §7.1): no torch.distributed
        call, but the same validation rules still apply to the single message.

        Raises:
            ConfirmationTimeoutError: If a rank does not reach the barrier.
        """
        if self.world_size == 1:
            return (message,)

        import datetime

        import torch.distributed as dist  # function-local (RFC 01 §1.2/§7.1)

        try:
            dist.monitored_barrier(
                group=self.cpu_group,
                timeout=datetime.timedelta(seconds=self.timeout_s),
                wait_all_ranks=True,
            )
        except Exception as exc:  # noqa: BLE001 - surface as a typed timeout
            raise ConfirmationTimeoutError(
                f"rank(s) did not reach barrier at {point}",
                where=point,
            ) from exc
        gathered: list[RoundMessage | None] = [None] * self.world_size
        dist.all_gather_object(gathered, message, group=self.cpu_group)
        return tuple(gathered)  # type: ignore[arg-type]


def read_rank_coordinates() -> RankCoordinates:
    """Read this rank's position on every parallel axis at point 0 (RFC 01 §7.2).

    Reads diffusion ``parallel_state`` and vLLM TP state function-locally. In a
    non-distributed (study/test) environment it falls back to a single-rank
    coordinate so the lifecycle can run at world size 1.
    """
    from vllm_omni.diffusion.weight_management.interfaces import ParallelCoordinate

    one = ParallelCoordinate(size=1, rank=0)
    try:  # pragma: no cover - exercised only under a real distributed launch
        from vllm_omni.diffusion.distributed import parallel_state as ps

        def coord(get_size, get_rank) -> ParallelCoordinate:
            return ParallelCoordinate(size=get_size(), rank=get_rank())

        return RankCoordinates(
            tp=coord(ps.get_tp_world_size, ps.get_tp_rank),
            sp=coord(ps.get_sp_world_size, ps.get_sp_rank),
            ulysses=one,
            ring=one,
            cfg=one,
            dp=coord(ps.get_dp_world_size, ps.get_dp_rank),
            pp=one,
            ep=None,
            fs=None,
        )
    except Exception:  # noqa: BLE001 - no distributed env: single-rank default
        return RankCoordinates(
            tp=one, sp=one, ulysses=one, ring=one, cfg=one, dp=one, pp=one,
            ep=None, fs=None,
        )


def assemble_topology(
    gathered: tuple[Point0Payload, ...], deployment_scoped_default: bool,
    stage_id: int,
) -> Topology:
    """Build the identical Topology every rank holds after point 0 (RFC 01 §7.2).

    Adopts rank 0's (deployment_id, scoped) pair; any rank that claims a
    different propagated ID is a configuration error.

    Raises:
        GroupConfirmationError: If deployment IDs are inconsistent across ranks.
    """
    rank0_id, rank0_scoped = gathered[0].deployment
    for i, p in enumerate(gathered):
        dep_id, _ = p.deployment
        if dep_id != rank0_id:
            raise GroupConfirmationError(
                f"rank {i} deployment id {dep_id!r} != rank 0 {rank0_id!r}",
                where="point0_topology",
            )
    return Topology(
        deployment_id=rank0_id,
        deployment_scoped=rank0_scoped,
        stage_id=stage_id,
        world_size=len(gathered),
        records=tuple(p.record for p in gathered),
        coordinates=tuple(p.coordinates for p in gathered),
    )


# Process-level AllGather group table (RFC 01 §5.1). Keyed by (members, backend);
# reused across startups in the same process. Closing a transaction or runtime
# never touches this table; only DiffusionWorker.shutdown destroys it.
_group_table: dict[tuple[tuple[int, ...], str], object] = {}


def group_table_keys() -> frozenset:
    """Existing (members, backend) keys, gathered at point 1 so every rank makes
    the same reuse-vs-create decision."""
    return frozenset(_group_table.keys())


def create_allgather_groups(
    all_fingerprints: tuple[RankFingerprints, ...],
    gathered: tuple[Point1Payload, ...],
) -> dict[str, tuple[AllGatherGroup, ...]]:
    """Partition declaring ranks by content identity into AllGather groups.

    Rejects singleton groups (RFC 01 §7.2), so world size 1 rejects any AllGather
    component. Real ``torch.distributed.new_group`` creation is function-local and
    only runs at world size > 1; this study build returns the computed grouping.

    Raises:
        GroupConfirmationError: If any component yields a singleton group.
    """
    # Collect, per component, the ranks that declare it and their identity.
    by_component: dict[str, dict[int, str]] = {}
    for fp in all_fingerprints:
        for component in fp.collective:
            identity = fp.content_identities.get(component)
            if identity is not None:
                by_component.setdefault(component, {})[fp.rank] = identity

    groups: dict[str, tuple[AllGatherGroup, ...]] = {}
    for component in sorted(by_component):
        rank_to_identity = by_component[component]
        partitions: dict[str, list[int]] = {}
        for rank, identity in rank_to_identity.items():
            partitions.setdefault(identity, []).append(rank)
        comp_groups = []
        for identity in sorted(partitions, key=lambda k: min(partitions[k])):
            members = tuple(sorted(partitions[identity]))
            if len(members) < 2:
                raise GroupConfirmationError(
                    f"singleton AllGather group for component {component!r} "
                    f"at rank(s) {members}",
                    where="point1_fingerprints",
                    component=component,
                )
            comp_groups.append(
                AllGatherGroup(component=component, ranks=members, group=None)
            )
        groups[component] = tuple(comp_groups)
    return groups


def check_unmatched_policy_keys(gathered: tuple[Point1Payload, ...]) -> None:
    """Fail only if a required policy key is unmatched on EVERY rank (RFC 01 §7.2).

    Rank-specific declarations may be absent on some ranks, so the error fires
    only on the intersection across ranks.

    Raises:
        PlanningError: If the intersection of unmatched required keys is nonempty.
    """
    from vllm_omni.diffusion.weight_management.errors import PlanningError

    sets = [p.unmatched_policy_keys for p in gathered]
    common = set(sets[0]).intersection(*sets[1:]) if sets else set()
    if common:
        raise PlanningError(
            f"required policy keys unmatched on all ranks: {sorted(common)}",
            where="point1_fingerprints",
        )


def check_collective_parts(all_fingerprints: tuple[RankFingerprints, ...]) -> None:
    """Enforce AllGather collective consistency across ranks (RFC 01 §7.2).

    If any rank lists a component in ``collective``, every rank that declares it
    in ``content_identities`` must also list it in ``collective``.

    Raises:
        GroupConfirmationError: On an inconsistent collective declaration.
    """
    collective_components = set()
    for fp in all_fingerprints:
        collective_components.update(fp.collective)
    for component in collective_components:
        for fp in all_fingerprints:
            declares = component in fp.content_identities
            in_collective = component in fp.collective
            if declares and not in_collective:
                raise GroupConfirmationError(
                    f"rank {fp.rank} declares {component!r} but omits it from "
                    f"its collective set",
                    where="point1_fingerprints",
                    component=component,
                )


def destroy_allgather_groups() -> None:
    """Destroy every process-table group. Called by DiffusionWorker.shutdown
    before destroy_distributed_env() (RFC 01 §5.1)."""
    if not _group_table:
        return
    import torch.distributed as dist  # function-local

    for group in _group_table.values():
        try:
            dist.destroy_process_group(group)
        except Exception:  # noqa: BLE001 - best-effort teardown
            pass
    _group_table.clear()
