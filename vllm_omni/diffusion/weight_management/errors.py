# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Error types for the unified weight management.

PR 01-1 of RFC 01 (child of umbrella RFC #9536).
    - Spec: RFC 01 section 3 "Error types". The subclass list below and the
      message format in ``WeightManagementError.__str__`` come straight from there.
    - Which startup step raises which type: RFC 01 section 3's two tables
      (one per validation phase, one for wrapping foreign exceptions).

Depends only on the standard library (RFC 01 section 3 requires this so the
errors can be imported anywhere without pulling in torch).

Why a whole taxonomy instead of plain exceptions: the startup driver runs on
every rank and must report *predictable* errors before the first model write
(umbrella RFC goal "Weight loading" #4). A typed, picklable ``ErrorSummary``
lets a failing rank's reason be collected at coordination points and re-raised
on peers as ``PeerFailureError``.

There is no legacy equivalent to point at: today offload failures surface as
whatever ad-hoc exception the backend happened to raise.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

_MAX_MESSAGE = 4096


@dataclass(frozen=True)
class ErrorSummary:
    """Picklable snapshot collected at coordination points."""

    error_type: str
    phase: str
    rank: int
    message: str


class WeightManagementError(RuntimeError):
    """Base error. Subclasses set ``phase``; the driver fills identity fields.

    Args:
        what: The facts, including component names and conflicting values.
        where: A file path, module path, tensor name, or socket name; omitted
            when no location applies.
        fix: A concrete argument, recipe field, or command; omitted when no
            practical action exists.
        component: The component the error concerns, if any.
    """

    phase: ClassVar[str] = "unknown"

    def __init__(
        self,
        what: str,
        *,
        where: str | None = None,
        fix: str | None = None,
        component: str | None = None,
    ) -> None:
        super().__init__(what)
        self.what = what
        self.where = where
        self.fix = fix
        self.component = component
        # Filled in by the startup driver before the error propagates.
        self.stage_id: int | None = None
        self.rank: int | None = None
        self.world_size: int | None = None

    def __str__(self) -> str:
        stage = self.stage_id if self.stage_id is not None else "?"
        rank = self.rank if self.rank is not None else "?"
        world = self.world_size if self.world_size is not None else "?"
        line = (
            f"[weight-management][stage {stage}][rank {rank}/{world}]"
            f"[{self.phase}] {self.what}"
        )
        if self.where is not None:
            line += f"; at {self.where}"
        if self.fix is not None:
            line += f"; fix: {self.fix}"
        return line

    def summary(self) -> ErrorSummary:
        return ErrorSummary(
            error_type=type(self).__name__,
            phase=self.phase,
            rank=self.rank if self.rank is not None else -1,
            message=str(self)[:_MAX_MESSAGE],
        )


class ConfigError(WeightManagementError, ValueError):
    phase: ClassVar[str] = "config"


class ConstructionError(WeightManagementError):
    phase: ClassVar[str] = "build_structure"


class PlanningError(WeightManagementError):
    phase: ClassVar[str] = "plan"


class PreflightError(WeightManagementError):
    phase: ClassVar[str] = "preflight"


class GroupConfirmationError(WeightManagementError):
    phase: ClassVar[str] = "confirm"


class HostCapacityError(GroupConfirmationError):
    phase: ClassVar[str] = "admit"


class PrepareError(WeightManagementError):
    phase: ClassVar[str] = "prepare"


class DeviceCapacityError(PrepareError):
    phase: ClassVar[str] = "prepare"


class StackUnavailableError(WeightManagementError):
    phase: ClassVar[str] = "construct_modules"


class AllocationError(WeightManagementError):
    phase: ClassVar[str] = "allocate"


class WriteError(WeightManagementError):
    phase: ClassVar[str] = "write"


class FinalizeError(WeightManagementError):
    phase: ClassVar[str] = "finalize"


class CommitError(WeightManagementError):
    phase: ClassVar[str] = "commit"


class PeerFailureError(WeightManagementError):
    phase: ClassVar[str] = "peer"


class ConfirmationTimeoutError(WeightManagementError):
    phase: ClassVar[str] = "confirm"


class LifecycleError(WeightManagementError):
    phase: ClassVar[str] = "lifecycle"


class RuntimeFault(WeightManagementError):
    phase: ClassVar[str] = "service"
