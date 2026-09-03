"""
Adaptive traffic-signal controller.

Two things drive a change in which lane is green:

1. advance() — called when the current lane's green countdown reaches
   zero (or once, to kick off the very first cycle). This moves through
   lanes in round-robin order (fair: no lane starves forever), with each
   lane's green duration scaled by its own last-known vehicle density.
   A lane flagged with an emergency vehicle preempts the round-robin
   order immediately — but only once per detection, so it doesn't hog
   green forever on a single stale detection.

2. update_lane_data() — called whenever a new frame is analyzed for a
   lane. It only updates that lane's stored density/emergency flag; it
   does NOT interrupt whichever lane is currently mid-countdown, except
   to kick off the very first cycle if nothing has run yet. This mirrors
   real intersections: a camera updates continuously, but the signal
   doesn't visibly jump around every time a new frame comes in.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.config import settings


@dataclass
class LaneState:
    name: str
    vehicle_count: int = 0
    emergency: bool = False


@dataclass
class SignalState:
    lanes: list[LaneState]
    active_lane: int = 0
    green_seconds: int = settings.min_green_seconds
    reason: str = "initial"

    def as_dict(self) -> dict:
        return {
            "active_lane": self.active_lane,
            "active_lane_name": self.lanes[self.active_lane].name,
            "green_seconds": self.green_seconds,
            "reason": self.reason,
            "lanes": [
                {"name": l.name, "vehicle_count": l.vehicle_count, "emergency": l.emergency}
                for l in self.lanes
            ],
        }


def _green_duration_for(density: int) -> int:
    return min(
        settings.max_green_seconds,
        max(settings.min_green_seconds, settings.min_green_seconds + density * 2),
    )


class SignalController:
    """Holds live state for a single intersection with N lanes."""

    def __init__(self, lane_names: list[str]):
        self.state = SignalState(lanes=[LaneState(name=n) for n in lane_names])

    def update_lane_data(self, lane_id: int, count: int, emergency: bool) -> SignalState:
        lane = self.state.lanes[lane_id]
        lane.vehicle_count = count
        lane.emergency = emergency

        # Nothing has run yet (server just started) — kick off the cycle
        # immediately instead of waiting on a countdown that never began.
        if self.state.reason == "initial":
            self.advance()
        return self.state

    def advance(self) -> SignalState:
        """Move to the next lane. Call this when the current green
        countdown reaches zero (or once, to start the first cycle)."""
        emergency_idx = next(
            (i for i, l in enumerate(self.state.lanes) if l.emergency), None
        )

        if emergency_idx is not None:
            chosen = emergency_idx
            self.state.green_seconds = settings.max_green_seconds
            self.state.reason = "emergency_vehicle"
            # Treat this emergency as served so the same stale detection
            # doesn't preempt the queue again on every future advance().
            self.state.lanes[chosen].emergency = False
        else:
            num_lanes = len(self.state.lanes)
            start = self.state.active_lane if self.state.reason != "initial" else -1
            chosen = (start + 1) % num_lanes
            self.state.green_seconds = _green_duration_for(self.state.lanes[chosen].vehicle_count)
            self.state.reason = "round_robin"

        self.state.active_lane = chosen
        return self.state
