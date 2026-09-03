"""
Adaptive traffic-signal controller.

Replaces the original project's approach (a global array persisted to a
pickle file on every read/write) with an explicit, in-memory state machine
that's easy to unit test and reason about.

Rule: the lane with the highest vehicle density gets the green light, for
a duration scaled by that density (clamped between MIN/MAX green seconds).
An emergency vehicle in any lane immediately pre-empts the signal in its
favor, overriding density-based selection.
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


class SignalController:
    """Holds live state for a single intersection with N lanes."""

    def __init__(self, lane_names: list[str]):
        self.state = SignalState(lanes=[LaneState(name=n) for n in lane_names])

    def update(self, counts: list[int], emergencies: list[bool]) -> SignalState:
        for lane, count, emergency in zip(self.state.lanes, counts, emergencies):
            lane.vehicle_count = count
            lane.emergency = emergency

        emergency_idx = next(
            (i for i, l in enumerate(self.state.lanes) if l.emergency), None
        )

        if emergency_idx is not None:
            self.state.active_lane = emergency_idx
            self.state.green_seconds = settings.max_green_seconds
            self.state.reason = "emergency_vehicle"
            return self.state

        densest_idx = max(
            range(len(self.state.lanes)),
            key=lambda i: self.state.lanes[i].vehicle_count,
        )
        density = self.state.lanes[densest_idx].vehicle_count
        green = min(
            settings.max_green_seconds,
            max(settings.min_green_seconds, settings.min_green_seconds + density * 2),
        )
        self.state.active_lane = densest_idx
        self.state.green_seconds = green
        self.state.reason = "density"
        return self.state
