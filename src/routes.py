"""Taught-path representation and geometric reference utilities.

This module returns route information only. It does not clamp policy speeds.
"""

import numpy as np
from config import SPAN

def angle_wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


class Route:
    def __init__(self, scenario):
        self.figure8 = scenario.figure8
        if scenario.figure8:
            # Analytic proxy for the CSA repeated loop. Its lead-in is tangent to the crossing.
            t = np.linspace(0, 4 * np.pi, 12001)
            a, b = 22.0, 11.0
            heading = np.arctan2(2 * b, a)
            u, v = a * np.sin(t), b * np.sin(2 * t)
            x = 8 + u * np.cos(heading) + v * np.sin(heading)
            y = -u * np.sin(heading) + v * np.cos(heading)
            x, y = np.r_[np.linspace(0, 8, 161)[:-1], x], np.r_[np.zeros(160), y]
            distance = np.r_[0.0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
            self.finish = distance[-1] - SPAN - 0.2
        else:
            end = max(30, scenario.distance + 12)
            x = np.linspace(-1, end, round((end + 1) / 0.05) + 1)
            curve_x = np.maximum(x - 4.5, 0)
            y = scenario.amplitude * (1 - np.cos(scenario.frequency * curve_x))
            distance = np.r_[0.0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))] - 1
            self.finish = scenario.distance
        self.s = np.arange(distance[0], distance[-1], 0.05)
        self.xy = np.column_stack([np.interp(self.s, distance, x), np.interp(self.s, distance, y)])
        derivative = np.gradient(self.xy, axis=0)
        self.yaw = np.unwrap(np.arctan2(derivative[:, 1], derivative[:, 0]))
        self.curvature = np.gradient(self.yaw, self.s)

    def track(self, xy, last):
        # An arclength window disambiguates a known route at its self-intersection.
        lo, hi = max(0, last - 14), min(len(self.s), last + 45)
        segment = self.xy[lo:hi] - xy
        index = lo + np.argmin(np.sum(segment * segment, axis=1))
        ref_yaw = self.yaw[index]
        dx, dy = xy - self.xy[index]
        cross = -dx * np.sin(ref_yaw) + dy * np.cos(ref_yaw)
        progress = self.s[index] + dx * np.cos(ref_yaw) + dy * np.sin(ref_yaw)
        return index, progress, cross


def route_poses(route, distance):
    return np.column_stack([np.interp(distance, route.s, column)
                            for column in (*route.xy.T, route.yaw)])


def projected_distance(route, xy, *, hint=None):
    """Project onto the path, using arclength to select the branch on a loop."""
    if route.figure8 and hint is None:
        raise ValueError('Figure-eight projection requires an arclength hint.')
    distance = (np.asarray(hint).copy() if hint is not None
                else np.interp(xy[:, 0], route.xy[:, 0], route.s))
    for _ in range(4):
        poses = route_poses(route, distance)
        dx, dy = (xy - poses[:, :2]).T
        yaw = poses[:, 2]
        distance = np.clip(distance + dx * np.cos(yaw) + dy * np.sin(yaw), route.s[0], route.s[-1])
    return distance


def follower_references(route, lead_xy, *, lead_distance_hint=None):
    """Find a point on the path a nominal Euclidean chord behind each lead pose."""
    distance = projected_distance(route, lead_xy, hint=lead_distance_hint) - SPAN
    for _ in range(5):
        poses = route_poses(route, distance)
        dx, dy = (lead_xy - poses[:, :2]).T
        separation = np.hypot(dx, dy)
        tangent = dx * np.cos(poses[:, 2]) + dy * np.sin(poses[:, 2])
        distance += (separation - SPAN) * separation / np.maximum(tangent, 0.5)
    return route_poses(route, distance)

