"""
geometry.py

Reusable CadQuery geometry and placement helpers for the spacecraft model.

Coordinate convention
---------------------
+X : telescope / forward
-X : propulsion / aft
+Y/-Y : solar-array sides
+Z/-Z : remaining body faces

Most primitives are created with their *local +X axis* as the functional axis.
The orient_x_axis_to_vector() helper then rotates local +X onto a requested
body-frame direction vector.
"""

from __future__ import annotations

import math
from typing import Iterable, Tuple

import cadquery as cq


Vector3 = Tuple[float, float, float]


def _normalise(v: Iterable[float]) -> Vector3:
    x, y, z = (float(a) for a in v)
    mag = math.sqrt(x * x + y * y + z * z)
    if mag <= 1e-12:
        raise ValueError("Orientation vector must be non-zero.")
    return x / mag, y / mag, z / mag


def make_box(length: float, width: float, height: float) -> cq.Workplane:
    """
    Create a box centred at the local origin.

    Dimensions map directly to spacecraft axes before any rotation:
      length -> local X
      width  -> local Y
      height -> local Z
    """
    return cq.Workplane("XY").box(length, width, height, centered=(True, True, True))


def make_capsule_tank(length: float, width: float, height: float) -> cq.Workplane:
    """
    Approximate an axisymmetric pressure vessel aligned with local +X.

    For the current PEPT-200 envelope (225 x 207 x 207 mm), this produces a
    capsule-like tank using a short cylindrical barrel and spherical end regions.

    If width and height differ, the smaller transverse dimension is used as the
    conservative circular diameter.
    """
    diameter = min(width, height)
    radius = diameter / 2.0
    barrel_length = max(0.0, length - 2.0 * radius)

    if barrel_length <= 1e-9:
        return cq.Workplane("XY").sphere(length / 2.0)

    barrel = (
        cq.Workplane("YZ")
        .circle(radius)
        .extrude(barrel_length)
        .translate((-barrel_length / 2.0, 0.0, 0.0))
    )

    left = cq.Workplane("XY").sphere(radius).translate(
        (-barrel_length / 2.0, 0.0, 0.0)
    )
    right = cq.Workplane("XY").sphere(radius).translate(
        (barrel_length / 2.0, 0.0, 0.0)
    )

    return barrel.union(left).union(right)


def orient_x_axis_to_vector(
    shape: cq.Workplane,
    direction: Iterable[float],
) -> cq.Workplane:
    """
    Rotate a shape so its local +X axis points along `direction`.

    Uses an axis-angle rotation and handles parallel/anti-parallel edge cases.
    """
    tx, ty, tz = _normalise(direction)

    # Local reference axis.
    rx, ry, rz = 1.0, 0.0, 0.0

    dot = max(-1.0, min(1.0, rx * tx + ry * ty + rz * tz))

    # Already aligned.
    if dot > 1.0 - 1e-12:
        return shape

    # Exactly opposite: rotate 180 degrees around +Z.
    if dot < -1.0 + 1e-12:
        return shape.rotate((0, 0, 0), (0, 0, 1), 180.0)

    # cross(local_x, target)
    ax = ry * tz - rz * ty
    ay = rz * tx - rx * tz
    az = rx * ty - ry * tx
    axis_mag = math.sqrt(ax * ax + ay * ay + az * az)
    ax, ay, az = ax / axis_mag, ay / axis_mag, az / axis_mag

    angle_deg = math.degrees(math.acos(dot))
    return shape.rotate((0, 0, 0), (ax, ay, az), angle_deg)


def place(
    shape: cq.Workplane,
    position_mm: Iterable[float],
    orientation: Iterable[float] | None = None,
) -> cq.Workplane:
    """
    Orient and translate a shape into the spacecraft body frame.

    Parameters
    ----------
    shape
        Shape whose functional/local reference axis is +X.
    position_mm
        Body-frame component-centre position (x, y, z), mm.
    orientation
        Optional body-frame unit vector that local +X should point toward.
    """
    result = shape

    if orientation is not None:
        result = orient_x_axis_to_vector(result, orientation)

    x, y, z = (float(v) for v in position_mm)
    return result.translate((x, y, z))


def make_bus_frame(
    length: float,
    width: float,
    height: float,
    member: float = 12.0,
) -> cq.Workplane:
    """
    Create a simple 12-edge structural frame for visualising the bus envelope.

    This is intentionally *not* a structural design.  It is preferable to a
    solid bus block for V1 because the internal payload/tank/battery remain
    visible in STEP viewers.
    """
    hx, hy, hz = length / 2.0, width / 2.0, height / 2.0

    result = None

    def add(shape):
        nonlocal result
        result = shape if result is None else result.union(shape)

    # Four X-directed members.
    for y in (-hy, hy):
        for z in (-hz, hz):
            add(make_box(length, member, member).translate((0, y, z)))

    # Four Y-directed members.
    for x in (-hx, hx):
        for z in (-hz, hz):
            add(make_box(member, width, member).translate((x, 0, z)))

    # Four Z-directed members.
    for x in (-hx, hx):
        for y in (-hy, hy):
            add(make_box(member, member, height).translate((x, y, 0)))

    return result


def make_axis_indicator(length: float = 120.0, radius: float = 2.5):
    """
    Return three thin cylinders representing +X/+Y/+Z axes.

    The return value is a dict so they can be added to an Assembly separately.
    """
    x_axis = (
        cq.Workplane("YZ")
        .circle(radius)
        .extrude(length)
    )
    y_axis = (
        cq.Workplane("XZ")
        .circle(radius)
        .extrude(length)
    )
    z_axis = (
        cq.Workplane("XY")
        .circle(radius)
        .extrude(length)
    )
    return {"X axis": x_axis, "Y axis": y_axis, "Z axis": z_axis}
