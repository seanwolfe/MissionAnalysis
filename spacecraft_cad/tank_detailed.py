"""
tank_detailed.py

Higher-craftsmanship CadQuery model of the Rafael PEPT-200-style propellant tank,
based on the user-provided dimensional drawing and reference photograph.

Modeling intent
---------------
This is a spacecraft-level external CAD representation, not manufacturing CAD.
It captures the features that materially improve visual realism:

- near-spherical pressure vessel
- layered equatorial flange / seam
- six mounting lugs with bolt holes
- upper and lower port bosses
- upper and lower tubing / feed stubs
- softened edges where practical

Coordinate convention for THIS COMPONENT
----------------------------------------
Local +X is the tank polar / axial direction.
The equatorial flange lies in the local YZ plane.

That convention is intentional: it matches the spacecraft project's convention
where component "length" maps to spacecraft X before placement.

Drawing-derived / drawing-guided dimensions used here
------------------------------------------------------
The supplied drawing visibly indicates approximately:
- vessel/equatorial body diameter: 193.1 mm
- overall axial height: ~200 mm
- equatorial envelope / flange region: ~207.35 mm
- six mounting holes: Ø5.2 mm
- mounting pattern: 6-fold / 60 deg
- upper tube radial reach dimension: ~115 mm
- lower boss width: ~44 mm

A few small secondary dimensions are represented by visually faithful engineering
defaults because the raster drawing does not expose every tolerance/feature
unambiguously.  All such values are parameters below and easy to revise.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import cadquery as cq


@dataclass(frozen=True)
class TankParameters:
    # Main vessel
    shell_diameter: float = 193.1
    overall_axial_height: float = 200.0

    # Equatorial flange / seam
    flange_outer_diameter: float = 207.35
    flange_thickness: float = 7.0
    seam_ring_outer_diameter: float = 201.0
    seam_ring_thickness: float = 2.2
    seam_ring_offset: float = 5.0

    # Six mounting lugs / holes
    lug_count: int = 6
    lug_radial_length: float = 13.0
    lug_tangential_width: float = 18.0
    lug_thickness: float = 6.0
    lug_hole_diameter: float = 5.2
    lug_hole_radial_offset: float = 8.5

    # Polar bosses
    upper_boss_diameter: float = 34.0
    upper_boss_length: float = 11.0
    upper_boss_neck_diameter: float = 22.0
    upper_boss_neck_length: float = 6.0

    lower_boss_diameter: float = 44.0
    lower_boss_length: float = 9.0
    lower_boss_neck_diameter: float = 22.0
    lower_boss_neck_length: float = 6.0

    # Tubing
    tube_outer_diameter: float = 5.5
    upper_tube_length: float = 92.0
    lower_tube_length: float = 35.0
    upper_tube_angle_deg: float = 18.0
    lower_tube_angle_deg: float = -12.0

    # Small visual fillets
    lug_fillet: float = 3.0


def _x_cylinder(length: float, diameter: float, x0: float = 0.0) -> cq.Workplane:
    """Cylinder whose axis is local +X, starting at x=x0."""
    return (
        cq.Workplane("YZ")
        .circle(diameter / 2.0)
        .extrude(length)
        .translate((x0, 0.0, 0.0))
    )


def _centered_x_cylinder(length: float, diameter: float) -> cq.Workplane:
    return _x_cylinder(length, diameter, -length / 2.0)


def _make_lug(p: TankParameters, angle_deg: float) -> cq.Workplane:
    """
    One equatorial mounting lug with a through-hole.

    It is sketched directly in the YZ plane and extruded along X.  Building it
    this way is much more robust than rotating a box and then performing a
    boolean cut in a differently oriented workplane.
    """
    r_flange = p.flange_outer_diameter / 2.0
    radial_overlap = 2.0
    r_center = r_flange + p.lug_radial_length / 2.0 - radial_overlap

    # Start with the lug centred on +Y.
    lug = (
        cq.Workplane("YZ")
        .center(r_center, 0.0)
        .rect(p.lug_radial_length, p.lug_tangential_width)
        .extrude(p.lug_thickness / 2.0, both=True)
    )

    # Through-hole, also authored in YZ so the drill axis is X.
    r_hole = r_flange + p.lug_hole_radial_offset
    hole = (
        cq.Workplane("YZ")
        .center(r_hole, 0.0)
        .circle(p.lug_hole_diameter / 2.0)
        .extrude(p.lug_thickness, both=True)
    )

    lug = lug.cut(hole)

    # Distribute identical lugs around the tank X axis.
    if abs(angle_deg) > 1e-12:
        lug = lug.rotate((0, 0, 0), (1, 0, 0), angle_deg)

    return lug


def _make_tube(
    start_x: float,
    start_y: float,
    start_z: float,
    length: float,
    diameter: float,
    angle_deg: float,
    azimuth_deg: float = 0.0,
) -> cq.Workplane:
    """
    Straight feed/fill tube represented as a swept cylinder.

    Tube begins at a boss and points mostly radially outward in +Y with a small
    axial X inclination.  `azimuth_deg` rotates the tube around the X axis.
    """
    a = math.radians(angle_deg)

    # Direction before azimuth rotation: axial X component + radial +Y component.
    dx = math.sin(a)
    dr = math.cos(a)

    az = math.radians(azimuth_deg)
    dy = dr * math.cos(az)
    dz = dr * math.sin(az)

    end = (
        start_x + length * dx,
        start_y + length * dy,
        start_z + length * dz,
    )

    path = cq.Workplane("XY").moveTo(start_x, start_y).workplane(offset=start_z)
    # Robust 3D tube construction using a cylinder aligned to direction vector:
    v = cq.Vector(end[0] - start_x, end[1] - start_y, end[2] - start_z)
    mag = v.Length
    direction = v.normalized()

    # Build along +X then rotate local +X to direction.
    tube = _x_cylinder(mag, diameter, 0.0)

    ref = cq.Vector(1, 0, 0)
    dot = max(-1.0, min(1.0, ref.dot(direction)))
    if dot < 1.0 - 1e-10:
        if dot < -1.0 + 1e-10:
            tube = tube.rotate((0, 0, 0), (0, 0, 1), 180)
        else:
            axis = ref.cross(direction)
            angle = math.degrees(math.acos(dot))
            tube = tube.rotate((0, 0, 0), axis.toTuple(), angle)

    return tube.translate((start_x, start_y, start_z))



def make_detailed_tank(p: TankParameters = TankParameters()) -> cq.Workplane:
    """
    Build the complete external tank representation as a CadQuery compound.

    IMPORTANT:
    The pressure vessel is kept as a complete sphere.  The flange, rings, lugs,
    bosses, and tubes are separate solids collected into one compound instead
    of being fused into the sphere.  This avoids OpenCascade boolean-fusion
    artifacts that can accidentally remove one hemisphere when several solids
    meet tangentially at the equator.

    The returned object behaves like a normal CadQuery Workplane for placement
    and STEP export, while preserving both upper and lower hemispheres.
    """
    shell_r = p.shell_diameter / 2.0
    parts = []

    # Full spherical pressure vessel.
    shell = cq.Workplane("XY").sphere(shell_r)
    parts.append(shell.val())

    # Main equatorial flange.
    flange = _centered_x_cylinder(
        p.flange_thickness,
        p.flange_outer_diameter,
    )
    parts.append(flange.val())

    # Secondary seam / stiffening rings.
    for sgn in (-1.0, 1.0):
        ring = _centered_x_cylinder(
            p.seam_ring_thickness,
            p.seam_ring_outer_diameter,
        ).translate((sgn * p.seam_ring_offset, 0.0, 0.0))
        parts.append(ring.val())

    # Six mounting lugs with through-holes.
    for i in range(p.lug_count):
        lug = _make_lug(p, i * 360.0 / p.lug_count)
        parts.append(lug.val())

    # +X polar boss stack.
    upper_shell_x = shell_r
    upper_neck = _x_cylinder(
        p.upper_boss_neck_length,
        p.upper_boss_neck_diameter,
        upper_shell_x - 1.0,
    )
    upper_boss = _x_cylinder(
        p.upper_boss_length,
        p.upper_boss_diameter,
        upper_shell_x + p.upper_boss_neck_length - 2.0,
    )
    parts.extend([upper_neck.val(), upper_boss.val()])

    # -X polar boss stack.
    lower_neck = _x_cylinder(
        p.lower_boss_neck_length,
        p.lower_boss_neck_diameter,
        -upper_shell_x - p.lower_boss_neck_length + 1.0,
    )
    lower_boss = _x_cylinder(
        p.lower_boss_length,
        p.lower_boss_diameter,
        -upper_shell_x - p.lower_boss_neck_length - p.lower_boss_length + 2.0,
    )
    parts.extend([lower_neck.val(), lower_boss.val()])

    # Feed / fill tubes.
    upper_tube = _make_tube(
        start_x=upper_shell_x + p.upper_boss_neck_length + p.upper_boss_length * 0.65,
        start_y=p.upper_boss_diameter * 0.35,
        start_z=0.0,
        length=p.upper_tube_length,
        diameter=p.tube_outer_diameter,
        angle_deg=p.upper_tube_angle_deg,
        azimuth_deg=12.0,
    )
    parts.append(upper_tube.val())

    lower_tube = _make_tube(
        start_x=-upper_shell_x - p.lower_boss_neck_length - p.lower_boss_length * 0.65,
        start_y=-p.lower_boss_diameter * 0.30,
        start_z=0.0,
        length=p.lower_tube_length,
        diameter=p.tube_outer_diameter,
        angle_deg=p.lower_tube_angle_deg,
        azimuth_deg=205.0,
    )
    parts.append(lower_tube.val())

    compound = cq.Compound.makeCompound(parts)
    return cq.Workplane("XY").newObject([compound])

def make_tank_assembly(p: TankParameters = TankParameters()) -> cq.Assembly:
    """
    Coloured assembly wrapper for viewers that preserve STEP assembly colours.
    """
    assy = cq.Assembly(name="Detailed_PEPT200_Tank")
    assy.add(
        make_detailed_tank(p),
        name="PEPT-200 tank",
        color=cq.Color(0.48, 0.50, 0.52),
    )
    return assy


if __name__ == "__main__":
    from pathlib import Path

    out = Path(__file__).resolve().parent / "outputs"
    out.mkdir(exist_ok=True)

    tank = make_detailed_tank()
    assy = make_tank_assembly()

    cq.exporters.export(tank, str(out / "pept200_detailed_tank.step"))
    assy.save(str(out / "pept200_detailed_tank_assembly.step"))
    cq.exporters.export(tank, str(out / "pept200_detailed_tank.stl"))

    print("Exported detailed tank geometry to:", out)
