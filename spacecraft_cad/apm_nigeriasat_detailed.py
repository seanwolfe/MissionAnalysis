"""
apm_nigeriasat_detailed.py

Detailed, parametric spacecraft-level representation of the SSTL NigeriaSat-2
heritage two-axis antenna pointing mechanism (APM), based on the user-provided
reference photographs/CAD images.

The available references clearly show the architecture but do NOT provide a
complete manufacturing drawing.  Therefore this model reproduces the major
flight-hardware features and proportions without claiming exact vendor geometry.

Architecture represented
-------------------------
- circular spacecraft mounting flange
- cylindrical lower electronics module
- azimuth drive / bearing ring
- geared azimuth turntable
- elevation yoke / side frames
- elevation motor and gear housings
- rectangular horn antenna
- thin front radome
- septum-polarizer / throat block
- representative cable / harness runs
- mounting bolts and bearing details

Coordinate convention
---------------------
LOCAL +X = nominal antenna boresight.
The mounting interface is at local X = 0.
The entire APM extends predominantly in +X, away from the spacecraft.

This is deliberate: the spacecraft placement sheet puts the APM reference point
on the +Z bus face and stores orientation (0,0,+1).  The generic placement helper
therefore maps local +X directly to spacecraft +Z and keeps the mounting flange
on the spacecraft surface.

Representative envelope
-----------------------
Because the NigeriaSat-2 reference does not provide a complete dimensioned
drawing in the supplied material, the default envelope is an engineering
approximation chosen from the photo proportions:
    mounting flange OD  : 150 mm
    horn mouth          : 92 x 92 mm
    overall height      : ~225 mm from mounting plane to radome
    max transverse span : ~175 mm including motors/yoke

All dimensions are parameters below and can be changed once a better mechanical
drawing is available.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import cadquery as cq


@dataclass(frozen=True)
class APMParameters:
    # Mount / lower electronics
    flange_diameter: float = 150.0
    flange_thickness: float = 7.0
    flange_hole_count: int = 8
    flange_hole_circle_diameter: float = 126.0
    flange_hole_diameter: float = 5.0

    electronics_diameter: float = 116.0
    electronics_height: float = 46.0
    electronics_ring_diameter: float = 124.0
    electronics_ring_height: float = 5.0

    # Azimuth drive / bearing
    azimuth_bearing_diameter: float = 132.0
    azimuth_bearing_height: float = 16.0
    azimuth_gear_diameter: float = 138.0
    azimuth_gear_height: float = 7.0
    gear_tooth_count: int = 36
    gear_tooth_radial: float = 3.0
    gear_tooth_tangential: float = 5.0

    # Elevation structure
    yoke_base_height: float = 10.0
    yoke_span: float = 126.0
    yoke_arm_width: float = 14.0
    yoke_arm_depth: float = 16.0
    elevation_axis_height: float = 123.0

    # Elevation drive
    elevation_motor_diameter: float = 30.0
    elevation_motor_length: float = 47.0
    elevation_gear_diameter: float = 42.0
    elevation_gear_thickness: float = 8.0

    # Horn / RF front end
    horn_length: float = 82.0
    horn_throat_y: float = 28.0
    horn_throat_z: float = 25.0
    horn_mouth_y: float = 92.0
    horn_mouth_z: float = 92.0
    horn_wall: float = 2.5
    horn_center_x: float = 163.0

    throat_block_length: float = 27.0
    throat_block_y: float = 36.0
    throat_block_z: float = 36.0

    radome_thickness: float = 2.5
    radome_overhang: float = 4.0

    # Small actuator / cable hardware
    az_motor_diameter: float = 26.0
    az_motor_length: float = 42.0
    cable_diameter: float = 4.0


def _x_cylinder(length, diameter, x_center, y=0.0, z=0.0):
    return (
        cq.Workplane("YZ")
        .circle(diameter/2.0)
        .extrude(length/2.0, both=True)
        .translate((x_center,y,z))
    )


def _x_ring(length, outer_d, inner_d, x_center):
    return (
        cq.Workplane("YZ")
        .circle(outer_d/2.0)
        .circle(inner_d/2.0)
        .extrude(length/2.0, both=True)
        .translate((x_center,0.0,0.0))
    )


def _box(l,w,h,x=0.0,y=0.0,z=0.0):
    return (
        cq.Workplane("XY")
        .box(l,w,h,centered=(True,True,True))
        .translate((x,y,z))
    )


def _rect_frustum(length, rear_y, rear_z, front_y, front_z, x_start):
    """
    Solid rectangular frustum aligned along +X.
    """
    return (
        cq.Workplane("YZ")
        .workplane(offset=x_start)
        .rect(rear_y, rear_z)
        .workplane(offset=length)
        .rect(front_y, front_z)
        .loft(combine=True)
    )


def _hollow_rect_frustum(
    length, rear_y, rear_z, front_y, front_z, wall, x_start
):
    outer = _rect_frustum(
        length, rear_y, rear_z, front_y, front_z, x_start
    )
    inner = _rect_frustum(
        length + 2.0,
        max(2.0, rear_y - 2*wall),
        max(2.0, rear_z - 2*wall),
        max(2.0, front_y - 2*wall),
        max(2.0, front_z - 2*wall),
        x_start - 1.0
    )
    return outer.cut(inner)


def make_detailed_apm(p: APMParameters = APMParameters()) -> cq.Workplane:
    parts = []

    # ------------------------------------------------------------------
    # Spacecraft mounting flange at X=0.
    # ------------------------------------------------------------------
    flange = _x_cylinder(
        p.flange_thickness,
        p.flange_diameter,
        p.flange_thickness/2.0
    )
    parts.append(flange.val())

    # Mounting bolt heads around flange.
    bolt_r = p.flange_hole_circle_diameter/2.0
    for i in range(p.flange_hole_count):
        a = math.radians(i*360.0/p.flange_hole_count)
        y = bolt_r*math.cos(a)
        z = bolt_r*math.sin(a)
        bolt = _x_cylinder(2.0, 6.5, -0.8, y, z)
        parts.append(bolt.val())

    # ------------------------------------------------------------------
    # Lower electronics module.
    # ------------------------------------------------------------------
    elec_x = p.flange_thickness + p.electronics_height/2.0
    electronics = _x_cylinder(
        p.electronics_height,
        p.electronics_diameter,
        elec_x
    )
    parts.append(electronics.val())

    # Upper/lower retaining rings.
    for x in (
        p.flange_thickness + 4.0,
        p.flange_thickness + p.electronics_height - 4.0
    ):
        ring = _x_ring(
            p.electronics_ring_height,
            p.electronics_ring_diameter,
            p.electronics_diameter - 4.0,
            x
        )
        parts.append(ring.val())

    # Electronics connector block.
    conn = _box(
        12.0, 20.0, 16.0,
        x=elec_x - 6.0,
        y=p.electronics_diameter/2.0 + 6.0,
        z=-20.0
    )
    parts.append(conn.val())

    # ------------------------------------------------------------------
    # Azimuth drive/bearing.
    # ------------------------------------------------------------------
    az_x = (
        p.flange_thickness + p.electronics_height
        + p.azimuth_bearing_height/2.0
    )
    bearing = _x_ring(
        p.azimuth_bearing_height,
        p.azimuth_bearing_diameter,
        92.0,
        az_x
    )
    parts.append(bearing.val())

    gear_x = (
        p.flange_thickness + p.electronics_height
        + p.azimuth_bearing_height
        + p.azimuth_gear_height/2.0
    )
    gear_ring = _x_ring(
        p.azimuth_gear_height,
        p.azimuth_gear_diameter,
        p.azimuth_bearing_diameter - 10.0,
        gear_x
    )
    parts.append(gear_ring.val())

    # Visual gear teeth around azimuth ring.
    rr = p.azimuth_gear_diameter/2.0 + p.gear_tooth_radial/2.0
    for i in range(p.gear_tooth_count):
        ang = i*360.0/p.gear_tooth_count
        a = math.radians(ang)
        y = rr*math.cos(a)
        z = rr*math.sin(a)
        tooth = _box(
            p.azimuth_gear_height,
            p.gear_tooth_radial,
            p.gear_tooth_tangential,
            x=gear_x,
            y=y,
            z=z
        ).rotate((gear_x,0,0),(1,0,0),ang)
        parts.append(tooth.val())

    # Azimuth motor, offset to one side of turntable.
    az_motor = _x_cylinder(
        p.az_motor_length,
        p.az_motor_diameter,
        gear_x + 10.0,
        y=-(p.azimuth_gear_diameter/2.0 + p.az_motor_diameter/2.0 - 5.0),
        z=-26.0
    )
    parts.append(az_motor.val())

    # ------------------------------------------------------------------
    # Elevation yoke.
    # ------------------------------------------------------------------
    yoke_base_x = (
        p.flange_thickness + p.electronics_height
        + p.azimuth_bearing_height + p.azimuth_gear_height
        + p.yoke_base_height/2.0
    )
    yoke_base = _box(
        p.yoke_base_height,
        p.yoke_span,
        18.0,
        x=yoke_base_x
    )
    parts.append(yoke_base.val())

    axis_x = p.elevation_axis_height

    # Two yoke arms rising toward the elevation axis.
    for ysign in (-1.0, 1.0):
        y = ysign*(p.yoke_span/2.0 - p.yoke_arm_width/2.0)
        arm_len = axis_x - yoke_base_x
        arm = _box(
            arm_len,
            p.yoke_arm_width,
            p.yoke_arm_depth,
            x=(axis_x + yoke_base_x)/2.0,
            y=y,
            z=0.0
        )
        parts.append(arm.val())

    # Elevation-axis bearing housings.
    for ysign in (-1.0, 1.0):
        y = ysign*(p.yoke_span/2.0 - p.yoke_arm_width/2.0)
        bearing_h = _x_cylinder(
            13.0,
            24.0,
            axis_x,
            y=y,
            z=0.0
        )
        parts.append(bearing_h.val())

    # Elevation motor and gear on one side.
    elev_motor = (
        cq.Workplane("XZ")
        .circle(p.elevation_motor_diameter/2.0)
        .extrude(p.elevation_motor_length/2.0, both=True)
        .translate((axis_x, p.yoke_span/2.0 + p.elevation_motor_length/2.0 - 8.0, 0.0))
    )
    parts.append(elev_motor.val())

    elev_gear = (
        cq.Workplane("XZ")
        .circle(p.elevation_gear_diameter/2.0)
        .extrude(p.elevation_gear_thickness/2.0, both=True)
        .translate((axis_x, p.yoke_span/2.0 - 2.0, 0.0))
    )
    parts.append(elev_gear.val())

    # Opposite-side passive bearing cap.
    cap = (
        cq.Workplane("XZ")
        .circle(18.0)
        .extrude(6.0/2.0, both=True)
        .translate((axis_x, -p.yoke_span/2.0 + 2.0, 0.0))
    )
    parts.append(cap.val())

    # ------------------------------------------------------------------
    # RF throat / polarizer and horn.
    # ------------------------------------------------------------------
    throat_x = axis_x + p.throat_block_length/2.0
    throat = _box(
        p.throat_block_length,
        p.throat_block_y,
        p.throat_block_z,
        x=throat_x
    )
    try:
        throat = throat.edges("|X").fillet(2.0)
    except Exception:
        pass
    parts.append(throat.val())

    horn_start = axis_x + p.throat_block_length - 2.0
    horn = _hollow_rect_frustum(
        p.horn_length,
        p.horn_throat_y,
        p.horn_throat_z,
        p.horn_mouth_y,
        p.horn_mouth_z,
        p.horn_wall,
        horn_start
    )
    parts.append(horn.val())

    # Front radome, represented as a thin plate over the horn mouth.
    horn_front = horn_start + p.horn_length
    radome = _box(
        p.radome_thickness,
        p.horn_mouth_y + 2*p.radome_overhang,
        p.horn_mouth_z + 2*p.radome_overhang,
        x=horn_front + p.radome_thickness/2.0
    )
    try:
        radome = radome.edges("|X").fillet(1.0)
    except Exception:
        pass
    parts.append(radome.val())

    # ------------------------------------------------------------------
    # Representative cable loops / harness segments.
    # Straight/angled segments are enough to make the assembly read as
    # real flight hardware without creating fragile spline geometry.
    # ------------------------------------------------------------------
    cable1 = _box(
        48.0,
        p.cable_diameter,
        p.cable_diameter,
        x=100.0,
        y=-72.0,
        z=24.0
    ).rotate((100.0,-72.0,24.0),(0,1,0),-22.0)
    parts.append(cable1.val())

    cable2 = _box(
        38.0,
        p.cable_diameter,
        p.cable_diameter,
        x=139.0,
        y=-60.0,
        z=28.0
    ).rotate((139.0,-60.0,28.0),(0,1,0),18.0)
    parts.append(cable2.val())

    return cq.Workplane("XY").newObject([cq.Compound.makeCompound(parts)])


def make_apm_assembly(p: APMParameters = APMParameters()) -> cq.Assembly:
    assy = cq.Assembly(name="SSTL_NigeriaSat2_APM")
    assy.add(
        make_detailed_apm(p),
        name="NigeriaSat-2 Heritage APM",
        color=cq.Color(0.62,0.62,0.60),
    )
    return assy


if __name__ == "__main__":
    from pathlib import Path

    out = Path(__file__).resolve().parent / "outputs"
    out.mkdir(exist_ok=True)

    apm = make_detailed_apm()
    assy = make_apm_assembly()

    cq.exporters.export(apm, str(out / "nigeriasat2_apm_detailed.step"))
    assy.save(str(out / "nigeriasat2_apm_detailed_assembly.step"))
    cq.exporters.export(apm, str(out / "nigeriasat2_apm_detailed.stl"))

    print("Exported detailed NigeriaSat-2 heritage APM geometry to:", out)
