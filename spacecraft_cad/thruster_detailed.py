"""
thruster_detailed.py

Higher-craftsmanship CadQuery representation of the Aerojet Rocketdyne MR-103G
1-N-class hydrazine thruster selected in the spacecraft systems workbook.

Source geometry
---------------
The user-provided dimensional drawing shows the following key envelope/interface
values:
    overall length        : 173 mm  (113 mm + 60 mm around the mounting interface)
    envelope width/height : approximately 40 mm in the current system workbook
    mounting-interface to nozzle-side extreme : 113 mm max
    mounting-interface to valve-side extreme  : 60 mm max
    small aft feature     : 7 mm max shown on drawing

The photo also shows the characteristic architecture:
    - long nozzle / chamber assembly
    - multiple cylindrical thermal / structural bands
    - mounting interface ring / bracket
    - compact valve/inlet hardware behind the mounting plane
    - small propellant tube / electrical lead detail

This is spacecraft-configuration CAD, not manufacturing CAD.

Local component coordinates
---------------------------
+X : PLUME / nozzle-exit direction
-X : valve / spacecraft feed-interface direction
Y,Z: transverse directions

This convention is deliberate because the master workbook's thruster
orientation vectors are plume-direction vectors.  Therefore the standard
`place()` helper can align the detailed thruster local +X axis directly to the
spreadsheet orientation vector.

The component is centred on its 173 mm overall envelope so the existing
placement vectors remain valid.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import cadquery as cq


@dataclass(frozen=True)
class ThrusterParameters:
    # Authoritative current system envelope
    overall_length: float = 173.0
    max_diameter: float = 40.0

    # Drawing interface location
    nozzle_side_from_mount: float = 113.0
    valve_side_from_mount: float = 60.0

    # Nozzle / chamber
    nozzle_exit_diameter: float = 38.0
    nozzle_throat_diameter: float = 12.0
    nozzle_length: float = 76.0
    chamber_diameter: float = 25.0
    chamber_length: float = 30.0

    # Exterior bands visible in photo
    thermal_band_count: int = 5
    thermal_band_width: float = 4.0
    thermal_band_raise: float = 1.4

    # Mounting interface
    mount_flange_diameter: float = 40.0
    mount_flange_axial: float = 6.0
    mount_hole_count: int = 4
    mount_hole_diameter: float = 3.2
    mount_hole_circle_diameter: float = 31.0

    # Valve / aft hardware
    valve_body_diameter: float = 20.0
    valve_body_length: float = 28.0
    aft_neck_diameter: float = 12.0
    aft_neck_length: float = 19.0
    inlet_stub_diameter: float = 7.0
    inlet_stub_length: float = 13.0

    # Side tube / lead details
    side_tube_diameter: float = 3.0
    side_tube_length: float = 58.0
    clamp_count: int = 3
    clamp_width: float = 5.0
    clamp_thickness: float = 2.5


def _x_cylinder(length: float, diameter: float, x_center: float, y=0.0, z=0.0):
    return (
        cq.Workplane("YZ")
        .circle(diameter / 2.0)
        .extrude(length / 2.0, both=True)
        .translate((x_center, y, z))
    )


def _x_ring(length: float, outer_d: float, inner_d: float, x_center: float):
    return (
        cq.Workplane("YZ")
        .circle(outer_d / 2.0)
        .circle(inner_d / 2.0)
        .extrude(length / 2.0, both=True)
        .translate((x_center, 0.0, 0.0))
    )


def _frustum(length: float, d1: float, d2: float, x_start: float):
    """
    Solid frustum aligned along +X.
    """
    return (
        cq.Workplane("YZ")
        .workplane(offset=x_start)
        .circle(d1 / 2.0)
        .workplane(offset=length)
        .circle(d2 / 2.0)
        .loft(combine=True)
    )


def _box(l, w, h, x=0.0, y=0.0, z=0.0):
    return (
        cq.Workplane("XY")
        .box(l, w, h, centered=(True, True, True))
        .translate((x, y, z))
    )


def _orient_x_to_vector(shape: cq.Workplane, vec):
    """
    Local helper for the small side-tube detail.
    """
    tx, ty, tz = [float(v) for v in vec]
    mag = math.sqrt(tx*tx + ty*ty + tz*tz)
    tx, ty, tz = tx/mag, ty/mag, tz/mag

    dot = max(-1.0, min(1.0, tx))  # dot((1,0,0), target)
    if dot > 1.0 - 1e-12:
        return shape
    if dot < -1.0 + 1e-12:
        return shape.rotate((0,0,0),(0,0,1),180.0)

    # cross((1,0,0), target) = (0,-tz,ty)
    axis = (0.0, -tz, ty)
    angle = math.degrees(math.acos(dot))
    return shape.rotate((0,0,0), axis, angle)


def make_detailed_thruster(
    p: ThrusterParameters = ThrusterParameters(),
) -> cq.Workplane:
    """
    Build one detailed MR-103G-style thruster as a compound.

    The 173 mm envelope is centred at x=0.  The mounting plane therefore lies
    at x = +86.5 - 113 = -26.5 mm.
    """
    parts = []

    x_nozzle_tip = p.overall_length / 2.0
    x_aft_tip = -p.overall_length / 2.0
    x_mount = x_nozzle_tip - p.nozzle_side_from_mount

    # ---------------------------------------------------------
    # Nozzle: plume exits toward +X.
    # ---------------------------------------------------------
    nozzle_x0 = x_nozzle_tip - p.nozzle_length
    nozzle = _frustum(
        p.nozzle_length,
        p.nozzle_throat_diameter,
        p.nozzle_exit_diameter,
        nozzle_x0,
    )
    parts.append(nozzle.val())

    # Thin exit lip makes the nozzle silhouette read more like real hardware.
    exit_lip = _x_ring(
        2.4,
        p.nozzle_exit_diameter + 1.5,
        p.nozzle_exit_diameter - 2.5,
        x_nozzle_tip - 1.2,
    )
    parts.append(exit_lip.val())

    # ---------------------------------------------------------
    # Combustion chamber / catalyst bed region.
    # ---------------------------------------------------------
    chamber_center = nozzle_x0 - p.chamber_length / 2.0 + 2.0
    chamber = _x_cylinder(
        p.chamber_length,
        p.chamber_diameter,
        chamber_center,
    )
    parts.append(chamber.val())

    # Shoulder between chamber and nozzle.
    shoulder = _frustum(
        10.0,
        p.chamber_diameter + 2.0,
        p.nozzle_throat_diameter + 3.0,
        nozzle_x0 - 8.0,
    )
    parts.append(shoulder.val())

    # Repeated thermal / structural bands visible along the chamber.
    band_start = chamber_center - p.chamber_length / 2.0 + 5.0
    usable = p.chamber_length - 10.0
    for i in range(p.thermal_band_count):
        xb = band_start + i * usable / max(1, p.thermal_band_count - 1)
        band = _x_ring(
            p.thermal_band_width,
            p.chamber_diameter + 2.0 * p.thermal_band_raise,
            p.chamber_diameter - 1.5,
            xb,
        )
        parts.append(band.val())

    # ---------------------------------------------------------
    # Mounting interface.
    # ---------------------------------------------------------
    mount = _x_ring(
        p.mount_flange_axial,
        p.mount_flange_diameter,
        p.valve_body_diameter - 2.0,
        x_mount,
    )
    parts.append(mount.val())

    # Four bolt heads / interface bosses around mounting flange.
    br = p.mount_hole_circle_diameter / 2.0
    for i in range(p.mount_hole_count):
        a = math.radians(i * 360.0 / p.mount_hole_count)
        y = br * math.cos(a)
        z = br * math.sin(a)
        boss = _x_cylinder(3.2, 6.0, x_mount - 1.5, y, z)
        parts.append(boss.val())

    # ---------------------------------------------------------
    # Valve and feed-interface side (-X of mounting plane).
    # ---------------------------------------------------------
    valve_center = x_mount - p.mount_flange_axial/2.0 - p.valve_body_length/2.0 + 1.0
    valve_body = _x_cylinder(
        p.valve_body_length,
        p.valve_body_diameter,
        valve_center,
    )
    parts.append(valve_body.val())

    # Valve shoulders / collars.
    valve_front_ring = _x_ring(
        4.0,
        p.valve_body_diameter + 5.0,
        p.valve_body_diameter - 2.0,
        x_mount - 7.0,
    )
    parts.append(valve_front_ring.val())

    aft_neck_center = (
        valve_center
        - p.valve_body_length / 2.0
        - p.aft_neck_length / 2.0
        + 2.0
    )
    aft_neck = _x_cylinder(
        p.aft_neck_length,
        p.aft_neck_diameter,
        aft_neck_center,
    )
    parts.append(aft_neck.val())

    inlet_center = x_aft_tip + p.inlet_stub_length / 2.0
    inlet = _x_cylinder(
        p.inlet_stub_length,
        p.inlet_stub_diameter,
        inlet_center,
    )
    parts.append(inlet.val())

    # Small aft fitting ring.
    aft_ring = _x_ring(
        4.0,
        13.0,
        7.5,
        x_aft_tip + p.inlet_stub_length + 1.0,
    )
    parts.append(aft_ring.val())

    # ---------------------------------------------------------
    # Side feed tube / electrical lead visual details.
    # ---------------------------------------------------------
    # Keep the representative side line inside the selected ~40 mm transverse
    # package envelope.  It runs longitudinally along the chamber/valve body,
    # matching the photo without creating a large artificial side protrusion.
    tube_center_x = chamber_center - 12.0
    tube = _x_cylinder(
        p.side_tube_length,
        p.side_tube_diameter,
        tube_center_x,
        y=p.chamber_diameter/2.0 + 4.0,
        z=3.0,
    )
    parts.append(tube.val())

    # Three visible clamp straps around the chamber/nozzle body.
    for i in range(p.clamp_count):
        xc = chamber_center - 9.0 + i * 12.0
        clamp = _x_ring(
            p.clamp_width,
            p.chamber_diameter + 5.0,
            p.chamber_diameter + 1.0,
            xc,
        )
        parts.append(clamp.val())

        # Tiny clamp foot extending on +Z side.
        foot = _box(
            p.clamp_width,
            6.0,
            8.0,
            x=xc,
            y=0.0,
            z=p.chamber_diameter/2.0 + 5.0,
        )
        parts.append(foot.val())

    # Representative rectangular electrical connector close to valve body.
    connector = _box(
        12.0,
        10.0,
        8.0,
        x=x_mount - 15.0,
        y=11.0,
        z=10.0,
    )
    parts.append(connector.val())

    compound = cq.Compound.makeCompound(parts)
    return cq.Workplane("XY").newObject([compound])


def make_thruster_assembly(
    p: ThrusterParameters = ThrusterParameters(),
) -> cq.Assembly:
    """
    Assembly wrapper with metallic / thermal colours.
    """
    assy = cq.Assembly(name="MR103G_Detailed_Thruster")
    assy.add(
        make_detailed_thruster(p),
        name="MR-103G thruster",
        color=cq.Color(0.42, 0.37, 0.28),
    )
    return assy


if __name__ == "__main__":
    from pathlib import Path

    out = Path(__file__).resolve().parent / "outputs"
    out.mkdir(exist_ok=True)

    thruster = make_detailed_thruster()
    assy = make_thruster_assembly()

    cq.exporters.export(thruster, str(out / "mr103g_detailed_thruster.step"))
    assy.save(str(out / "mr103g_detailed_thruster_assembly.step"))
    cq.exporters.export(thruster, str(out / "mr103g_detailed_thruster.stl"))

    print("Exported detailed MR-103G-style thruster geometry to:", out)
