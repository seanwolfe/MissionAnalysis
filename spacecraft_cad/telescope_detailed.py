"""
telescope_detailed.py

Higher-craftsmanship CadQuery model of the optical telescope assembly.

Available design information
----------------------------
The current systems workbook gives the telescope assembly envelope as:
    Length : 580 mm
    Width  : 390 mm
    Height : 340 mm

The only visual reference supplied is a photograph showing a compact
space-telescope optical head with:
- cylindrical optical barrel
- large circular front aperture
- thick front retaining/flange ring
- recessed primary-mirror region
- central secondary-mirror / detector support
- three spider vanes
- rear barrel / adapter section
- external electronics / harness hardware along one side

This model therefore aims for *credible spacecraft-level craftsmanship* rather
than pretending to reproduce a proprietary vendor drawing.

Local component frame
---------------------
+X : optical boresight / forward
-X : rear spacecraft mounting side
Y,Z: transverse telescope cross-section

The component is centred at the local origin so the existing workbook position
vector can be used directly.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import cadquery as cq


@dataclass(frozen=True)
class TelescopeParameters:
    # Master-workbook envelope
    length: float = 580.0
    envelope_width: float = 390.0
    envelope_height: float = 340.0

    # Main optical barrel
    barrel_outer_diameter: float = 315.0
    barrel_length: float = 470.0
    barrel_wall_thickness: float = 7.0

    # Front aperture / flange
    front_flange_outer_diameter: float = 335.0
    front_flange_axial: float = 18.0
    front_lip_outer_diameter: float = 326.0
    front_lip_axial: float = 7.0
    clear_aperture_diameter: float = 276.0

    # Rear adapter
    rear_adapter_diameter: float = 286.0
    rear_adapter_length: float = 72.0
    rear_mount_flange_diameter: float = 306.0
    rear_mount_flange_axial: float = 12.0

    # Primary / secondary optical visual geometry
    primary_mirror_diameter: float = 270.0
    primary_mirror_thickness: float = 8.0
    primary_recess: float = 20.0
    secondary_hub_diameter: float = 66.0
    secondary_hub_length: float = 28.0
    secondary_face_diameter: float = 52.0

    # Spider
    spider_count: int = 3
    spider_thickness: float = 5.0
    spider_width: float = 12.0

    # Front flange fasteners
    front_fastener_count: int = 8
    fastener_head_diameter: float = 9.0
    fastener_head_depth: float = 3.0
    fastener_circle_diameter: float = 307.0

    # External longitudinal ribs
    rib_count: int = 4
    rib_width: float = 9.0
    rib_height: float = 6.0
    rib_length: float = 390.0

    # Representative side electronics / harness tray
    side_box_length: float = 120.0
    side_box_width: float = 44.0
    side_box_height: float = 58.0
    side_box_x: float = 105.0
    side_box_y: float = 168.0
    side_box_z: float = 72.0

    harness_rail_length: float = 235.0
    harness_rail_width: float = 18.0
    harness_rail_height: float = 9.0

    # Mounting feet
    foot_length: float = 46.0
    foot_width: float = 24.0
    foot_height: float = 15.0


def _x_cylinder(
    length: float,
    diameter: float,
    x_center: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
) -> cq.Workplane:
    return (
        cq.Workplane("YZ")
        .circle(diameter / 2.0)
        .extrude(length / 2.0, both=True)
        .translate((x_center, y, z))
    )


def _x_ring(
    length: float,
    outer_diameter: float,
    inner_diameter: float,
    x_center: float,
) -> cq.Workplane:
    return (
        cq.Workplane("YZ")
        .circle(outer_diameter / 2.0)
        .circle(inner_diameter / 2.0)
        .extrude(length / 2.0, both=True)
        .translate((x_center, 0.0, 0.0))
    )


def _box(l, w, h, x=0.0, y=0.0, z=0.0) -> cq.Workplane:
    return (
        cq.Workplane("XY")
        .box(l, w, h, centered=(True, True, True))
        .translate((x, y, z))
    )


def _make_spider_vane(
    x_center: float,
    inner_radius: float,
    outer_radius: float,
    thickness_x: float,
    width_tangential: float,
    angle_deg: float,
) -> cq.Workplane:
    """
    One rectangular spider vane in the front aperture plane.
    """
    radial_len = max(1.0, outer_radius - inner_radius)
    radial_center = (outer_radius + inner_radius) / 2.0

    vane = _box(
        thickness_x,
        radial_len,
        width_tangential,
        x=x_center,
        y=radial_center,
        z=0.0,
    )
    if abs(angle_deg) > 1e-12:
        vane = vane.rotate((0, 0, 0), (1, 0, 0), angle_deg)
    return vane


def make_detailed_telescope(
    p: TelescopeParameters = TelescopeParameters(),
) -> cq.Workplane:
    """
    Build the external telescope representation as a compound.

    Using a compound instead of a single giant boolean-fused solid keeps STEP
    export stable and preserves assembly-like visual detail.
    """
    parts = []

    x_front = p.length / 2.0
    x_rear = -p.length / 2.0

    # ---------------------------------------------------------
    # Main cylindrical optical barrel.
    # ---------------------------------------------------------
    barrel_center = 20.0
    barrel = _x_cylinder(
        p.barrel_length,
        p.barrel_outer_diameter,
        x_center=barrel_center,
    )
    parts.append(barrel.val())

    # A slightly smaller raised barrel band near the front.
    front_band = _x_cylinder(
        14.0,
        p.barrel_outer_diameter + 8.0,
        x_center=x_front - 47.0,
    )
    parts.append(front_band.val())

    # Rear reinforcement band.
    rear_band = _x_cylinder(
        13.0,
        p.barrel_outer_diameter + 5.0,
        x_center=x_rear + 88.0,
    )
    parts.append(rear_band.val())

    # ---------------------------------------------------------
    # Front optical aperture assembly.
    # ---------------------------------------------------------
    front_flange_center = x_front - p.front_flange_axial / 2.0
    front_flange = _x_ring(
        p.front_flange_axial,
        p.front_flange_outer_diameter,
        p.clear_aperture_diameter,
        front_flange_center,
    )
    parts.append(front_flange.val())

    front_lip = _x_ring(
        p.front_lip_axial,
        p.front_lip_outer_diameter,
        p.clear_aperture_diameter - 7.0,
        x_front - p.front_lip_axial / 2.0 + 1.0,
    )
    parts.append(front_lip.val())

    # Primary mirror / reflective disk recessed just behind front opening.
    primary_x = x_front - p.primary_recess
    primary = _x_cylinder(
        p.primary_mirror_thickness,
        p.primary_mirror_diameter,
        x_center=primary_x,
    )
    parts.append(primary.val())

    # Central secondary mirror / hub.
    secondary_x = x_front - 7.0
    secondary_hub = _x_cylinder(
        p.secondary_hub_length,
        p.secondary_hub_diameter,
        x_center=secondary_x - p.secondary_hub_length / 2.0,
    )
    parts.append(secondary_hub.val())

    secondary_face = _x_cylinder(
        4.0,
        p.secondary_face_diameter,
        x_center=x_front - 2.0,
    )
    parts.append(secondary_face.val())

    # Three spider vanes.
    inner_r = p.secondary_hub_diameter / 2.0
    outer_r = p.clear_aperture_diameter / 2.0 - 4.0
    spider_x = x_front - 11.0
    for i in range(p.spider_count):
        angle = i * 360.0 / p.spider_count
        vane = _make_spider_vane(
            spider_x,
            inner_r,
            outer_r,
            p.spider_thickness,
            p.spider_width,
            angle,
        )
        parts.append(vane.val())

    # Eight front fastener heads around the flange.
    bolt_r = p.fastener_circle_diameter / 2.0
    fastener_x = x_front + p.fastener_head_depth / 2.0 - 2.0
    for i in range(p.front_fastener_count):
        a = math.radians(i * 360.0 / p.front_fastener_count)
        y = bolt_r * math.cos(a)
        z = bolt_r * math.sin(a)
        head = _x_cylinder(
            p.fastener_head_depth,
            p.fastener_head_diameter,
            x_center=fastener_x,
            y=y,
            z=z,
        )
        parts.append(head.val())

    # ---------------------------------------------------------
    # Rear adapter and spacecraft mounting interface.
    # ---------------------------------------------------------
    rear_adapter_center = x_rear + p.rear_adapter_length / 2.0 + 6.0
    rear_adapter = _x_cylinder(
        p.rear_adapter_length,
        p.rear_adapter_diameter,
        x_center=rear_adapter_center,
    )
    parts.append(rear_adapter.val())

    rear_mount = _x_ring(
        p.rear_mount_flange_axial,
        p.rear_mount_flange_diameter,
        p.rear_adapter_diameter - 24.0,
        x_rear + p.rear_mount_flange_axial / 2.0,
    )
    parts.append(rear_mount.val())

    # ---------------------------------------------------------
    # Longitudinal external ribs on the main barrel.
    # ---------------------------------------------------------
    rib_r = p.barrel_outer_diameter / 2.0 + p.rib_height / 2.0 - 1.0
    for i in range(p.rib_count):
        angle = 45.0 + i * 360.0 / p.rib_count
        rib = _box(
            p.rib_length,
            p.rib_width,
            p.rib_height,
            x=25.0,
            y=rib_r,
            z=0.0,
        ).rotate((0, 0, 0), (1, 0, 0), angle)
        parts.append(rib.val())

    # ---------------------------------------------------------
    # External electronics / harness tray.
    #
    # The earlier V5 version placed several small boxes at arbitrary Y/Z
    # coordinates; in an oblique view some appeared to float above the barrel.
    # Here the equipment is mounted on one deliberate +Z tray with visible
    # standoffs, so every external item has an obvious structural attachment.
    # ---------------------------------------------------------
    barrel_r = p.barrel_outer_diameter / 2.0

    # Two support standoffs that visibly bridge the barrel to the tray.
    tray_z = barrel_r + 15.0
    for x in (-72.0, 72.0):
        standoff = _box(
            18.0,
            54.0,
            22.0,
            x=x,
            y=0.0,
            z=barrel_r + 6.0,
        )
        parts.append(standoff.val())

    # Long equipment tray sitting directly on the standoffs.
    tray = _box(
        220.0,
        92.0,
        8.0,
        x=5.0,
        y=0.0,
        z=tray_z + 10.0,
    )
    try:
        tray = tray.edges("|X").fillet(2.0)
    except Exception:
        pass
    parts.append(tray.val())

    # Two electronics boxes mounted on the tray.
    box1 = _box(
        92.0,
        62.0,
        36.0,
        x=65.0,
        y=0.0,
        z=tray_z + 32.0,
    )
    try:
        box1 = box1.edges("|Z").fillet(2.5)
    except Exception:
        pass
    parts.append(box1.val())

    box2 = _box(
        66.0,
        56.0,
        30.0,
        x=-65.0,
        y=0.0,
        z=tray_z + 29.0,
    )
    try:
        box2 = box2.edges("|Z").fillet(2.0)
    except Exception:
        pass
    parts.append(box2.val())

    # One continuous cable/harness rail; no disconnected "floating" clips.
    harness = _box(
        196.0,
        12.0,
        8.0,
        x=2.0,
        y=-37.0,
        z=tray_z + 16.0,
    )
    parts.append(harness.val())

    # Four small tie-down blocks attached directly to the rail.
    for x in (-72.0, -24.0, 24.0, 72.0):
        tie = _box(
            8.0,
            18.0,
            12.0,
            x=x,
            y=-37.0,
            z=tray_z + 21.0,
        )
        parts.append(tie.val())

    # ---------------------------------------------------------
    # Four mounting feet below/side of rear barrel.
    # ---------------------------------------------------------
    foot_x_positions = (-170.0, 115.0)
    for x in foot_x_positions:
        for ysgn in (-1.0, 1.0):
            y = ysgn * 138.0
            z = -p.envelope_height / 2.0 + p.foot_height / 2.0
            foot = _box(
                p.foot_length,
                p.foot_width,
                p.foot_height,
                x=x,
                y=y,
                z=z,
            )
            parts.append(foot.val())

    compound = cq.Compound.makeCompound(parts)
    return cq.Workplane("XY").newObject([compound])


def make_telescope_assembly(
    p: TelescopeParameters = TelescopeParameters(),
) -> cq.Assembly:
    """
    Multi-colour assembly wrapper for STEP viewers that preserve colours.
    """
    assy = cq.Assembly(name="Detailed_Telescope_Assembly")

    # For now the detailed telescope is exported as one compound.  The colour
    # represents the dark anodized / painted optical-head finish in the photo.
    assy.add(
        make_detailed_telescope(p),
        name="Telescope optical head",
        color=cq.Color(0.08, 0.09, 0.10),
    )
    return assy


if __name__ == "__main__":
    from pathlib import Path

    out = Path(__file__).resolve().parent / "outputs"
    out.mkdir(exist_ok=True)

    telescope = make_detailed_telescope()
    assy = make_telescope_assembly()

    cq.exporters.export(telescope, str(out / "telescope_detailed.step"))
    assy.save(str(out / "telescope_detailed_assembly.step"))
    cq.exporters.export(telescope, str(out / "telescope_detailed.stl"))

    print("Exported detailed telescope geometry to:", out)
