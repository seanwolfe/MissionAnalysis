"""
battery_detailed.py

Higher-craftsmanship CadQuery model of the EnerSys / ABSL 8s3p spacecraft
battery shown in the user-provided datasheet image.

The battery specification block gives:
    footprint : 176 x 96 mm
    height    : 98 mm
    mass      : 1.66 kg

This model is intended for spacecraft configuration/CAD figures rather than
manufacturing.  It captures the visually important external features:

- gold rectangular battery enclosure
- thick front/end plate
- four external corner mounting rails / feet
- repeated fasteners
- recessed top connector block
- side/end reliefs and raised structural ribs
- small chamfers/fillets to eliminate the "primitive box" appearance

Local component coordinates
---------------------------
+X : 176 mm battery length
+Y : 96 mm battery width
+Z : 98 mm battery height

The part is centred at the local origin, matching the spacecraft placement
convention used in the master workbook.
"""

from __future__ import annotations

from dataclasses import dataclass
import cadquery as cq


@dataclass(frozen=True)
class BatteryParameters:
    # Datasheet envelope
    length: float = 176.0
    width: float = 96.0
    height: float = 98.0

    # Main housing
    end_plate_thickness: float = 4.0
    side_wall_inset: float = 4.0
    body_corner_radius: float = 2.0

    # Four longitudinal external rails / feet
    rail_width: float = 10.0
    rail_projection_y: float = 4.0
    rail_projection_z: float = 5.0
    rail_end_margin: float = 7.0

    # End-plate fasteners
    screw_diameter: float = 4.0
    screw_head_diameter: float = 7.0
    screw_head_depth: float = 1.8
    screw_edge_y: float = 13.0
    screw_edge_z: float = 14.0

    # Top connector
    connector_length: float = 35.0
    connector_width: float = 13.0
    connector_height: float = 8.0
    connector_recess_depth: float = 2.5
    connector_x: float = -54.0

    # Raised top rim / lid
    lid_thickness: float = 2.0
    lid_inset: float = 5.0
    lid_raise: float = 1.5

    # Side structural bands / ribs
    rib_thickness: float = 3.0
    rib_width: float = 7.0
    rib_locations_x: tuple[float, ...] = (-61.0, 61.0)


def _box(l: float, w: float, h: float, x=0.0, y=0.0, z=0.0) -> cq.Workplane:
    return (
        cq.Workplane("XY")
        .box(l, w, h, centered=(True, True, True))
        .translate((x, y, z))
    )


def _x_cylinder(length: float, diameter: float, x_center: float, y: float, z: float) -> cq.Workplane:
    return (
        cq.Workplane("YZ")
        .circle(diameter / 2.0)
        .extrude(length / 2.0, both=True)
        .translate((x_center, y, z))
    )


def make_detailed_battery(p: BatteryParameters = BatteryParameters()) -> cq.Workplane:
    """
    Return the detailed battery as a compound of robust external solids.

    A compound is used rather than aggressively fusing every decorative feature;
    it is more reliable for STEP export and preserves crisp assembly-like detail.
    """
    parts = []

    # Main case, slightly inset from the full envelope so rails and plates read
    # as separate structural hardware.
    body_len = p.length - 2.0 * p.end_plate_thickness
    body_w = p.width - 2.0 * p.side_wall_inset
    body_h = p.height - 7.0

    body = _box(body_len, body_w, body_h, z=-1.5)
    try:
        body = body.edges("|X").fillet(p.body_corner_radius)
    except Exception:
        pass
    parts.append(body.val())

    # Thick end plates at ±X.
    for sgn in (-1.0, 1.0):
        xp = sgn * (p.length / 2.0 - p.end_plate_thickness / 2.0)
        plate = _box(
            p.end_plate_thickness,
            p.width - 4.0,
            p.height - 8.0,
            x=xp,
            z=-1.0,
        )
        parts.append(plate.val())

        # Four raised screw heads on each end plate.
        for y in (-p.width / 2.0 + p.screw_edge_y, p.width / 2.0 - p.screw_edge_y):
            for z in (-p.height / 2.0 + p.screw_edge_z, p.height / 2.0 - p.screw_edge_z):
                head = _x_cylinder(
                    p.screw_head_depth,
                    p.screw_head_diameter,
                    xp + sgn * (p.end_plate_thickness / 2.0 + p.screw_head_depth / 2.0),
                    y,
                    z,
                )
                parts.append(head.val())

    # Top lid: slightly smaller raised plate.
    lid = _box(
        p.length - 2.0 * p.lid_inset,
        p.width - 2.0 * p.lid_inset,
        p.lid_thickness,
        z=p.height / 2.0 - p.lid_thickness / 2.0 + p.lid_raise,
    )
    parts.append(lid.val())

    # Longitudinal mounting rails/feet near the four Y/Z corners.
    rail_len = p.length - 2.0 * p.rail_end_margin
    for ysgn in (-1.0, 1.0):
        for zsgn in (-1.0, 1.0):
            y = ysgn * (
                p.width / 2.0 - p.rail_width / 2.0 + p.rail_projection_y / 2.0
            )
            z = zsgn * (
                p.height / 2.0 - p.rail_width / 2.0 + p.rail_projection_z / 2.0
            )
            rail = _box(
                rail_len,
                p.rail_width,
                p.rail_width,
                y=y,
                z=z,
            )
            try:
                rail = rail.edges("|X").fillet(1.2)
            except Exception:
                pass
            parts.append(rail.val())

    # Two vertical structural ribs/bands around the long body.
    for x in p.rib_locations_x:
        # side bars on ±Y
        for ysgn in (-1.0, 1.0):
            y = ysgn * (p.width / 2.0 - p.rib_thickness / 2.0)
            rib = _box(
                p.rib_width,
                p.rib_thickness,
                p.height - 14.0,
                x=x,
                y=y,
                z=-1.0,
            )
            parts.append(rib.val())

        # top bridge makes each rib read as a clamp/band
        top_rib = _box(
            p.rib_width,
            p.width - 5.0,
            p.rib_thickness,
            x=x,
            z=p.height / 2.0 - 7.0,
        )
        parts.append(top_rib.val())

    # Recessed connector pocket on the top face.
    pocket = _box(
        p.connector_length + 6.0,
        p.connector_width + 6.0,
        p.connector_recess_depth,
        x=p.connector_x,
        z=p.height / 2.0 + p.lid_raise + 0.4,
    )
    parts.append(pocket.val())

    connector = _box(
        p.connector_length,
        p.connector_width,
        p.connector_height,
        x=p.connector_x,
        z=p.height / 2.0 + p.connector_height / 2.0 + p.lid_raise,
    )
    try:
        connector = connector.edges("|Z").fillet(2.0)
    except Exception:
        pass
    parts.append(connector.val())

    # Connector face insert, slightly smaller and raised.
    insert = _box(
        p.connector_length - 5.0,
        p.connector_width - 4.0,
        1.5,
        x=p.connector_x,
        z=p.height / 2.0 + p.connector_height + p.lid_raise + 0.75,
    )
    parts.append(insert.val())

    # Small corner feet at the end plates, matching the obvious external lugs
    # visible in the product image.
    foot_x = p.length / 2.0 - 11.0
    for x in (-foot_x, foot_x):
        for ysgn in (-1.0, 1.0):
            y = ysgn * (p.width / 2.0 + 2.0)
            foot = _box(18.0, 6.0, 17.0, x=x, y=y, z=-20.0)
            parts.append(foot.val())

    compound = cq.Compound.makeCompound(parts)
    return cq.Workplane("XY").newObject([compound])


def make_battery_assembly(p: BatteryParameters = BatteryParameters()) -> cq.Assembly:
    assy = cq.Assembly(name="ABSL_8s3p_Battery")
    assy.add(
        make_detailed_battery(p),
        name="ABSL 8s3p battery",
        color=cq.Color(0.72, 0.52, 0.16),
    )
    return assy


if __name__ == "__main__":
    from pathlib import Path

    out = Path(__file__).resolve().parent / "outputs"
    out.mkdir(exist_ok=True)

    battery = make_detailed_battery()
    assy = make_battery_assembly()

    cq.exporters.export(battery, str(out / "absl_8s3p_detailed_battery.step"))
    assy.save(str(out / "absl_8s3p_detailed_battery_assembly.step"))
    cq.exporters.export(battery, str(out / "absl_8s3p_detailed_battery.stl"))

    print("Exported detailed ABSL 8s3p battery geometry to:", out)
