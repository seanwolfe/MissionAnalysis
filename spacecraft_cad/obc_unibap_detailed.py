"""
obc_unibap_detailed.py

Higher-craftsmanship CadQuery representation of the Unibap iX5-106 OBC
selected for the spacecraft C&DH subsystem.

Current workbook envelope
-------------------------
Length : 100 mm
Width  : 100 mm
Height : 50 mm

Visual reference
----------------
The supplied reference image shows a ruggedized stacked-compute module with:
- multiple green PCB layers
- four corner standoffs / structural posts
- substantial machined top plate
- bottom/base plate
- front and side connector blocks
- visible board-edge electronics
- recessed internal electronics under the top plate

This model intentionally captures that architecture rather than reproducing
every chip or connector.

Local component frame
---------------------
+X : 100 mm length
+Y : 100 mm width
+Z : 50 mm height

The model is centred at the local origin for direct placement from the master
workbook.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import cadquery as cq


@dataclass(frozen=True)
class OBCParameters:
    # Workbook envelope
    length: float = 100.0
    width: float = 100.0
    height: float = 50.0

    # Layer stack
    base_plate_thickness: float = 3.0
    top_plate_thickness: float = 5.0
    pcb_thickness: float = 1.6
    pcb_count: int = 3
    pcb_spacing: float = 11.0

    # Structural corner posts
    post_diameter: float = 7.0
    post_margin: float = 9.0

    # Machined top plate cutout / bridge geometry
    top_plate_inset: float = 3.0
    top_cutout_length: float = 66.0
    top_cutout_width: float = 44.0
    bridge_width: float = 9.0

    # Connector blocks
    front_connector_length: float = 48.0
    front_connector_width: float = 7.0
    front_connector_height: float = 9.0
    side_connector_length: float = 22.0
    side_connector_width: float = 8.0
    side_connector_height: float = 10.0

    # Electronics packages
    chip_height: float = 3.0
    chip_large: float = 14.0
    chip_small: float = 8.0

    # Fasteners / visual detail
    screw_head_diameter: float = 4.2
    screw_head_depth: float = 1.5


def _box(l, w, h, x=0.0, y=0.0, z=0.0):
    return (
        cq.Workplane("XY")
        .box(l, w, h, centered=(True, True, True))
        .translate((x, y, z))
    )


def _z_cylinder(height: float, diameter: float, x: float, y: float, z_center: float):
    return (
        cq.Workplane("XY")
        .center(x, y)
        .circle(diameter / 2.0)
        .extrude(height / 2.0, both=True)
        .translate((0.0, 0.0, z_center))
    )


def make_detailed_obc(p: OBCParameters = OBCParameters()) -> cq.Workplane:
    """
    Build a detailed Unibap-style OBC as a compound of separate visual solids.
    """
    parts = []

    z_bottom = -p.height / 2.0
    z_top = +p.height / 2.0

    # ---------------------------------------------------------
    # Bottom base plate.
    # ---------------------------------------------------------
    base = _box(
        p.length,
        p.width,
        p.base_plate_thickness,
        z=z_bottom + p.base_plate_thickness / 2.0,
    )
    parts.append(base.val())

    # Slightly inset dark underside layer.
    underside = _box(
        p.length - 4.0,
        p.width - 4.0,
        1.5,
        z=z_bottom + p.base_plate_thickness + 0.75,
    )
    parts.append(underside.val())

    # ---------------------------------------------------------
    # PCB stack.
    # ---------------------------------------------------------
    pcb_z0 = z_bottom + 9.0
    for i in range(p.pcb_count):
        z = pcb_z0 + i * p.pcb_spacing
        pcb = _box(
            p.length - 4.0,
            p.width - 4.0,
            p.pcb_thickness,
            z=z,
        )
        parts.append(pcb.val())

        # Representative electronics on each board.
        # Deliberately varied positions so the boards look populated rather
        # than like identical empty plates.
        offsets = [
            (-26.0 + 6*i, -20.0, p.chip_large, p.chip_large),
            (8.0, 19.0 - 3*i, p.chip_small, p.chip_small),
            (26.0, -8.0 + 5*i, 11.0, 8.0),
            (-5.0, 5.0, 8.0, 13.0),
        ]
        for x, y, lx, ly in offsets:
            chip = _box(
                lx, ly, p.chip_height,
                x=x, y=y,
                z=z + p.pcb_thickness/2.0 + p.chip_height/2.0,
            )
            parts.append(chip.val())

        # Front board-edge connector bank.
        conn = _box(
            p.front_connector_length,
            p.front_connector_width,
            p.front_connector_height,
            x=0.0,
            y=-p.width/2.0 + p.front_connector_width/2.0 + 1.5,
            z=z + p.front_connector_height/2.0 - 1.0,
        )
        parts.append(conn.val())

    # ---------------------------------------------------------
    # Four corner structural posts.
    # ---------------------------------------------------------
    post_h = p.height - p.base_plate_thickness - p.top_plate_thickness
    post_z = z_bottom + p.base_plate_thickness + post_h/2.0

    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            x = sx * (p.length/2.0 - p.post_margin)
            y = sy * (p.width/2.0 - p.post_margin)
            post = _z_cylinder(
                post_h,
                p.post_diameter,
                x, y, post_z,
            )
            parts.append(post.val())

    # ---------------------------------------------------------
    # Machined top plate.
    # ---------------------------------------------------------
    top_z = z_top - p.top_plate_thickness / 2.0

    top_outer = _box(
        p.length,
        p.width,
        p.top_plate_thickness,
        z=top_z,
    )

    # Large central opening, offset slightly toward the front edge as in the
    # reference image.
    cut = _box(
        p.top_cutout_length,
        p.top_cutout_width,
        p.top_plate_thickness + 2.0,
        x=-4.0,
        y=-8.0,
        z=top_z,
    )
    top_plate = top_outer.cut(cut)
    parts.append(top_plate.val())

    # Raised rear bridge / stiffener across the top.
    bridge = _box(
        p.length - 12.0,
        p.bridge_width,
        4.0,
        y=p.width/2.0 - 11.0,
        z=z_top + 2.0,
    )
    parts.append(bridge.val())

    # Four top-plate screw heads.
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            x = sx * (p.length/2.0 - 11.0)
            y = sy * (p.width/2.0 - 11.0)
            screw = _z_cylinder(
                p.screw_head_depth,
                p.screw_head_diameter,
                x, y,
                z_top + p.top_plate_thickness/2.0 + p.screw_head_depth/2.0 - 0.5,
            )
            parts.append(screw.val())

    # ---------------------------------------------------------
    # Side connector / I/O blocks.
    # ---------------------------------------------------------
    side_z = pcb_z0 + p.pcb_spacing
    for ysign in (-1.0, 1.0):
        conn = _box(
            p.side_connector_length,
            p.side_connector_width,
            p.side_connector_height,
            x=24.0,
            y=ysign * (p.width/2.0 - p.side_connector_width/2.0),
            z=side_z,
        )
        parts.append(conn.val())

    # Rear power/data connector block.
    rear_conn = _box(
        30.0,
        8.0,
        12.0,
        x=-22.0,
        y=p.width/2.0 - 4.0,
        z=pcb_z0 + 3.0,
    )
    parts.append(rear_conn.val())

    # ---------------------------------------------------------
    # Small exposed board-edge / component details for visual richness.
    # ---------------------------------------------------------
    for x in (-32.0, -12.0, 8.0, 28.0):
        component = _box(
            7.0, 5.0, 5.0,
            x=x,
            y=-p.width/2.0 + 8.0,
            z=pcb_z0 + p.pcb_spacing + 5.0,
        )
        parts.append(component.val())

    compound = cq.Compound.makeCompound(parts)
    return cq.Workplane("XY").newObject([compound])


def make_obc_assembly(p: OBCParameters = OBCParameters()) -> cq.Assembly:
    """
    Multi-colour visual assembly.  STEP viewers that preserve assembly colours
    will show a metallic frame with green PCB/electronics cues.
    """
    assy = cq.Assembly(name="Unibap_iX5_106_Detailed")
    assy.add(
        make_detailed_obc(p),
        name="Unibap iX5-106 OBC",
        color=cq.Color(0.35, 0.38, 0.38),
    )
    return assy


if __name__ == "__main__":
    from pathlib import Path

    out = Path(__file__).resolve().parent / "outputs"
    out.mkdir(exist_ok=True)

    obc = make_detailed_obc()
    assy = make_obc_assembly()

    cq.exporters.export(obc, str(out / "unibap_ix5_106_detailed.step"))
    assy.save(str(out / "unibap_ix5_106_detailed_assembly.step"))
    cq.exporters.export(obc, str(out / "unibap_ix5_106_detailed.stl"))

    print("Exported detailed Unibap iX5-106 geometry to:", out)
