"""
baffle_detailed.py

Higher-craftsmanship CadQuery model of the optical baffle.

The user's spacecraft sizing sheet defines the baffle envelope as:
    Length : 150 mm
    Width  : 325 mm
    Height : 325 mm

The supplied reference image shows the relevant form: a hollow tapered
frustum / cone rather than a rectangular box.  This model therefore keeps the
150 x 325 x 325 mm maximum envelope but replaces the old box with a realistic
optical-baffle architecture:

- hollow conical/frustum shell
- large forward aperture
- smaller telescope-side throat
- forward stiffening lip
- rear mounting collar
- internal knife-edge vanes
- several longitudinal external stiffeners

This is configuration / publication CAD, not manufacturing CAD.

Local component frame
---------------------
+X : optical / telescope boresight direction
     rear/telescope interface at -X
     open forward aperture at +X
Y,Z: circular baffle cross-section

The component geometric centre remains at x=0 so it can be positioned using
the existing master-workbook placement vector.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import cadquery as cq


@dataclass(frozen=True)
class BaffleParameters:
    # Master-workbook envelope
    length: float = 150.0
    max_outer_diameter: float = 325.0

    # Taper
    rear_outer_diameter: float = 245.0
    wall_thickness: float = 4.0

    # End rings / mounting interfaces
    front_lip_axial: float = 5.0
    front_lip_radial: float = 7.0
    rear_collar_axial: float = 10.0
    rear_collar_outer_diameter: float = 260.0

    # Internal knife-edge vane system
    vane_count: int = 4
    vane_axial_thickness: float = 2.0
    vane_radial_depth: float = 13.0
    vane_end_margin: float = 22.0

    # External longitudinal stiffeners
    stiffener_count: int = 4
    stiffener_width: float = 8.0
    stiffener_height: float = 5.0
    stiffener_end_margin: float = 12.0

    # Rear bolt / interface detail
    rear_bolt_count: int = 8
    rear_bolt_diameter: float = 4.0
    rear_bolt_circle_diameter: float = 232.0
    rear_bolt_head_diameter: float = 7.0
    rear_bolt_head_depth: float = 2.0


def _radius_at_x(p: BaffleParameters, x: float) -> float:
    """Outer shell radius at local x."""
    x0 = -p.length / 2.0
    t = (x - x0) / p.length
    r0 = p.rear_outer_diameter / 2.0
    r1 = p.max_outer_diameter / 2.0
    return r0 + t * (r1 - r0)


def _frustum(
    length: float,
    rear_diameter: float,
    front_diameter: float,
    x_start: float,
) -> cq.Workplane:
    """
    Solid frustum aligned along +X, made by lofting two circular sections.
    """
    wp = (
        cq.Workplane("YZ")
        .workplane(offset=x_start)
        .circle(rear_diameter / 2.0)
        .workplane(offset=length)
        .circle(front_diameter / 2.0)
        .loft(combine=True)
    )
    return wp


def _x_ring(
    x_center: float,
    axial_thickness: float,
    outer_diameter: float,
    inner_diameter: float,
) -> cq.Workplane:
    """Annular ring whose axis is X."""
    outer = (
        cq.Workplane("YZ")
        .circle(outer_diameter / 2.0)
        .circle(inner_diameter / 2.0)
        .extrude(axial_thickness / 2.0, both=True)
        .translate((x_center, 0.0, 0.0))
    )
    return outer


def _x_cylinder(
    length: float,
    diameter: float,
    x_center: float,
    y: float,
    z: float,
) -> cq.Workplane:
    return (
        cq.Workplane("YZ")
        .circle(diameter / 2.0)
        .extrude(length / 2.0, both=True)
        .translate((x_center, y, z))
    )


def make_detailed_baffle(p: BaffleParameters = BaffleParameters()) -> cq.Workplane:
    """
    Build a detailed tapered optical baffle as a compound of robust solids.
    """
    parts = []

    x_rear = -p.length / 2.0
    x_front = p.length / 2.0

    # ---------------------------------------------------------
    # Main hollow frustum shell.
    # ---------------------------------------------------------
    outer = _frustum(
        p.length,
        p.rear_outer_diameter,
        p.max_outer_diameter,
        x_rear,
    )

    rear_inner_d = p.rear_outer_diameter - 2.0 * p.wall_thickness
    front_inner_d = p.max_outer_diameter - 2.0 * p.wall_thickness

    # Inner frustum is extended very slightly at both ends to ensure a clean cut.
    inner = _frustum(
        p.length + 2.0,
        rear_inner_d,
        front_inner_d,
        x_rear - 1.0,
    )

    shell = outer.cut(inner)
    parts.append(shell.val())

    # ---------------------------------------------------------
    # Forward aperture lip.
    # ---------------------------------------------------------
    front_lip = _x_ring(
        x_front - p.front_lip_axial / 2.0,
        p.front_lip_axial,
        p.max_outer_diameter,
        p.max_outer_diameter - 2.0 * (p.wall_thickness + p.front_lip_radial),
    )
    parts.append(front_lip.val())

    # ---------------------------------------------------------
    # Rear mounting collar.
    # ---------------------------------------------------------
    rear_collar = _x_ring(
        x_rear + p.rear_collar_axial / 2.0,
        p.rear_collar_axial,
        p.rear_collar_outer_diameter,
        rear_inner_d - 5.0,
    )
    parts.append(rear_collar.val())

    # ---------------------------------------------------------
    # Internal knife-edge vanes.
    # ---------------------------------------------------------
    if p.vane_count > 0:
        usable = p.length - 2.0 * p.vane_end_margin
        spacing = usable / (p.vane_count + 1)

        for i in range(1, p.vane_count + 1):
            xv = x_rear + p.vane_end_margin + i * spacing
            outer_r = _radius_at_x(p, xv) - p.wall_thickness - 1.0
            inner_r = max(10.0, outer_r - p.vane_radial_depth)

            vane = _x_ring(
                xv,
                p.vane_axial_thickness,
                2.0 * outer_r,
                2.0 * inner_r,
            )
            parts.append(vane.val())

    # ---------------------------------------------------------
    # External longitudinal stiffeners / rails.
    #
    # Each is a slender bar following the tapered shell approximately.
    # They add the visual "flight hardware" craftsmanship seen in the
    # reference cross-section without pretending to reproduce a vendor part.
    # ---------------------------------------------------------
    stiff_len = p.length - 2.0 * p.stiffener_end_margin
    x_mid = 0.0
    r_mid = _radius_at_x(p, x_mid) + p.stiffener_height / 2.0 - 1.0

    for i in range(p.stiffener_count):
        angle = i * 360.0 / p.stiffener_count
        a = math.radians(angle)
        y = r_mid * math.cos(a)
        z = r_mid * math.sin(a)

        rib = (
            cq.Workplane("XY")
            .box(
                stiff_len,
                p.stiffener_width,
                p.stiffener_height,
                centered=(True, True, True),
            )
            .translate((0.0, r_mid, 0.0))
            .rotate((0, 0, 0), (1, 0, 0), angle)
        )
        parts.append(rib.val())

    # ---------------------------------------------------------
    # Rear mounting-bolt heads.
    # ---------------------------------------------------------
    bolt_r = p.rear_bolt_circle_diameter / 2.0
    bolt_x = x_rear - p.rear_bolt_head_depth / 2.0

    for i in range(p.rear_bolt_count):
        a = math.radians(i * 360.0 / p.rear_bolt_count)
        y = bolt_r * math.cos(a)
        z = bolt_r * math.sin(a)
        head = _x_cylinder(
            p.rear_bolt_head_depth,
            p.rear_bolt_head_diameter,
            bolt_x,
            y,
            z,
        )
        parts.append(head.val())

    compound = cq.Compound.makeCompound(parts)
    return cq.Workplane("XY").newObject([compound])


def make_baffle_assembly(p: BaffleParameters = BaffleParameters()) -> cq.Assembly:
    assy = cq.Assembly(name="Detailed_Optical_Baffle")
    assy.add(
        make_detailed_baffle(p),
        name="Optical baffle",
        color=cq.Color(0.08, 0.09, 0.10),
    )
    return assy


if __name__ == "__main__":
    from pathlib import Path

    out = Path(__file__).resolve().parent / "outputs"
    out.mkdir(exist_ok=True)

    baffle = make_detailed_baffle()
    assy = make_baffle_assembly()

    cq.exporters.export(baffle, str(out / "optical_baffle_detailed.step"))
    assy.save(str(out / "optical_baffle_detailed_assembly.step"))
    cq.exporters.export(baffle, str(out / "optical_baffle_detailed.stl"))

    print("Exported detailed optical baffle geometry to:", out)
