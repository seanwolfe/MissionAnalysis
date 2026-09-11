"""
xband_lga_detailed.py

Detailed spacecraft-level CadQuery representation of the IQ spacecom-style
X-band 2x2 quad patch antenna shown in the user-provided product sheet.

Drawing/reference dimensions from the supplied image:
    Quad patch antenna : 60 x 40 x 1.8 mm
    Mass               : 20 g
    Polarization       : RHCP (LHCP optional)
    Frequency          : 8.025-8.400 GHz or 7.145-7.250 GHz
    Connector          : SMA female
    Type               : patch antenna

For this spacecraft model, this component is used as the fixed broad-coverage
LGA / safe-mode antenna representation.

IMPORTANT:
The supplied product sheet differs from the previously entered spreadsheet
placeholder for the safe-mode antenna. This CAD follows the NEW supplied image:
60 x 40 x 1.8 mm, not the older 35 x 35 x 1.8 mm placeholder.

Orientation convention
----------------------
The placement sheet stores antenna boresight / face-normal vectors.  This model
is authored with:
    local +X = antenna boresight / outward normal
    local Y  = 60 mm
    local Z  = 40 mm
    local X  = 1.8 mm substrate thickness

This lets the existing `place()` helper use the workbook antenna orientation
vectors directly.
"""

from __future__ import annotations

from dataclasses import dataclass
import cadquery as cq


@dataclass(frozen=True)
class LGAPatchParameters:
    panel_y: float = 60.0
    panel_z: float = 40.0
    thickness_x: float = 1.8

    # Four square radiating patches in a 2 x 2 arrangement
    patch_y: float = 10.5
    patch_z: float = 10.5
    patch_spacing_y: float = 22.0
    patch_spacing_z: float = 16.5
    patch_raise_x: float = 0.45

    # Mounting holes
    hole_diameter: float = 2.5
    hole_offset_y: float = 26.0
    hole_offset_z: float = 16.0
    hole_insert_depth: float = 0.8

    # Feed / center hardware
    center_feed_diameter: float = 2.2
    center_feed_depth: float = 0.8
    feed_spacing: float = 4.0

    # Rear SMA connector representation
    sma_body_diameter: float = 5.5
    sma_body_length: float = 7.0
    sma_pin_diameter: float = 1.2
    sma_pin_length: float = 3.5


def _box_x(thickness, y_size, z_size, x=0.0, y=0.0, z=0.0):
    return (
        cq.Workplane("XY")
        .box(thickness, y_size, z_size, centered=(True,True,True))
        .translate((x,y,z))
    )


def _x_cyl(length, diameter, x, y=0.0, z=0.0):
    return (
        cq.Workplane("YZ")
        .circle(diameter/2.0)
        .extrude(length/2.0, both=True)
        .translate((x,y,z))
    )


def make_detailed_lga(
    p: LGAPatchParameters = LGAPatchParameters(),
) -> cq.Workplane:
    parts = []

    # Main RF substrate / ground plane.
    substrate = _box_x(
        p.thickness_x,
        p.panel_y,
        p.panel_z,
        x=0.0,
    )
    try:
        substrate = substrate.edges("|X").fillet(0.8)
    except Exception:
        pass
    parts.append(substrate.val())

    # Front (+X) radiating patches, 2x2.
    x_front = p.thickness_x/2.0 + p.patch_raise_x/2.0
    for sy in (-1.0, 1.0):
        for sz in (-1.0, 1.0):
            patch = _box_x(
                p.patch_raise_x,
                p.patch_y,
                p.patch_z,
                x=x_front,
                y=sy*p.patch_spacing_y/2.0,
                z=sz*p.patch_spacing_z/2.0,
            )
            parts.append(patch.val())

    # Four corner mounting-hole inserts / bores.
    for sy in (-1.0, 1.0):
        for sz in (-1.0, 1.0):
            y = sy*p.hole_offset_y
            z = sz*p.hole_offset_z
            insert = _x_cyl(
                p.hole_insert_depth,
                p.hole_diameter,
                x=p.thickness_x/2.0 + p.hole_insert_depth/2.0 - 0.2,
                y=y,
                z=z,
            )
            parts.append(insert.val())

    # Small central feed / phasing details visible in product photography.
    for z in (-p.feed_spacing/2.0, 0.0, p.feed_spacing/2.0):
        feed = _x_cyl(
            p.center_feed_depth,
            p.center_feed_diameter,
            x=p.thickness_x/2.0 + p.center_feed_depth/2.0,
            y=0.0,
            z=z,
        )
        parts.append(feed.val())

    # Rear (-X) SMA connector body.
    sma = _x_cyl(
        p.sma_body_length,
        p.sma_body_diameter,
        x=-p.thickness_x/2.0 - p.sma_body_length/2.0,
        y=0.0,
        z=0.0,
    )
    parts.append(sma.val())

    # Small center pin / feed protrusion.
    pin = _x_cyl(
        p.sma_pin_length,
        p.sma_pin_diameter,
        x=-p.thickness_x/2.0 - p.sma_body_length - p.sma_pin_length/2.0 + 0.5,
        y=0.0,
        z=0.0,
    )
    parts.append(pin.val())

    return cq.Workplane("XY").newObject([cq.Compound.makeCompound(parts)])


def make_lga_assembly(
    p: LGAPatchParameters = LGAPatchParameters(),
) -> cq.Assembly:
    assy = cq.Assembly(name="IQ_Spacecom_XBand_Quad_Patch")
    assy.add(
        make_detailed_lga(p),
        name="X-band 2x2 Quad Patch LGA",
        color=cq.Color(0.76,0.76,0.72),
    )
    return assy


if __name__ == "__main__":
    from pathlib import Path
    out = Path(__file__).resolve().parent / "outputs"
    out.mkdir(exist_ok=True)

    lga = make_detailed_lga()
    assy = make_lga_assembly()

    cq.exporters.export(lga, str(out / "xband_quad_patch_lga_detailed.step"))
    assy.save(str(out / "xband_quad_patch_lga_detailed_assembly.step"))
    cq.exporters.export(lga, str(out / "xband_quad_patch_lga_detailed.stl"))
    print("Exported detailed X-band quad-patch LGA geometry to:", out)
