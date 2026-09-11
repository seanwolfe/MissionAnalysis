"""
solar_array_detailed.py

Detailed deployable solar-array model for the microsatellite.

Locked geometry from the design discussion
------------------------------------------
Per panel:
    panel size      : 860 x 300 mm
    active cells    : 18 (6 x 3)
    individual cell : 138.9 x 70.5 mm
    configuration   : single-sided
    CAD state       : deployed

The 6-cell direction follows the long panel dimension.  The active cell field is
centred longitudinally within the 860 mm panel rather than pushed toward the aft
hinge.  This keeps the photovoltaic field away from the local hinge/thruster
interface as much as the available margin permits.

The panel is represented with:
- sandwich substrate
- perimeter frame
- 6 x 3 individual solar cells
- small inter-cell gaps
- two aft-edge hinge fittings
- short deployment yoke/standoff
- outboard sun-sensor mounting bracket

Local panel frame
-----------------
+X : 860 mm panel longitudinal direction
+Y : panel outward normal / active-face normal
+Z : 300 mm panel short direction

A single standalone panel is authored in this convenient local frame.  The full
spacecraft script applies the deployed transform so that:
    panel long axis -> +/- spacecraft Y
    active face     -> spacecraft -X
    short axis      -> spacecraft Z
"""

from __future__ import annotations
from dataclasses import dataclass
import cadquery as cq


@dataclass(frozen=True)
class SolarArrayParameters:
    panel_length: float = 860.0
    panel_width: float = 300.0
    panel_thickness: float = 7.0

    # perimeter/frame
    frame_width: float = 8.0
    frame_raise: float = 1.5

    # cells
    cell_length: float = 138.9
    cell_width: float = 70.5
    cell_thickness: float = 0.7
    cells_long: int = 6
    cells_short: int = 3
    gap_long: float = 2.0
    gap_short: float = 4.0

    # hinge/yoke hardware on the -X edge of the STOWED panel
    hinge_block_length: float = 28.0
    hinge_block_width: float = 24.0
    hinge_block_height: float = 18.0
    hinge_offset_z: float = 92.0

    yoke_length: float = 38.0
    yoke_width: float = 18.0
    yoke_height: float = 14.0

    # sun sensor bracket at deployed outboard edge
    css_bracket_length: float = 24.0
    css_bracket_width: float = 36.0
    css_bracket_height: float = 5.0


def _box(l,w,h,x=0.0,y=0.0,z=0.0):
    return (
        cq.Workplane("XY")
        .box(l,w,h,centered=(True,True,True))
        .translate((x,y,z))
    )


def make_detailed_solar_panel(
    p: SolarArrayParameters = SolarArrayParameters(),
) -> cq.Workplane:
    parts=[]

    # Main sandwich panel. Local Y is thickness/normal.
    substrate = _box(
        p.panel_length,
        p.panel_thickness,
        p.panel_width,
    )
    try:
        substrate = substrate.edges("|Y").fillet(2.0)
    except Exception:
        pass
    parts.append(substrate.val())

    # Perimeter frame rails, slightly proud of the active face (+Y).
    y_frame = p.panel_thickness/2.0 + p.frame_raise/2.0
    for zsign in (-1,1):
        rail = _box(
            p.panel_length,
            p.frame_raise,
            p.frame_width,
            y=y_frame,
            z=zsign*(p.panel_width/2.0-p.frame_width/2.0),
        )
        parts.append(rail.val())

    for xsign in (-1,1):
        rail = _box(
            p.frame_width,
            p.frame_raise,
            p.panel_width,
            x=xsign*(p.panel_length/2.0-p.frame_width/2.0),
            y=y_frame,
        )
        parts.append(rail.val())

    # ---------------------------------------------------------
    # 6 x 3 cell field, centred on the panel.
    # ---------------------------------------------------------
    field_L = p.cells_long*p.cell_length + (p.cells_long-1)*p.gap_long
    field_W = p.cells_short*p.cell_width + (p.cells_short-1)*p.gap_short

    x0 = -field_L/2.0 + p.cell_length/2.0
    z0 = -field_W/2.0 + p.cell_width/2.0
    y_cell = p.panel_thickness/2.0 + p.cell_thickness/2.0 + 0.15

    for i in range(p.cells_long):
        for j in range(p.cells_short):
            x = x0 + i*(p.cell_length+p.gap_long)
            z = z0 + j*(p.cell_width+p.gap_short)

            cell = _box(
                p.cell_length,
                p.cell_thickness,
                p.cell_width,
                x=x,
                y=y_cell,
                z=z,
            )
            parts.append(cell.val())

            # Three subtle busbar strips per cell.
            for frac in (-0.25,0.0,0.25):
                strip = _box(
                    p.cell_length-5.0,
                    0.25,
                    0.6,
                    x=x,
                    y=y_cell+p.cell_thickness/2.0+0.1,
                    z=z+frac*p.cell_width,
                )
                parts.append(strip.val())

    # ---------------------------------------------------------
    # Two hinge fittings on the panel's aft/stowed -X edge.
    # ---------------------------------------------------------
    x_hinge = -p.panel_length/2.0 - p.hinge_block_length/2.0 + 2.0
    for zsign in (-1,1):
        hinge = _box(
            p.hinge_block_length,
            p.hinge_block_width,
            p.hinge_block_height,
            x=x_hinge,
            y=0.0,
            z=zsign*p.hinge_offset_z,
        )
        parts.append(hinge.val())

    # Short central yoke/standoff visual element.
    yoke = _box(
        p.yoke_length,
        p.yoke_width,
        p.yoke_height,
        x=-p.panel_length/2.0-p.yoke_length/2.0+3.0,
        y=0.0,
        z=0.0,
    )
    parts.append(yoke.val())

    # Sun-sensor support bracket at the outboard/opposite panel edge.
    css_bracket = _box(
        p.css_bracket_length,
        p.css_bracket_height,
        p.css_bracket_width,
        x=p.panel_length/2.0-p.css_bracket_length/2.0,
        y=p.panel_thickness/2.0+p.css_bracket_height/2.0,
        z=0.0,
    )
    parts.append(css_bracket.val())

    return cq.Workplane("XY").newObject([cq.Compound.makeCompound(parts)])


def make_solar_panel_assembly(p: SolarArrayParameters = SolarArrayParameters()):
    assy=cq.Assembly(name="Detailed_Solar_Array_Panel")
    assy.add(
        make_detailed_solar_panel(p),
        name="860x300 mm 6x3 solar panel",
        color=cq.Color(0.18,0.22,0.35),
    )
    return assy


if __name__=="__main__":
    from pathlib import Path
    out=Path(__file__).resolve().parent/"outputs"
    out.mkdir(exist_ok=True)

    panel=make_detailed_solar_panel()
    assy=make_solar_panel_assembly()

    cq.exporters.export(panel,str(out/"solar_panel_860x300_detailed.step"))
    assy.save(str(out/"solar_panel_860x300_detailed_assembly.step"))
    cq.exporters.export(panel,str(out/"solar_panel_860x300_detailed.stl"))

    print("Exported detailed 860x300 mm solar panel to:",out)
