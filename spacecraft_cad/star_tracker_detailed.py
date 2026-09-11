"""
star_tracker_detailed.py

Detailed spacecraft-level CadQuery representation of the selected
BCT Mid Extension Nano Star Tracker.

Current workbook envelope:
    170 x 85 x 70 mm

Orientation convention
----------------------
The spacecraft workbook stores the star-tracker boresight vector. The generic
placement helper aligns a component's local +X axis to that vector.

Therefore this model is authored with:
    local +X = optical boresight / aperture direction
    local Y  = 85 mm side-to-side
    local Z  = 70 mm top-to-bottom
    local X  = 170 mm length

This is publication/configuration CAD, not vendor manufacturing CAD.

Reference-image features reproduced:
- gold/anodized front optical bezel
- black recessed aperture/window
- long silver sensor body
- top cover plate with screws
- side relief pocket
- rear connector plate with two small ports
- small side mounting/interface lugs
"""

from __future__ import annotations
from dataclasses import dataclass
import cadquery as cq


@dataclass(frozen=True)
class StarTrackerParameters:
    length: float = 170.0
    width: float = 85.0
    height: float = 70.0

    front_bezel_thickness: float = 8.0
    front_bezel_overhang_y: float = 10.0
    front_bezel_overhang_z: float = 8.0

    body_length: float = 150.0
    body_width: float = 68.0
    body_height: float = 58.0
    body_x_center: float = 18.0

    top_cover_thickness: float = 3.0
    top_cover_inset: float = 3.0

    aperture_width: float = 52.0
    aperture_height: float = 40.0
    aperture_depth: float = 4.0
    aperture_corner: float = 8.0

    side_pocket_length: float = 36.0
    side_pocket_height: float = 24.0
    side_pocket_depth: float = 6.0

    rear_plate_thickness: float = 4.0
    port_diameter: float = 5.0

    screw_diameter: float = 3.0
    screw_height: float = 1.2
    bezel_screw_y: float = 34.0
    bezel_screw_z: float = 28.0

    lug_length: float = 8.0
    lug_width: float = 10.0
    lug_height: float = 8.0


def _box(l, w, h, x=0.0, y=0.0, z=0.0):
    return cq.Workplane("XY").box(l, w, h, centered=(True, True, True)).translate((x, y, z))

def _z_cyl(h, d, x, y, z):
    return cq.Workplane("XY").center(x, y).circle(d/2).extrude(h/2, both=True).translate((0,0,z))

def _x_cyl(h, d, x, y, z):
    return cq.Workplane("YZ").circle(d/2).extrude(h/2, both=True).translate((x,y,z))

def make_detailed_star_tracker(p: StarTrackerParameters = StarTrackerParameters()) -> cq.Workplane:
    parts = []

    x_front = -p.length/2.0
    x_rear =  p.length/2.0

    # Main silver body
    body = _box(
        p.body_length, p.body_width, p.body_height,
        x=p.body_x_center
    )
    try:
        body = body.edges("|X").fillet(1.8)
    except Exception:
        pass
    parts.append(body.val())

    # Front gold bezel plate
    bezel = _box(
        p.front_bezel_thickness,
        p.width + p.front_bezel_overhang_y,
        p.height + p.front_bezel_overhang_z,
        x=x_front + p.front_bezel_thickness/2.0
    )
    try:
        bezel = bezel.edges("|X").fillet(1.5)
    except Exception:
        pass
    parts.append(bezel.val())

    # Black recessed optical aperture/window
    aperture_frame = _box(
        p.aperture_depth,
        p.aperture_width,
        p.aperture_height,
        x=x_front + p.front_bezel_thickness + p.aperture_depth/2.0 - 1.0
    )
    try:
        aperture_frame = aperture_frame.edges("|X").fillet(p.aperture_corner)
    except Exception:
        pass
    parts.append(aperture_frame.val())

    # Rear connector plate
    rear_plate = _box(
        p.rear_plate_thickness,
        p.body_width,
        p.body_height,
        x=x_rear - p.rear_plate_thickness/2.0
    )
    parts.append(rear_plate.val())

    # Top cover
    top_cover = _box(
        p.body_length - 6.0,
        p.body_width - 2*p.top_cover_inset,
        p.top_cover_thickness,
        x=p.body_x_center,
        z=p.body_height/2.0 + p.top_cover_thickness/2.0
    )
    parts.append(top_cover.val())

    # Side relief pocket (visual cue from reference)
    side_pocket = _box(
        p.side_pocket_length,
        p.side_pocket_depth,
        p.side_pocket_height,
        x=22.0,
        y=-p.body_width/2.0 - p.side_pocket_depth/2.0 + 0.4,
        z=-3.0
    )
    parts.append(side_pocket.val())

    # Rear ports/connectors
    for y, z in [(16.0, 10.0), (16.0, -10.0)]:
        port = _x_cyl(
            3.0, p.port_diameter,
            x_rear + 1.2, y, z
        )
        parts.append(port.val())

    # Front bezel screws
    for sy in (-1, 1):
        for sz in (-1, 1):
            screw = _x_cyl(
                p.screw_height, p.screw_diameter,
                x_front + p.front_bezel_thickness/2.0 + 0.3,
                sy * p.bezel_screw_y,
                sz * p.bezel_screw_z
            )
            parts.append(screw.val())

    # Top cover screws
    for x in (-50, -20, 20, 50):
        screw = _z_cyl(
            p.screw_height, p.screw_diameter,
            x, 0.0, p.body_height/2.0 + p.top_cover_thickness + 0.2
        )
        parts.append(screw.val())

    # Small side mounting/interface lugs
    for ysign in (-1, 1):
        lug = _box(
            p.lug_length, p.lug_width, p.lug_height,
            x=-15.0,
            y=ysign*(p.body_width/2.0 + p.lug_width/2.0 - 1.5),
            z=-18.0
        )
        parts.append(lug.val())

    return cq.Workplane("XY").newObject([cq.Compound.makeCompound(parts)])


def make_star_tracker_assembly(p: StarTrackerParameters = StarTrackerParameters()) -> cq.Assembly:
    assy = cq.Assembly(name="BCT_Mid_Extension_Nano_Star_Tracker")
    assy.add(
        make_detailed_star_tracker(p),
        name="BCT Mid Extension Nano Star Tracker",
        color=cq.Color(0.78, 0.69, 0.45)
    )
    return assy


if __name__ == "__main__":
    from pathlib import Path
    out = Path(__file__).resolve().parent / "outputs"
    out.mkdir(exist_ok=True)

    st = make_detailed_star_tracker()
    assy = make_star_tracker_assembly()

    cq.exporters.export(st, str(out / "star_tracker_detailed.step"))
    assy.save(str(out / "star_tracker_detailed_assembly.step"))
    cq.exporters.export(st, str(out / "star_tracker_detailed.stl"))
    print("Exported detailed star tracker geometry to:", out)
