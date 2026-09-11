"""
iris_v2_detailed.py

Higher-craftsmanship CadQuery representation of the SDL / Space Dynamics
Laboratory IRIS V2 radio transponder.

Current authoritative envelope used in the spacecraft workbook:
    Length : 101 mm
    Width  : 101 mm
    Height : 56 mm

Visual reference cues from the supplied photograph:
- square stacked avionics enclosure
- gold/anodized top cover
- blue/grey side walls
- multiple stacked side plates
- upper connector rail / interface bar
- repeated perimeter fasteners
- side mounting ears / slots
- lower stacked connector/interface layers

This is spacecraft-configuration CAD, not manufacturing CAD.

Local component frame
---------------------
+X : 101 mm length
+Y : 101 mm width
+Z : 56 mm height
"""

from __future__ import annotations

from dataclasses import dataclass
import cadquery as cq


@dataclass(frozen=True)
class IRISParameters:
    length: float = 101.0
    width: float = 101.0
    height: float = 56.0

    # Main enclosure
    body_height: float = 41.0
    top_plate_thickness: float = 4.0
    bottom_plate_thickness: float = 3.0

    # Side stacked layers / seams
    seam_count: int = 3
    seam_thickness: float = 1.3
    seam_offset_z: tuple[float, ...] = (-17.0, -10.0, -3.0)

    # Upper rear connector rail
    rail_length: float = 92.0
    rail_width: float = 11.0
    rail_height: float = 8.0
    rail_y: float = 43.0

    # Fasteners
    top_screw_diameter: float = 4.2
    top_screw_height: float = 1.5
    top_screw_margin: float = 8.0
    side_screw_diameter: float = 3.4
    side_screw_depth: float = 1.5

    # Mounting ears
    ear_length: float = 9.0
    ear_width: float = 8.0
    ear_height: float = 16.0

    # Side connector slots
    side_slot_length: float = 29.0
    side_slot_height: float = 5.0
    side_slot_depth: float = 2.0


def _box(l,w,h,x=0.0,y=0.0,z=0.0):
    return cq.Workplane("XY").box(l,w,h,centered=(True,True,True)).translate((x,y,z))

def _z_cyl(h,d,x,y,z):
    return cq.Workplane("XY").center(x,y).circle(d/2).extrude(h/2,both=True).translate((0,0,z))

def _x_cyl(h,d,x,y,z):
    return cq.Workplane("YZ").circle(d/2).extrude(h/2,both=True).translate((x,y,z))

def make_detailed_iris(p: IRISParameters = IRISParameters()) -> cq.Workplane:
    parts=[]

    # Main side-wall body
    body=_box(p.length-4.0,p.width-4.0,p.body_height,z=-3.0)
    parts.append(body.val())

    # Bottom plate
    bottom=_box(p.length,p.width,p.bottom_plate_thickness,
                z=-p.height/2+p.bottom_plate_thickness/2)
    parts.append(bottom.val())

    # Gold top cover
    top=_box(p.length,p.width,p.top_plate_thickness,
             z=p.height/2-p.top_plate_thickness/2)
    parts.append(top.val())

    # Repeated side-wall seam plates
    for z in p.seam_offset_z:
        seam=_box(p.length+1.0,p.width+1.0,p.seam_thickness,z=z)
        parts.append(seam.val())

    # Raised rear connector/interface rail, matching the distinctive photo.
    rail=_box(p.rail_length,p.rail_width,p.rail_height,
              y=p.rail_y,z=p.height/2+1.0)
    parts.append(rail.val())

    # Rail fastener heads
    for x in (-39,-27,-15,-3,9,21,33,43):
        screw=_z_cyl(1.2,3.5,x,p.rail_y,p.height/2+p.rail_height/2+1.0)
        parts.append(screw.val())

    # Top-cover perimeter screws
    positions = [
        (-42,-42),(-18,-42),(18,-42),(42,-42),
        (-42,42),(-18,42),(18,42),(42,42),
        (-42,-12),(-42,12),(42,-12),(42,12)
    ]
    for x,y in positions:
        screw=_z_cyl(
            p.top_screw_height,p.top_screw_diameter,
            x,y,p.height/2+p.top_screw_height/2-0.4
        )
        parts.append(screw.val())

    # Side mounting ears / vertical edge blocks.
    for sx in (-1,1):
        for sy in (-1,1):
            ear=_box(
                p.ear_length,p.ear_width,p.ear_height,
                x=sx*(p.length/2+p.ear_length/2-2.0),
                y=sy*(p.width/2-p.ear_width/2),
                z=-10.0
            )
            parts.append(ear.val())

    # Side connector openings / slots.
    for sx in (-1,1):
        for z in (-16,-5,6):
            slot=_box(
                p.side_slot_depth,p.side_slot_length,p.side_slot_height,
                x=sx*(p.length/2-p.side_slot_depth/2+0.2),
                y=0.0,z=z
            )
            parts.append(slot.val())

    # Front/bottom connector bays.
    for x in (-30,-10,10,30):
        bay=_box(15.0,2.2,7.0,x=x,y=-p.width/2-0.5,z=-18.0)
        parts.append(bay.val())

    # Small circular side fasteners.
    for sx in (-1,1):
        for z in (-18,0,18):
            fast=_x_cyl(
                p.side_screw_depth,p.side_screw_diameter,
                sx*(p.length/2+0.2),0.0,z
            )
            parts.append(fast.val())

    compound=cq.Compound.makeCompound(parts)
    return cq.Workplane("XY").newObject([compound])


def make_iris_assembly(p: IRISParameters = IRISParameters()) -> cq.Assembly:
    assy=cq.Assembly(name="SDL_IRIS_V2")
    assy.add(
        make_detailed_iris(p),
        name="IRIS V2 Radio Transponder",
        color=cq.Color(0.55,0.48,0.30)
    )
    return assy


if __name__=="__main__":
    from pathlib import Path
    out=Path(__file__).resolve().parent/"outputs"
    out.mkdir(exist_ok=True)

    iris=make_detailed_iris()
    assy=make_iris_assembly()

    cq.exporters.export(iris,str(out/"iris_v2_detailed.step"))
    assy.save(str(out/"iris_v2_detailed_assembly.step"))
    cq.exporters.export(iris,str(out/"iris_v2_detailed.stl"))

    print("Exported detailed IRIS V2 geometry to:",out)
