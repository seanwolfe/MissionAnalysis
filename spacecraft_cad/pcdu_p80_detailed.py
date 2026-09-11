from dataclasses import dataclass
from pathlib import Path
import cadquery as cq

@dataclass(frozen=True)
class P80Parameters:
    length: float = 95.0
    width: float = 94.9
    height: float = 38.8
    body_inset_xy: float = 4.0
    top_plate_thickness: float = 2.5
    bottom_plate_thickness: float = 2.5
    rail_width: float = 7.0
    rail_projection: float = 2.0
    rail_hole_diameter: float = 2.5
    rail_hole_count: int = 6
    rail_hole_spacing: float = 5.56
    slot_height: float = 6.0
    slot_depth: float = 2.5
    long_slot_length: float = 40.0
    short_slot_length: float = 22.0
    screw_head_diameter: float = 4.0
    screw_head_height: float = 1.5
    screw_margin: float = 7.0

def _box(l,w,h,x=0,y=0,z=0):
    return cq.Workplane("XY").box(l,w,h,centered=(True,True,True)).translate((x,y,z))

def _z_cyl(h,d,x,y,z):
    return cq.Workplane("XY").center(x,y).circle(d/2).extrude(h/2,both=True).translate((0,0,z))

def _x_cyl(h,d,x,y,z):
    return cq.Workplane("YZ").circle(d/2).extrude(h/2,both=True).translate((x,y,z))

def make_detailed_p80(p=P80Parameters()):
    parts=[]
    body=_box(p.length-2*p.body_inset_xy,p.width-2*p.body_inset_xy,
              p.height-p.top_plate_thickness-p.bottom_plate_thickness)
    parts.append(body.val())
    parts.append(_box(p.length-2,p.width-2,p.top_plate_thickness,
                      z=p.height/2-p.top_plate_thickness/2).val())
    parts.append(_box(p.length-2,p.width-2,p.bottom_plate_thickness,
                      z=-p.height/2+p.bottom_plate_thickness/2).val())

    rx=p.length/2-p.rail_width/2+p.rail_projection/2
    ry=p.width/2-p.rail_width/2+p.rail_projection/2
    for sx in (-1,1):
        for sy in (-1,1):
            parts.append(_box(p.rail_width,p.rail_width,p.height+2,
                              x=sx*rx,y=sy*ry).val())
            z0=-(p.rail_hole_count-1)*p.rail_hole_spacing/2
            for i in range(p.rail_hole_count):
                z=z0+i*p.rail_hole_spacing
                parts.append(_x_cyl(1.4,p.rail_hole_diameter,
                                    sx*(p.length/2+0.3),
                                    sy*(p.width/2-3.5),z).val())

    for sx in (-1,1):
        for sy in (-1,1):
            parts.append(_z_cyl(p.screw_head_height,p.screw_head_diameter,
                                sx*(p.length/2-p.screw_margin),
                                sy*(p.width/2-p.screw_margin),
                                p.height/2+p.screw_head_height/2-0.4).val())

    for sy in (-1,1):
        for z in (-9,9):
            parts.append(_box(p.long_slot_length,p.slot_depth,p.slot_height,
                              y=sy*(p.width/2-p.slot_depth/2+0.2),z=z).val())
            parts.append(_box(p.long_slot_length-3,3.0,1.6,
                              y=sy*(p.width/2+0.4),z=z).val())

    for sx in (-1,1):
        for z in (-8,8):
            parts.append(_box(p.slot_depth,p.short_slot_length,p.slot_height,
                              x=sx*(p.length/2-p.slot_depth/2+0.2),z=z).val())

    for x,lx in [(-24,9),(-8,7),(10,5),(26,5)]:
        parts.append(_box(lx,2.0,5.0,x=x,y=-p.width/2-0.8,z=-11).val())

    return cq.Workplane("XY").newObject([cq.Compound.makeCompound(parts)])

def make_p80_assembly(p=P80Parameters()):
    assy=cq.Assembly(name="GomSpace_NanoPower_P80")
    assy.add(make_detailed_p80(p),name="NanoPower P80",color=cq.Color(0.08,0.08,0.08))
    return assy

if __name__=="__main__":
    out=Path(__file__).resolve().parent/"outputs"
    out.mkdir(exist_ok=True)
    pcdu=make_detailed_p80()
    assy=make_p80_assembly()
    cq.exporters.export(pcdu,str(out/"gomspace_p80_detailed.step"))
    assy.save(str(out/"gomspace_p80_detailed_assembly.step"))
    cq.exporters.export(pcdu,str(out/"gomspace_p80_detailed.stl"))
    print("Exported detailed GomSpace NanoPower P80 geometry to:",out)
