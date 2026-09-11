"""
integration_structure_detailed.py

Concept-level secondary structure and equipment mounting for the microsatellite.

Purpose
-------
Turn the open-frame subsystem packaging model into a mechanically credible,
"secured and complete" pre-Phase-A spacecraft without pretending to have
manufacturing-level structural definition.

Included:
- four longitudinal shear-panel frames (+/-Y and +/-Z)
- forward optical bulkhead ring
- aft propulsion bulkhead
- -Y avionics equipment tray and support angles
- dual-ring tank cradle with radial support struts
- battery shelf and retention brackets
- star-tracker local doubler plates
- APM circular mounting doubler
- aft thruster mounting doublers
- solar-array hinge/yoke support brackets
- local reaction-wheel mounting pedestals

Geometry is derived from the spacecraft envelope and current component
placements wherever practical.
"""

from __future__ import annotations
from dataclasses import dataclass
import math
import cadquery as cq


@dataclass(frozen=True)
class StructureParameters:
    skin_thickness: float = 4.0
    bulkhead_thickness: float = 6.0
    panel_frame_width: float = 34.0

    tray_thickness: float = 4.0
    angle_leg: float = 14.0
    angle_thickness: float = 3.0

    tank_ring_x1: float = -345.0
    tank_ring_x2: float = -190.0
    tank_ring_od: float = 226.0
    tank_ring_id: float = 211.0
    tank_ring_thickness: float = 8.0

    battery_shelf_l: float = 205.0
    battery_shelf_w: float = 125.0
    battery_shelf_t: float = 4.0

    star_tracker_plate_l: float = 190.0
    star_tracker_plate_w: float = 108.0
    star_tracker_plate_t: float = 5.0

    apm_doubler_d: float = 174.0
    apm_doubler_t: float = 6.0


def _box(l,w,h,x=0,y=0,z=0):
    return cq.Workplane("XY").box(l,w,h,centered=(True,True,True)).translate((x,y,z))


def _x_ring(thickness, od, id_, x):
    return (
        cq.Workplane("YZ")
        .circle(od/2.0)
        .circle(id_/2.0)
        .extrude(thickness/2.0,both=True)
        .translate((x,0,0))
    )


def _yz_plate(thickness, ysize, zsize, x):
    return cq.Workplane("YZ").rect(ysize,zsize).extrude(thickness/2,both=True).translate((x,0,0))


def _xz_plate(xsize, thickness, zsize, y):
    return cq.Workplane("XZ").rect(xsize,zsize).extrude(thickness/2,both=True).translate((0,y,0))


def _xy_plate(xsize, ysize, thickness, z):
    return cq.Workplane("XY").rect(xsize,ysize).extrude(thickness/2,both=True).translate((0,0,z))


def _frame_xz(bus_l,bus_h,t,fw,y):
    outer = _xz_plate(bus_l,t,bus_h,y)
    inner = _xz_plate(bus_l-2*fw,t+2,bus_h-2*fw,y)
    return outer.cut(inner)


def _frame_xy(bus_l,bus_w,t,fw,z):
    outer = _xy_plate(bus_l,bus_w,t,z)
    inner = _xy_plate(bus_l-2*fw,bus_w-2*fw,t+2,z)
    return outer.cut(inner)


def _strut_between(p1,p2,width=10.0):
    """
    Rectangular strut aligned between two 3-D points.
    """
    import numpy as np
    a=np.array(p1,dtype=float)
    b=np.array(p2,dtype=float)
    v=b-a
    L=float(np.linalg.norm(v))
    if L < 1e-6:
        return _box(width,width,width,*a)

    mid=(a+b)/2
    # Start with local +X along strut.
    solid=_box(L,width,width)
    ux=v/L
    # quaternion-like axis-angle from +X to ux
    ex=np.array([1.0,0.0,0.0])
    dot=max(-1.0,min(1.0,float(np.dot(ex,ux))))
    if dot < 0.999999:
        axis=np.cross(ex,ux)
        n=np.linalg.norm(axis)
        if n < 1e-9:
            axis=np.array([0.0,0.0,1.0])
            ang=180.0
        else:
            axis=axis/n
            ang=math.degrees(math.acos(dot))
        solid=solid.rotate((0,0,0),tuple(axis),ang)
    return solid.translate(tuple(mid))


def make_integration_structure(bus_l=850.0,bus_w=450.0,bus_h=450.0,
                               p: StructureParameters=StructureParameters()):
    parts=[]

    # ---------------------------------------------------------
    # Longitudinal shear-panel frames.  These give the bus a
    # closed, secured appearance while preserving large windows
    # so internal equipment remains visible in the concept CAD.
    # ---------------------------------------------------------
    parts.append(_frame_xz(bus_l,bus_h,p.skin_thickness,p.panel_frame_width,+bus_w/2).val())
    parts.append(_frame_xz(bus_l,bus_h,p.skin_thickness,p.panel_frame_width,-bus_w/2).val())
    parts.append(_frame_xy(bus_l,bus_w,p.skin_thickness,p.panel_frame_width,+bus_h/2).val())
    parts.append(_frame_xy(bus_l,bus_w,p.skin_thickness,p.panel_frame_width,-bus_h/2).val())

    # Forward optical bulkhead: ring around telescope/baffle region.
    x_front=bus_l/2
    front_outer=_yz_plate(p.bulkhead_thickness,bus_w,bus_h,x_front)
    front_inner=_yz_plate(p.bulkhead_thickness+2,360.0,360.0,x_front)
    parts.append(front_outer.cut(front_inner).val())

    # Aft propulsion bulkhead, full plate.
    x_aft=-bus_l/2
    parts.append(_yz_plate(p.bulkhead_thickness,bus_w,bus_h,x_aft).val())

    # ---------------------------------------------------------
    # -Y avionics tray supporting OBC1/OBC2/PCDU/IRIS/IMU.
    # Current equipment occupies roughly x=-368..-160 and z=-105..105.
    # ---------------------------------------------------------
    tray_x=-270.0
    tray_l=250.0
    tray_h=245.0
    tray_y=-216.0
    tray=_xz_plate(tray_l,p.tray_thickness,tray_h,tray_y).translate((tray_x,0,0))
    parts.append(tray.val())

    # Tray edge rails / angle-like stiffeners.
    for z in (-tray_h/2+8.0, tray_h/2-8.0):
        parts.append(_box(tray_l,10.0,10.0,tray_x,tray_y+4.0,z).val())
    for x in (tray_x-tray_l/2+8.0,tray_x+tray_l/2-8.0):
        parts.append(_box(10.0,10.0,tray_h,x,tray_y+4.0,0).val())

    # Four stand-offs from tray to bus frame / side panel.
    for x in (tray_x-105,tray_x+105):
        for z in (-100,100):
            parts.append(_box(14.0,18.0,14.0,x,-225+9.0,z).val())

    # ---------------------------------------------------------
    # Tank cradle: two annular hoops and diagonal radial struts.
    # Tank ~207 mm diameter centred at y=z=0.
    # ---------------------------------------------------------
    for x in (p.tank_ring_x1,p.tank_ring_x2):
        parts.append(_x_ring(p.tank_ring_thickness,p.tank_ring_od,p.tank_ring_id,x).val())

        # Four diagonal support struts to ±Y/±Z structural regions.
        r=p.tank_ring_od/2.0
        attach=[
            ((x, +r*0.70, +r*0.70),(x,+bus_w/2-18,+bus_h/2-18)),
            ((x, -r*0.70, +r*0.70),(x,-bus_w/2+18,+bus_h/2-18)),
            ((x, +r*0.70, -r*0.70),(x,+bus_w/2-18,-bus_h/2+18)),
            ((x, -r*0.70, -r*0.70),(x,-bus_w/2+18,-bus_h/2+18)),
        ]
        for a,b in attach:
            parts.append(_strut_between(a,b,8.0).val())

    # Two longitudinal cradle tie-bars.
    for y,z in [(0,118),(0,-118),(118,0),(-118,0)]:
        parts.append(_box(abs(p.tank_ring_x2-p.tank_ring_x1),8,8,
                          (p.tank_ring_x1+p.tank_ring_x2)/2,y,z).val())

    # ---------------------------------------------------------
    # Battery shelf below battery at x=-267.5, z~162.5.
    # ---------------------------------------------------------
    batt_x=-267.5
    batt_z=109.5   # battery bottom ~113.5 mm; leave small mounting interface
    shelf=_xy_plate(p.battery_shelf_l,p.battery_shelf_w,p.battery_shelf_t,batt_z).translate((batt_x,0,0))
    parts.append(shelf.val())

    # Battery retention corners.
    for xsign in (-1,1):
        for ysign in (-1,1):
            parts.append(_box(
                14,14,55,
                batt_x+xsign*(p.battery_shelf_l/2-10),
                ysign*(p.battery_shelf_w/2-10),
                batt_z+27
            ).val())

    # Shelf struts down to tank cradle / side structure.
    for ysign in (-1,1):
        parts.append(_strut_between(
            (batt_x,ysign*52,batt_z),
            (batt_x,ysign*165,35),
            8
        ).val())

    # ---------------------------------------------------------
    # Reaction wheel mounting pedestals.  Compact blocks behind
    # each current wheel centre; visually communicate hard mounting.
    # ---------------------------------------------------------
    rw_positions=[
        (-267.5,155,50),
        (-267.5,155,-50),
        (-267.5,70,-155),
        (-267.5,-20,-155),
    ]
    for x,y,z in rw_positions:
        # pedestal biased toward nearest structural wall in Y/Z.
        dy=10 if y >= 0 else -10
        dz=10 if z >= 0 else -10
        parts.append(_box(22,22,22,x,y-dy,z-dz).val())

    # ---------------------------------------------------------
    # External local doubler plates.
    # ---------------------------------------------------------
    # Star trackers on ±Z.
    for z in (+bus_h/2,-bus_h/2):
        parts.append(_box(
            p.star_tracker_plate_l,
            p.star_tracker_plate_w,
            p.star_tracker_plate_t,
            250.0,
            -120.0 if z>0 else 120.0,
            z
        ).val())

    # APM circular doubler on +Z at x=-75, y=0.
    apm = (
        cq.Workplane("XY")
        .circle(p.apm_doubler_d/2.0)
        .extrude(p.apm_doubler_t/2.0,both=True)
        .translate((-75.0,0.0,bus_h/2))
    )
    parts.append(apm.val())

    # Aft thruster mounting doublers at mounting-interface locations.
    thruster_mounts=[
        (140,0),(-140,0),
        (-175,125),(175,125),(-175,-125),(175,-125)
    ]
    for y,z in thruster_mounts:
        pad=(
            cq.Workplane("YZ")
            .circle(29.0)
            .extrude(4.0/2,both=True)
            .translate((x_aft-1.0,y,z))
        )
        parts.append(pad.val())

    # ---------------------------------------------------------
    # Solar-array hinge support at aft (-X) edge of each ±Y face.
    # Two hinge blocks per side, axis parallel Z.
    # ---------------------------------------------------------
    for side in (-1,1):
        y=side*(bus_w/2+7.0)
        for z in (-92,92):
            parts.append(_box(34,22,30,x_aft+12,y,z).val())
            # small outboard yoke link
            parts.append(_box(28,28,12,x_aft,y+side*15,z).val())

    return cq.Workplane("XY").newObject([cq.Compound.makeCompound(parts)])


if __name__=="__main__":
    from pathlib import Path
    out=Path(__file__).resolve().parent/"outputs"
    out.mkdir(exist_ok=True)
    s=make_integration_structure()
    cq.exporters.export(s,str(out/"integration_structure_detailed.step"))
    cq.exporters.export(s,str(out/"integration_structure_detailed.stl"))
    print("Exported integration structure to:",out)
