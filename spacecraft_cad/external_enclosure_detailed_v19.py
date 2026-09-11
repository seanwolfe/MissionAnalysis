"""
external_enclosure_detailed_v19.py

V19 enclosure:
- full removable skins on +/-Y, +/-Z and +X
- +X skin is now a true full front panel around the telescope with a circular
  optical aperture and reinforced aperture ring
- short Scout-like casings for BOTH axial thrusters
- angled Scout-like casings for the FOUR canted corner thrusters

The casing geometry is conceptual structural fairing / mounting hardware,
not an attempt to reproduce proprietary NEA Scout manufacturing geometry.
"""

from __future__ import annotations
from dataclasses import dataclass
import math
import cadquery as cq


@dataclass(frozen=True)
class EnclosureParameters:
    skin_thickness: float = 3.0

    # +X optical panel
    optical_aperture_diameter: float = 342.0
    optical_ring_outer_diameter: float = 378.0
    optical_ring_thickness: float = 5.0
    optical_ring_axial_depth: float = 4.0

    # Thruster casing
    axial_casing_length: float = 34.0
    canted_casing_length: float = 38.0
    casing_outer_diameter: float = 66.0
    casing_inner_diameter: float = 44.0
    casing_flange_diameter: float = 82.0
    casing_flange_thickness: float = 4.0
    casing_lip_length: float = 5.0
    casing_lip_outer_diameter: float = 72.0

    axial_y: float = 140.0
    axial_z: float = 0.0

    # Canted mounting points on the aft face.
    corner_mount_y: float = 175.0
    corner_mount_z: float = 125.0


def _box(l,w,h,x=0.0,y=0.0,z=0.0):
    return cq.Workplane("XY").box(l,w,h,centered=(True,True,True)).translate((x,y,z))


def make_y_skin(bus_l,bus_w,bus_h,side=+1,p=EnclosureParameters()):
    y=side*(bus_w/2+p.skin_thickness/2)
    return _box(bus_l,p.skin_thickness,bus_h,y=y)


def make_z_skin(bus_l,bus_w,bus_h,side=+1,p=EnclosureParameters()):
    z=side*(bus_h/2+p.skin_thickness/2)
    return _box(bus_l,bus_w,p.skin_thickness,z=z)


def make_plus_x_skin(bus_l,bus_w,bus_h,p=EnclosureParameters()):
    """
    Full +X closure panel around the telescope/baffle.

    The central optical aperture is circular and only slightly larger than the
    current ~325 mm baffle envelope.  A separate reinforcement ring gives the
    front face a much more flight-like appearance.
    """
    x=bus_l/2+p.skin_thickness/2

    outer=(
        cq.Workplane("YZ")
        .rect(bus_w,bus_h)
        .extrude(p.skin_thickness/2,both=True)
        .translate((x,0,0))
    )

    aperture=(
        cq.Workplane("YZ")
        .circle(p.optical_aperture_diameter/2)
        .extrude((p.skin_thickness+6)/2,both=True)
        .translate((x,0,0))
    )
    panel=outer.cut(aperture)

    # External reinforcing ring around telescope opening.
    ring_x = bus_l/2 + p.skin_thickness + p.optical_ring_axial_depth/2
    ring=(
        cq.Workplane("YZ")
        .circle(p.optical_ring_outer_diameter/2)
        .circle((p.optical_aperture_diameter+3.0)/2)
        .extrude(p.optical_ring_axial_depth/2,both=True)
        .translate((ring_x,0,0))
    )

    # Eight visible fastener heads around aperture ring.
    bolts=[]
    r=(p.optical_ring_outer_diameter+p.optical_aperture_diameter)/4
    for i in range(8):
        a=math.radians(i*45)
        y=r*math.cos(a)
        z=r*math.sin(a)
        b=(
            cq.Workplane("YZ")
            .circle(3.2)
            .extrude(2.0/2,both=True)
            .translate((ring_x+2.0,y,z))
        )
        bolts.append(b.val())

    return cq.Workplane("XY").newObject(
        [cq.Compound.makeCompound([panel.val(),ring.val()]+bolts)]
    )


def _x_annulus(length,od,id_,x_center,y,z):
    return (
        cq.Workplane("YZ")
        .circle(od/2)
        .circle(id_/2)
        .extrude(length/2,both=True)
        .translate((x_center,y,z))
    )


def _orient_x_to_vector(shape, vec):
    """Rotate a local +X aligned shape to a unit vector."""
    import numpy as np
    v=np.array(vec,dtype=float)
    v=v/np.linalg.norm(v)
    ex=np.array([1.0,0.0,0.0])
    dot=max(-1.0,min(1.0,float(np.dot(ex,v))))
    if dot > 0.999999:
        return shape
    if dot < -0.999999:
        return shape.rotate((0,0,0),(0,0,1),180.0)

    axis=np.cross(ex,v)
    axis=axis/np.linalg.norm(axis)
    ang=math.degrees(math.acos(dot))
    return shape.rotate((0,0,0),tuple(axis),ang)


def _local_annulus_x(length,od,id_,x0):
    return (
        cq.Workplane("YZ")
        .circle(od/2)
        .circle(id_/2)
        .extrude(length/2,both=True)
        .translate((x0,0,0))
    )


def make_oriented_thruster_casing(mount_xyz, direction,
                                  length=38.0,
                                  p=EnclosureParameters()):
    """
    Casing centered on a thruster mounting interface and aligned with the
    thruster/plume axis.

    Local +X is outward along the supplied direction.  A short annular sleeve
    covers the first part of the thruster body, with a broad aft-face flange and
    an outer lip, giving the canted jets the same integrated visual language as
    the NEA Scout corner-thruster housings.
    """
    mx,my,mz = mount_xyz

    # In local frame, mount plane at local x=0 and casing extends in +X.
    sleeve=_local_annulus_x(
        length,p.casing_outer_diameter,p.casing_inner_diameter,length/2
    )
    flange=_local_annulus_x(
        p.casing_flange_thickness,
        p.casing_flange_diameter,
        p.casing_inner_diameter,
        p.casing_flange_thickness/2
    )
    lip=_local_annulus_x(
        p.casing_lip_length,
        p.casing_lip_outer_diameter,
        p.casing_inner_diameter,
        length-p.casing_lip_length/2
    )

    # Four little fasteners around the local flange.
    bolts=[]
    r=p.casing_flange_diameter/2-8
    for y,z in [(r,0),(-r,0),(0,r),(0,-r)]:
        b=(
            cq.Workplane("YZ")
            .circle(3.0)
            .extrude(2.0/2,both=True)
            .translate((1.0,y,z))
        )
        bolts.append(b.val())

    comp=cq.Workplane("XY").newObject([
        cq.Compound.makeCompound([sleeve.val(),flange.val(),lip.val()]+bolts)
    ])
    comp=_orient_x_to_vector(comp,direction)
    return comp.translate((mx,my,mz))


def make_all_thruster_casings(bus_l,p=EnclosureParameters()):
    xface=-bus_l/2
    solids=[]

    # Axial pair.
    for y in (+p.axial_y,-p.axial_y):
        c=make_oriented_thruster_casing(
            (xface,y,p.axial_z),
            (-1.0,0.0,0.0),
            p.axial_casing_length,p
        )
        solids.append(c.val())

    # Four NEA-Scout-like canted corner jets.
    corner_specs=[
        # mount Y, mount Z, plume direction
        (-p.corner_mount_y,+p.corner_mount_z,(-0.258819,+0.683013,+0.683013)),
        (+p.corner_mount_y,+p.corner_mount_z,(-0.258819,-0.683013,+0.683013)),
        (-p.corner_mount_y,-p.corner_mount_z,(-0.258819,+0.683013,-0.683013)),
        (+p.corner_mount_y,-p.corner_mount_z,(-0.258819,-0.683013,-0.683013)),
    ]
    for y,z,d in corner_specs:
        c=make_oriented_thruster_casing(
            (xface,y,z),d,p.canted_casing_length,p
        )
        solids.append(c.val())

    return cq.Workplane("XY").newObject([cq.Compound.makeCompound(solids)])


if __name__=="__main__":
    from pathlib import Path
    out=Path(__file__).resolve().parent/"outputs"
    out.mkdir(exist_ok=True)

    L,W,H=850.0,450.0,450.0
    cq.exporters.export(make_plus_x_skin(L,W,H),str(out/"skin_plusX_full_telescope_panel.step"))
    cq.exporters.export(make_all_thruster_casings(L),str(out/"all_six_thruster_casings.step"))
    print("Exported V19 enclosure geometry to:",out)
