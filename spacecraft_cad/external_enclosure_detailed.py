"""
external_enclosure_detailed.py

External removable spacecraft skins plus short axial-thruster mounting sleeves.

Requested V18 changes
---------------------
1. Add full removable skin panels on:
   +Y, -Y, +Z, -Z, +X
   The -X face remains the dedicated propulsion bulkhead / thruster face.

2. Add Scout-like short casings around the two axial thrusters where they pass
   through the aft propulsion panel.  These are represented as structural
   mounting sleeves / fairing collars, not as pressure vessels.

The external skins are intentionally separate solids so they can be hidden or
removed in a STEP assembly when an internal packaging view is desired.
"""

from __future__ import annotations
from dataclasses import dataclass
import cadquery as cq


@dataclass(frozen=True)
class EnclosureParameters:
    skin_thickness: float = 3.0
    edge_relief: float = 2.0

    # +X optical aperture
    optical_opening_y: float = 370.0
    optical_opening_z: float = 370.0
    optical_corner_radius: float = 8.0

    # Axial-thruster sleeve/casing
    casing_length: float = 34.0
    casing_outer_diameter: float = 66.0
    casing_inner_diameter: float = 45.0
    casing_flange_diameter: float = 82.0
    casing_flange_thickness: float = 4.0
    casing_lip_length: float = 5.0
    casing_lip_outer_diameter: float = 72.0

    # Axial thruster mount locations on aft face.
    axial_y: float = 140.0
    axial_z: float = 0.0


def _box(l,w,h,x=0.0,y=0.0,z=0.0):
    return (
        cq.Workplane("XY")
        .box(l,w,h,centered=(True,True,True))
        .translate((x,y,z))
    )


def make_y_skin(bus_l, bus_w, bus_h, side=+1,
                p: EnclosureParameters=EnclosureParameters()):
    """Full removable +/-Y skin."""
    y = side * (bus_w/2.0 + p.skin_thickness/2.0)
    return _box(bus_l, p.skin_thickness, bus_h, y=y)


def make_z_skin(bus_l, bus_w, bus_h, side=+1,
                p: EnclosureParameters=EnclosureParameters()):
    """Full removable +/-Z skin."""
    z = side * (bus_h/2.0 + p.skin_thickness/2.0)
    return _box(bus_l, bus_w, p.skin_thickness, z=z)


def make_plus_x_skin(bus_l, bus_w, bus_h,
                     p: EnclosureParameters=EnclosureParameters()):
    """
    +X closure panel with a large central optical opening.
    This reads visually as a complete external panel while preserving the
    telescope/baffle aperture.
    """
    x = bus_l/2.0 + p.skin_thickness/2.0

    outer = (
        cq.Workplane("YZ")
        .rect(bus_w, bus_h)
        .extrude(p.skin_thickness/2.0, both=True)
        .translate((x,0,0))
    )

    # Square-ish optical opening with softened corners.
    inner = (
        cq.Workplane("YZ")
        .rect(p.optical_opening_y, p.optical_opening_z)
        .extrude((p.skin_thickness+4.0)/2.0, both=True)
        .translate((x,0,0))
    )
    try:
        inner = inner.edges("|X").fillet(p.optical_corner_radius)
    except Exception:
        pass

    return outer.cut(inner)


def _x_annulus(length, od, id_, x_center, y, z):
    return (
        cq.Workplane("YZ")
        .circle(od/2.0)
        .circle(id_/2.0)
        .extrude(length/2.0, both=True)
        .translate((x_center,y,z))
    )


def make_axial_thruster_casing(bus_l, y,
                               p: EnclosureParameters=EnclosureParameters()):
    """
    Short structural sleeve around one axial thruster.

    The aft panel lies at X=-bus_l/2.
    The sleeve projects outward in -X and hides the awkward first portion of
    the thruster body where it passes through the propulsion bulkhead.
    """
    x_face = -bus_l/2.0

    # Main cylindrical mounting sleeve, entirely outboard of the aft panel.
    sleeve_center = x_face - p.casing_length/2.0
    sleeve = _x_annulus(
        p.casing_length,
        p.casing_outer_diameter,
        p.casing_inner_diameter,
        sleeve_center,
        y,
        p.axial_z
    )

    # Broad flange flush against the spacecraft aft panel.
    flange_center = x_face - p.casing_flange_thickness/2.0
    flange = _x_annulus(
        p.casing_flange_thickness,
        p.casing_flange_diameter,
        p.casing_inner_diameter,
        flange_center,
        y,
        p.axial_z
    )

    # Small outer lip gives the casing a more flight-hardware appearance.
    lip_center = x_face - p.casing_length + p.casing_lip_length/2.0
    lip = _x_annulus(
        p.casing_lip_length,
        p.casing_lip_outer_diameter,
        p.casing_inner_diameter,
        lip_center,
        y,
        p.axial_z
    )

    # Four flange fastener heads.
    bolts = []
    r = p.casing_flange_diameter/2.0 - 8.0
    for yy,zz in [(y+r,p.axial_z),(y-r,p.axial_z),
                  (y,p.axial_z+r),(y,p.axial_z-r)]:
        bolt = (
            cq.Workplane("YZ")
            .circle(3.2)
            .extrude(2.0/2.0,both=True)
            .translate((x_face-1.0,yy,zz))
        )
        bolts.append(bolt.val())

    solids = [sleeve.val(), flange.val(), lip.val()] + bolts
    return cq.Workplane("XY").newObject([cq.Compound.makeCompound(solids)])


def make_axial_thruster_casings(bus_l,
                                p: EnclosureParameters=EnclosureParameters()):
    solids=[]
    for y in (+p.axial_y,-p.axial_y):
        solids.append(make_axial_thruster_casing(bus_l,y,p).val())
    return cq.Workplane("XY").newObject([cq.Compound.makeCompound(solids)])


if __name__=="__main__":
    from pathlib import Path
    out=Path(__file__).resolve().parent/"outputs"
    out.mkdir(exist_ok=True)

    L,W,H=850.0,450.0,450.0
    cq.exporters.export(make_y_skin(L,W,H,+1),str(out/"skin_plusY.step"))
    cq.exporters.export(make_y_skin(L,W,H,-1),str(out/"skin_minusY.step"))
    cq.exporters.export(make_z_skin(L,W,H,+1),str(out/"skin_plusZ.step"))
    cq.exporters.export(make_z_skin(L,W,H,-1),str(out/"skin_minusZ.step"))
    cq.exporters.export(make_plus_x_skin(L,W,H),str(out/"skin_plusX_optical.step"))
    cq.exporters.export(make_axial_thruster_casings(L),str(out/"axial_thruster_casings.step"))

    print("Exported V18 enclosure parts to:",out)
