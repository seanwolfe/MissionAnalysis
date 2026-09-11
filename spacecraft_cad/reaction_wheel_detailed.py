"""
reaction_wheel_detailed.py

Detailed spacecraft-level CadQuery representation of the selected
Blue Canyon Technologies RWP100-style reaction wheel.

Current workbook envelope:
    70 x 70 x 25 mm

Visual reference cues from the supplied image:
- square silver structural frame
- four thick corner supports
- circular exposed wheel / rotor on the top face
- recessed dark annular gap around the rotor
- central hub / fastener
- visible curved internal wheel housing beneath the top plate
- small top fasteners at the corners

This is configuration / publication CAD, not vendor manufacturing CAD.

Orientation convention
----------------------
The spacecraft workbook stores each reaction-wheel SPIN AXIS vector.
The generic placement helper aligns a component's local +X axis to that vector.

Therefore this model is authored with:
    local +X = wheel spin axis
    local Y/Z = 70 x 70 mm mounting face
    local X = 25 mm package thickness

This allows all four tetrahedral wheel-axis vectors in the workbook to be used
directly.
"""

from __future__ import annotations

from dataclasses import dataclass
import cadquery as cq


@dataclass(frozen=True)
class ReactionWheelParameters:
    face_y: float = 70.0
    face_z: float = 70.0
    thickness_x: float = 25.0

    # Structural plates
    front_plate_thickness: float = 3.0
    rear_plate_thickness: float = 2.5
    plate_inset: float = 1.5

    # Central wheel / rotor
    rotor_outer_diameter: float = 52.0
    rotor_inner_detail_diameter: float = 34.0
    rotor_depth: float = 4.0
    rotor_recess: float = 2.0
    hub_diameter: float = 9.0
    hub_depth: float = 2.0

    # Annular gap / inner housing
    housing_outer_diameter: float = 58.0
    housing_inner_diameter: float = 48.0
    housing_depth: float = 5.0

    # Four corner posts
    post_y: float = 8.0
    post_z: float = 8.0
    post_margin: float = 7.0
    post_depth: float = 19.0

    # Side frame openings
    side_bar_width: float = 7.0
    side_bar_depth: float = 4.0

    # Fasteners
    screw_diameter: float = 3.2
    screw_depth: float = 1.2
    screw_offset_y: float = 29.0
    screw_offset_z: float = 29.0

    # Decorative rotor slots / scallop cues
    slot_count: int = 12
    slot_length: float = 7.0
    slot_width: float = 2.2


def _box_x(thickness, y_size, z_size, x=0.0, y=0.0, z=0.0):
    return (
        cq.Workplane("XY")
        .box(thickness, y_size, z_size, centered=(True, True, True))
        .translate((x, y, z))
    )


def _x_cylinder(length, diameter, x, y=0.0, z=0.0):
    return (
        cq.Workplane("YZ")
        .circle(diameter / 2.0)
        .extrude(length / 2.0, both=True)
        .translate((x, y, z))
    )


def _x_ring(length, outer_d, inner_d, x):
    return (
        cq.Workplane("YZ")
        .circle(outer_d / 2.0)
        .circle(inner_d / 2.0)
        .extrude(length / 2.0, both=True)
        .translate((x, 0.0, 0.0))
    )


def make_detailed_reaction_wheel(
    p: ReactionWheelParameters = ReactionWheelParameters(),
) -> cq.Workplane:
    parts = []

    x_front = -p.thickness_x / 2.0
    x_rear = +p.thickness_x / 2.0

    # ---------------------------------------------------------
    # Front/top plate (the face with the visible wheel)
    # ---------------------------------------------------------
    front = _box_x(
        p.front_plate_thickness,
        p.face_y,
        p.face_z,
        x=x_front + p.front_plate_thickness / 2.0,
    )
    try:
        front = front.edges("|X").fillet(1.5)
    except Exception:
        pass
    parts.append(front.val())

    # Rear/base plate
    rear = _box_x(
        p.rear_plate_thickness,
        p.face_y - 3.0,
        p.face_z - 3.0,
        x=x_rear - p.rear_plate_thickness / 2.0,
    )
    parts.append(rear.val())

    # ---------------------------------------------------------
    # Four corner structural posts
    # ---------------------------------------------------------
    x_post = 0.5
    for sy in (-1.0, 1.0):
        for sz in (-1.0, 1.0):
            y = sy * (p.face_y / 2.0 - p.post_margin)
            z = sz * (p.face_z / 2.0 - p.post_margin)
            post = _box_x(
                p.post_depth,
                p.post_y,
                p.post_z,
                x=x_post,
                y=y,
                z=z,
            )
            try:
                post = post.edges("|X").fillet(1.0)
            except Exception:
                pass
            parts.append(post.val())

    # Side frame bars connecting the corners.
    for zsign in (-1.0, 1.0):
        bar = _box_x(
            p.side_bar_depth,
            p.face_y - 16.0,
            p.side_bar_width,
            x=4.0,
            y=0.0,
            z=zsign * (p.face_z / 2.0 - p.side_bar_width / 2.0),
        )
        parts.append(bar.val())

    for ysign in (-1.0, 1.0):
        bar = _box_x(
            p.side_bar_depth,
            p.side_bar_width,
            p.face_z - 16.0,
            x=4.0,
            y=ysign * (p.face_y / 2.0 - p.side_bar_width / 2.0),
            z=0.0,
        )
        parts.append(bar.val())

    # ---------------------------------------------------------
    # Rotor / wheel system on visible front face
    # ---------------------------------------------------------
    rotor_x = x_front - p.rotor_depth / 2.0 + 0.6

    # Dark annular housing gap under the rotor.
    housing = _x_ring(
        p.housing_depth,
        p.housing_outer_diameter,
        p.housing_inner_diameter,
        rotor_x + 1.2,
    )
    parts.append(housing.val())

    # Main circular rotor disc.
    rotor = _x_cylinder(
        p.rotor_depth,
        p.rotor_outer_diameter,
        rotor_x,
    )
    parts.append(rotor.val())

    # Raised inner annulus / engraved circular detail.
    inner_ring = _x_ring(
        1.0,
        p.rotor_inner_detail_diameter + 5.0,
        p.rotor_inner_detail_diameter - 2.0,
        rotor_x - p.rotor_depth/2.0 - 0.2,
    )
    parts.append(inner_ring.val())

    # Central hub.
    hub = _x_cylinder(
        p.hub_depth,
        p.hub_diameter,
        rotor_x - p.rotor_depth/2.0 - p.hub_depth/2.0 + 0.2,
    )
    parts.append(hub.val())

    # ---------------------------------------------------------
    # Rotor slot/scallop details.
    # These sit just above the rotor face and give the visible circular
    # machined pattern seen in the reference image.
    # ---------------------------------------------------------
    import math
    rr = p.rotor_inner_detail_diameter / 2.0 + 5.0
    for i in range(p.slot_count):
        a = math.radians(i * 360.0 / p.slot_count)
        y = rr * math.cos(a)
        z = rr * math.sin(a)

        slot = _box_x(
            0.7,
            p.slot_length,
            p.slot_width,
            x=rotor_x - p.rotor_depth/2.0 - 0.45,
            y=y,
            z=z,
        ).rotate((0,0,0),(1,0,0), i * 360.0 / p.slot_count)
        parts.append(slot.val())

    # ---------------------------------------------------------
    # Four visible corner fasteners on the front plate
    # ---------------------------------------------------------
    for sy in (-1.0, 1.0):
        for sz in (-1.0, 1.0):
            screw = _x_cylinder(
                p.screw_depth,
                p.screw_diameter,
                x_front - p.screw_depth/2.0 + 0.2,
                sy * p.screw_offset_y,
                sz * p.screw_offset_z,
            )
            parts.append(screw.val())

    return cq.Workplane("XY").newObject([cq.Compound.makeCompound(parts)])


def make_reaction_wheel_assembly(
    p: ReactionWheelParameters = ReactionWheelParameters(),
) -> cq.Assembly:
    assy = cq.Assembly(name="BCT_RWP100_Detailed")
    assy.add(
        make_detailed_reaction_wheel(p),
        name="BCT RWP100 Reaction Wheel",
        color=cq.Color(0.72,0.72,0.72),
    )
    return assy


if __name__ == "__main__":
    from pathlib import Path
    out = Path(__file__).resolve().parent / "outputs"
    out.mkdir(exist_ok=True)

    rw = make_detailed_reaction_wheel()
    assy = make_reaction_wheel_assembly()

    cq.exporters.export(rw, str(out / "rwp100_reaction_wheel_detailed.step"))
    assy.save(str(out / "rwp100_reaction_wheel_detailed_assembly.step"))
    cq.exporters.export(rw, str(out / "rwp100_reaction_wheel_detailed.stl"))

    print("Exported detailed reaction wheel geometry to:", out)
