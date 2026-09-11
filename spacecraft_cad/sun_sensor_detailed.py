"""
sun_sensor_detailed.py

Detailed spacecraft-level CAD representation of the selected Bradford Space /
Cosine coarse sun sensor, based on the supplied product photograph and the
current workbook envelope.

Workbook envelope:
    30 x 30 x 14 mm

IMPORTANT orientation convention
--------------------------------
The spacecraft placement sheet stores sensor boresight / sensing-normal vectors.
The generic placement helper aligns a component's LOCAL +X axis to that vector.

Therefore this sun-sensor model is authored with:
    local +X = sensing normal / outward face
    local YZ = 30 x 30 mm mounting footprint
    local X thickness = 14 mm

This makes all six workbook orientation vectors directly usable.

Visual features represented:
- square gold mounting plate
- raised rectangular sensor housing
- dark/black optical detector plate
- gold sensing-circle marking represented geometrically
- central protective/cover block
- four mounting holes
- small polarity / interface details
"""

from __future__ import annotations

from dataclasses import dataclass
import cadquery as cq


@dataclass(frozen=True)
class SunSensorParameters:
    footprint_y: float = 30.0
    footprint_z: float = 30.0
    total_thickness_x: float = 14.0

    base_thickness: float = 2.5
    base_corner_radius: float = 1.5

    frame_y: float = 19.0
    frame_z: float = 18.0
    frame_depth: float = 8.0
    frame_wall: float = 2.0

    detector_y: float = 14.0
    detector_z: float = 9.0
    detector_depth: float = 1.2

    cover_y: float = 8.5
    cover_z: float = 10.0
    cover_depth: float = 2.4

    mount_hole_diameter: float = 2.6
    mount_hole_offset_y: float = 10.0
    mount_hole_offset_z: float = 10.0

    sensing_ring_outer_diameter: float = 6.5
    sensing_ring_inner_diameter: float = 5.6
    sensing_ring_depth: float = 0.6


def _box_x(thickness, y_size, z_size, x=0.0, y=0.0, z=0.0):
    """
    Rectangular solid where X is thickness and Y/Z are footprint dimensions.
    """
    return (
        cq.Workplane("XY")
        .box(thickness, y_size, z_size, centered=(True, True, True))
        .translate((x, y, z))
    )


def _x_cylinder(length, diameter, x, y, z):
    return (
        cq.Workplane("YZ")
        .circle(diameter / 2.0)
        .extrude(length / 2.0, both=True)
        .translate((x, y, z))
    )


def make_detailed_sun_sensor(
    p: SunSensorParameters = SunSensorParameters(),
) -> cq.Workplane:
    parts = []

    # Base plate centered so its rear surface lies at the rear of the envelope.
    x_back = -p.total_thickness_x / 2.0
    x_base = x_back + p.base_thickness / 2.0

    base = _box_x(
        p.base_thickness,
        p.footprint_y,
        p.footprint_z,
        x=x_base,
    )
    try:
        base = base.edges("|X").fillet(p.base_corner_radius)
    except Exception:
        pass
    parts.append(base.val())

    # Four visible mounting-hole recesses / inserts.
    for sy in (-1.0, 1.0):
        for sz in (-1.0, 1.0):
            y = sy * p.mount_hole_offset_y
            z = sz * p.mount_hole_offset_z
            hole_insert = _x_cylinder(
                1.0,
                p.mount_hole_diameter,
                x_back + 0.7,
                y,
                z,
            )
            parts.append(hole_insert.val())

    # Raised gold frame around sensing element.
    frame_x = x_back + p.base_thickness + p.frame_depth / 2.0
    frame_outer = _box_x(
        p.frame_depth,
        p.frame_y,
        p.frame_z,
        x=frame_x,
    )
    frame_inner = _box_x(
        p.frame_depth + 1.0,
        p.frame_y - 2.0 * p.frame_wall,
        p.frame_z - 2.0 * p.frame_wall,
        x=frame_x + 0.3,
    )
    frame = frame_outer.cut(frame_inner)
    parts.append(frame.val())

    # Dark optical detector plate inside the frame, near the outward face.
    detector_x = x_back + p.base_thickness + p.frame_depth - p.detector_depth / 2.0
    detector = _box_x(
        p.detector_depth,
        p.detector_y,
        p.detector_z,
        x=detector_x,
        y=-1.0,
    )
    parts.append(detector.val())

    # Central/side protective cover block seen in the reference image.
    cover_x = detector_x + p.cover_depth / 2.0 + 0.3
    cover = _box_x(
        p.cover_depth,
        p.cover_y,
        p.cover_z,
        x=cover_x,
        y=3.8,
    )
    try:
        cover = cover.edges("|X").fillet(0.8)
    except Exception:
        pass
    parts.append(cover.val())

    # Geometric sensing-ring detail on the dark detector surface.
    ring = (
        cq.Workplane("YZ")
        .circle(p.sensing_ring_outer_diameter / 2.0)
        .circle(p.sensing_ring_inner_diameter / 2.0)
        .extrude(p.sensing_ring_depth / 2.0, both=True)
        .translate((detector_x + p.detector_depth / 2.0 + p.sensing_ring_depth / 2.0, -3.0, 0.0))
    )
    parts.append(ring.val())

    # Thin reference line extending from the sensing ring, as visible in photo.
    line = _box_x(
        0.6,
        7.0,
        0.8,
        x=detector_x + p.detector_depth / 2.0 + 0.3,
        y=1.6,
        z=0.0,
    )
    parts.append(line.val())

    # Small polarity / interface studs on exposed base.
    for y, z in [(-11.2, -6.0), (-11.2, +6.0)]:
        stud = _x_cylinder(
            0.9,
            1.6,
            x_back + p.base_thickness + 0.3,
            y,
            z,
        )
        parts.append(stud.val())

    return cq.Workplane("XY").newObject([cq.Compound.makeCompound(parts)])


def make_sun_sensor_assembly(
    p: SunSensorParameters = SunSensorParameters(),
) -> cq.Assembly:
    assy = cq.Assembly(name="Bradford_Cosine_Sun_Sensor")
    assy.add(
        make_detailed_sun_sensor(p),
        name="Coarse Sun Sensor",
        color=cq.Color(0.65, 0.49, 0.24),
    )
    return assy


if __name__ == "__main__":
    from pathlib import Path
    out = Path(__file__).resolve().parent / "outputs"
    out.mkdir(exist_ok=True)

    sensor = make_detailed_sun_sensor()
    assy = make_sun_sensor_assembly()

    cq.exporters.export(sensor, str(out / "cosine_sun_sensor_detailed.step"))
    assy.save(str(out / "cosine_sun_sensor_detailed_assembly.step"))
    cq.exporters.export(sensor, str(out / "cosine_sun_sensor_detailed.stl"))
    print("Exported detailed coarse sun sensor geometry to:", out)
