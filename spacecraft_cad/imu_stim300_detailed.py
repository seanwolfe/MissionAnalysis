"""
imu_stim300_detailed.py

Detailed spacecraft-level CadQuery representation of the Safran STIM300 IMU.

The current spacecraft workbook carries the STIM300 envelope as:
    65 x 55 x 10 mm

The user-provided mechanical drawing shows the visually important features:
- compact rectangular sensor body
- asymmetric raised top/interface bridge
- three circular mounting/interface features
- small top fasteners
- gold-toned side/end structure
- black central sensor enclosure

This is configuration / publication CAD, not manufacturing CAD.

Local component frame
---------------------
+X : 65 mm length
+Y : 55 mm width
+Z : 10 mm nominal thickness

The current spacecraft placement sheet uses body-aligned orientation (1,0,0)
for the IMU, so this local frame maps directly to the spacecraft body frame.
"""

from __future__ import annotations

from dataclasses import dataclass
import cadquery as cq


@dataclass(frozen=True)
class STIM300Parameters:
    length: float = 65.0
    width: float = 55.0
    height: float = 10.0

    # Main body
    body_length: float = 46.0
    body_width: float = 38.0
    body_height: float = 8.5
    body_x_offset: float = 3.0
    body_y_offset: float = 1.0

    # Rear/top mounting bridge
    bridge_length: float = 34.0
    bridge_width: float = 8.0
    bridge_height: float = 3.0
    bridge_x: float = -8.0
    bridge_y: float = 18.0

    # Mounting/interface ears
    ear_diameter: float = 7.5
    ear_height: float = 2.5
    ear_positions: tuple[tuple[float,float], ...] = (
        (-18.0, 17.0),
        (18.0, 17.0),
        (0.0, -18.0),
    )

    # Through / visible mounting bores
    hole_diameter: float = 3.0

    # Connector / service block
    connector_length: float = 18.0
    connector_width: float = 6.0
    connector_height: float = 3.2
    connector_x: float = -5.0
    connector_y: float = 20.0

    # Small screws / cover details
    screw_diameter: float = 2.2
    screw_height: float = 0.8


def _box(l,w,h,x=0.0,y=0.0,z=0.0):
    return (
        cq.Workplane("XY")
        .box(l,w,h,centered=(True,True,True))
        .translate((x,y,z))
    )


def _z_cyl(h,d,x,y,z):
    return (
        cq.Workplane("XY")
        .center(x,y)
        .circle(d/2.0)
        .extrude(h/2.0,both=True)
        .translate((0.0,0.0,z))
    )


def make_detailed_stim300(
    p: STIM300Parameters = STIM300Parameters(),
) -> cq.Workplane:
    parts = []

    # Thin base/reference plate.
    base = _box(
        p.length,
        p.width,
        1.8,
        z=-p.height/2.0 + 0.9,
    )
    try:
        base = base.edges("|Z").fillet(1.2)
    except Exception:
        pass
    parts.append(base.val())

    # Main dark sensor enclosure.
    body = _box(
        p.body_length,
        p.body_width,
        p.body_height,
        x=p.body_x_offset,
        y=p.body_y_offset,
        z=-0.2,
    )
    try:
        body = body.edges("|Z").fillet(2.0)
    except Exception:
        pass
    parts.append(body.val())

    # Raised top/rear structural bridge, clearly visible in the drawing.
    bridge = _box(
        p.bridge_length,
        p.bridge_width,
        p.bridge_height,
        x=p.bridge_x,
        y=p.bridge_y,
        z=p.height/2.0 + p.bridge_height/2.0 - 1.0,
    )
    try:
        bridge = bridge.edges("|Z").fillet(1.0)
    except Exception:
        pass
    parts.append(bridge.val())

    # Three mounting/interface bosses.
    for x,y in p.ear_positions:
        boss = _z_cyl(
            p.ear_height,
            p.ear_diameter,
            x,y,
            p.height/2.0 + p.ear_height/2.0 - 1.2,
        )
        parts.append(boss.val())

        # Dark bore insert for visual clarity.
        bore = _z_cyl(
            p.ear_height + 0.5,
            p.hole_diameter,
            x,y,
            p.height/2.0 + p.ear_height/2.0 - 1.0,
        )
        parts.append(bore.val())

    # Connector / interface bar.
    connector = _box(
        p.connector_length,
        p.connector_width,
        p.connector_height,
        x=p.connector_x,
        y=p.connector_y,
        z=p.height/2.0 + p.connector_height/2.0 - 0.8,
    )
    parts.append(connector.val())

    # Small repeated top fasteners.
    for x,y in [(-22,11), (22,11), (-16,-12), (16,-12)]:
        screw = _z_cyl(
            p.screw_height,
            p.screw_diameter,
            x,y,
            p.height/2.0 + p.screw_height/2.0 - 0.2,
        )
        parts.append(screw.val())

    # Small side rail to break up the simple box silhouette.
    side_rail = _box(
        8.0,
        p.body_width - 5.0,
        2.0,
        x=p.body_x_offset - p.body_length/2.0 + 2.5,
        y=p.body_y_offset,
        z=3.0,
    )
    parts.append(side_rail.val())

    return cq.Workplane("XY").newObject([cq.Compound.makeCompound(parts)])


def make_stim300_assembly(
    p: STIM300Parameters = STIM300Parameters(),
) -> cq.Assembly:
    assy = cq.Assembly(name="Safran_STIM300_IMU")
    assy.add(
        make_detailed_stim300(p),
        name="STIM300 IMU",
        color=cq.Color(0.12,0.12,0.12),
    )
    return assy


if __name__ == "__main__":
    from pathlib import Path
    out = Path(__file__).resolve().parent / "outputs"
    out.mkdir(exist_ok=True)

    imu = make_detailed_stim300()
    assy = make_stim300_assembly()

    cq.exporters.export(imu, str(out / "stim300_detailed.step"))
    assy.save(str(out / "stim300_detailed_assembly.step"))
    cq.exporters.export(imu, str(out / "stim300_detailed.stl"))

    print("Exported detailed STIM300 geometry to:", out)
