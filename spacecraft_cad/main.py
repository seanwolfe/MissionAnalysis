import cadquery as cq
from pathlib import Path


# ============================================================
# SIMPLE CADQUERY -> STEP -> FREECAD WORKFLOW TEST
# Units: mm
# ============================================================

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

STEP_FILE = OUTPUT_DIR / "test_spacecraft.step"


# ------------------------------------------------------------
# 1. MAIN SPACECRAFT BUS
# ------------------------------------------------------------

bus_length = 400
bus_width = 300
bus_height = 300

bus = (
    cq.Workplane("XY")
    .box(bus_length, bus_width, bus_height)
)


# ------------------------------------------------------------
# 2. SIMPLE TELESCOPE / PAYLOAD
# ------------------------------------------------------------

telescope_length = 250
telescope_diameter = 140

# Cylinder initially along Z, then rotate it so it points along +X
telescope = (
    cq.Workplane("XY")
    .cylinder(telescope_length, telescope_diameter / 2)
    .rotate((0, 0, 0), (0, 1, 0), 90)
)

# Put it on the +X side of the spacecraft
telescope = telescope.translate(
    (
        bus_length / 2 + telescope_length / 2,
        0,
        0,
    )
)


# ------------------------------------------------------------
# 3. TWO SOLAR PANELS
# ------------------------------------------------------------

panel_length = 350
panel_width = 180
panel_thickness = 8

solar_panel_1 = (
    cq.Workplane("XY")
    .box(panel_width, panel_length, panel_thickness)
    .translate(
        (
            0,
            bus_width / 2 + panel_length / 2,
            0,
        )
    )
)

solar_panel_2 = (
    cq.Workplane("XY")
    .box(panel_width, panel_length, panel_thickness)
    .translate(
        (
            0,
            -(bus_width / 2 + panel_length / 2),
            0,
        )
    )
)


# ------------------------------------------------------------
# 4. INTERNAL COMPONENT BOXES
# ------------------------------------------------------------

battery = (
    cq.Workplane("XY")
    .box(150, 100, 80)
    .translate((-50, 0, -70))
)

obc = (
    cq.Workplane("XY")
    .box(100, 80, 40)
    .translate((40, 60, 80))
)

pcdu = (
    cq.Workplane("XY")
    .box(90, 70, 35)
    .translate((40, -70, 80))
)


# ------------------------------------------------------------
# 5. SIMPLE PROPULSION TANK
# ------------------------------------------------------------

tank = (
    cq.Workplane("XY")
    .sphere(70)
    .translate((-80, 0, 40))
)


# ------------------------------------------------------------
# 6. BUILD ASSEMBLY
# ------------------------------------------------------------

spacecraft = cq.Assembly(name="TestSpacecraft")

spacecraft.add(
    bus,
    name="Bus",
    color=cq.Color(0.65, 0.65, 0.68)
)

spacecraft.add(
    telescope,
    name="Telescope",
    color=cq.Color(0.25, 0.25, 0.30)
)

spacecraft.add(
    solar_panel_1,
    name="SolarPanel_PosY",
    color=cq.Color(0.10, 0.20, 0.65)
)

spacecraft.add(
    solar_panel_2,
    name="SolarPanel_NegY",
    color=cq.Color(0.10, 0.20, 0.65)
)

spacecraft.add(
    battery,
    name="Battery",
    color=cq.Color(0.20, 0.70, 0.20)
)

spacecraft.add(
    obc,
    name="OBC",
    color=cq.Color(0.85, 0.55, 0.10)
)

spacecraft.add(
    pcdu,
    name="PCDU",
    color=cq.Color(0.90, 0.80, 0.10)
)

spacecraft.add(
    tank,
    name="PropellantTank",
    color=cq.Color(0.70, 0.70, 0.75)
)


# ------------------------------------------------------------
# 7. EXPORT TO STEP
# ------------------------------------------------------------

spacecraft.export(str(STEP_FILE))

print("STEP export complete.")
print(f"File created at: {STEP_FILE.resolve()}")