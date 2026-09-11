"""
spacecraft_v19_complete_enclosure.py

V19:
- V17 structural integration retained
- removable +/-Y and +/-Z skins retained
- +X front face upgraded to a full telescope-surrounding closure panel with
  circular aperture and reinforcement ring
- ALL SIX thrusters receive short mounting casings:
  two axial sleeves + four angled Scout-like corner sleeves
- corrected deployed solar arrays retained
"""

from pathlib import Path
import cadquery as cq

from excel_import import load_components, require_component, require_dimensions, require_position
from geometry import make_axis_indicator, make_bus_frame, place
from tank_detailed import make_detailed_tank
from battery_detailed import make_detailed_battery
from baffle_detailed import make_detailed_baffle
from telescope_detailed import make_detailed_telescope
from thruster_detailed import make_detailed_thruster
from obc_unibap_detailed import make_detailed_obc
from pcdu_p80_detailed import make_detailed_p80
from iris_v2_detailed import make_detailed_iris
from sun_sensor_detailed import make_detailed_sun_sensor
from imu_stim300_detailed import make_detailed_stim300
from star_tracker_detailed import make_detailed_star_tracker
from reaction_wheel_detailed import make_detailed_reaction_wheel
from xband_lga_detailed import make_detailed_lga
from apm_nigeriasat_detailed import make_detailed_apm
from solar_array_detailed import make_detailed_solar_panel, SolarArrayParameters
from integration_structure_detailed import make_integration_structure
from external_enclosure_detailed_v19 import (
    make_y_skin,make_z_skin,make_plus_x_skin,make_all_thruster_casings
)

HERE = Path(__file__).resolve().parent
WORKBOOK = r"C:\Users\seanw\OneDrive - University of Toronto\Thesis\Journal Paper 5 - Mission Concept\Results\Systems\budget_closing_consolidated_master_ADCS_massprops_no_circular.xlsx"
OUTPUT_DIR = HERE / "outputs"
APM_NAME="Steerable Tx APM / Directional Antenna"


def add_component(assy,c,name,fn,color,assy_name=None):
    s=require_component(c,name)
    assy.add(
        place(fn(),require_position(s),s.orientation),
        name=assy_name or name,
        color=color
    )


def _safe_mode_patch_names(c):
    for pair in [
        ("Safe-mode Rx Patch 1","Safe-mode Rx Patch 2"),
        ("LGA 1","LGA 2"),
    ]:
        if all(name in c for name in pair):
            return pair
    return []


def _deployed_panel(panel,side_sign,bus_length,bus_width):
    if side_sign>0:
        deployed=panel.rotate((0,0,0),(0,0,1),90.0)
    else:
        deployed=(
            panel
            .rotate((0,0,0),(1,0,0),180.0)
            .rotate((0,0,0),(0,0,1),-90.0)
        )
    x=-bus_length/2
    y=side_sign*(bus_width/2+SolarArrayParameters().panel_length/2+18)
    return deployed.translate((x,y,0))


def _array_css_position(side_sign,bus_length,bus_width):
    p=SolarArrayParameters()
    return (-bus_length/2,
            side_sign*(bus_width/2+p.panel_length+18),
            0)


def build_v19(workbook_path=WORKBOOK,include_skins=True):
    c=load_components(workbook_path)
    bus=require_component(c,"Microsatellite bus")
    L,W,H=require_dimensions(bus)

    assy=cq.Assembly(name="Microsatellite_V19_Complete")

    assy.add(make_bus_frame(L,W,H,member=12.0),
             name="Primary bus frame",
             color=cq.Color(0.64,0.64,0.64))

    assy.add(make_integration_structure(L,W,H),
             name="Secondary structure, trays, cradles & brackets",
             color=cq.Color(0.48,0.50,0.52))

    for name,fn,color in [
        ("Telescope Assembly",make_detailed_telescope,cq.Color(0.08,0.09,0.10)),
        ("Baffle",make_detailed_baffle,cq.Color(0.06,0.07,0.08)),
        ("Tank",make_detailed_tank,cq.Color(0.48,0.50,0.52)),
        ("Battery",make_detailed_battery,cq.Color(0.72,0.52,0.16)),
        ("PCDU",make_detailed_p80,cq.Color(0.08,0.08,0.08)),
        ("IRIS Transponder",make_detailed_iris,cq.Color(0.55,0.48,0.30)),
        ("IMU",make_detailed_stim300,cq.Color(0.18,0.16,0.13)),
    ]:
        add_component(assy,c,name,fn,color)

    for i in range(1,7):
        add_component(assy,c,f"Thruster {i}",make_detailed_thruster,cq.Color(0.45,0.38,0.27))

    for i,name in enumerate(["OBC 1","OBC 2"]):
        add_component(
            assy,c,name,make_detailed_obc,cq.Color(0.32,0.38,0.34),
            assy_name=f"{name} — {'Primary' if i==0 else 'Cold Redundant'}"
        )

    for i in range(1,5):
        add_component(assy,c,f"Sun Sensor {i}",make_detailed_sun_sensor,cq.Color(0.67,0.50,0.22))

    for i in range(1,3):
        add_component(assy,c,f"Star Tracker {i}",make_detailed_star_tracker,cq.Color(0.70,0.70,0.72))

    for i in range(1,5):
        add_component(assy,c,f"Reaction Wheel {i}",make_detailed_reaction_wheel,cq.Color(0.72,0.72,0.72))

    for idx,name in enumerate(_safe_mode_patch_names(c),start=1):
        add_component(assy,c,name,make_detailed_lga,cq.Color(0.78,0.78,0.73),
                      assy_name=f"X-band LGA {idx}")

    if APM_NAME in c:
        add_component(assy,c,APM_NAME,make_detailed_apm,cq.Color(0.61,0.61,0.59),
                      assy_name="NigeriaSat-2 Heritage APM + Tx Horn")

    for side_sign,label in [(+1,"+Y"),(-1,"-Y")]:
        assy.add(
            _deployed_panel(make_detailed_solar_panel(),side_sign,L,W),
            name=f"{label} Deployed Solar Array — 6x3 cells",
            color=cq.Color(0.16,0.20,0.36)
        )
        css=place(make_detailed_sun_sensor(),
                  _array_css_position(side_sign,L,W),
                  (0.0,float(side_sign),0.0))
        assy.add(css,name=f"{label} Array-tip Sun Sensor",
                 color=cq.Color(0.67,0.50,0.22))

    # All six thruster panel casings.
    assy.add(
        make_all_thruster_casings(L),
        name="All Six Thruster Mounting Casings / Fairing Sleeves",
        color=cq.Color(0.54,0.55,0.56)
    )

    if include_skins:
        skin_color=cq.Color(0.43,0.45,0.46)
        assy.add(make_y_skin(L,W,H,+1),name="REMOVABLE +Y Outer Skin",color=skin_color)
        assy.add(make_y_skin(L,W,H,-1),name="REMOVABLE -Y Outer Skin",color=skin_color)
        assy.add(make_z_skin(L,W,H,+1),name="REMOVABLE +Z Outer Skin",color=skin_color)
        assy.add(make_z_skin(L,W,H,-1),name="REMOVABLE -Z Outer Skin",color=skin_color)
        assy.add(make_plus_x_skin(L,W,H),
                 name="REMOVABLE +X Telescope Surround Panel",
                 color=skin_color)

    axes=make_axis_indicator(120.0,2.5)
    assy.add(axes["X axis"],name="+X axis",color=cq.Color(1,0,0))
    assy.add(axes["Y axis"],name="+Y axis",color=cq.Color(0,0.8,0))
    assy.add(axes["Z axis"],name="+Z axis",color=cq.Color(0,0.2,1))

    return assy


if __name__=="__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)

    full=OUTPUT_DIR/"spacecraft_v19_complete_enclosure.step"
    build_v19(include_skins=True).save(str(full))

    service=OUTPUT_DIR/"spacecraft_v19_service_view_no_skins.step"
    build_v19(include_skins=False).save(str(service))

    print("Exported:",full)
    print("Exported:",service)
