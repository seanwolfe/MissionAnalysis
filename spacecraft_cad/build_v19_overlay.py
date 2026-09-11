from pathlib import Path
import math
import cadquery as cq
import numpy as np

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)

L,W,H = 850.0,450.0,450.0
XFACE = -L/2.0

# ---------- helpers ----------
def orient_x_to_vector(shape, vec):
    v=np.array(vec,dtype=float); v=v/np.linalg.norm(v)
    ex=np.array([1.,0.,0.])
    dot=max(-1.0,min(1.0,float(np.dot(ex,v))))
    if dot > 0.999999:
        return shape
    if dot < -0.999999:
        return shape.rotate((0,0,0),(0,0,1),180)
    axis=np.cross(ex,v); axis=axis/np.linalg.norm(axis)
    ang=math.degrees(math.acos(dot))
    return shape.rotate((0,0,0),tuple(axis),ang)

def local_annulus(length, od, id_, xcenter):
    return cq.Workplane("YZ").circle(od/2).circle(id_/2).extrude(length/2,both=True).translate((xcenter,0,0))

def make_casing(mount, direction, length=38.0):
    od,id_=66.0,44.0
    sleeve=local_annulus(length,od,id_,length/2)
    flange=local_annulus(4.0,82.0,id_,2.0)
    lip=local_annulus(5.0,72.0,id_,length-2.5)
    solids=[sleeve.val(),flange.val(),lip.val()]
    r=33.0
    for y,z in [(r,0),(-r,0),(0,r),(0,-r)]:
        bolt=cq.Workplane("YZ").circle(3.0).extrude(1.0,both=True).translate((1.0,y,z))
        solids.append(bolt.val())
    comp=cq.Workplane("XY").newObject([cq.Compound.makeCompound(solids)])
    return orient_x_to_vector(comp,direction).translate(mount)

# ---------- four canted Scout-like casings ----------
corner_specs=[
    ((XFACE,-175,+125),(-0.258819,+0.683013,+0.683013)),
    ((XFACE,+175,+125),(-0.258819,-0.683013,+0.683013)),
    ((XFACE,-175,-125),(-0.258819,+0.683013,-0.683013)),
    ((XFACE,+175,-125),(-0.258819,-0.683013,-0.683013)),
]
canted=[]
for m,d in corner_specs:
    canted.append(make_casing(m,d).val())
canted_comp=cq.Workplane("XY").newObject([cq.Compound.makeCompound(canted)])
cq.exporters.export(canted_comp,str(OUT/"four_canted_thruster_casings.step"))

# ---------- full +X face around telescope ----------
skin_t=3.0
x=L/2+skin_t/2
outer=cq.Workplane("YZ").rect(W,H).extrude(skin_t/2,both=True).translate((x,0,0))
aperture_d=342.0
hole=cq.Workplane("YZ").circle(aperture_d/2).extrude((skin_t+6)/2,both=True).translate((x,0,0))
panel=outer.cut(hole)
ring_x=L/2+skin_t+2.0
ring=cq.Workplane("YZ").circle(189.0).circle((aperture_d+3)/2).extrude(2.0,both=True).translate((ring_x,0,0))
bolts=[]
r=180.0
for i in range(8):
    a=math.radians(i*45)
    yy=r*math.cos(a); zz=r*math.sin(a)
    b=cq.Workplane("YZ").circle(3.2).extrude(1.0,both=True).translate((ring_x+2,yy,zz))
    bolts.append(b.val())
front=cq.Workplane("XY").newObject([cq.Compound.makeCompound([panel.val(),ring.val()]+bolts)])
cq.exporters.export(front,str(OUT/"plusX_full_telescope_panel.step"))

# ---------- upgrade existing V18 full model ----------
src=Path("/mnt/data/spacecraft_v18_complete_enclosure.step")
base=cq.importers.importStep(str(src))

assy=cq.Assembly(name="Microsatellite_V19")
assy.add(base,name="V18 spacecraft base",color=cq.Color(0.7,0.7,0.7))
assy.add(canted_comp,name="Four canted NEA-Scout-like thruster casings",color=cq.Color(0.54,0.55,0.56))
assy.add(front,name="Full +X telescope surround panel",color=cq.Color(0.43,0.45,0.46))
assy.save(str(OUT/"spacecraft_v19_complete_enclosure.step"))

print("done")
