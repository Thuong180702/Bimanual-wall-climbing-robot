"""One joint, exploded and assembled: parent bracket, actuator, child bracket."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from cadgen import build123d as bd
from lib.links import BUILDERS, CHILD_OF, PJOINT, flange_of
from lib.shapes import _plane
from lib.spec import BODY_DIR, MOTOR_OF

MM = 0.001
STEPS = {"AK80-64": "ak80_64", "AK60-39": "ak60_39", "AK45-36": "ak45_36"}
HERE = os.path.dirname(os.path.abspath(__file__))

jn = sys.argv[1]
par = [j[1] for j in PJOINT.values() if j[0] == jn][0]
chi = [j[2] for j in PJOINT.values() if j[0] == jn][0]
key = MOTOR_OF[jn]
cj = [c for c in CHILD_OF[par] if c[0] == jn][0]

v = bd.import_step(os.path.join(HERE, "..", "STEP", "purchased",
                                STEPS[key] + ".step")).scale(MM)
m = bd.Location(_plane(flange_of(cj), tuple(-x for x in BODY_DIR[jn]))) * v
m.label = "%s  %s" % (jn, key)

p = BUILDERS[par]()
p.label = "parent  " + par
c = bd.Location(bd.Plane(origin=cj[4])) * BUILDERS[chi]()
c.label = "child  " + chi

out = bd.Compound(children=[bd.Scale(x, 1000.0) if False else x for x in (p, m, c)])
out = bd.Compound(children=[x.scale(1000.0) for x in (p, m, c)])
bd.export_step(out, os.path.join(HERE, "..", "tmp", "joint_%s.step" % jn))
print("wrote tmp/joint_%s.step" % jn)
