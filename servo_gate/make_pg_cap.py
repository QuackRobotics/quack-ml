"""Copy Playground open_duck_mini_v2.xml with forcerange capped on leg actuators only.
Usage: make_pg_cap.py <xmls_dir> <cap_Nm>. Writes open_duck_mini_v2_cap.xml + scene_flat_terrain_cap.xml next to originals."""
import sys, re, pathlib
d = pathlib.Path(sys.argv[1]); cap = float(sys.argv[2])
LEG = ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle")
src = (d / "open_duck_mini_v2.xml").read_text()
n = 0
def sub(m):
    global n
    tag = m.group(0)
    name = re.search(r'name="([^"]+)"', tag)
    if name and any(k in name.group(1) for k in LEG):
        n += 1
        tag = re.sub(r'\s*forcerange="[^"]*"', "", tag)
        return tag.replace("<position", f'<position forcerange="-{cap} {cap}"', 1)
    return tag
inside = src.split("<actuator>", 1)
head, rest = inside[0], inside[1]
act, tail = rest.split("</actuator>", 1)
act = re.sub(r"<position\b[^>]*>", sub, act)
(d / "open_duck_mini_v2_cap.xml").write_text(head + "<actuator>" + act + "</actuator>" + tail)
(d / "scene_flat_terrain_cap.xml").write_text((d / "scene_flat_terrain.xml").read_text().replace('file="open_duck_mini_v2.xml"', 'file="open_duck_mini_v2_cap.xml"'))
print("capped leg actuators:", n)
if n != 10:
    sys.exit(f"expected 10 leg actuators, got {n}")
