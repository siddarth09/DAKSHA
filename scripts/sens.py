"""Sensitivity: does moving the table down or the arms forward buy G1 reach margin?"""
import numpy as np, mujoco, importlib
import zero_layout as L, reach_gate as G

rng = np.random.default_rng(0)
rb_raw, _ = G.rebot_cloud(rng)          # clouds are mount-relative once we subtract the mount
base_pts = rb_raw["left"] - np.array(L.MOUNTS["left"])
g1 = G.g1_cloud_pelvis(rng)
cw, sw = np.cos(G.G1_YAW), np.sin(G.G1_YAW)
R = np.array([[cw,-sw,0],[sw,cw,0],[0,0,1.0]])
gl, gr = g1["left"] @ R.T, g1["right"] @ R.T
hx, hy, _ = L.TABLE_HALF; cx, cy = L.TABLE_CENTER_XY

print(f"{'table_z':>8} {'base_x':>7} | overlap (L) at standoff 0.10 / 0.20 / 0.30")
print("-"*66)
for tz in (0.65, 0.70, 0.75):
    for bxm in (-0.25, -0.15, -0.05):
        mounts = {"left": (bxm, +L.BASE_SEP/2, tz), "right": (bxm, -L.BASE_SEP/2, tz)}
        both = G.inter(*[G.keys(base_pts + np.array(m)) for m in mounts.values()])
        c = (G.unkey(both) + 0.5) * G.VOXEL
        on = ((c[:,0]>=cx-hx)&(c[:,0]<=cx+hx)&(c[:,1]>=cy-hy)&(c[:,1]<=cy+hy)
              &(c[:,2]>=tz)&(c[:,2]<=tz+G.SLAB))
        for mx,my,_ in mounts.values():
            on &= np.hypot(c[:,0]-mx, c[:,1]-my) > G.BASE_KEEPOUT
        both = both[on]
        row=[]
        for so in (0.10,0.20,0.30):
            base=np.array([cx+hx+so,0.0,G.G1_PELVIS_Z])
            k=G.inter(both, G.keys(gl+base), G.keys(gr+base))
            row.append(len(k)*G.VOXEL**3*1000)
        star = "  <=" if (tz,bxm)==(0.75,-0.25) else ""
        print(f"{tz:8.2f} {bxm:7.2f} | {row[0]:6.2f} {row[1]:6.2f} {row[2]:6.2f}{star}")
