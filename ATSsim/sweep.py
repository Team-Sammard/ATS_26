import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
# ---------------- PARAMETERS ----------------

start_time = time.time()
theta = 0
angle_step = np.deg2rad(20)
b = np.deg2rad(1.5)

r1 = 4000
r2 = 4000

beam_radius1 = 350
beam_radius2 = 525

max_rho = np.deg2rad(60)

az0 = np.deg2rad(45)
el0 = np.deg2rad(45)

running = True

# ---------------- FIGURE ----------------

fig = plt.figure()

ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

for ax, r in [(ax1, r1), (ax2, r2)]:
    ax.set_proj_type('ortho')
    ax.set_xlim(0, r)
    ax.set_ylim(0, r)
    ax.set_zlim(0, r)
    ax.set_box_aspect([1,1,1])
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")

ax1.set_title("Archimedes Spiral")
ax2.set_title("Jalebi bai")

# ---------------- PLOTS ----------------

line1, = ax1.plot([], [], [])
circle1, = ax1.plot([], [], [], color='red')
trail1, = ax1.plot([], [], [], color='red')

line2, = ax2.plot([], [], [])
circle2, = ax2.plot([], [], [], color='red')
trail2, = ax2.plot([], [], [], color='red')

trail_x1, trail_y1, trail_z1 = [], [], []
trail_x2, trail_y2, trail_z2 = [], [], []

# ---------------- HELPERS ----------------

def perpendicular_circle(x, y, z, radius):
    v = np.array([x, y, z])
    v /= np.linalg.norm(v)

    ref = np.array([1,0,0]) if abs(v[0]) < 0.9 else np.array([0,1,0])
    u1 = np.cross(v, ref)
    u1 /= np.linalg.norm(u1)
    u2 = np.cross(v, u1)
    u2 /= np.linalg.norm(u2)

    phi = np.linspace(0, 2*np.pi, 50)
    pts = (np.outer(np.cos(phi), u1) + np.outer(np.sin(phi), u2)) * radius
    pts += np.array([x, y, z])

    return pts[:,0], pts[:,1], pts[:,2]

# ---------------- SECOND ANGLE STATE ----------------

az2 = 0
el2 = 0
az_step = np.deg2rad(5)
el_step = np.deg2rad(5)

# ---------------- UPDATE 1 (SPHERICAL SPIRAL) ----------------

def update_spiral():
    global theta, running

    if not running:
        return

    # ---- Grow spiral ----
    theta += angle_step
    rho = b * theta
    phi = theta

    if rho > max_rho:
        running = False

        elapsed_time = time.time() - start_time

        area = 2 * np.pi * r1**2 * (1 - np.cos(rho))

        print("Spiral finished")
        print("Time elapsed:", round(elapsed_time, 2), "seconds")
        print("spiral 1 area covered (m^2):", round(area, 2))
        print("Area per time (m^2/s): ", round(area / elapsed_time, 2))
        return


    # ---- Center unit vector ----
    v0 = np.array([
        np.cos(el0) * np.cos(az0),
        np.cos(el0) * np.sin(az0),
        np.sin(el0)
    ])

    # ---- Build orthonormal basis ----
    if abs(v0[0]) < 0.9:
        ref = np.array([1, 0, 0])
    else:
        ref = np.array([0, 1, 0])

    u1 = np.cross(v0, ref)
    u1 /= np.linalg.norm(u1)
    u2 = np.cross(v0, u1)

    # ---- Exact spherical rotation (Rodrigues) ----
    tangent = np.cos(phi) * u1 + np.sin(phi) * u2

    v = (
        v0 * np.cos(rho)
        + np.cross(tangent, v0) * np.sin(rho)
        + tangent * np.dot(tangent, v0) * (1 - np.cos(rho))
    )

    # ---- Scale to radius ----
    x1, y1, z1 = r1 * v

    # ---- Update pointing line ----
    line1.set_data([0, x1], [0, y1])
    line1.set_3d_properties([0, z1])

    # ---- Perpendicular beam disk ----
    cx1, cy1, cz1 = perpendicular_circle(x1, y1, z1, beam_radius1)
    circle1.set_data(cx1, cy1)
    circle1.set_3d_properties(cz1)

    # ---- Trail of swept area (beam rim) ----
    trail_x1.extend(cx1)
    trail_y1.extend(cy1)
    trail_z1.extend(cz1)

    trail1.set_data(trail_x1, trail_y1)
    trail1.set_3d_properties(trail_z1)

# ---------------- UPDATE 2 (NESTED ANGLE SWEEP) ----------------
az2 = az0                      # start at center
el2 = 0                        # starting elevation

az_step = np.deg2rad(3)        # sweep speed
el_step_slow = np.deg2rad(0.35) # slow rise speed
az_span = np.deg2rad(20)

el_step = np.deg2rad(3)
max_el = np.deg2rad(90)

az_direction = 1

start_time2 = None
finished2 = False
el_center = np.deg2rad(20)
el_span = np.deg2rad(8)

drift_rate = np.deg2rad(5)   # slow upward climb per unit time

def update_nested(frame):
    global az2, el2, az_direction, az_step
    global start_time2, finished2

    if finished2:
        return

    if start_time2 is None:
        start_time2 = time.time()

    # ---- Azimuth back-and-forth sweep ----
    t = frame * 0.3   # time parameter (adjust speed)

    # Constant width az sweep
    az2 = az0 + az_span * np.sin(t)

    # Infinity motion in elevation
    el2 = el_center + el_span * np.sin(2 * t)

    # Slow upward drift
    el2 += drift_rate * t


    # ---- Stop condition when elevation finished ----
    if el2 > max_el:
        finished2 = True

        elapsed = time.time() - start_time2

        # Sector spherical area calculation
        az_width = 2 * az_span
        area = r2**2 * az_width * (1 - np.cos(max_el))

        print("\nNested sector sweep finished")
        print("Time elapsed:", round(elapsed, 2), "seconds")
        print("Area covered (m^2 approx):", round(area, 2))
        print("Area per time (m^2/s): ", round(area / elapsed, 2))
        return

    # ---- Convert spherical to Cartesian ----
    x2 = r2 * np.cos(el2) * np.cos(az2)
    y2 = r2 * np.cos(el2) * np.sin(az2)
    z2 = r2 * np.sin(el2)

    # ---- Update pointing line ----
    line2.set_data([0, x2], [0, y2])
    line2.set_3d_properties([0, z2])

    # ---- Beam disk ----
    cx2, cy2, cz2 = perpendicular_circle(x2, y2, z2, beam_radius2)
    circle2.set_data(cx2, cy2)
    circle2.set_3d_properties(cz2)

    # ---- Trail ----
    trail_x2.extend(cx2)
    trail_y2.extend(cy2)
    trail_z2.extend(cz2)

    trail2.set_data(trail_x2, trail_y2)
    trail2.set_3d_properties(trail_z2)


# ---------------- MASTER UPDATE ----------------

def update(frame):

    update_spiral()
    update_nested(frame)

    return (
        line1, circle1,
        line2, circle2
    )

# ---------------- ANIMATION ----------------

ani = FuncAnimation(
    fig,
    update,
    interval=2,
    blit=False,
    cache_frame_data=False
)

plt.show()
