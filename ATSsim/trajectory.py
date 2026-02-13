import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rocketpy import Environment, SolidMotor, Rocket, Flight
from rocketpy import Accelerometer, Barometer, GnssReceiver, Gyroscope
import datetime
from filterpy.kalman import ExtendedKalmanFilter

#ENVIRONMENT:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

env = Environment(latitude=31.042639, longitude=-103.541944, elevation=1400)

tomorrow = datetime.date.today() + datetime.timedelta(days=1)
env.set_date(
    (tomorrow.year, tomorrow.month, tomorrow.day, 12)
)  # Hour given in UTC time

env.set_atmospheric_model(
    type="custom_atmosphere", wind_u=[(0, 3), (10000, 8)], wind_v=[(0, 5), (10000, -5)]
)

#ROCKET:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

Cesaroni_M1670 = SolidMotor(
    thrust_source=r"Cesaroni_M1670.eng",
    dry_mass=1.815,
    dry_inertia=(0.125, 0.125, 0.002),
    nozzle_radius=33 / 1000,
    grain_number=5,
    grain_density=1815,
    grain_outer_radius=33 / 1000,
    grain_initial_inner_radius=15 / 1000,
    grain_initial_height=120 / 1000,
    grain_separation=5 / 1000,
    grains_center_of_mass_position=0.397,
    center_of_dry_mass_position=0.317,
    nozzle_position=0,
    burn_time=3.9,
    throat_radius=11 / 1000,
    coordinate_system_orientation="nozzle_to_combustion_chamber",
)

calisto = Rocket(
    radius=127 / 2000,
    mass=14.426,
    inertia=(6.321, 6.321, 0.034),
    power_off_drag=r"powerOffCurveDrag.csv",
    power_on_drag=r"powerOnCurveDrag.csv",
    center_of_mass_without_motor=0,
    coordinate_system_orientation="tail_to_nose",
)

calisto.add_motor(Cesaroni_M1670, position=-1.255)

main = calisto.add_parachute(
    name="main",
    cd_s=10.0,
    trigger=800,      # ejection altitude in meters
    sampling_rate=105,
    lag=1.5,
    noise=(0, 8.3, 0.5),
    radius=1.5,
    height=1.5,
    porosity=0.0432,
)

drogue = calisto.add_parachute(
    name="drogue",
    cd_s=1.0,
    trigger="apogee",  # ejection at apogee
    sampling_rate=105,
    lag=1.5,
    noise=(0, 8.3, 0.5),
    radius=1.5,
    height=1.5,
    porosity=0.0432,
)

nose_cone = calisto.add_nose(
    length=0.55829, kind="von karman", position=1.278
)

fin_set = calisto.add_trapezoidal_fins(
    n=4,
    root_chord=0.120,
    tip_chord=0.060,
    span=0.110,
    position=-1.04956,
    cant_angle=0.5,
    airfoil=(r"radians.csv","radians"),
)

tail = calisto.add_tail(
    top_radius=0.0635, bottom_radius=0.0435, length=0.060, position=-1.194656
)

#SENSORS:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

accel_noisy_nosecone = Accelerometer(
    sampling_rate=50,
    consider_gravity=False,
    orientation=(60, 60, 60),
    measurement_range=70,
    resolution=0.4882,
    noise_density=0.05,
    random_walk_density=0.02,
    constant_bias=1,
    operating_temperature=25,
    temperature_bias=0.02,
    temperature_scale_factor=0.02,
    cross_axis_sensitivity=0.02,
    name="Accelerometer in Nosecone",
)

accel_clean_cdm = Accelerometer(
    sampling_rate=50,
    consider_gravity=False,
    orientation=[
        [0.25, -0.0581, 0.9665],
        [0.433, 0.8995, -0.0581],
        [-0.8661, 0.433, 0.25],
    ],
    name="Accelerometer in CDM",
)

gyro_clean = Gyroscope(sampling_rate=100)
gyro_noisy = Gyroscope(
    sampling_rate=100,
    resolution=0.001064225153655079, 
    orientation=(-60, -60, -60),
    noise_density=[0, 0.03, 0.05],
    noise_variance=1.01,
    random_walk_density=[0, 0.01, 0.02],
    random_walk_variance=[1, 1, 1.05],
    constant_bias=[0, 0.3, 0.5],
    operating_temperature=25,
    temperature_bias=[0, 0.01, 0.02],
    temperature_scale_factor=[0, 0.01, 0.02],
    cross_axis_sensitivity=0.5,
    acceleration_sensitivity=[0, 0.0008, 0.0017],
    name="Gyroscope",
)

calisto.add_sensor(gyro_clean, -0.10482544178314143)  # +0.5, 127/2000)
calisto.add_sensor(gyro_noisy, (1.278 - 0.4, 127 / 2000 - 127 / 4000, 0))

barometer_clean = Barometer(
    sampling_rate=50,
    measurement_range=100000,
    resolution=0.16,
    noise_density=19,
    noise_variance=19,
    random_walk_density=0.01,
    constant_bias=1,
    operating_temperature=25,
    temperature_bias=0.02,
    temperature_scale_factor=0.02,
)
calisto.add_sensor(barometer_clean, (-0.10482544178314143 + 0.5, -127 / 2000, 0))

gnss_clean = GnssReceiver(
    sampling_rate=1,
    position_accuracy=1,
    altitude_accuracy=1,
)
calisto.add_sensor(gnss_clean, (-0.10482544178314143 + 0.5, +127 / 2000, 0))

calisto.add_sensor(accel_noisy_nosecone, 1.278)
calisto.add_sensor(accel_clean_cdm, -0.10482544178314143)  # , 127/2000)

test_flight = Flight(
    rocket=calisto, environment=env, rail_length=5.2, inclination=85, heading=0, time_overshoot=False
    )
#test_flight.plots.trajectory_3d()

# To export sensor data to a csv file:
#barometer_clean.export_measured_data("exported_barometer_data.csv")
#gyro_clean.export_measured_data("exported_gyro_data.csv")
#accel_noisy_nosecone.export_measured_data("exported_accel_noisy_nosecone_data.csv")
#accel_clean_cdm.export_measured_data("exported_accel_clean_cdm_data.csv")
#gnss_clean.export_measured_data("exported_gnss_clean_data.csv")

#NOISY GPS AND EKFILTERED GPS::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

x_true = np.asarray(test_flight.x)[:, 1] #true trajetory
y_true = np.asarray(test_flight.y)[:, 1]
z_true = np.asarray(test_flight.z)[:, 1]
t      = np.asarray(test_flight.time)

GPS_STD = 30.0   #gps noise 

noise_x = np.random.normal(0, GPS_STD, size=len(x_true))
noise_y = np.random.normal(0, GPS_STD, size=len(y_true))
noise_z = np.random.normal(0, GPS_STD, size=len(z_true))

x_gps = x_true + noise_x #Noisy GPS
y_gps = y_true + noise_y
z_gps = z_true + noise_z

#RSSI:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
gcs = np.array([0.0, -2000.0, 0.0])
pos = np.column_stack((x_gps, y_gps, z_gps))
d = np.linalg.norm(pos - gcs, axis=1) 
refd = 50 #reference distanec for lora. 50m in this case
refsig = 50 #sinalfg strength 50dbm for lora at 50m
d = np.maximum(d, 50) 
pathLoss = refsig + 10*2.2*np.log10(d / refd)

TX_POWER_DBM = 17.0   #for lora
rssi = TX_POWER_DBM - pathLoss

RSSI_MAX = -90    # "good link" at long range
RSSI_MIN = -115   # near noise floor
weight = (rssi - RSSI_MIN) / (RSSI_MAX - RSSI_MIN)
weights = np.clip(weight, 0.05, 1.0)

#noisy gps values simulated every 1Hz
gps_rate_hz = 1.0          # 1 Hz GPS
sim_dt = np.mean(np.diff(t))
gps_step = int(1 / (gps_rate_hz * sim_dt))

x_gps_plot = x_gps[::gps_step]
y_gps_plot = y_gps[::gps_step]
z_gps_plot = z_gps[::gps_step]

x_true_plot = x_true[::gps_step]
y_true_plot = y_true[::gps_step]
z_true_plot = z_true[::gps_step]

#EKF FOR GPS
dt_gps = gps_step * sim_dt


def fx(x, dt):
    """State transition"""
    F = np.array([
        [1, 0, 0, dt, 0,  0],
        [0, 1, 0, 0,  dt, 0],
        [0, 0, 1, 0,  0,  dt],
        [0, 0, 0, 1,  0,  0],
        [0, 0, 0, 0,  1,  0],
        [0, 0, 0, 0,  0,  1],
    ])
    return F @ x


def F_jacobian(x, dt):
    return np.array([
        [1, 0, 0, dt, 0,  0],
        [0, 1, 0, 0,  dt, 0],
        [0, 0, 1, 0,  0,  dt],
        [0, 0, 0, 1,  0,  0],
        [0, 0, 0, 0,  1,  0],
        [0, 0, 0, 0,  0,  1],
    ])


def hx(x):
    """GPS measures position only"""
    return x[:3]


def H_jacobian(x):
    H = np.zeros((3, 6))
    H[0, 0] = 1
    H[1, 1] = 1
    H[2, 2] = 1
    return H

ekf = ExtendedKalmanFilter(dim_x=6, dim_z=3)

ekf.x = np.array([
    x_gps_plot[0],
    y_gps_plot[0],
    z_gps_plot[0],
    0.0, 0.0, 0.0
])

ekf.P = np.eye(6) * 100.0
ekf.R = np.eye(3) * GPS_STD**2 
ekf.Q = np.eye(6) * 2.0

x_ekf, y_ekf, z_ekf = [], [], []
print("weights min/max:", weights.min(), weights.max())

for i in range(len(x_gps_plot)):
    # 1. State prediction
    ekf.x = fx(ekf.x, dt_gps)

    # 2. Jacobian update
    ekf.F = F_jacobian(ekf.x, dt_gps)
    
    # 3. Covariance prediction
    ekf.predict()
    ekf.R = np.eye(3) * (GPS_STD**2) / max(weights[i], 0.55)

    z_meas = np.array([
        x_gps_plot[i],
        y_gps_plot[i],
        z_gps_plot[i]
    ])

    ekf.update(
    z=z_meas,
    HJacobian=H_jacobian,
    Hx=hx)


    x_ekf.append(ekf.x[0])
    y_ekf.append(ekf.x[1])
    z_ekf.append(ekf.x[2])

# =========================================================
# 7. PLOT EVERYTHING
# =========================================================

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

# True trajectory
ax.plot(
    x_true, y_true, z_true,
    color="blue", linewidth=2, label="True trajectory"
)

# Noisy GPS
#ax.plot(
#    x_gps_plot, y_gps_plot, z_gps_plot,
#    color="red", linewidth=2, label="Noisy GPS")

# EKF estimate
ax.plot(
    x_ekf, y_ekf, z_ekf,
    color="green", linewidth=2, label="EKF estimate"
)

#plot error:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
true_pos = np.column_stack((x_true_plot, y_true_plot, z_true_plot))
ekf_pos  = np.column_stack((x_ekf, y_ekf, z_ekf))
pos_error = np.linalg.norm(true_pos - ekf_pos, axis=1)
rmse = np.sqrt(np.mean(pos_error**2))
print("Position RMSE (m):", rmse)


ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.legend()
ax.set_title("True vs Noisy GPS vs EKF Trajectory")

plt.show()
