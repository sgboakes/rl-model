import numpy as np
import matplotlib.pyplot as plt
from pysatellite import Transformations
import pysatellite.config as cfg


def generate_measurements():

    rad_arr = 7e6 * np.ones((num_sats, 1), dtype='float64')
    omega_arr = 1 / np.sqrt(rad_arr ** 3 / mu)
    theta_arr = np.array((2 * pi * np.random.rand(num_sats, 1)), dtype='float64')
    k_arr = np.ones((num_sats, 3), dtype='float64')
    k_arr[:, :] = 1 / np.sqrt(3)

    # Make data structures
    sat_eci = {chr(i + 97): np.zeros((3, simLength)) for i in range(num_sats)}
    sat_aer = {chr(i + 97): np.zeros((3, simLength)) for i in range(num_sats)}
    not_vis_count = 0
    for i in range(num_sats):
        c = chr(i + 97)
        for j in range(simLength):
            v = np.array([[rad_arr[i] * sin(omega_arr[i] * (j + 1) * stepLength)],
                          [0],
                          [rad_arr[i] * cos(omega_arr[i] * (j + 1) * stepLength)]], dtype='float64')

            sat_eci[c][:, j] = (v @ cos(theta_arr[i])) + (np.cross(k_arr[i, :].T, v.T) * sin(theta_arr[i])) + (
                               k_arr[i, :].T * np.dot(k_arr[i, :].T, v) * (1 - cos(theta_arr[i])))

            sat_aer[c][:, j:j + 1] = Transformations.eci_to_aer(sat_eci[c][:, j], stepLength, j + 1, sensECEF,
                                                                sensLLA[0], sensLLA[1])

            if not trans_earth:
                if sat_aer[c][1, j] < 0:
                    sat_aer[c][:, j:j + 1] = np.array([[np.nan], [np.nan], [np.nan]])

        if np.isnan(sat_aer[c]).all():
            not_vis_count +=1

    print("There are {i} non visible satellites of {s} satellites".format(i=not_vis_count, s=num_sats))

    # Add small deviations for measurements
    # Using calculated max measurement deviations for LT:
    # Based on 0.15"/pixel, sat size = 2m, max range = 1.38e7
    # sigma = 1/2 * 0.15" for it to be definitely on that pixel
    # Add angle devs to Az/Elev, and range devs to Range

    ang_meas_dev, range_meas_dev = 1e-6, 20

    sat_aer_mes = {chr(i + 97): np.zeros((3, simLength)) for i in range(num_sats)}
    for i in range(num_sats):
        c = chr(i + 97)
        sat_aer_mes[c][0, :] = sat_aer[c][0, :] + (ang_meas_dev * np.random.randn(1, simLength))
        sat_aer_mes[c][1, :] = sat_aer[c][1, :] + (ang_meas_dev * np.random.randn(1, simLength))
        sat_aer_mes[c][2, :] = sat_aer[c][2, :] + (range_meas_dev * np.random.randn(1, simLength))

    sat_eci_mes = {chr(i + 97): np.zeros((3, simLength)) for i in range(num_sats)}
    for i in range(num_sats):
        c = chr(i + 97)
        for j in range(simLength):
            sat_eci_mes[c][:, j:j + 1] = Transformations.aer_to_eci(sat_aer_mes[c][:, j], stepLength, j + 1, sensECEF,
                                                                    sensLLA[0], sensLLA[1])

    return sat_aer, sat_eci, sat_aer_mes, sat_eci_mes


def generate_orbits():
    sat_eci = {chr(i + 97): np.zeros((3, simLength)) for i in range(num_sats)}
    sat_aer = {chr(i + 97): np.zeros((3, simLength)) for i in range(num_sats)}
    sat_aer_mes = {chr(i + 97): np.zeros((3, simLength)) for i in range(num_sats)}
    sat_eci_mes = {chr(i + 97): np.zeros((3, simLength)) for i in range(num_sats)}

    return sat_aer, sat_eci, sat_aer_mes, sat_eci_mes


if __name__ == "__main__":

    plt.close('all')
    np.random.seed(1)
    # ~~~~ Variables

    sin = np.sin
    cos = np.cos
    pi = np.float64(np.pi)

    sensLat = np.float64(28.300697)
    sensLon = np.float64(-16.509675)
    sensAlt = np.float64(2390)
    sensLLA = np.array([[sensLat * pi / 180], [sensLon * pi / 180], [sensAlt]], dtype='float64')
    # sensLLA = np.array([[pi/2], [0], [1000]], dtype='float64')
    sensECEF = Transformations.lla_to_ecef(sensLLA)
    sensECEF.shape = (3, 1)

    # simLength = cfg.simLength
    simLength = 20
    stepLength = cfg.stepLength

    mu = cfg.mu

    trans_earth = False

    # ~~~~ Satellite Conversion

    # Define sat pos in ECI and convert to AER
    # radArr: radii for each sat metres
    # omegaArr: orbital rate for each sat rad/s
    # thetaArr: inclination angle for each sat rad
    # kArr: normal vector for each sat metres

    num_sats = 100
    # global satAER, satECI, satAERMes, satECIMes
    satAER, satECI, satAERMes, satECIMes = generate_measurements()

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.set(theta_zero_location='N', theta_direction=-1)
    ax.set_rlim(90, 0)
    for i in range(num_sats):
        c = chr(i+97)
        ax.plot(satAER[c][0, :], np.rad2deg(satAERMes[c][1, :]))
        ax.plot(satAER[c][0, 0], np.rad2deg(satAERMes[c][1, 0]), 'gx')
        ax.plot(satAER[c][0, -1], np.rad2deg(satAERMes[c][1, -1]), 'rx')

    plt.show()
