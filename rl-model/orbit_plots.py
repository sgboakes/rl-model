import numpy as np
import matplotlib.pyplot as plt
from pysatellite import Filters, Transformations, Functions
import pysatellite.config as cfg


def generate_measurements(num_sats, simLength):

    radArr = 7e6 * np.ones((num_sats, 1), dtype='float64')
    omegaArr = 1 / np.sqrt(radArr ** 3 / mu)
    thetaArr = np.array((2 * pi * np.random.rand(num_sats, 1)), dtype='float64')
    kArr = np.ones((num_sats, 3), dtype='float64')
    kArr[:, :] = 1 / np.sqrt(3)

    # Make data structures
    satECI = {chr(i + 97): np.zeros((3, simLength)) for i in range(num_sats)}
    satAER = {chr(i + 97): np.zeros((3, simLength)) for i in range(num_sats)}

    for i in range(num_sats):
        c = chr(i + 97)
        for j in range(simLength):
            v = np.array([[radArr[i] * sin(omegaArr[i] * (j + 1) * stepLength)],
                          [0],
                          [radArr[i] * cos(omegaArr[i] * (j + 1) * stepLength)]], dtype='float64')

            satECI[c][:, j] = (v @ cos(thetaArr[i])) + (np.cross(kArr[i, :].T, v.T) * sin(thetaArr[i])) + (
                    kArr[i, :].T * np.dot(kArr[i, :].T, v) * (1 - cos(thetaArr[i])))

            satAER[c][:, j:j + 1] = Transformations.ECItoAER(satECI[c][:, j], stepLength, j + 1, sensECEF, sensLLA[0],
                                                             sensLLA[1])

            if not trans_earth:
                if satAER[c][1, j] < 0:
                    satAER[c][:, j:j + 1] = np.array([[np.nan], [np.nan], [np.nan]])

        if np.isnan(satAER[c]).all():
            print('Satellite {s} is not observable'.format(s=c))

    # Add small deviations for measurements
    # Using calculated max measurement deviations for LT:
    # Based on 0.15"/pixel, sat size = 2m, max range = 1.38e7
    # sigma = 1/2 * 0.15" for it to be definitely on that pixel
    # Add angle devs to Az/Elev, and range devs to Range

    angMeasDev, rangeMeasDev = 1e-6, 20

    satAERMes = {chr(i + 97): np.zeros((3, simLength)) for i in range(num_sats)}
    for i in range(num_sats):
        c = chr(i + 97)
        satAERMes[c][0, :] = satAER[c][0, :] + (angMeasDev * np.random.randn(1, simLength))
        satAERMes[c][1, :] = satAER[c][1, :] + (angMeasDev * np.random.randn(1, simLength))
        satAERMes[c][2, :] = satAER[c][2, :] + (rangeMeasDev * np.random.randn(1, simLength))

    satECIMes = {chr(i + 97): np.zeros((3, simLength)) for i in range(num_sats)}
    for i in range(num_sats):
        c = chr(i + 97)
        for j in range(simLength):
            satECIMes[c][:, j:j + 1] = Transformations.AERtoECI(satAERMes[c][:, j], stepLength, j + 1, sensECEF,
                                                                sensLLA[0], sensLLA[1])

    return satAER, satECI, satAERMes, satECIMes


if __name__ == "__main__":

    plt.close('all')
    np.random.seed(2)
    # ~~~~ Variables

    sin = np.sin
    cos = np.cos
    pi = np.float64(np.pi)

    sensLat = np.float64(28.300697)
    sensLon = np.float64(-16.509675)
    sensAlt = np.float64(2390)
    sensLLA = np.array([[sensLat * pi / 180], [sensLon * pi / 180], [sensAlt]], dtype='float64')
    # sensLLA = np.array([[pi/2], [0], [1000]], dtype='float64')
    sensECEF = Transformations.LLAtoECEF(sensLLA)
    sensECEF.shape = (3, 1)

    # simLength = cfg.simLength
    simLength = 20
    stepLength = cfg.stepLength

    mu = cfg.mu

    trans_earth = True

    # ~~~~ Satellite Conversion

    # Define sat pos in ECI and convert to AER
    # radArr: radii for each sat metres
    # omegaArr: orbital rate for each sat rad/s
    # thetaArr: inclination angle for each sat rad
    # kArr: normal vector for each sat metres

    num_sats = 25
    # global satAER, satECI, satAERMes, satECIMes
    satAER, satECI, satAERMes, satECIMes = generate_measurements(num_sats, simLength)

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.set(theta_zero_location='N', theta_direction=-1)
    ax.set_rlim(90, 0)
    for i in range(num_sats):
        c = chr(i+97)
        ax.plot(satAER[c][0, :], np.rad2deg(satAERMes[c][1, :]))

    plt.show()
