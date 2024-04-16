# -*- coding: utf-8 -*-

import numpy as np


def mse(reconstruction: np.ndarray, sample: np.ndarray):
    return (reconstruction - sample) ** 2 / np.sum(sample ** 2)


def dmse(reconstruction: np.ndarray, sample: np.ndarray):
    return 2 * (reconstruction - sample) / np.sum(sample ** 2)


def corr(convolved: np.ndarray, sample: np.ndarray) -> float:
    """ Función de fitness como coeficiente de correlación entre ambas funciones """
    var_sample = np.sum(sample ** 2)
    var_convolved = np.sum(convolved ** 2)
    cross = sample * convolved
    return 1 - cross / np.sqrt(var_sample * var_convolved)


def dcorr(convolved: np.ndarray, sample: np.ndarray) -> float:
    var_sample = np.sum(sample ** 2)
    var_convolved = np.sum(convolved ** 2)
    var_cross = np.sum(sample * convolved)
    return (- sample + convolved * var_cross / var_convolved) / np.sqrt(var_sample * var_convolved)
