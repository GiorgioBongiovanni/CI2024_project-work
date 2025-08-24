# Copyright © 2024 Giovanni Squillero <giovanni.squillero@polito.it>
# https://github.com/squillero/computational-intelligence
# Free under certain conditions — see the license for details.

import numpy as np

# All numpy's mathematical functions can be used in formulas
# see: https://numpy.org/doc/stable/reference/routines.math.html


# Notez bien: No need to include f0 -- it's just an example!
def f0(x: np.ndarray) -> np.ndarray:
    return x[0] + np.sin(x[1]) / 5


def f1(x: np.ndarray) -> np.ndarray: 
    return np.sin(x[0])


def f2(x: np.ndarray) -> np.ndarray:
    return (
        (x[0] * 1.887173 + (x[2] + x[1])) *
        (1.0143482e6 - ((x[2] + x[1]) * (x[0] * 46669.336)))
    )


def f3(x: np.ndarray) -> np.ndarray:
    return (
        (np.square(x[0]) * 2.0) -
        ((np.square(x[1]) * x[1]) - (x[2] * -3.5))
    ) + 4.0


def f4(x: np.ndarray) -> np.ndarray:
    return (np.cos(x[1]) * 7.0) + ((x[0] * -0.09090909) + 3.2794166)


def f5(x: np.ndarray) -> np.ndarray: 
    return (np.square(np.square((x[0] * (np.square(x[1]) + x[0])) * 0.01635572)) - 0.15140401) * -7.9823986e-10


def f6(x: np.ndarray) -> np.ndarray:
    return x[1] + ((x[1] - x[0]) * 0.6945204)


def f7(x: np.ndarray) -> np.ndarray:
    return (
        np.exp(
            np.abs(x[1] + x[0]) - np.sqrt(np.abs((x[0] - x[1]) * 22.61408))
        ) * 18.085468
    ) + 2.655571


def f8(x: np.ndarray) -> np.ndarray:
    return (
        ((x[5] * 5.0000405) * np.square(np.square(x[5]))) -
        np.square((np.square(x[4]) * 2.011731) + np.sin(x[3]))
    ) - (x[3] * -43.06929)
