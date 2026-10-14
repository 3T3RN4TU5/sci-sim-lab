#!/usr/bin/env python3
"""
Independent reference values from SciPy / NumPy for the TypeScript test suite.

Writes tests/fixtures/scipy.json, which tests/crosscheck.test.ts consumes (the test is
skipped when the fixture is absent). Re-generate with:

    pip install scipy numpy        # or: pip install --break-system-packages scipy
    npm run crosscheck
"""
import json
import math
import os

import numpy as np
import scipy
from scipy import signal, special
from scipy.integrate import solve_ivp

TIGHT = dict(rtol=1e-13, atol=1e-13)


def lorenz(t, y, s=10.0, r=28.0, b=8.0 / 3.0):
    return [s * (y[1] - y[0]), y[0] * (r - y[2]) - y[1], y[0] * y[1] - b * y[2]]


def vdp(mu):
    return lambda t, y: [y[1], mu * (1 - y[0] ** 2) * y[1] - y[0]]


def kepler(t, y):
    r3 = (y[0] ** 2 + y[1] ** 2) ** 1.5
    return [y[2], y[3], -y[0] / r3, -y[1] / r3]


def robertson(t, y):
    a, b, c = 0.04 * y[0], 1e4 * y[1] * y[2], 3e7 * y[1] ** 2
    return [-a + b, a - b - c, c]


def robertson_jac(t, y):
    return [[-0.04, 1e4 * y[2], 1e4 * y[1]], [0.04, -1e4 * y[2] - 6e7 * y[1], -1e4 * y[1]], [0, 6e7 * y[1], 0]]


def sir(beta, gamma, N):
    return lambda t, y: [-beta * y[0] * y[1] / N, beta * y[0] * y[1] / N - gamma * y[1], gamma * y[1]]


def final(sol):
    return [float(v) for v in sol.y[:, -1]]


def main():
    out = {"generator": f"scipy {scipy.__version__}, numpy {np.__version__}"}

    out["lorenz_t2"] = final(solve_ivp(lorenz, (0, 2), [1, 1, 1], method="DOP853", **TIGHT))
    out["vdp10_t20"] = final(solve_ivp(vdp(10), (0, 20), [2, 0], method="DOP853", **TIGHT))
    e = 0.9
    y0 = [1 - e, 0, 0, math.sqrt((1 + e) / (1 - e))]
    out["kepler09_t3"] = final(solve_ivp(kepler, (0, 3), y0, method="DOP853", **TIGHT))
    out["robertson_t40"] = final(
        solve_ivp(robertson, (0, 40), [1, 0, 0], method="Radau", jac=robertson_jac, rtol=1e-10, atol=1e-14)
    )

    # Step count of SciPy's own Dormand–Prince (RK45) for the same tolerance: the
    # TypeScript controller should land in the same ballpark.
    e = 0.6
    y0 = [1 - e, 0, 0, math.sqrt((1 + e) / (1 - e))]
    rk45 = solve_ivp(kepler, (0, 2 * math.pi), y0, method="RK45", rtol=1e-8, atol=1e-8)
    out["rk45_kepler06_steps"] = int(len(rk45.t) - 1)

    N = 1e6
    s = solve_ivp(sir(0.3, 0.1, N), (0, 100), [N - 10, 10, 0], method="DOP853", **TIGHT, t_eval=[100])
    out["sir_day100"] = final(s)

    n = np.arange(64)
    x = np.sin(0.3 * n) + 0.5 * np.cos(1.7 * n**1.1) + 0.01 * n
    X = np.fft.fft(x)
    out["fft64_input"] = x.tolist()
    out["fft64_re"] = X.real.tolist()
    out["fft64_im"] = X.imag.tolist()

    out["hann16"] = signal.windows.hann(16, sym=True).tolist()
    out["blackman16"] = signal.windows.blackman(16, sym=True).tolist()
    out["firwin_lowpass_101_100Hz_fs1000_blackman"] = signal.firwin(101, 100, fs=1000, window="blackman").tolist()

    temps = [1.0, 1.5, 2.0, 2.2, 2.5, 3.0, 4.0]
    energies = []
    for T in temps:
        b = 1 / T
        k = 2 * math.sinh(2 * b) / math.cosh(2 * b) ** 2
        K = special.ellipk(k * k)  # SciPy uses the parameter m = k²
        t = math.tanh(2 * b)
        energies.append(-(1 / math.tanh(2 * b)) * (1 + (2 / math.pi) * (2 * t * t - 1) * K))
    out["onsager_T"] = temps
    out["onsager_energy"] = energies

    path = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures", "scipy.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {os.path.normpath(path)} ({out['generator']})")


if __name__ == "__main__":
    main()
