import numpy as np
import matplotlib.pyplot as plt
import sgp4.api
from math import sin, cos, sqrt, pi

mu = 398600.4415
Re = 6378.1365
J2 = 0.108262668355315E-02

def rv_to_orbital_elements(r, v):
    r = np.array(r)
    v = np.array(v)
    c = np.cross(r, v)
    c_abs = np.linalg.norm(c)
    r_abs = np.linalg.norm(r)
    v_abs = np.linalg.norm(v)
    fl = np.cross(v, c) - mu * r / r_abs
    fl_abs = np.linalg.norm(fl)
    e = fl_abs / mu
    p = c_abs ** 2 / mu
    a = p / (1 - e ** 2)
    e_z = np.array([0, 0, 1])
    i = np.arccos(np.dot(c, e_z) / c_abs)
    node = np.cross(e_z, c)
    node = node / np.linalg.norm(node)
    Omega = np.arctan2(node[1], node[0])
    omega = np.arccos(np.dot(node, fl) / fl_abs)
    if fl[2] < 0:
        omega = 2 * np.pi - omega
    nu = np.arccos(np.dot(fl, r) / (fl_abs * r_abs))
    if np.dot(r, v) < 0:
        nu = 2 * np.pi - nu

    E = 2 * np.arctan(np.sqrt((1 - e) / (1 + e)) * np.tan(nu / 2))
    M = E - e * np.sin(E)

    return a, e, i, Omega, omega, M


def orbital_elements_to_rv(a, e, i, Omega, omega, M):
    E = M
    E_prev = 0
    first = True
    i = 0
    while first or abs(E_prev - E) > 1e-8:
        E_prev = E
        E = E - (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
        i += 1
        if i >= 10:
            break
        if first:
            first = False

    nu = 2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(E / 2))

    P = np.array([
        cos(omega) * cos(Omega) - sin(omega) * cos(i) * sin(Omega),
        cos(omega) * sin(Omega) + sin(omega) * cos(i) * cos(Omega),
        sin(omega) * sin(i)
    ])

    Q = np.array([
        -sin(omega) * cos(Omega) - cos(omega) * cos(i) * sin(Omega),
        -sin(omega) * sin(Omega) + cos(omega) * cos(i) * cos(Omega),
        cos(omega) * sin(i)
    ])
    p = a * (1 - e ** 2)
    r = p / (1 + e * np.cos(nu))
    x = r * np.cos(nu)
    y = r * np.sin(nu)
    v_coeff = np.sqrt(mu / p)
    x_dot = -v_coeff * np.sin(nu)
    y_dot = v_coeff * (e + np.cos(nu))
    r_n = P * x + Q * y
    v_n = x_dot * P + y_dot * Q

    return r_n, v_n


def kepler_plus_model(r0, v0, n_dot_tle, n_ddot_tle, delta_t):
    r0 = np.array(r0)
    v0 = np.array(v0)

    a0, e0, i0, Omega0, omega0, M0 = rv_to_orbital_elements(r0, v0)
    n0 = sqrt(mu / a0 ** 3)
    p0 = a0 * (1 - e0 ** 2)

    n_dot = 2 * n_dot_tle * (2 * pi) / (86400 ** 2)  # [рад/с^2]
    n_ddot = 6 * n_ddot_tle * (2 * pi) / (86400 ** 3)  # [рад/с^3]

    a = a0 - (2 * a0 / (3 * n0)) * n_dot * delta_t
    e = e0 - (2 * (1 - e0) / (3 * n0)) * n_dot * delta_t
    Omega = Omega0 - (3 * n0 * Re ** 2 * J2 / (2 * p0 ** 2)) * cos(i0) * delta_t
    omega = omega0 + (3 * n0 * Re ** 2 * J2 / (4 * p0 ** 2)) * (4 - 5 * sin(i0) ** 2) * delta_t
    M = M0 + n0 * delta_t + (n_dot / 2) * delta_t ** 2 + (n_ddot / 6) * delta_t ** 3
    i = i0

    r, v = orbital_elements_to_rv(a, e, i, Omega, omega, M)
    return r, v


def main():
    r0_tle = np.array([6873.987, 3036.358, -16.893])
    v0_tle = np.array([-1.836760, 2.715854, 6.435654])

    n_dot_tle = .00007016
    n_ddot_tle = 0.0

    delta_t_total = 30 * 86400  # 30 суток в секундах
    delta_t_step = 10 * 60  # 10 минут в секундах
    t_points = np.arange(0, delta_t_total + delta_t_step, delta_t_step)

    results_kp = []
    for t in t_points:
        r_kp, v_kp = kepler_plus_model(r0_tle, v0_tle, n_dot_tle, n_ddot_tle, t)
        a_kp, e_kp, i_kp, Omega_kp, omega_kp, _ = rv_to_orbital_elements(r_kp, v_kp)
        results_kp.append((a_kp, e_kp, Omega_kp, omega_kp))

    results_kp = np.array(results_kp)

    t_days = t_points / 86400

    # Построение графиков
    plt.figure(figsize=(14, 10))

    # Большая полуось
    plt.subplot(2, 2, 1)
    plt.plot(t_days, results_kp[:, 0], label='Кеплер+', color='red')
    plt.xlabel('Время, сут')
    plt.ylabel('a, км')
    plt.title('Большая полуось')
    plt.grid()
    plt.legend()

    # Эксцентриситет
    plt.subplot(2, 2, 2)
    plt.plot(t_days, results_kp[:, 1], label='Кеплер+', color='red')
    plt.xlabel('Время, сут')
    plt.ylabel('e')
    plt.title('Эксцентриситет')
    plt.grid()
    plt.legend()

    # Долгота восходящего узла
    plt.subplot(2, 2, 3)
    plt.plot(t_days, (results_kp[:, 2]), label='Кеплер+', color='red')
    plt.xlabel('Время, сут')
    plt.ylabel('Ω, градусы')
    plt.title('Долгота восходящего узла')
    plt.grid()
    plt.legend()

    # Аргумент перицентра
    plt.subplot(2, 2, 4)
    plt.plot(t_days, (results_kp[:, 3]), label='Кеплер+', color='red')
    plt.xlabel('Время, сут')
    plt.ylabel('ω, градусы')
    plt.title('Аргумент перицентра')
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()