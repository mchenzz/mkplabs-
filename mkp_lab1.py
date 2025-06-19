import numpy as np
from typing import Tuple, Optional
from datetime import datetime, timedelta

# Константы
mu = 398600.4418  # гравитационный параметр Земли, км³/с²
omega0 = 7.292115085e-5  # угловая скорость вращения Земли, рад/с
Re = 6378.1365  # экваториальный радиус Земли, км
F = 1 / 298.2525784  # сжатие Земли
sigma = 1.0  # баллистический коэффициент, м²/кг

atm_table = [
    (150, 180, 2.070e-9, 22.523),
    (180, 200, 5.464e-10, 29.740),
    (200, 250, 2.789e-10, 37.105),
    (250, 300, 7.248e-11, 45.546),
    (300, 350, 2.418e-11, 53.628),
    (350, 400, 9.518e-12, 53.298),
    (400, 450, 3.725e-12, 58.515),
    (450, 500, 1.585e-12, 60.828),
    (500, 600, 6.967e-13, 63.822),
    (600, 700, 1.454e-13, 71.835),
    (700, 800, 3.614e-14, 88.667),
    (800, 900, 1.170e-14, 124.64),
    (900, 1000, 5.245e-15, 181.05),
    (1000, 1500, 3.019e-15, 268.00)
]


def eci_to_ecef(r_eci: np.ndarray, t: float, t0: float = 0.0) -> np.ndarray:
    theta = omega0 * (t - t0)  # Угол поворота с учётом t0
    rotation_matrix = np.array([
        [np.cos(theta), np.sin(theta), 0],
        [-np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    return rotation_matrix @ r_eci


def state_to_kepler(x: np.ndarray):
    r = x[:3]
    r_abs = np.linalg.norm(r)
    v = x[3:]
    v_abs = np.linalg.norm(v)
    c = np.cross(r, v)
    c_abs = np.linalg.norm(c)
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

    return np.array([a, e, i, Omega, omega, M])


def solve_kepler(e: float, M: float, tol: float = 1e-8):
    E = M
    E_prev = 0
    first = True
    i = 0
    while first or abs(E_prev - E) > tol:
        E_prev = E
        E = E - (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
        i += 1
        if i >= 10:
            break
        if first:
            first = False
    return E


def elements2state(elem: np.ndarray):
    a, e, i, Omega, omega, M = elem
    E = solve_kepler(e, M)
    nu = 2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(E / 2))

    P = np.array([np.cos(omega) * np.cos(Omega) - np.sin(omega) * np.cos(i) * np.sin(Omega),
                  np.cos(omega) * np.sin(Omega) + np.sin(omega) * np.cos(i) * np.cos(Omega),
                  np.sin(omega) * np.sin(i)])
    Q = np.array([-np.sin(omega) * np.cos(Omega) - np.cos(omega) * np.cos(i) * np.sin(Omega),
                  -np.sin(omega) * np.sin(Omega) + np.cos(omega) * np.cos(i) * np.cos(Omega),
                  np.cos(omega) * np.sin(i)])

    p = a * (1 - e ** 2)
    r = p / (1 + e * np.cos(nu))
    x = r * np.cos(nu)
    y = r * np.sin(nu)
    v_coeff = np.sqrt(mu / p)

    x_dot = -v_coeff * np.sin(nu)
    y_dot = v_coeff * (e + np.cos(nu))

    state = np.zeros(6)
    state[:3] = P * x + Q * y
    state[3:] = x_dot * P + y_dot * Q
    return state


def calc_rho_atm(h: float) -> float:
    if h > 1500:
        return 0.0

    for row in atm_table:
        h0, hmax, rho0, H = row
        if h0 <= h < hmax:
            return rho0 * np.exp(-(h - h0) / H)


    return 0.0


def xy22llh(r: np.ndarray, Re: float = Re, F: float = F, tol: float = 5e-8) -> Tuple[float, float, float]:
    x, y, z = r
    e = np.sqrt(2 * F - F * F)
    dz = 0.0
    phi0 = -10
    i = 0

    xy2 = x * x + y * y

    while i < 10:
        stu_phi = (z + dz) / np.sqrt(xy2 + (z + dz) * (z + dz))
        N = Re / np.sqrt(1 - e * e * stu_phi * stu_phi)
        dz = N * e * e * stu_phi

        phi = np.arctan(stu_phi)
        if abs(phi - phi0) < tol:
            break

        phi0 = phi
        i += 1
    phi = np.arctan((z + dz) / np.sqrt(xy2))
    lam = np.arctan2(y, x)
    h = np.sqrt(xy2 + (z + dz) * (z + dz)) - N
    return phi, lam, h


def rhs_fun(t: float, alpha: np.ndarray) -> np.ndarray:
    a, e, i, Omega, omega, M = alpha

    p = a * (1 - e * e)
    n = np.sqrt(mu / (a * a * a))
    E = solve_kepler(e, M)
    nu = 2 * np.arctan(np.sqrt((1. + e) / (1. - e)) * np.tan(E / 2.))
    u = nu + omega
    r = p / (1. + e * np.cos(nu))

    state = elements2state(alpha)
    r_vec = state[:3]

    phi, lam, h = xy22llh(r_vec)
    rho = calc_rho_atm(h)

    v_r = np.sqrt(mu / p) * e * np.sin(nu)
    v_nu = np.sqrt(mu / p) * (1 + e * np.cos(nu))

    v_rel = np.array([
        v_r,
        v_nu - omega0 * r * np.cos(i),
        omega0 * r * np.sin(i) * np.cos(u)
    ])

    v_rel_mps = v_rel * 1000
    v_rel_norm_mps = np.linalg.norm(v_rel_mps)
    a_dist_mps2 = - sigma * rho * v_rel_norm_mps * v_rel_mps
    a_dist = a_dist_mps2 / 1000

    S = a_dist[0]
    T = a_dist[1]
    W = a_dist[2]

    da_dt = 2 / (n * np.sqrt(1 - e * e)) * (e * S * np.sin(nu) + T * (1 + e * np.cos(nu)))
    de_dt = np.sqrt(p / mu) * (S * np.sin(nu) + T * (np.cos(nu) + np.cos(E)))
    di_dt = r * W * np.cos(u) / np.sqrt(mu * p)
    dOmega_dt = r * W * np.sin(u) / (np.sqrt(mu * p) * np.sin(i))
    domega_dt = (1 / e) * np.sqrt(p / mu) * (-S * np.cos(nu) + T * np.sin(nu) * (1 + 1 / (1 + e * np.cos(nu)))) - dOmega_dt * np.cos(i)
    dM_dt = n - (2 * S * r) / (np.sqrt(mu * a)) + (np.sqrt(1 - e * e) / (e)) * np.sqrt(p / mu) * (S * np.cos(nu) - T * np.sin(nu) * (1 + 1 / (1 + e * np.cos(nu))))

    return np.array([da_dt, de_dt, di_dt, dOmega_dt, domega_dt, dM_dt])


def rk4(func, t0: float, x0: np.ndarray, h: float) -> np.ndarray:
    k1 = func(t0, x0)
    k2 = func(t0 + h * 0.5, x0 + k1 * h * 0.5)
    k3 = func(t0 + h * 0.5, x0 + k2 * h * 0.5)
    k4 = func(t0 + h, x0 + k3 * h)

    return x0 + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def predict_fall(r0: np.ndarray, v0: np.ndarray, t0_simulation: float = 0.0, t0_ecef: float = 0.0,
                 max_time: float = 7776000.0, h_threshold: float = 150.0) -> dict:
    x0 = np.concatenate([r0, v0])
    alpha = state_to_kepler(x0)
    t = t0_simulation
    trajectory = []
    result = {
        'fall_detected': False,
        'fall_time': None,
        'lat': None,
        'lon': None,
        'height': None,
        'trajectory': []
    }
    state0 = elements2state(alpha)
    r_ecef0 = eci_to_ecef(state0[:3], t, t0_ecef)
    _, _, h0 = xy22llh(r_ecef0)

    tau = 60.0


    while True:
        alpha = rk4(rhs_fun, t, alpha, tau)
        t += tau

        state = elements2state(alpha)
        r_vec = state[:3]
        r_ecef = eci_to_ecef(r_vec, t, t0_ecef)
        phi, lam, h = xy22llh(r_ecef)
        lam_deg = np.degrees(lam)


        if h < h_threshold:
            result['fall_detected'] = True
            result['fall_time'] = t
            result['lat'] = np.degrees(phi)
            result['lon'] = lam_deg
            result['height'] = h
            break

        if t - t0_simulation > max_time:
            result['fall_detected'] = False
            result['fall_time'] = t
            result['lat'] = np.degrees(phi)
            result['lon'] = lam_deg
            result['height'] = h
            break

    result['trajectory'] = np.array(trajectory)
    return result

if __name__ == "__main__":
    r0 = np.array([6873.987, 3036.358, -16.893])
    v0 = np.array([-1.836760, 2.715854, 6.435654])

    # Временные параметры
    initial_time = datetime(2025, 2, 2, 22, 45, 15, 438000)
    t_earth = datetime(2025, 2, 3, 3, 7, 4)
    delta_t_earth = (t_earth - initial_time).total_seconds()

    result = predict_fall(r0, v0, t0_simulation=0.0, t0_ecef=delta_t_earth)
    fall_time_utc = initial_time + timedelta(seconds=result['fall_time'])

    # Вывод результатов
    print("Результаты прогноза падения:")
    print(f"Обнаружено падение: {'Да' if result['fall_detected'] else 'Нет'}")
    print(f"Время падения (UTC): {fall_time_utc.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
    print(f"Широта: {result['lat']:.3f} град, долгота: {result['lon']:.3f} град")
    print(f"Высота точки падения: {result['height']:.3f} км")
