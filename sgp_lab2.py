import numpy as np
import matplotlib.pyplot as plt
from sgp4.api import Satrec, WGS84, jday

mu = 398600.4418

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

    return a, e, omega, Omega

def main():
    s = '1 16625U 86014F   25032.41232112  .00007016  00000-0  59476-3 0  9995'
    t = '2 16625  63.2991  28.3167 0835444 106.5059 262.9372 13.67372417274143'
    sat = Satrec.twoline2rv(s, t, WGS84)
    year = 2025
    month = 2
    day = 2
    hour = 22
    minute = 45
    second = 15.438

    jd_start, fr_start = jday(year, month, day, hour, minute, second)
    jd, dt = jday(year, month, day, hour, minute, second)
    e, r, v = sat.sgp4(jd, dt)


    time_steps = np.arange(0, 30 * 86400, 600)
    a_list, e_list, omega_list, Omega_list = [], [], [], []

    for t in time_steps:
        jd = jd_start
        fr = fr_start + t / 86400
        error, r, v = sat.sgp4(jd, fr)
        print(r, v)


        if error == 0:
            x = np.concatenate([r, v])
            a, e, omega, Omega = state_to_kepler(x)
            a_list.append(a)
            e_list.append(e)
            omega_list.append(omega)
            Omega_list.append(Omega)

    days = time_steps / 86400
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(days, a_list, color='purple')
    plt.xlabel('Время (дни)')
    plt.ylabel('a (км)')
    plt.title('Большая полуось')

    plt.subplot(2, 2, 2)
    plt.plot(days, e_list, color='purple')
    plt.xlabel('Время (дни)')
    plt.ylabel('e')
    plt.title('Эксцентриситет')

    plt.subplot(2, 2, 4)
    plt.plot(days, omega_list, color='purple')
    plt.xlabel('Время (дни)')
    plt.ylabel('ω (рад)')
    plt.title('Аргумент перицентра')

    plt.subplot(2, 2, 3)
    plt.plot(days, Omega_list, color='purple')
    plt.xlabel('Время (дни)')
    plt.ylabel('Ω (рад)')
    plt.title('Долгота восходящего узла')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()