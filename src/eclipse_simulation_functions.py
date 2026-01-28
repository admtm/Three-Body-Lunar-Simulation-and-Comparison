import numpy as np
from matplotlib import pyplot as plt
from astropy import constants as const

G = const.G.value
M_SUN = const.M_sun.value
M_EARTH = const.M_earth.value
M_MOON = 7.34767309e22

R_SUN = const.R_sun.value
R_EARTH = const.R_earth.value
R_MOON = 1737400.0

AU = const.au.value

a_Earth = 1.000001018 * AU
e_Earth = 0.0167086

a_Moon = 384748000.0
e_Moon = 0.0549006

DAY = 24 * 3600
MONTH = 27.321661 * DAY
YEAR = 365.25636 * DAY

MASSES = np.array([M_SUN, M_EARTH, M_MOON])
T_FINAL = YEAR
DT = 15
t_array = np.arange(0, T_FINAL, DT)

def get_kepler_state(f, a, e, M_cent, t, M0=0):
    T = 2 * np.pi * np.sqrt(a**3 / (G * M_cent))
    M = M0 + 2 * np.pi * (t / T)
    E, _ = f(M, e)

    r_x = a * (np.cos(E) - e) # equals a*cos(E) - a*e
    r_y = a * np.sqrt(1 - e**2) * np.sin(E)
    
    n = np.sqrt(G*M_cent / a**3)
    v_x = -a * n * np.sin(E) / (1 - e * np.cos(E))
    v_y = a * n * np.sqrt(1 - e**2) * np.cos(E) / (1 - e * np.cos(E))
    
    r = np.array([r_x, r_y, 0.0])
    v = np.array([v_x, v_y, 0.0])
    
    return r, v

def kepler_history(f, a, e, M_central, t_array, M0=0):
    mu = G * M_central

    r_hist = np.zeros((len(t_array), 3))
    v_hist = np.zeros((len(t_array), 3))

    for i, t in enumerate(t_array):
        r, v = get_kepler_state(f, a, e, M_central, t, M0=M0)
        r_hist[i] = r
        v_hist[i] = v

    return r_hist, v_hist

def rotate_to_3d(flat_history, i, Omega, omega):
    R_omega = np.array([[np.cos(omega), -np.sin(omega), 0], [np.sin(omega), np.cos(omega), 0], [0, 0, 1]])
    R_i     = np.array([[1, 0, 0], [0, np.cos(i), -np.sin(i)], [0, np.sin(i), np.cos(i)]])
    R_Omega = np.array([[np.cos(Omega), -np.sin(Omega), 0], [np.sin(Omega), np.cos(Omega), 0], [0, 0, 1]])
    
    R_total = R_Omega @ R_i @ R_omega
    
    return np.einsum('ij,kj->ki', R_total, flat_history)

def reconstruct_history(rel_history, t_arr):
    T = len(t_arr)

    angles = 2 * np.pi * (t_arr / YEAR)
    r_earth = np.stack([
        a_Earth * np.cos(angles),
        a_Earth * np.sin(angles),
        np.zeros(T)
    ], axis=1)

    r_sun = np.zeros((T, 3))

    r_moon = r_earth + rel_history[:, 0, :]

    history = np.stack([r_sun, r_earth, r_moon], axis=1)

    return history

def get_eclipse_data_vectorized(history, t_array):
    r_sun   = history[:, 0, :]
    r_earth = history[:, 1, :]
    r_moon  = history[:, 2, :]

    s_e = r_earth - r_sun
    e_m = r_moon - r_earth

    d_se = np.linalg.norm(s_e, axis=1)
    d_em = np.linalg.norm(e_m, axis=1)

    axis_unit = s_e / d_se[:, None]

    projection = np.einsum('ij,ij->i', e_m, axis_unit)

    perp_vec = e_m - projection[:, None] * axis_unit
    d_perp = np.linalg.norm(perp_vec, axis=1)

    r_umbra = R_EARTH - (R_SUN - R_EARTH) * (d_em / d_se)

    mask = (projection > 0) & (d_perp + R_MOON < r_umbra)


    idx = np.where(mask)[0]

    eclipses = [
        {
            'time': t_array[i] / 3600,
            'dist': d_perp[i] / 1000
        }
        for i in idx
    ]

    return eclipses

def line_sphere_intersection(p1, p2, sphere_center, sphere_radius):
    d = p2 - p1
    f = p1 - sphere_center

    a = np.dot(d, d)
    b = 2 * np.dot(f, d)
    c = np.dot(f, f) - sphere_radius**2

    discriminant = b**2 - 4*a*c

    if discriminant < 0:
        return False
    else:
        discriminant_sqrt = np.sqrt(discriminant)
        t1 = (-b - discriminant_sqrt) / (2*a)
        t2 = (-b + discriminant_sqrt) / (2*a)

        if (0 <= t1 <= 1) or (0 <= t2 <= 1):
            return True, (t1, t2)
        else:
            return False

def get_double_mc_illumination(r_sun, r_earth, r_moon, num_samples=3000):
    hits = 0
    light_pts, shadow_pts, umbra_pts = [], [], []
    
    # Moon-Earth direction and perpendicular vectors for the disk representation
    v_ray = r_moon - r_earth
    v_dir = v_ray / np.linalg.norm(v_ray)
    
    # Auxiliary vectors for drawing/orienting the Moon's disk
    v1 = np.cross(np.array([0, 0, 1]), v_dir)
    if np.linalg.norm(v1) < 1e-6:
        v1 = np.cross(np.array([0, 1, 0]), v_dir)
    v1 /= np.linalg.norm(v1)
    v2 = np.cross(v_dir, v1)

    r_m_rel = r_moon - r_earth
    r_s_rel = r_sun - r_earth
    d_se = np.linalg.norm(r_s_rel)
    d_em = np.linalg.norm(r_m_rel)

    # Shadow cone direction and radius at the Moon's distance
    axis_unit = -r_s_rel / d_se
    proj_center = np.dot(r_m_rel, axis_unit)
    r_umbra = R_EARTH - (R_SUN - R_EARTH) * (d_em / d_se)

    for _ in range(num_samples):
        # Random point on the Sun's surface (Spherical)
        phi = np.random.uniform(0, 2*np.pi)
        costheta = np.random.uniform(-1, 1)
        sun_p = r_sun + R_SUN * np.array([
            np.sqrt(1 - costheta**2) * np.cos(phi),
            np.sqrt(1 - costheta**2) * np.sin(phi),
            costheta
        ])

        # Random point on the Moon's disk (Uniform area)
        r_val = R_MOON * np.sqrt(np.random.uniform(0, 1))
        theta = np.random.uniform(0, 2*np.pi)
        dy, dz = r_val * np.cos(theta), r_val * np.sin(theta)
        moon_p = r_moon + dy * v1 + dz * v2

        # Geometric check (Umbra cone)
        moon_rel_p = moon_p - r_earth
        proj_p = np.dot(moon_rel_p, axis_unit)
        d_perp_p = np.linalg.norm(moon_rel_p - proj_p * axis_unit)

        # DEBUG: If the point is within the geometric umbra cone
        if proj_center > 0 and d_perp_p < r_umbra:
            hits += 1
            umbra_pts.append([dy, dz])
            continue

        # Ray tracing (If outside the cone, check if Earth obstructs the ray)
        if line_sphere_intersection(sun_p, moon_p, r_earth, R_EARTH):
            hits += 1
            shadow_pts.append([dy, dz]) # (intersection-based shadow)
        else:
            light_pts.append([dy, dz])   # (actual light)

    illumination = 1.0 - (hits / num_samples)
    return illumination, np.array(light_pts), np.array(shadow_pts), np.array(umbra_pts)


def find_all_lunar_eclipses(history):
    history = np.asarray(history)

    r_s = history[:, 0, :]
    r_e = history[:, 1, :]
    r_m = history[:, 2, :]

    # Geocentric frame (Earth at origin)
    r_m_rel = r_m - r_e
    r_s_rel = r_s - r_e

    d_se = np.linalg.norm(r_s_rel, axis=1)
    d_em = np.linalg.norm(r_m_rel, axis=1)

    # Shadow axis (away from Sun)
    axis_unit = -r_s_rel / d_se[:, None]

    # Projection onto shadow axis
    proj_len = np.einsum("ij,ij->i", r_m_rel, axis_unit)
    projection = proj_len[:, None] * axis_unit
    d_perp = np.linalg.norm(r_m_rel - projection, axis=1)

    # Umbra radius at Moon distance
    r_umbra = R_EARTH - (R_SUN - R_EARTH) * (d_em / d_se)
    # Totality condition (FULL Moon inside umbra)
    in_total_eclipse = (proj_len > 0) & ((d_perp + R_MOON) <= r_umbra)

    eclipses = []
    inside = False

    for i in range(len(in_total_eclipse)):
        if in_total_eclipse[i] and not inside:
            start = i
            inside = True

        if inside and (not in_total_eclipse[i] or i == len(in_total_eclipse) - 1):
            end = i
            inside = False

            segment = slice(start, end)

            # Peak = deepest totality
            margin = r_umbra[segment] - (d_perp[segment] + R_MOON)
            peak = start + np.argmax(margin)

            eclipses.append({
                "start": start,
                "peak": peak,
                "end": end,
                "is_total": True
            })

    # Drop artificial eclipse at t=0
    if eclipses and eclipses[0]["start"] == 0:
        eclipses = eclipses[1:]

    return eclipses

