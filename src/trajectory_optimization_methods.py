import numpy as np
import matplotlib.pyplot as plt
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

def rv_to_all_kepler_elements(r_vec, v_vec, mu):
    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)
    h_vec = np.cross(r_vec, v_vec)
    h = np.linalg.norm(h_vec)
    
    # Inclination
    i = np.arccos(np.clip(h_vec[2] / h, -1, 1))
    
    # Right Ascension of the Ascending Node (RAAN / Omega)
    n_vec = np.cross([0, 0, 1], h_vec)
    n = np.linalg.norm(n_vec)
    if n != 0:
        Omega = np.arccos(np.clip(n_vec[0] / n, -1, 1))
        if n_vec[1] < 0: Omega = 2 * np.pi - Omega
    else:
        Omega = 0
        
    # Eccentricity
    e_vec = ((v**2 - mu/r) * r_vec - np.dot(r_vec, v_vec) * v_vec) / mu
    e = np.linalg.norm(e_vec)
    
    # Argument of Periapsis (omega)
    if n != 0 and e > 1e-10:
        omega = np.arccos(np.clip(np.dot(n_vec, e_vec) / (n * e), -1, 1))
        if e_vec[2] < 0: omega = 2 * np.pi - omega
    else:
        omega = 0
        
    # Semi-major axis (a)
    energy = (v**2 / 2) - (mu / r)
    a = -mu / (2 * energy)
    
    # Mean Anomaly (M0) - fixed with arctan2
    cos_nu = np.dot(e_vec, r_vec) / (e * r)
    nu = np.arccos(np.clip(cos_nu, -1, 1))
    if np.dot(r_vec, v_vec) < 0: nu = 2 * np.pi - nu
        
    cos_E = (e + np.cos(nu)) / (1 + e * np.cos(nu))
    sin_E = (np.sqrt(1 - e**2) * np.sin(nu)) / (1 + e * np.cos(nu))
    E = np.arctan2(sin_E, cos_E) # Handles the full 360-degree range
    
    M0 = np.mod(E - e * np.sin(E), 2 * np.pi)
    
    return {"a": a, "e": e, "i": i, "Omega": Omega, "omega": omega, "M0": M0}

def solve_kepler_w_newton_backtrack_bisection(M, e, tol=1e-12, max_iter=50):
    M = np.mod(M, 2*np.pi)
    
    if e < 0.8:
        E = M
    else:
        E = np.pi
    
    for i in range(1, max_iter + 1):
        f = E - e*np.sin(E) - M
        fp = 1 - e*np.cos(E)
        
        if abs(f) < tol:
            return E, i
        
        step = -f / fp
        E_new = E + step
        
        lmbda = 1.0
        while abs(E_new - E) > np.pi:
            lmbda *= 0.5
            E_new = E + lmbda*step
            if lmbda < 1e-6:
                break
        
        if abs(E_new - E) > np.pi:
            E_new = 0.5 * (E + M)
        
        E = E_new
    
    return E, max_iter

def get_acc_3body_batch(r_batch):
    
    # r_batch: (batch_size, 3, 3), coordinates for [Sun, Earth, Moon]

    batch_size = r_batch.shape[0]
    acc = np.zeros_like(r_batch)
    
    # Extract positions
    r_s = r_batch[:, 0, :]
    r_e = r_batch[:, 1, :]
    r_m = r_batch[:, 2, :]
    
    # Relative vectors
    r_se = r_e - r_s
    r_sm = r_m - r_s
    r_em = r_m - r_e
    
    # Cubed distances (vectorized)
    d_se3 = np.linalg.norm(r_se, axis=1, keepdims=True)**3
    d_sm3 = np.linalg.norm(r_sm, axis=1, keepdims=True)**3
    d_em3 = np.linalg.norm(r_em, axis=1, keepdims=True)**3
    
    # Sun's acceleration (Pull from Earth and Moon)
    acc[:, 0, :] = G * M_EARTH * r_se / d_se3 + G * M_MOON * r_sm / d_sm3
    
    # Earth's acceleration (Pull from Sun and Moon)
    acc[:, 1, :] = -G * M_SUN * r_se / d_se3 + G * M_MOON * r_em / d_em3
    
    # Moon's acceleration (Pull from Earth and Sun)
    acc[:, 2, :] = -G * M_EARTH * r_em / d_em3 - G * M_SUN * r_sm / d_sm3
    
    return acc

def velocity_verlet_3body_batch(v_moon_batch, r_moon_rel0, r_earth0, v_earth0, t_arr, save_history=False):
    dt = t_arr[1] - t_arr[0]
    n_steps = len(t_arr)
    v_moon_batch = np.atleast_2d(v_moon_batch)
    n_batch = v_moon_batch.shape[0]
    
    # Constructing initial state (batch_size, 3 bodies, 3 coordinates)
    r_state = np.zeros((n_batch, 3, 3))
    v_state = np.zeros((n_batch, 3, 3))
    
    for b in range(n_batch):
        r_state[b, 0, :] = [0, 0, 0]               # Sun
        r_state[b, 1, :] = r_earth0                # Earth
        r_state[b, 2, :] = r_earth0 + r_moon_rel0  # Moon (Earth + Relative Moon)
        
        v_state[b, 0, :] = [0, 0, 0]                  # Sun
        v_state[b, 1, :] = v_earth0                   # Earth
        v_state[b, 2, :] = v_earth0 + v_moon_batch[b] # Individual velocity for each batch item
    
    if save_history:
        history_r = np.zeros((n_steps, n_batch, 3, 3))
        history_v = np.zeros((n_steps, n_batch, 3, 3))
        history_r[0] = r_state
        history_v[0] = v_state

    # Calculate initial acceleration
    a_current = get_acc_3body_batch(r_state)
    
    for i in range(n_steps - 1):
        v_half = v_state + 0.5 * a_current * dt
    
        r_state = r_state + v_half * dt

        a_next = get_acc_3body_batch(r_state)
        
        v_state = v_half + 0.5 * a_next * dt
        
        if save_history:
            history_r[i+1] = r_state
            history_v[i+1] = v_state
            
        a_current = a_next
        
    if save_history:
        return history_r, history_v
    else:
        # Return only the final state
        return r_state, v_state
    
def get_kepler_state(a, e, M_cent, t, M0=0):
    T = 2 * np.pi * np.sqrt(a**3 / G*M_cent)
    M = M0 + 2 * np.pi * (t / T)
    E, _ = solve_kepler_w_newton_backtrack_bisection(M, e)
    
    r_x = a * (np.cos(E) - e) # equals a*cos(E) - a*e
    r_y = a * np.sqrt(1 - e**2) * np.sin(E)
    
    n = np.sqrt(G*M_cent / a**3)
    v_x = -a * n * np.sin(E) / (1 - e * np.cos(E))
    v_y = a * n * np.sqrt(1 - e**2) * np.cos(E) / (1 - e * np.cos(E))
    
    r = np.array([r_x, r_y, 0.0])
    v = np.array([v_x, v_y, 0.0])
    
    return r, v


def calculate_loss_batch_3body(v_moon_batch, r0_rel, r_earth0, v_earth0, t_loss):
    # history_r shape: (time, batch, 3_bodies, 3_coordinates)
    history_r, history_v = velocity_verlet_3body_batch(
        v_moon_batch, r0_rel, r_earth0, v_earth0, t_loss, save_history=True
    )
    
    # Extracting relative motion (Moon - Earth)
    # index 2: Moon, index 1: Earth
    rel_r = history_r[:, :, 2, :] - history_r[:, :, 1, :]
    rel_v = history_v[:, :, 2, :] - history_v[:, :, 1, :]
    
    skip = len(t_loss) // 2
    
    # Orbital closure error
    diffs = rel_r[skip:] - r0_rel
    dists_sq = np.sum(diffs**2, axis=2)
    dist_closure = np.sqrt(np.min(dists_sq, axis=0))

    # Energy error (Relative energy with respect to Earth)
    v_sq = np.sum(rel_v[-1]**2, axis=-1)
    r_mag = np.linalg.norm(rel_r[-1], axis=-1)
    
    # Specific energy of the Earth-Moon system 
    energy_final = 0.5 * v_sq - (G * M_EARTH) / r_mag
    
    v0_sq = np.sum(v_moon_batch**2, axis=-1)
    energy_initial = 0.5 * v0_sq - (G * M_EARTH) / np.linalg.norm(r0_rel)
    
    energy_error = np.abs(energy_final - energy_initial)

    return (dist_closure / 1000.0) + (energy_error / 10.0)

def df_batch(f_batch, r0_rel, t_loss, v, h=1e-4):
    H = np.eye(len(v)) * h
    
    # Send x+h and x-h points at once (6-row batch)
    v_batch = np.vstack([v + H, v - H])  # (6, 3) matrix
    
    # Single call to the batch function
    losses = f_batch(v_batch, r0_rel, t_loss)
    
    loss_plus = losses[:3]
    loss_minus = losses[3:]
    
    return (loss_plus - loss_minus) / (2 * h)

def gradient_nd(loss_f, v_start, r0_rel, t_loss, gamma=0.6, niter=400, tol=100):
    v = np.atleast_2d(v_start)
    
    min_loss = float(loss_f(v, r0_rel, t_loss)[0])
    best_v = v.copy()
    
    smooth_g = np.zeros(3)
    beta = 0.9 
    
    print(f"Starting search (Simulated Gradient)... Initial error: {min_loss/1000:,.2f} km")

    for i in range(niter):
        g = df_batch(loss_f, r0_rel, t_loss, v.flatten(), h=0.001)
        
        g_norm = np.linalg.norm(g)
        g_unit = g / g_norm if g_norm > 1e-9 else g
        
        smooth_g = beta * smooth_g + (1 - beta) * g_unit
        
        v_trial = v - gamma * smooth_g
        new_loss = float(loss_f(v_trial, r0_rel, t_loss)[0])

        if new_loss < min_loss:
            gamma *= 1.1
            v = v_trial.copy()
            min_loss = new_loss
            best_v = v_trial.copy()
        else:
            gamma *= 0.5 
            smooth_g *= 0.5
            
        if i % 10 == 0:
            print(f"Iter {i:3} | Error: {min_loss/1000:8.4f} km | Gamma: {gamma:.6f}")

        if min_loss < tol:
            print(f"\n--- Success! Error: {min_loss:.2f} m ---")
            break
            
    return best_v.flatten()