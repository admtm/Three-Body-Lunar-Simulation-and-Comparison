from matplotlib import pyplot as plt
import numpy as np
from astropy import constants as const
from scipy.signal import find_peaks
from matplotlib.animation import FuncAnimation

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

def print_period_std_dev(history, t_array, label):
    r_rel = history[:,2,:] - history[:,1,:] 
    r_norm = np.linalg.norm(r_rel, axis=1)
    
    actual_dt = t_array[1] - t_array[0]
    
    peaks, _ = find_peaks(-r_norm, distance=20*DAY/actual_dt)
    
    if len(peaks) < 2:
        print(f"---------------------------------------------------------------------")
        print(f"{label}: Not enough data (found {len(peaks)} peaks). Check if t_array matches history!")
        return

    periods = np.diff(t_array[peaks]) / DAY

    print("---------------------------------------------------------------------")
    print(f"{label} average period and standard deviation statistics:")
    print("---------------------------------------------------------------------")
    print(f"Average period:  {periods.mean():,.2f} days")
    print(f"Standard deviation: {periods.std():,.4f} days")

def plot_moon_geocentric(history, label, color, alpha=1.0, linewidth=1.0):
    r_rel = history[:, 2, :] - history[:, 1, :]
    plt.plot(
        r_rel[:, 0] / 1000,
        r_rel[:, 1] / 1000,
        label=label,
        color=color,
        alpha=alpha,
        linewidth=linewidth
    )

def plot_distance_time(history, t_array, label, color):
    r_rel = history[:, 2, :] - history[:, 1, :]
    r = np.linalg.norm(r_rel, axis=1)
    plt.plot(t_array/DAY, r/1000, label=label, color=color)

    plt.axhline(r.min()/1000, color=color, linestyle="--", alpha=0.4)
    plt.axhline(r.max()/1000, color=color, linestyle="--", alpha=0.4)
    plt.axhline(r.mean()/1000, color=color, linestyle=":",  alpha=0.8)

    print("----------------------------------------")
    print(f"{label} distance statistics:")
    print("----------------------------------------")
    print(f"Min r: {r.min()/1000:,.2f} km")
    print(f"Max r: {r.max()/1000:,.2f} km")
    print(f"Mean r: {r.mean()/1000:,.2f} km")

def plot_lunar_eclipse(f, history, t_array, eclipse, color, label, num_samples=3000):
    fig, axes = plt.subplots(1, 7, figsize=(22, 6), facecolor="black")

    totality_start = int(eclipse["start"])
    totality_end = int(eclipse["end"])
    totality_duration = totality_end - totality_start
    if totality_duration == 0: totality_duration = 50 

    expansion = int(totality_duration * 2.5)
    
    idx_1 = max(0, totality_start - expansion)
    idx_2 = max(0, totality_start - int(totality_duration * 0.5))
    idx_3 = max(0, totality_start - int(totality_duration * 0.3)) 
    idx_4 = int(eclipse["peak"])
    idx_5 = min(len(t_array)-1, totality_end + int(totality_duration * 0.3))
    idx_6 = min(len(t_array)-1, totality_end + int(totality_duration * 0.5))
    idx_7 = min(len(t_array)-1, totality_end + expansion)

    indices = [idx_1, idx_2, idx_3, idx_4, idx_5, idx_6, idx_7]

    for ax, idx in zip(axes, indices):
        idx = int(idx)
        ax.set_facecolor("black")
        
        r_s, r_e, r_m = history[idx]
        illum, light, shadow, umbra = f(r_s, r_e, r_m, num_samples=num_samples)

        if len(umbra) > 0: ax.scatter(umbra[:, 0], umbra[:, 1], s=1.2, c="#120321", alpha=0.5)
        if len(shadow) > 0: ax.scatter(shadow[:, 0], shadow[:, 1], s=1, c="#444444", alpha=0.4)
        if len(light) > 0: ax.scatter(light[:, 0], light[:, 1], s=0.8, c=color, alpha=0.8)

        time_h = t_array[idx] / 3600

        ax.set_title(f"Time: {time_h:.2f} h\nMC Coverage: {illum*100:.1f}%", 
                     color="white", fontsize=10)
        ax.set_aspect('equal')
        ax.axis("off")

    main_title = f"{label.upper()}\n" \
             f"Lunar Eclipse Sequence | Peak: {t_array[int(eclipse['peak'])] / 3600:.2f} h"

    fig.suptitle(main_title, color="white", fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.show()

def plot_all_lunar_eclipses(f, history, t_array, eclipses, color, label, num_samples=3000, max_events=None):
    for i, e in enumerate(eclipses):
        if max_events is not None and i >= max_events:
            break
        plot_lunar_eclipse(f, history, t_array, e, color, label, num_samples=num_samples)

'''
3D animation functions
'''

def expand_eclipse_window(eclipse, n_total, pad_factor=3.0):
    # Expand the eclipse indices with padding for smoother animation context.
    start = int(eclipse["start"])
    end   = int(eclipse["end"])
    dur = max(1, end - start)
    pad = int(pad_factor * dur)
    a = max(0, start - pad)
    b = min(n_total - 1, end + pad)
    return a, b


def compute_earth_facecolors(earth_local, sun_vec_km):
    # Earth shading brighter on the sun-facing side,
    # using dot product of local normals and Sun direction.
    sun_unit = sun_vec_km / (np.linalg.norm(sun_vec_km) + 1e-12)
    # Dot product with surface normals (earth_local directions)
    normals = earth_local / (np.linalg.norm(earth_local, axis=-1, keepdims=True) + 1e-12)
    dots = np.sum(normals * sun_unit[None, None, :], axis=-1)
    # Clamp to [0,1]
    intensity = np.clip(dots, 0, 1)
    # Blue-ish color
    r = 0.2 + 0.3 * intensity
    g = 0.4 + 0.4 * intensity
    b = 0.8 + 0.2 * intensity
    rgba = np.stack([r, g, b, np.full_like(r, 0.8)], axis=-1)
    return rgba


def compute_facecolors(moon_center_km, sun_vec_km, moon_local, sphere_res, scale_factor=1.0):
    # Moon shading based on Sun direction + Earth's umbra shadow.
    # scale_factor: the factor by which moon_local is scaled for visualization
    
    sun_unit = sun_vec_km / (np.linalg.norm(sun_vec_km) + 1e-12)
    
    # Calculate surface normals from Moon center
    normals = moon_local / (np.linalg.norm(moon_local, axis=-1, keepdims=True) + 1e-12)
    
    # Dot product with Sun direction for basic illumination
    dots = np.sum(normals * sun_unit[None, None, :], axis=-1)
    intensity = np.clip(dots, 0, 1)
    
    # Check umbra shadow (Earth's shadow cone)
    # Use UNSCALED moon_center for shadow calculation (actual Moon position)
    d_es = np.linalg.norm(sun_vec_km) + 1e-12
    axis_unit = -sun_vec_km / d_es  # Direction away from Sun

    # If the Moon center enters the umbra/penumbra, dim the whole disk a bit
    proj_center = np.dot(moon_center_km, axis_unit)
    d_perp_center = np.linalg.norm(moon_center_km - proj_center * axis_unit)

    # Umbra/penumbra radii at Moon distance (unscaled)
    r_umbra_center = (R_EARTH / 1000.0) - (R_SUN / 1000.0 - R_EARTH / 1000.0) * (np.linalg.norm(moon_center_km) / d_es)
    r_penumbra_center = r_umbra_center + (R_MOON / 1000.0) * 2.0  # soft edge

    center_shadow_factor = 1.0
    if proj_center > 0:  # behind Earth, anti-sun side
        if d_perp_center < r_umbra_center + (R_MOON / 1000.0):
            center_shadow_factor = 0.25  # deep umbra on whole disk
        elif d_perp_center < r_penumbra_center:
            center_shadow_factor = 0.6   # penumbra dimming
    
    # Calculate shadow in original (unscaled) coordinates
    # moon_local is scaled by scale_factor, so unscale it for shadow calc
    moon_local_unscaled = moon_local / scale_factor  # Back to original size
    pts_unscaled = moon_center_km[None, None, :] + moon_local_unscaled  # Original coords
    
    # Distance from Earth (origin)
    d_pts = np.linalg.norm(pts_unscaled, axis=-1)
    
    # Projection along shadow axis
    proj_len = np.sum(pts_unscaled * axis_unit[None, None, :], axis=-1)
    perp_vec = pts_unscaled - proj_len[:, :, None] * axis_unit[None, None, :]
    d_perp = np.linalg.norm(perp_vec, axis=-1)
    
    # Umbra radius at each distance (unscaled, like in eclipse detection)
    r_umbra = (R_EARTH / 1000.0) - (R_SUN / 1000.0 - R_EARTH / 1000.0) * (d_pts / d_es)
    
    # Points in shadow: behind Earth (proj_len > 0) AND within cone (d_perp < r_umbra)
    in_shadow = (proj_len > 0) & (d_perp < r_umbra)
    
    # Darken points in shadow significantly + apply whole-disk shadow factor
    intensity = np.where(in_shadow, intensity * 0.2, intensity)
    intensity = intensity * center_shadow_factor
    
    # Gray Moon color with shading
    r = 0.3 + 0.5 * intensity
    g = 0.3 + 0.5 * intensity
    b = 0.3 + 0.5 * intensity
    
    rgba = np.stack([r, g, b, np.ones_like(r)], axis=-1)
    return rgba


def make_update(ax, lim, earth_local, moon_local, r_m_rel, r_s_rel, sun_line, Re, Rs, Rm, title, t, idx, sphere_res):
    # Create surfaces once, then update their data and colors only
    
    # Initial plot setup
    # Earth surface (static position)
    sun_v_0 = r_s_rel[0]
    earth_facecolors_0 = compute_earth_facecolors(earth_local, sun_v_0)
    earth_surf = ax.plot_surface(
        earth_local[..., 0], earth_local[..., 1], earth_local[..., 2],
        facecolors=earth_facecolors_0, linewidth=0, antialiased=False, shade=False
    )
    
    # Moon surface (initial position)
    moon_c_0 = r_m_rel[0]
    X_0 = moon_c_0[0] + moon_local[..., 0]
    Y_0 = moon_c_0[1] + moon_local[..., 1]
    Z_0 = moon_c_0[2] + moon_local[..., 2]
    scale_factor = Rm / (R_MOON / 1000.0)
    moon_facecolors_0 = compute_facecolors(moon_c_0, sun_v_0, moon_local, sphere_res, scale_factor)
    moon_surf = ax.plot_surface(
        X_0, Y_0, Z_0,
        facecolors=moon_facecolors_0, linewidth=0, antialiased=False, shade=False
    )

    def update(frame_i):
        nonlocal earth_surf, moon_surf
        
        # Get current frame data
        sun_v = r_s_rel[frame_i]
        moon_c = r_m_rel[frame_i]
        
        # Update sun direction line
        s_norm = np.linalg.norm(sun_v)
        sdir = sun_v / (s_norm + 1e-12)
        sun_line.set_data([0, -sdir[0]*lim], [0, -sdir[1]*lim])
        sun_line.set_3d_properties([0, -sdir[2]*lim])

        # Update Earth (remove and replot with new colors)
        earth_facecolors = compute_earth_facecolors(earth_local, sun_v)
        if earth_surf is not None:
            earth_surf.remove()
            
        earth_surf = ax.plot_surface(
            earth_local[..., 0], earth_local[..., 1], earth_local[..., 2],
            facecolors=earth_facecolors, linewidth=0, antialiased=False, shade=False
        )

        # Update Moon (remove and replot with new position and colors)
        X = moon_c[0] + moon_local[..., 0]
        Y = moon_c[1] + moon_local[..., 1]
        Z = moon_c[2] + moon_local[..., 2]
        # Pass scale factor (Rm/R_MOON) to account for visualization scaling
        scale_factor = Rm / (R_MOON / 1000.0)
        facecolors = compute_facecolors(moon_c, sun_v, moon_local, sphere_res, scale_factor)

        if moon_surf is not None:
            moon_surf.remove()

        moon_surf = ax.plot_surface(
            X, Y, Z,
            facecolors=facecolors,
            linewidth=0,
            antialiased=False,
            shade=False
        )

        # Update Title with current time
        array_idx = idx[frame_i]
        ax.set_title(f"{title}\n t = {t[array_idx]/3600:.2f} h (from start)")
        
        return (sun_line, earth_surf, moon_surf)

    return update


def animate_eclipse_3d_geocentric_with_cone_shadow(history, t, eclipse, title="", sphere_res=100):
    # Geocentric 3D animation
    #  - Earth as sphere at origin
    #  - Moon as translated sphere with shading
    #  - Sun direction indicated by a line
    # Units: kilometers for plotting (converted from meters).

    a, b = expand_eclipse_window(eclipse, len(t), pad_factor=5.0)
    idx = np.arange(a, b + 1)

    # km for plotting
    r_s = history[idx, 0, :] / 1000.0
    r_e = history[idx, 1, :] / 1000.0
    r_m = history[idx, 2, :] / 1000.0

    r_s_rel = r_s - r_e   # Earth->Sun (km)
    r_m_rel = r_m - r_e   # Earth->Moon (km)

    Re = R_EARTH / 1000.0 * 20  # Scaled up for visibility
    Rs = R_SUN   / 1000.0
    Rm = R_MOON  / 1000.0 * 50  # Scaled up for visibility

    # Sphere mesh
    u = np.linspace(0, 2*np.pi, sphere_res)
    v = np.linspace(0, np.pi, sphere_res)
    uu, vv = np.meshgrid(u, v)

    xs = np.cos(uu) * np.sin(vv)
    ys = np.sin(uu) * np.sin(vv)
    zs = np.cos(vv)

    moon_local = np.stack([xs, ys, zs], axis=-1) * Rm  # (res,res,3)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_zlabel("z (km)")

    # Set limits based on actual Moon distance range
    moon_dist_max = np.max(np.linalg.norm(r_m_rel, axis=1))
    lim = max(moon_dist_max * 1.5, 500000)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)

    # Earth sphere mesh (earth-centered)
    earth_local = np.stack([xs, ys, zs], axis=-1) * Re  # (res,res,3)

    # Sun direction line
    sun_line, = ax.plot([0, 0], [0, 0], [0, 0], 'r-', linewidth=2, label='Sun direction')
    ax.legend()

    # Pass the full time array, not just t[idx]
    update_func = make_update(ax, lim, earth_local, moon_local, r_m_rel, r_s_rel, sun_line, Re, Rs, Rm, title, t, idx, sphere_res)

    anim = FuncAnimation(fig, update_func, frames=len(idx), interval=25, blit=False, repeat=True)
    return anim

def refine_all_eclipses(history, t, eclipses, factor=5, pad_factor=2.0):
    # Refines the entire history but applies upsampling only around detected eclipses.
    # Returns a full, refined timeline.
    # Collect all indices that need refining
    refined_segments_t = []
    refined_segments_h = []
    
    # We identify all indices that fall into any eclipse window
    to_refine = set()
    for eclipse in eclipses:
        a, b = expand_eclipse_window(eclipse, len(t), pad_factor=pad_factor)
        for i in range(a, b): # range to b because we interpolate between i and i+1
            to_refine.add(i)

    for i in range(len(t) - 1):
        t0, t1 = t[i], t[i+1]
        h0, h1 = history[i], history[i+1]
        
        # If this interval is part of an eclipse, upsample it
        current_factor = factor if i in to_refine else 1
        
        t_seg = np.linspace(t0, t1, current_factor + 1)
        h_seg = np.linspace(h0, h1, current_factor + 1)
        
        # Avoid duplicating the end points
        if i < len(t) - 2:
            t_seg = t_seg[:-1]
            h_seg = h_seg[:-1]
            
        refined_segments_t.append(t_seg)
        refined_segments_h.append(h_seg)
        
    return np.concatenate(refined_segments_t), np.concatenate(refined_segments_h)