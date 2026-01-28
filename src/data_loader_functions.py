import numpy as np
from pathlib import Path

def read_horizons_vectors(path):
    # Reads NASA Horizons database 'Vectors' format files.
    # Returns times (JD), positions (km), and velocities (km/s).

    julian_dates, pos_list, vel_list = [], [], []
    inside = False

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line == "$$SOE":
                inside = True
                continue
            if line == "$$EOE":
                break
            if not inside:
                continue
            if not line or not (line[0].isdigit() or line[0] == "-"):
                continue

            parts = [p.strip() for p in line.split(",")]
            # Columns:
            # 0 JD, 1 Calendar, 2 X,3 Y,4 Z, 5 VX,6 VY,7 VZ, ...
            julian_dates.append(float(parts[0]))
            pos_list.append([float(parts[2]), float(parts[3]), float(parts[4])])
            vel_list.append([float(parts[5]), float(parts[6]), float(parts[7])])

    julian_dates = np.array(julian_dates, dtype=float)
    positions = np.array(pos_list, dtype=float)
    velocities = np.array(vel_list, dtype=float)
    return julian_dates, positions, velocities

def pick(jd, jd_common, r, v):
        idx = np.searchsorted(jd, jd_common)
        return r[idx], v[idx]


def load_horizons_history(sun_path, earth_path, moon_path):
    jd_s, r_s_km, v_s_km_s = read_horizons_vectors(sun_path)
    jd_e, r_e_km, v_e_km_s = read_horizons_vectors(earth_path)
    jd_m, r_m_km, v_m_km_s = read_horizons_vectors(moon_path)

    # Align by JD (they should match if exported consistently)
    jd_common = np.intersect1d(np.intersect1d(jd_s, jd_e), jd_m)

    r_s_km, v_s_km_s = pick(jd_s, jd_common, r_s_km, v_s_km_s)
    r_e_km, v_e_km_s = pick(jd_e, jd_common, r_e_km, v_e_km_s)
    r_m_km, v_m_km_s = pick(jd_m, jd_common, r_m_km, v_m_km_s)

    # Convert to SI
    r_s = r_s_km * 1000.0
    r_e = r_e_km * 1000.0
    r_m = r_m_km * 1000.0

    v_s = v_s_km_s * 1000.0
    v_e = v_e_km_s * 1000.0
    v_m = v_m_km_s * 1000.0

    # Time array in seconds from initial time
    t = (jd_common - jd_common[0]) * 86400.0

    # Position and velocity histories: (times, bodies, coordinates)
    history = np.stack([r_s, r_e, r_m], axis=1)      # (T, 3, 3)
    v_hist  = np.stack([v_s, v_e, v_m], axis=1)      # (T, 3, 3)
    return t, history, v_hist

def get_data_path(filename):
    #Returns the absolute path of a file located in the 'data' directory.
    #Works regardless of where the function is called from within the project.

    # Get the directory where this script resides (the 'src' folder)
    current_dir = Path(__file__).resolve().parent
    
    data_dir = current_dir.parent / "data"
    
    file_path = data_dir / filename
    
    if not file_path.exists():
        print(f"Warning: File not found at: {file_path}")
        return None
        
    return file_path