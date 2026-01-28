import sys
from pathlib import Path
import time

# Add parent directory to path for imports
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from src import plot_functions as pf

# Path to cached simulation results
# Absolute path to the script's directory
DATA_FILE = current_dir / "data" / "simulation_results.npz"

print(f"DEBUG: Looking for data at: {DATA_FILE}")

def load_cached_results():
    if not DATA_FILE.exists():
        return None
    
    with np.load(DATA_FILE, allow_pickle=True) as data:
        model_configs = []
        
        # Define the prefixes that were used during saving
        prefixes = ["nasa", "kepler", "verlet_raw", "verlet_opt"]
        
        for p in prefixes:
            h_key = f"h_{p}"
            if h_key in data:
                # Extract the eclipse list
                e_list = list(data[f"e_{p}"])
                
                model_configs.append({
                    "id": p.upper(),
                    "history": data[h_key],
                    "t": data[f"t_{p}"],
                    # Take only the first one if it eclipses, otherwise None
                    "eclipse": e_list[0] if len(e_list) > 0 else None,
                    # .item() is needed to convert from numpy array to string
                    "title": str(data[f"title_{p}"].item()) 
                })
        
        return model_configs

from scipy.interpolate import interp1d

def sync_to_nasa(model_list):
    # Find the NASA model, it's our reference
    nasa = next((m for m in model_list if m['id'] == 'NASA'), None)
    if not nasa:
        return model_list # If there's no NASA, we can't synchronize to anything
    
    # NASA time domain (this is the shortest)
    common_t = nasa['t'] 
    synced_models = []

    for m in model_list:
        try:
            # Create interpolation function based on original data
            # history shape: (time, object, coordinate) -> (N, 3, 3)
            interpolator = interp1d(m['t'], m['history'], axis=0, kind='linear', fill_value="extrapolate")
            
            # Calculate positions at NASA time points
            m['history'] = interpolator(common_t)
            m['t'] = common_t
            
            # Align the eclipse window to NASA as well
            # Since now all models have the same t, the indices will be the same
            m['eclipse'] = nasa['eclipse']
            
            synced_models.append(m)
        except Exception as e:
            print(f"Could not sync {m['id']}: {e}")
            synced_models.append(m)

    return synced_models

def main():
    # Load data (load_cached_results returns a list of model dictionaries)
    model_list = load_cached_results()
    


    if not model_list:
        print(f"ERROR: Could not find or load data file: {DATA_FILE}")
        return
    
    model_list = sync_to_nasa(model_list)
    
    print("\n" + "=" * 60)
    print(f"Starting Sequential Animations ({len(model_list)} models)")
    print("=" * 60)

    try:
        all_animations = []
        for model in model_list:
            # Check if a valid eclipse event exists for the model
            eclipse = model.get("eclipse")
            
            if eclipse is not None:
                print(f"\n>>> Running Animation: {model['title']}...")
                
                anim = pf.animate_eclipse_3d_geocentric_with_cone_shadow(
                model["history"], 
                model["t"], 
                model["eclipse"], 
                title=model["title"],
                )
                all_animations.append(anim)
                
                print(f"Close the '{model['id']}' window to proceed to the next model.")
                plt.show() 
            else:
                print(f"\n[!] No eclipse detected for {model['title']}, skipping.")

    except Exception as e:
        print(f"\nAn error occurred during animation: {e}")

    finally:
        # Cleanup: Ensure the temporary data file is deleted after use or on error
        if DATA_FILE.exists():
            try:
                DATA_FILE.unlink()
                print(f"\nCleanup: Temporary file '{DATA_FILE.name}' has been removed.")
            except Exception as cleanup_error:
                print(f"Warning: Cleanup failed: {cleanup_error}")

    print("\n" + "=" * 60)
    print("All animations completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()
