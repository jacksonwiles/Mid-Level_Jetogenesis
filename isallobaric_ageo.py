import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from netCDF4 import Dataset
from wrf import getvar, latlon_coords, extract_times
from scipy.ndimage import gaussian_filter  # For smoothing the interpolated fields
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter

def nan_gaussian_filter(a, sigma):
    a = np.array(a)
    # Create a mask of valid (non-NaN) entries
    mask = np.isnan(a) == False
    # Replace NaNs with 0
    a_filled = np.where(mask, a, 0)
    # Filter the filled array and the mask
    a_filtered = gaussian_filter(a_filled, sigma=sigma)
    mask_filtered = gaussian_filter(mask.astype(float), sigma=sigma)
    # Avoid division by zero; where mask_filtered is zero, result is NaN
    with np.errstate(invalid='ignore'):
        result = a_filtered / mask_filtered
    result[mask_filtered == 0] = np.nan
    return result

# Ensure an interactive backend is used (for on-screen display)
import matplotlib
matplotlib.use("TkAgg")  # or "Qt5Agg" if preferred

# Set directory containing WRF output files
wrf_directory = "./wrfout_d01/"  # Change to your actual directory
# Only files from 2023-08-20 are used
wrf_files = sorted(glob.glob(os.path.join(wrf_directory, "wrfout_d01_2023-08-*")))
output_dir = "./isallobaric_ageo"  # Directory to save plots
os.makedirs(output_dir, exist_ok=True)

if not wrf_files:
    print("No WRF output files found in the directory.")
    exit()

# Toggle options for saving and displaying the plot
save_plot = True       # Save the plot to a file if True
display_plot = False   # Display the plot on screen if True

# Define the target theta surface in Kelvin
target_theta = 308.0

# Use the known constant horizontal resolution of 2 km x 2 km
dx = 10000  # in meters
dy = 10000  # in meters

# Loop over each WRF file
for wrf_file in wrf_files:
    ncfile = Dataset(wrf_file)
    time_steps = extract_times(ncfile, timeidx=None, meta=False)
    # Create a base file name for saving plots
    wrf_filename = os.path.basename(wrf_file).replace(":", "-")
    
    for t_idx in range(len(time_steps)):
        # --- Extract fields: temperature (tk), geopotential height (z),
        # potential temperature (theta), Coriolis parameter (F), and wind components (u,v) ---
        tk    = getvar(ncfile, "tk", timeidx=t_idx)
        z     = getvar(ncfile, "z", timeidx=t_idx)
        theta = getvar(ncfile, "theta", timeidx=t_idx)
        F     = getvar(ncfile, "F", timeidx=t_idx)  # Coriolis parameter

        # Extract wind components (assumed to be named "ua" and "va")
        u = getvar(ncfile, "ua", timeidx=t_idx)
        v = getvar(ncfile, "va", timeidx=t_idx)

        # Retrieve horizontal lat/lon coordinates (using the variable with metadata)
        lats, lons = latlon_coords(tk)
        lats = np.array(lats)
        lons = np.array(lons)
        
        # Convert variables to numpy arrays
        tk    = np.array(tk)
        z     = np.array(z)
        theta = np.array(theta)
        F     = np.array(F)
        u     = np.array(u)
        v     = np.array(v)
        
        # Assume the fields have shape (nz, ny, nx)
        nz, ny, nx = tk.shape
        
        # Prepare arrays to hold the interpolated values at the 316 K theta surface
        tk_interp = np.full((ny, nx), np.nan)
        z_interp  = np.full((ny, nx), np.nan)
        u_interp  = np.full((ny, nx), np.nan)
        v_interp  = np.full((ny, nx), np.nan)
        
        # Loop over horizontal grid points to interpolate vertically
        for j in range(ny):
            for i in range(nx):
                theta_profile = theta[:, j, i]
                tk_profile = tk[:, j, i]
                z_profile  = z[:, j, i]
                u_profile = u[:, j, i]
                v_profile = v[:, j, i]
                if target_theta >= theta_profile.min() and target_theta <= theta_profile.max():
                    tk_interp[j, i] = np.interp(target_theta, theta_profile, tk_profile)
                    z_interp[j, i]  = np.interp(target_theta, theta_profile, z_profile)
                    u_interp[j, i]  = np.interp(target_theta, theta_profile, u_profile)
                    v_interp[j, i]  = np.interp(target_theta, theta_profile, v_profile)
                else:
                    tk_interp[j, i] = np.nan
                    z_interp[j, i]  = np.nan
                    u_interp[j, i]  = np.nan
                    v_interp[j, i]  = np.nan

        # --- Smooth the interpolated fields ---
        tk_interp = nan_gaussian_filter(tk_interp, sigma=15)
        z_interp  = nan_gaussian_filter(z_interp, sigma=15)
        u_interp  = nan_gaussian_filter(u_interp, sigma=15)
        v_interp  = nan_gaussian_filter(v_interp, sigma=15)
       
        # Compute the Montgomery stream function on the smoothed, interpolated surface:
        # M = 1004 * tk_interp + 9.8 * z_interp  (units: m²/s²)
        M_interp = 1004 * tk_interp + 9.8 * z_interp
        #M_interp = nan_gaussian_filter(M_interp, sigma=25)
        
        # --- Subset the data to the region [lat: 34–40, lon: -118 to -112] ---
        lat_inds = np.where((lats[:, 0] >= 34.0) & (lats[:, 0] <= 40.0))[0]
        lon_inds = np.where((lons[0, :] >= -118.0) & (lons[0, :] <= -112.0))[0]
        if len(lat_inds) == 0 or len(lon_inds) == 0:
            print(f"No data found in the specified region for time index: {t_idx}")
            continue
        lat_start, lat_end = lat_inds[0], lat_inds[-1] + 1
        lon_start, lon_end = lon_inds[0], lon_inds[-1] + 1
        
        lats_sub = lats[lat_start:lat_end, lon_start:lon_end]
        lons_sub = lons[lat_start:lat_end, lon_start:lon_end]
        M_interp_sub = M_interp[lat_start:lat_end, lon_start:lon_end]
        F_sub = F[lat_start:lat_end, lon_start:lon_end]  # Subset the Coriolis parameter
        u_interp_sub = u_interp[lat_start:lat_end, lon_start:lon_end]
        v_interp_sub = v_interp[lat_start:lat_end, lon_start:lon_end]
        
        # Print out min and max for M
        print(f"Time step {t_idx}: M: min {np.nanmin(M_interp_sub):.2f}, max {np.nanmax(M_interp_sub):.2f}")
        
        # --- Compute spatial gradient of M on the subset (first derivatives) ---
        dM_dy, dM_dx = np.gradient(M_interp_sub, dy, dx)
        print(f"Time step {t_idx}: dM_dx: min {np.nanmin(dM_dx):.2e}, max {np.nanmax(dM_dx):.2e}")
        print(f"Time step {t_idx}: dM_dy: min {np.nanmin(dM_dy):.2e}, max {np.nanmax(dM_dy):.2e}")
 
        dM_dx = gaussian_filter(dM_dx, sigma=15)
        dM_dy = gaussian_filter(dM_dy, sigma=15)


        # === Compute second derivatives of M for the advective term ===
        d2M_dx2 = np.gradient(dM_dx, dx, axis=1)
        d2M_dy2 = np.gradient(dM_dy, dy, axis=0)
        d2M_dxdy = np.gradient(dM_dx, dy, axis=0)

        d2M_dx2 = gaussian_filter(d2M_dx2, sigma=15)
        d2M_dy2 = gaussian_filter(d2M_dy2, sigma=15)
        d2M_dxdy = gaussian_filter(d2M_dxdy, sigma=15)


        print(f"Time step {t_idx}: d2M_dx2: min {np.nanmin(d2M_dx2):.2e}, max {np.nanmax(d2M_dx2):.2e}")
        print(f"Time step {t_idx}: d2M_dy2: min {np.nanmin(d2M_dy2):.2e}, max {np.nanmax(d2M_dy2):.2e}")
        print(f"Time step {t_idx}: d2M_dxdy: min {np.nanmin(d2M_dxdy):.2e}, max {np.nanmax(d2M_dxdy):.2e}")
        
        # Compute the advective term: (U·∇)(∇ψ) where ψ = M_interp
        adv_term_x = u_interp_sub * d2M_dx2 + v_interp_sub * d2M_dxdy
        adv_term_y = u_interp_sub * d2M_dxdy + v_interp_sub * d2M_dy2

        adv_term_x = gaussian_filter(adv_term_x, sigma=15)
        adv_term_y = gaussian_filter(adv_term_y, sigma=15)

        print(f"Time step {t_idx}: adv_term_x: min {np.nanmin(adv_term_x):.2e}, max {np.nanmax(adv_term_x):.2e}")
        print(f"Time step {t_idx}: adv_term_y: min {np.nanmin(adv_term_y):.2e}, max {np.nanmax(adv_term_y):.2e}")
        
        # --- Multiply by -1/F^2 ---
        final_adv_term_x = (-1.0 / (F_sub**2)) * adv_term_x
        final_adv_term_y = (-1.0 / (F_sub**2)) * adv_term_y
        print(f"Time step {t_idx}: final_adv_term_x: min {np.nanmin(final_adv_term_x):.2e}, max {np.nanmax(final_adv_term_x):.2e}")
        print(f"Time step {t_idx}: final_adv_term_y: min {np.nanmin(final_adv_term_y):.2e}, max {np.nanmax(final_adv_term_y):.2e}")

        final_adv_term_x = gaussian_filter(final_adv_term_x, sigma=15)
        final_adv_term_y = gaussian_filter(final_adv_term_y, sigma=15)

        # --- Plot the final advective term vector field ---
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
        time_str = np.datetime_as_string(time_steps[t_idx], unit='m').replace("T", ":")
        ax.set_title(f"Isallobaric Ageostrophic Term\n(308K Isentropic Surface) - {time_str}", fontsize=12)
        
        ax.add_feature(cfeature.COASTLINE, linewidth=1)
        ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.5)
        ax.add_feature(cfeature.STATES, linestyle="-", linewidth=2, alpha=0.5)
        ax.set_extent([-118.0, -112.0, 34.0, 40.0], crs=ccrs.PlateCarree())
        
        vector_step = 5  # Sample every 10 grid points to avoid clutter
        q = ax.quiver(lons_sub[::vector_step, ::vector_step],
                      lats_sub[::vector_step, ::vector_step],
                      final_adv_term_x[::vector_step, ::vector_step],
                      final_adv_term_y[::vector_step, ::vector_step],
                      transform=ccrs.PlateCarree(),
                      color='black',
                      scale_units='inches',
                      scale=25,
                      width=0.003,
                      headwidth=3,
                      headlength=4)
        ax.quiverkey(q, 0.9, -0.06, 5, "5 m/s", labelpos='E', transform=ax.transAxes)
        ax.set_xticks(np.linspace(lons_sub.min(), lons_sub.max(), num=5), crs=ccrs.PlateCarree())
        ax.set_yticks(np.linspace(lats_sub.min(), lats_sub.max(), num=5), crs=ccrs.PlateCarree())
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        
        fig.canvas.draw()
        output_filename = f"{output_dir}/isallobaric_ageostrophic_{time_str}.png"
        if save_plot:
            plt.savefig(output_filename, dpi=300, bbox_inches="tight")
            print(f"Saved final advective term plot: {output_filename}")
        if display_plot:
            plt.show()
            plt.pause(1)
        plt.close(fig)
    
    ncfile.close()
