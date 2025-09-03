import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from netCDF4 import Dataset
from wrf import getvar, latlon_coords, extract_times
from scipy.ndimage import gaussian_filter

def nan_gaussian_filter(a, sigma):
    """Apply a Gaussian filter, ignoring NaNs (as in reference scripts)."""
    a = np.array(a)
    mask = ~np.isnan(a)
    a_filled = np.where(mask, a, 0)
    a_filtered = gaussian_filter(a_filled, sigma=sigma)
    mask_filtered = gaussian_filter(mask.astype(float), sigma=sigma)
    with np.errstate(invalid='ignore'):
        result = a_filtered / mask_filtered
    result[mask_filtered == 0] = np.nan
    return result

# ------------------- User/Domain Settings -------------------
wrf_dir = "./wrfout_d01/"
wrf_files = sorted(glob.glob(os.path.join(wrf_dir, "wrfout_d01_2023-08-*")))
if not wrf_files:
    print("No WRF output files found!")
    exit()

# Make output directories for each level
plot_levels = [308, 310, 312, 314, 316, 318, 320, 322]
output_dirs = {lev: f"./total_ageostrophic_{lev}K" for lev in plot_levels}
for od in output_dirs.values():
    os.makedirs(od, exist_ok=True)

save_plot = True
display_plot = False

# Constants
dx = dy = 10000   # [m]
dt = 3600.0       # [s], time between steps, adjust if needed
cp, g = 1004.0, 9.8

# ------------------- Begin Processing -------------------
prev_M_interp_sub_dict = {lev: None for lev in plot_levels}
step_counter = 0

for wrf_file in wrf_files:
    ncfile = Dataset(wrf_file)
    times = extract_times(ncfile, timeidx=None, meta=False)
    print(f"Processing file: {wrf_file}")

    for t_idx, t in enumerate(times):
        print(f"\n  Time index: {t_idx}, Time: {t}")

        # 1. Extract 3D and 2D fields
        theta = getvar(ncfile, "theta", timeidx=t_idx)
        tk    = getvar(ncfile, "tk",    timeidx=t_idx)
        z     = getvar(ncfile, "z",     timeidx=t_idx)
        F     = getvar(ncfile, "F",     timeidx=t_idx)
        u     = getvar(ncfile, "ua",    timeidx=t_idx)
        v     = getvar(ncfile, "va",    timeidx=t_idx)
        try:
            Q = getvar(ncfile, "H_DIABATIC", timeidx=t_idx)
        except Exception:
            raise RuntimeError("Field H_DIABATIC not found. Required for diabatic term.")

        lats, lons = latlon_coords(theta)
        lats = np.array(lats)
        lons = np.array(lons)
        theta_np = np.array(theta)
        tk_np    = np.array(tk)
        z_np     = np.array(z)
        u_np     = np.array(u)
        v_np     = np.array(v)
        F_np     = np.array(F)
        Q_np     = np.array(Q)
        nz, ny, nx = theta_np.shape

        # --------- MAIN MULTILEVEL PLOT LOOP ----------
        for target_theta in plot_levels:
            # 2. Interpolate all fields to target_theta
            def interp_to_theta(field_3d):
                result = np.full((ny, nx), np.nan)
                for j in range(ny):
                    for i in range(nx):
                        th_prof = theta_np[:, j, i]
                        fld_prof = field_3d[:, j, i]
                        if target_theta >= th_prof.min() and target_theta <= th_prof.max():
                            result[j, i] = np.interp(target_theta, th_prof, fld_prof)
                return result

            tk_lev   = interp_to_theta(tk_np)
            z_lev    = interp_to_theta(z_np)
            u_lev    = interp_to_theta(u_np)
            v_lev    = interp_to_theta(v_np)
            Q_lev    = interp_to_theta(Q_np)

            # 3. Smoothing (sigma=15 for all fields)
            tk_lev   = nan_gaussian_filter(tk_lev, sigma=15)
            z_lev    = nan_gaussian_filter(z_lev, sigma=15)
            u_lev    = nan_gaussian_filter(u_lev, sigma=15)
            v_lev    = nan_gaussian_filter(v_lev, sigma=15)
            Q_lev    = nan_gaussian_filter(Q_lev, sigma=15)

            # 4. Compute Montgomery streamfunction
            M_lev = cp * tk_lev + g * z_lev

            # 5. Subset to domain
            lat_idx = np.where((lats[:, 0] >= 34) & (lats[:, 0] <= 40))[0]
            lon_idx = np.where((lons[0] >= -118) & (lons[0] <= -112))[0]
            if lat_idx.size == 0 or lon_idx.size == 0:
                continue
            j0, j1 = lat_idx[0], lat_idx[-1] + 1
            i0, i1 = lon_idx[0], lon_idx[-1] + 1
            ls = lats[j0:j1, i0:i1]
            lo = lons[j0:j1, i0:i1]
            M_lev_sub = M_lev[j0:j1, i0:i1]
            F_sub = F_np[j0:j1, i0:i1]
            u_lev_sub = u_lev[j0:j1, i0:i1]
            v_lev_sub = v_lev[j0:j1, i0:i1]
            Q_lev_sub = Q_lev[j0:j1, i0:i1]

            # 6. Isallobaric Ageostrophic Wind (advective)
            dM_dy, dM_dx = np.gradient(M_lev_sub, dy, dx)
            dM_dx = nan_gaussian_filter(dM_dx, sigma=15)
            dM_dy = nan_gaussian_filter(dM_dy, sigma=15)
            d2M_dx2 = np.gradient(dM_dx, dx, axis=1)
            d2M_dy2 = np.gradient(dM_dy, dy, axis=0)
            d2M_dxdy = np.gradient(dM_dx, dy, axis=0)
            d2M_dx2 = nan_gaussian_filter(d2M_dx2, sigma=15)
            d2M_dy2 = nan_gaussian_filter(d2M_dy2, sigma=15)
            d2M_dxdy = nan_gaussian_filter(d2M_dxdy, sigma=15)
            adv_term_x = u_lev_sub * d2M_dx2 + v_lev_sub * d2M_dxdy
            adv_term_y = u_lev_sub * d2M_dxdy + v_lev_sub * d2M_dy2
            adv_term_x = nan_gaussian_filter(adv_term_x, sigma=15)
            adv_term_y = nan_gaussian_filter(adv_term_y, sigma=15)
            with np.errstate(divide="ignore", invalid="ignore"):
                isallobaric_x = -adv_term_x / (F_sub**2)
                isallobaric_y = -adv_term_y / (F_sub**2)
            isallobaric_x = nan_gaussian_filter(isallobaric_x, sigma=15)
            isallobaric_y = nan_gaussian_filter(isallobaric_y, sigma=15)

            # 7. Inertial Advective Ageostrophic Wind (needs prev step)
            dM_dy, dM_dx = np.gradient(M_lev_sub, dy, dx)
            if step_counter == 0 or prev_M_interp_sub_dict[target_theta] is None:
                inertial_adv_x = np.zeros_like(M_lev_sub) * np.nan
                inertial_adv_y = np.zeros_like(M_lev_sub) * np.nan
            else:
                prev_dM_dy, prev_dM_dx = np.gradient(prev_M_interp_sub_dict[target_theta], dy, dx)
                d_dt_grad_dx = (dM_dx - prev_dM_dx) / dt
                d_dt_grad_dy = (dM_dy - prev_dM_dy) / dt
                inertial_adv_x = -d_dt_grad_dx / (F_sub**2)
                inertial_adv_y = -d_dt_grad_dy / (F_sub**2)

            # 8. Inertial Diabatic Ageostrophic Wind (vertical derivative in theta)
            # Use ±2 K around target_theta
            theta_levels_local = np.array([target_theta - 2, target_theta, target_theta + 2])
            dMdx_levs = []
            dMdy_levs = []
            for thval in theta_levels_local:
                tmp_tk = np.full((ny, nx), np.nan)
                tmp_z = np.full((ny, nx), np.nan)
                for j in range(ny):
                    for i in range(nx):
                        th_prof = theta_np[:, j, i]
                        if thval >= th_prof.min() and thval <= th_prof.max():
                            tmp_tk[j, i] = np.interp(thval, th_prof, tk_np[:, j, i])
                            tmp_z[j, i] = np.interp(thval, th_prof, z_np[:, j, i])
                tmp_tk = nan_gaussian_filter(tmp_tk, sigma=15)
                tmp_z  = nan_gaussian_filter(tmp_z, sigma=15)
                tmp_M  = cp * tmp_tk + g * tmp_z
                tmp_M  = nan_gaussian_filter(tmp_M, sigma=15)
                tmp_M_sub = tmp_M[j0:j1, i0:i1]
                dM_dy_tmp, dM_dx_tmp = np.gradient(tmp_M_sub, dy, dx)
                dMdx_levs.append(nan_gaussian_filter(dM_dx_tmp, sigma=15))
                dMdy_levs.append(nan_gaussian_filter(dM_dy_tmp, sigma=15))
            dMdx_dth = (dMdx_levs[2] - dMdx_levs[0]) / (theta_levels_local[2] - theta_levels_local[0])
            dMdy_dth = (dMdy_levs[2] - dMdy_levs[0]) / (theta_levels_local[2] - theta_levels_local[0])
            with np.errstate(divide="ignore", invalid="ignore"):
                diabatic_x = -Q_lev_sub / (F_sub ** 2) * dMdx_dth
                diabatic_y = -Q_lev_sub / (F_sub ** 2) * dMdy_dth
            diabatic_x = nan_gaussian_filter(diabatic_x, sigma=15)
            diabatic_y = nan_gaussian_filter(diabatic_y, sigma=15)

            # 9. Total Ageostrophic Wind Vector Sum
            total_x = isallobaric_x + inertial_adv_x + diabatic_x
            total_y = isallobaric_y + inertial_adv_y + diabatic_y

            # 10. Plot (skip very first time step, per physics)
            if step_counter > 0:
                mag = np.sqrt(total_x**2 + total_y**2)
                max_mag = np.nanmax(mag)
                print(f"    Max vector magnitude for TOTAL field at {target_theta} K: {max_mag:.3f} m/s")

                fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
                tstr = np.datetime_as_string(t, unit='m').replace('T', ':')
                ax.set_title(f"TOTAL Ageostrophic Wind ({target_theta} K)\n{tstr}", fontsize=14)
                ax.add_feature(cfeature.COASTLINE)
                ax.add_feature(cfeature.BORDERS, linestyle=':')
                ax.add_feature(cfeature.STATES, linestyle='-', alpha=0.5)
                ax.set_extent([-118, -112, 34, 40], crs=ccrs.PlateCarree())
                vector_step = 5
                q = ax.quiver(
                    lo[::vector_step, ::vector_step], ls[::vector_step, ::vector_step],
                    total_x[::vector_step, ::vector_step], total_y[::vector_step, ::vector_step],
                    transform=ccrs.PlateCarree(),
                    color='black', scale_units='inches', scale=25,
                    width=0.003, headwidth=3, headlength=4)
                ax.quiverkey(q, 0.9, -0.06, 5, "5 m/s", labelpos='E', transform=ax.transAxes)
                ax.set_xticks(np.linspace(lo.min(), lo.max(), 5), crs=ccrs.PlateCarree())
                ax.set_yticks(np.linspace(ls.min(), ls.max(), 5), crs=ccrs.PlateCarree())
                ax.set_xlabel("Longitude")
                ax.set_ylabel("Latitude")
                if save_plot:
                    fname = f"{output_dirs[target_theta]}/total_ageostrophic_{target_theta}K_{tstr}.png"
                    plt.savefig(fname, dpi=300, bbox_inches='tight')
                    print(f"    Saved: {fname}")
                if display_plot:
                    plt.show()
                    plt.pause(1)
                plt.close(fig)

            # Save prev M for this theta for next time step
            prev_M_interp_sub_dict[target_theta] = M_lev_sub.copy()

        step_counter += 1

    ncfile.close()

