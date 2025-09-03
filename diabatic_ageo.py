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
    a = np.array(a)
    mask = ~np.isnan(a)
    a_filled = np.where(mask, a, 0)
    a_filtered = gaussian_filter(a_filled, sigma=sigma)
    mask_filtered = gaussian_filter(mask.astype(float), sigma=sigma)
    with np.errstate(invalid='ignore'):
        result = a_filtered / mask_filtered
    result[mask_filtered == 0] = np.nan
    return result

# Settings
wrf_dir     = "./"
wrf_files   = sorted(glob.glob(os.path.join(wrf_dir, "wrfout_d01_2023-08-*")))
out_dir     = "./diabatic_plots"
os.makedirs(out_dir, exist_ok=True)

save_plot    = True
display_plot = False

# Isentropic levels (306–332 K every 2 K)
theta_levels = np.arange(306.0, 334.0, 2.0)   # [306, 308, ..., 332]
dx = dy = 10000  # horizontal resolution in m

cp, g = 1004.0, 9.8

# Output directory for 308 K wind
output_dir_308K = "./diabatic_fields_308K"
os.makedirs(output_dir_308K, exist_ok=True)

# --- Levels to plot and save as PNGs
plot_levels = [308, 310, 312, 314, 316, 318, 320, 322]

for wrf_file in wrf_files:
    print(f"\nProcessing file: {wrf_file}")
    nc    = Dataset(wrf_file)
    times = extract_times(nc, timeidx=None, meta=False)

    for ti, t in enumerate(times):
        print(f"\n  Time index: {ti}, Time: {t}")

        # 1. Read 3D fields (native model levels)
        theta = getvar(nc, "theta", timeidx=ti)   # [K]
        F     = getvar(nc, "F",     timeidx=ti)   # [1/s]
        tk    = getvar(nc, "tk",    timeidx=ti)   # [K]
        z     = getvar(nc, "z",     timeidx=ti)   # [m]
        h3d   = getvar(nc, "H_DIABATIC", timeidx=ti).values # [K/s] or [W/kg]

        theta_np = np.array(theta)
        F_np     = np.array(F)
        tk_np    = np.array(tk)
        z_np     = np.array(z)
        lats, lons = latlon_coords(theta)
        lats, lons = np.array(lats), np.array(lons)
        nz, ny, nx = theta_np.shape

        nlevs = len(theta_levels)

        # Allocate arrays for isentropic fields
        tk_isent = np.full((nlevs, ny, nx), np.nan)
        z_isent  = np.full((nlevs, ny, nx), np.nan)
        M_isent  = np.full((nlevs, ny, nx), np.nan)
        Q_isent  = np.full((nlevs, ny, nx), np.nan)

        # 2. Interpolate tk, z, and Q to each isentropic surface, then smooth (sigma=25)
        for k, thval in enumerate(theta_levels):
            for j in range(ny):
                for i in range(nx):
                    th_prof = theta_np[:,j,i]
                    tk_prof = tk_np[:,j,i]
                    z_prof  = z_np[:,j,i]
                    Q_prof  = h3d[:,j,i]
                    if np.any((th_prof < thval)) and np.any((th_prof > thval)):
                        try:
                            tk_isent[k,j,i] = np.interp(thval, th_prof, tk_prof)
                            z_isent[k,j,i]  = np.interp(thval, th_prof, z_prof)
                            Q_isent[k,j,i]  = np.interp(thval, th_prof, Q_prof)
                        except Exception:
                            continue
            tk_isent[k,:,:] = nan_gaussian_filter(tk_isent[k,:,:], sigma=15)
            z_isent[k,:,:]  = nan_gaussian_filter(z_isent[k,:,:], sigma=15)
            Q_isent[k,:,:]  = nan_gaussian_filter(Q_isent[k,:,:], sigma=15)
            M_isent[k,:,:]  = cp * tk_isent[k,:,:] + g * z_isent[k,:,:]

        # 3. Use 2D F for all isentropic surfaces (NO interpolation)
        if F_np.ndim == 3:
            F2d = F_np[0,:,:]
        else:
            F2d = F_np

        # 4. Convert Q_isent from W/kg to K/s if needed
        Q_units = getattr(nc.variables['H_DIABATIC'], 'units', '').lower()
        if 'w/kg' in Q_units or 'w m-2' in Q_units or np.nanmax(Q_isent) > 1:
            print("Converting Q_isent from W/kg to K/s")
            Q_isent = Q_isent / 1004.0  # cp = 1004 J/kg/K

        # 5. Compute and smooth horizontal gradients of M on each isentropic surface (sigma=15)
        dMdx_isent = np.empty_like(M_isent)
        dMdy_isent = np.empty_like(M_isent)
        for k in range(nlevs):
            dMdx_isent[k,:,:] = np.gradient(M_isent[k,:,:], axis=1) / dx
            dMdy_isent[k,:,:] = np.gradient(M_isent[k,:,:], axis=0) / dy
            dMdx_isent[k,:,:] = nan_gaussian_filter(dMdx_isent[k,:,:], sigma=15)
            dMdy_isent[k,:,:] = nan_gaussian_filter(dMdy_isent[k,:,:], sigma=15)

        # 6. Compute and smooth vertical derivatives of grad(M) with respect to theta (sigma=15)
        dMdx_dth = np.full_like(M_isent, np.nan)
        dMdy_dth = np.full_like(M_isent, np.nan)
        for k in range(1, nlevs):
            delta_theta = theta_levels[k] - theta_levels[k-1]  # [K]
            dMdx_dth[k,:,:] = (dMdx_isent[k,:,:] - dMdx_isent[k-1,:,:]) / delta_theta
            dMdy_dth[k,:,:] = (dMdy_isent[k,:,:] - dMdy_isent[k-1,:,:]) / delta_theta
            dMdx_dth[k,:,:] = nan_gaussian_filter(dMdx_dth[k,:,:], sigma=15)
            dMdy_dth[k,:,:] = nan_gaussian_filter(dMdy_dth[k,:,:], sigma=15)

        # 7. Compute and smooth 3D inertial ageostrophic response at each isentropic level (sigma=15)
        diab_x = np.full_like(M_isent, np.nan)
        diab_y = np.full_like(M_isent, np.nan)
        for k in range(nlevs):
            mask = (F2d != 0) & np.isfinite(Q_isent[k]) & np.isfinite(dMdx_dth[k]) & np.isfinite(dMdy_dth[k])
            diab_x[k][mask] = - Q_isent[k][mask] / (F2d[mask] ** 2) * dMdx_dth[k][mask]
            diab_y[k][mask] = - Q_isent[k][mask] / (F2d[mask] ** 2) * dMdy_dth[k][mask]
            diab_x[k,:,:] = nan_gaussian_filter(diab_x[k,:,:], sigma=15)
            diab_y[k,:,:] = nan_gaussian_filter(diab_y[k,:,:], sigma=15)

        # 8. Print min/max of all fields at every isentropic level
        for k, thval in enumerate(theta_levels):
            print(f"    θ = {thval:.1f} K:")
            print(f"      M_isent    [m^2/s^2]  min: {np.nanmin(M_isent[k]):.5e}, max: {np.nanmax(M_isent[k]):.5e}")
            print(f"      Q_isent    [K/s]      min: {np.nanmin(Q_isent[k]):.5e}, max: {np.nanmax(Q_isent[k]):.5e}")
            print(f"      dMdx_isent [m/s^2]    min: {np.nanmin(dMdx_isent[k]):.5e}, max: {np.nanmax(dMdx_isent[k]):.5e}")
            print(f"      dMdy_isent [m/s^2]    min: {np.nanmin(dMdy_isent[k]):.5e}, max: {np.nanmax(dMdy_isent[k]):.5e}")
            print(f"      dMdx_dth   [m/(s^2·K)] min: {np.nanmin(dMdx_dth[k]):.5e}, max: {np.nanmax(dMdx_dth[k]):.5e}")
            print(f"      dMdy_dth   [m/(s^2·K)] min: {np.nanmin(dMdy_dth[k]):.5e}, max: {np.nanmax(dMdy_dth[k]):.5e}")
            print(f"      diab_x     [m/s]      min: {np.nanmin(diab_x[k]):.5e}, max: {np.nanmax(diab_x[k]):.5e}")
            print(f"      diab_y     [m/s]      min: {np.nanmin(diab_y[k]):.5e}, max: {np.nanmax(diab_y[k]):.5e}")

        # -------- PLOT & SAVE PNGs FOR THE SELECTED LEVELS --------------
        tstr = np.datetime_as_string(t, unit='m').replace('T', ':')

        for plot_theta in plot_levels:
            if plot_theta not in theta_levels:
                continue
            k_plot = np.where(theta_levels == plot_theta)[0][0]
            u_plot = diab_x[k_plot, :, :]
            v_plot = diab_y[k_plot, :, :]
            ls = lats
            lo = lons

            # Restrict plotting to valid region as before:
            lat_idx = np.where((lats[:, 0] >= 34) & (lats[:, 0] <= 40))[0]
            lon_idx = np.where((lons[0] >= -118) & (lons[0] <= -112))[0]
            if lat_idx.size == 0 or lon_idx.size == 0:
                continue
            j0, j1 = lat_idx[0], lat_idx[-1] + 1
            i0, i1 = lon_idx[0], lon_idx[-1] + 1

            ls = lats[j0:j1, i0:i1]
            lo = lons[j0:j1, i0:i1]
            u_plot = u_plot[j0:j1, i0:i1]
            v_plot = v_plot[j0:j1, i0:i1]

            max_mag = np.nanmax(np.sqrt(u_plot**2 + v_plot**2))
            print(f"    Max vector magnitude for θ={plot_theta} K: {max_mag:.3f} m/s")

            fig, ax = plt.subplots(
                figsize=(10, 8),
                subplot_kw={'projection': ccrs.PlateCarree()}
            )
            ax.set_title(f"3D Inertial Diabatic Component {plot_theta} K — {tstr}")
            ax.add_feature(cfeature.COASTLINE)
            ax.add_feature(cfeature.BORDERS, linestyle=':')
            ax.add_feature(cfeature.STATES, linestyle='-', alpha=0.5)
            ax.set_extent([-118, -112, 34, 40], crs=ccrs.PlateCarree())

            vector_step = 5
            q = ax.quiver(
                lo[::vector_step, ::vector_step],
                ls[::vector_step, ::vector_step],
                u_plot[::vector_step, ::vector_step],
                v_plot[::vector_step, ::vector_step],
                transform=ccrs.PlateCarree(),
                color='black',
                scale_units='inches',
                scale=25,
                width=0.003,
                headwidth=3,
                headlength=4
            )

            ax.quiverkey(
                q, 0.9, -0.06, 5,
                "5 m/s", labelpos='E', transform=ax.transAxes
            )

            ax.set_xticks(np.linspace(lo.min(), lo.max(), 5), crs=ccrs.PlateCarree())
            ax.set_yticks(np.linspace(ls.min(), ls.max(), 5), crs=ccrs.PlateCarree())
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")

            if save_plot:
                fname = f"{out_dir}/inertial_ageo_{int(plot_theta)}K_{tstr}.png"
                plt.savefig(fname, dpi=300, bbox_inches='tight')
                print(f"    Saved: {fname}")
            if display_plot:
                plt.show()
                plt.pause(1)
            plt.close(fig)

        # 10. Output diab_x and diab_y at the 308 K isentrope
        k_target = np.where(theta_levels == 308)[0][0]
        u308 = diab_x[k_target, :, :]
        v308 = diab_y[k_target, :, :]

        # Use cropped domain for saving .npy and NetCDF as in original code
        lat_idx = np.where((lats[:, 0] >= 34) & (lats[:, 0] <= 40))[0]
        lon_idx = np.where((lons[0] >= -118) & (lons[0] <= -112))[0]
        if lat_idx.size > 0 and lon_idx.size > 0:
            j0, j1 = lat_idx[0], lat_idx[-1] + 1
            i0, i1 = lon_idx[0], lon_idx[-1] + 1
            u308_crop = u308[j0:j1, i0:i1]
            v308_crop = v308[j0:j1, i0:i1]
            np.save(os.path.join(output_dir_308K, f"diab_x_308K_{tstr}.npy"), u308_crop)
            np.save(os.path.join(output_dir_308K, f"diab_y_308K_{tstr}.npy"), v308_crop)
            print(f"    Saved diab_x and diab_y at 308 K to {output_dir_308K} as .npy files.")

            import netCDF4 as nc4
            ncfile = nc4.Dataset(os.path.join(output_dir_308K, f"ageo_308K_{tstr}.nc"), "w")
            ny_sub, nx_sub = u308_crop.shape
            ncfile.createDimension("lat", ny_sub)
            ncfile.createDimension("lon", nx_sub)
            xvar = ncfile.createVariable("diab_x_308K", "f4", ("lat", "lon"))
            yvar = ncfile.createVariable("diab_y_308K", "f4", ("lat", "lon"))
            xvar[:] = u308_crop
            yvar[:] = v308_crop
            xvar.units = "m/s"
            yvar.units = "m/s"
            ncfile.close()
            print(f"    Saved diab_x and diab_y at 308 K to {output_dir_308K} as NetCDF file.")

    nc.close()

