#!/usr/bin/env python3
"""portUrb supercell output (supercell_*.nc) -> oblique 3-D videos (no GPU needed).

Draws, for each output file:
  - storm outline: isosurface of total condensate mixing ratio
    (cloud_water + rain_water + cloud_ice + snow + graupel) / density_dry, coloured by height
  - updraft core: semi-transparent red isosurface of wvel
  - floor: column maximum of precipitation mixing ratio (rain + graupel), radar-like colours
The vertical axis is exaggerated (VERT_EXAG) for display only.

Usage:
    python viz_oblique_portUrb.py "<nc_glob>" <out_prefix> [--cond-gkg 0.5] [--w-ms 10]
Writes <out_prefix>_evolution.mp4 (one frame per file), <out_prefix>_rotation.mp4
(last file, 360-degree orbit) and <out_prefix>_final.png.

Rendering: PyVista off-screen (OSMesa); MP4 via imageio-ffmpeg. Tested with the
500x500x50 / 400 m / 2 h run on the RTX 3090 host (21 files, 2026-09-26).
"""
import argparse
import glob
import time

import imageio.v2 as imageio
import netCDF4 as nc
import numpy as np
import pyvista as pv

pv.OFF_SCREEN = True

VERT_EXAG = 4.0
CONDENSATE = ("cloud_water", "rain_water", "cloud_ice", "snow", "graupel")
PRECIP = ("rain_water", "graupel")


def load(path):
    """Return dict with grid info and the fields needed for one frame (g/kg, m/s)."""
    d = nc.Dataset(path)
    x = np.asarray(d.variables["x"][:], dtype=np.float64)
    y = np.asarray(d.variables["y"][:], dtype=np.float64)
    z = np.asarray(d.variables["z"][:], dtype=np.float64)
    rho = np.asarray(d.variables["density_dry"][:], dtype=np.float32)
    cond = np.zeros_like(rho)
    for name in CONDENSATE:
        cond += np.asarray(d.variables[name][:], dtype=np.float32)
    precip = np.zeros_like(rho)
    for name in PRECIP:
        precip += np.asarray(d.variables[name][:], dtype=np.float32)
    w = np.asarray(d.variables["wvel"][:], dtype=np.float32)
    etime = float(d.getncattr("etime")) if "etime" in d.ncattrs() else float("nan")
    d.close()
    # mass density (kg/m^3) -> mixing ratio (g/kg)
    cond_gkg = 1000.0 * cond / rho
    precip_gkg = 1000.0 * precip / rho
    return dict(x=x, y=y, z=z, cond=cond_gkg, precip=precip_gkg, w=w, etime=etime)


def build_scene(f, cond_gkg, w_ms, label):
    x, y, z = f["x"], f["y"], f["z"]
    nx, ny, nz = x.size, y.size, z.size
    dx = float(x[1] - x[0]); dy = float(y[1] - y[0]); dz = float(z[1] - z[0])
    grid = pv.ImageData(dimensions=(nx, ny, nz), spacing=(dx, dy, dz * VERT_EXAG),
                        origin=(float(x[0]), float(y[0]), float(z[0]) * VERT_EXAG))
    grid.point_data["cond"] = f["cond"].transpose(2, 1, 0).flatten(order="F")
    grid.point_data["w"] = f["w"].transpose(2, 1, 0).flatten(order="F")
    grid.point_data["height_km"] = grid.points[:, 2] / VERT_EXAG / 1000.0

    pl = pv.Plotter(off_screen=True, window_size=(1280, 960))
    pl.set_background("black")

    floor = pv.ImageData(dimensions=(nx, ny, 1), spacing=(dx, dy, 1),
                         origin=(float(x[0]), float(y[0]), -dz * VERT_EXAG * 0.5))
    floor.point_data["precip_col"] = np.max(f["precip"], axis=0).flatten(order="F")
    pl.add_mesh(floor, scalars="precip_col", cmap="turbo", clim=[0, 8],
                show_scalar_bar=False, opacity=0.55)

    try:
        storm = grid.contour(isosurfaces=[cond_gkg], scalars="cond")
        if storm.n_points > 0:
            pl.add_mesh(storm, scalars="height_km", cmap="viridis", clim=[0, 15],
                        show_scalar_bar=True,
                        scalar_bar_args={"title": "height km", "color": "white"},
                        smooth_shading=True, specular=0.3)
    except Exception:
        pass
    try:
        updraft = grid.contour(isosurfaces=[w_ms], scalars="w")
        if updraft.n_points > 0:
            pl.add_mesh(updraft, color="red", opacity=0.45, smooth_shading=True)
    except Exception:
        pass

    pl.add_text(label, position="upper_left", font_size=11, color="white")
    pl.add_text(f"cloud: condensate {cond_gkg:g} g/kg (colour = height)   red: w > {w_ms:g} m/s   "
                f"floor: column-max rain+graupel g/kg", position="lower_left", font_size=9, color="white")
    return pl, (nx * dx, ny * dy, nz * dz * VERT_EXAG)


def set_oblique_camera(pl, extent, azimuth_deg, elevation_deg=32.0, dist_factor=1.9):
    lx, ly, lz = extent
    cx, cy, cz = lx * 0.5, ly * 0.5, lz * 0.25
    r = lx * dist_factor
    az = np.deg2rad(azimuth_deg); el = np.deg2rad(elevation_deg)
    pl.camera_position = [(cx + r * np.cos(el) * np.cos(az), cy + r * np.cos(el) * np.sin(az),
                           cz + r * np.sin(el)), (cx, cy, cz), (0, 0, 1)]
    pl.camera.view_angle = 35


def write_mp4(frames, path, fps):
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=8, macro_block_size=None) as wr:
        for fr in frames:
            wr.append_data(fr)
    print(f"  wrote {path} ({len(frames)} frames @ {fps} fps)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("nc_glob")
    ap.add_argument("out_prefix")
    ap.add_argument("--cond-gkg", type=float, default=0.5, help="condensate isosurface (g/kg)")
    ap.add_argument("--w-ms", type=float, default=10.0, help="updraft isosurface (m/s)")
    ap.add_argument("--title", default="portUrb WK1982 supercell")
    ap.add_argument("--rot-frames", type=int, default=72)
    args = ap.parse_args()

    files = sorted(glob.glob(args.nc_glob))
    if not files:
        raise SystemExit(f"no files match {args.nc_glob}")
    print(f"{len(files)} files", flush=True)

    frames = []
    t0 = time.time()
    last = None
    for i, path in enumerate(files):
        f = load(path)
        last = f
        hours = f["etime"] / 3600.0
        label = f"t = {hours:.2f} h   {args.title}   {f['x'].size}x{f['y'].size}x{f['z'].size} @ {f['x'][1]-f['x'][0]:.0f} m (vert x{VERT_EXAG:.0f})"
        pl, extent = build_scene(f, args.cond_gkg, args.w_ms, label)
        set_oblique_camera(pl, extent, azimuth_deg=220 + i * 2.5)
        frames.append(pl.screenshot(return_img=True))
        if i == len(files) - 1:
            pl.screenshot(f"{args.out_prefix}_final.png")
        pl.close()
        print(f"  frame {i+1}/{len(files)} t={hours:.2f} h  ({time.time()-t0:.0f} s)", flush=True)
    write_mp4([fr for fr in frames for _ in range(4)], f"{args.out_prefix}_evolution.mp4", fps=6)

    hours = last["etime"] / 3600.0
    label = f"t = {hours:.2f} h   {args.title}   (vert x{VERT_EXAG:.0f})"
    pl, extent = build_scene(last, args.cond_gkg, args.w_ms, label)
    rot = []
    for i in range(args.rot_frames):
        set_oblique_camera(pl, extent, azimuth_deg=360.0 * i / args.rot_frames)
        rot.append(pl.screenshot(return_img=True))
    pl.close()
    write_mp4(rot, f"{args.out_prefix}_rotation.mp4", fps=24)
    print("done", flush=True)


if __name__ == "__main__":
    main()
