import fsspec
import xarray as xr
import scipy.spatial
import numpy as np
import argparse
from datetime import date
import metpy.calc as mpcalc
from metpy.units import units

# This script is a consolidation of several approaches to download and process
# ERA5 data from the Google Cloud ARCO-ERA5 dataset.
# It handles different variables, calculates derived quantities, and correctly
# processes geographical coordinates.
# Original source of inspiration: https://github.com/google-research/arco-era5/blob/main/docs/0-Surface-Reanalysis-Walkthrough.ipynb

def build_triangulation(x, y):
    """
    Creates a Delaunay tesselation
    
    """
    grid = np.stack([x, y], axis=1)
    return scipy.spatial.Delaunay(grid)

def interpolate(data, tri, mesh):
    """
    Interpolates the ERA5 grid using the Delaunay tesselation
    
    """
    indices = tri.find_simplex(mesh)
    ndim = tri.transform.shape[-1]
    T_inv = tri.transform[indices, :ndim, :]
    r = tri.transform[indices, ndim, :]
    c = np.einsum('...ij,...j', T_inv, mesh - r)
    c = np.concatenate([c, 1 - c.sum(axis=-1, keepdims=True)], axis=-1)
    result = np.einsum('...i,...i', data[:, tri.simplices[indices]], c)
    return np.where(indices == -1, np.nan, result)

def era5_processing(variable, year_start, year_end, dataset, lat_min, lat_max, lon_min, lon_max):
    """
    Code to process ERA5 Data over a specified bounding box.
    Source: https://github.com/google-research/arco-era5 
    
    Inputs:
        - variable (str) - Variable name to call, using ERA5 data names.
                           User can also call "vapor_pressure" (mb), "sfcWind" (m/s), 
                           and "relative_humidity" (decimal).
        - year_start (int) - First year for the data request.
        - year_end (int) - Last year for the data request (inclusive).
        - dataset (str) - Either "raw" or "analysis_ready" from ARCO-ERA5.
        - lat_min (float) - Minimum latitude.
        - lat_max (float) - Maximum latitude.
        - lon_min (float) - Minimum longitude (can be negative).
        - lon_max (float) - Maximum longitude (can be negative).
    Outputs:
        - fin_array (Dataarray) - Dataarray with appropriate ERA5 data for the 
                                  variable and timeframe chosen.
    """
    # Test bucket access
    fs = fsspec.filesystem('gs')
    fs.ls('gs://gcp-public-data-arco-era5/co/')
    
    # Define variables needed for special calculations
    calc = ''
    if variable == 'vapor_pressure':
        required_vars = ['2m_dewpoint_temperature']
        calc = 'vapor_pressure'
    elif variable == 'sfcWind':
        required_vars = ['10m_u_component_of_wind', '10m_v_component_of_wind']
        calc = 'sfcWind'
    elif variable == 'relative_humidity':
        required_vars = ['2m_temperature', '2m_dewpoint_temperature']
        calc = 'relative_humidity'
    else:
        required_vars = [variable]
        
    if dataset == 'raw':
        # Opening dataset with zarr
        reanalysis = xr.open_zarr(
            'gs://gcp-public-data-arco-era5/co/single-level-reanalysis.zarr', 
            chunks={'time': 48},
            consolidated=True,
            )
    
    if dataset == 'analysis_ready':
        # Opening dataset with zarr
        reanalysis = xr.open_zarr(
            'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3', 
            chunks={'time': 48},
            consolidated=True,
            )
    else:
        raise ValueError("`dataset` must be either 'raw' or 'analysis_ready'")

    # Dates
    i_date = str(year_start) + '-01-01'
    f_date = str(year_end) + '-12-31'
    
    # Handle longitude conversion for datasets on a 0-360 degree scale
    lon_min_360 = lon_min if lon_min >= 0 else lon_min + 360
    lon_max_360 = lon_max if lon_max >= 0 else lon_max + 360

    # Select time slice first
    time_slice = reanalysis.sel(time=slice(i_date, f_date))
    
    # Select desired variables and geographical area
    # Note: For descending latitude coordinates, slice(lat_max, lat_min) is correct.
    illinois_ds = time_slice[required_vars].where(
        (time_slice.longitude >= lon_min_360) & (time_slice.latitude >= lat_min) &
        (time_slice.longitude <= lon_max_360) & (time_slice.latitude <= lat_max),
        drop=True
    )
    
    # The 'raw' dataset requires interpolation to a regular grid.
    if dataset == 'raw':
        print("Raw dataset selected. Performing Delaunay triangulation...")
        # Note: This part of the code for interpolation was not fully tested in the notebooks.
        # It is preserved from the original era5.py script.
        tri = build_triangulation(illinois_ds.longitude, illinois_ds.latitude)
        longitude = np.linspace(lon_min_360, lon_max_360, num=round(lon_max_360 - lon_min_360) * 4 + 1)
        latitude = np.linspace(lat_min, lat_max, num=round(lat_max - lat_min) * 4 + 1)
            
        mesh = np.stack(np.meshgrid(longitude, latitude, indexing='ij'), axis=-1)
        
        # This part assumes a single variable for interpolation; may need adjustment for multi-variable calculations
        if len(required_vars) > 1:
            print(f"Warning: Interpolation for multi-variable calculation ({calc}) is not implemented. Only the first variable '{required_vars[0]}' will be interpolated.")
        
        mesh_int = interpolate(illinois_ds[required_vars[0]].values, tri, mesh)
        
        fin_array = xr.DataArray(mesh_int, 
                             coords=[('time', illinois_ds.time.data), ('longitude', longitude), ('latitude', latitude)])
    else:
        # The 'analysis_ready' dataset is already on a regular grid.
        fin_array = illinois_ds
        
    fin_array = fin_array.rename({'longitude':'lon', 'latitude':'lat'})
    print("Data loaded and subsetted:")
    print(fin_array)
    
    # Perform calculations using MetPy
    if calc == 'vapor_pressure':
        print("Calculating vapor pressure...")
        # Calculation requires units
        dewpoint = fin_array * units.kelvin
        fin_array = mpcalc.vapor_pressure(dewpoint).to('mbar')
    elif calc == 'sfcWind':
        print("Calculating surface wind speed...")
        u_wind = fin_array['10m_u_component_of_wind'] * units('m/s')
        v_wind = fin_array['10m_v_component_of_wind'] * units('m/s')
        fin_array = mpcalc.wind_speed(u_wind, v_wind)
    elif calc == 'relative_humidity':
        print("Calculating relative humidity...")
        temp = fin_array['2m_temperature'] * units.K
        dewpoint = fin_array['2m_dewpoint_temperature'] * units.K
        fin_array = mpcalc.relative_humidity_from_dewpoint(temp, dewpoint)
    
    return fin_array

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and process ERA5 data from Google Cloud.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--variable", required=True, type=str, 
                        help="Variable to download.\n"
                             "Can be a standard ERA5 variable name (e.g., '2m_temperature')\n"
                             "or a derived variable: 'vapor_pressure', 'sfcWind', 'relative_humidity'.")
    parser.add_argument("--year_start", required=True, type=int, help="Start year.")
    parser.add_argument("--year_end", required=True, type=int, help="End year.")
    parser.add_argument("--out_path", required=True, type=str, help="Path to save the output NetCDF file.")
    parser.add_argument("--dataset", required=True, type=str, help="Dataset to use: 'raw' or 'analysis_ready'.")
    parser.add_argument("--lat_min", required=True, type=float, help="Minimum latitude of bounding box.")
    parser.add_argument("--lat_max", required=True, type=float, help="Maximum latitude of bounding box.")
    parser.add_argument("--lon_min", required=True, type=float, help="Minimum longitude of bounding box (e.g., -91.5).")
    parser.add_argument("--lon_max", required=True, type=float, help="Maximum longitude of bounding box (e.g., -87.0).")
    
    args = parser.parse_args()

    dataarray = era5_processing(
        args.variable, 
        args.year_start, 
        args.year_end, 
        args.dataset,
        args.lat_min,
        args.lat_max,
        args.lon_min,
        args.lon_max
    )
    
    # Saving the dataset
    output_var_name = args.variable.replace('_', '-')
    output_file = (f"{args.out_path}/ERA5_{output_var_name}_{args.year_start}-{args.year_end}_"
                   f"{args.dataset}_{date.today()}.nc")
                   
    dataarray.to_netcdf(output_file)
    print(f"\nSuccess! Dataset saved to {output_file}")