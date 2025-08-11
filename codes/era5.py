import fsspec
import xarray as xr
import scipy.spatial
import numpy as np
import os
import argparse
from datetime import date
from calculations.calculations import vapor_pressure
from calculations.calculations import wind_tot
import metpy
from metpy.units import units 


# Using code from https://github.com/google-research/arco-era5/blob/main/docs/0-Surface-Reanalysis-Walkthrough.ipynb 

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

def era5_processing(variable, year_start, year_end, dataset):
    """
    Code to process ERA5 Data over the state of Illinois
    https://github.com/google-research/arco-era5 
    
    Inputs:
        - variable (str) - Variable name to call, using ERA5 data names
                           User can also call "vapor_pressure" (mb), "sfcWind" (m/s) and "relative_humidity" (decimal)
        - year_start (int) - First year you'd like to request
        - year_end (int) - Last year you'd like to request (inclusive)
        - dataset (str) - Either use the "raw" or "analysis_ready" dataset from ARCO-ERA5 (for specifics, see 
                            Github link)
    Outputs:
        - fin_array (Dataarray) - Dataarray with appropriate ERA5 data for the variable and timeframe chosen,
                                   interpolated using Delaunay triangulation
        
    """
    # Test bucket access
    fs = fsspec.filesystem('gs')
    fs.ls('gs://gcp-public-data-arco-era5/co/')
    
    # Pulling appropriate variables for calculations        
    if variable == 'vapor_pressure':
        variable = '2m_dewpoint_temperature'
        calc = 'vapor_pressure'
    elif variable == 'sfcWind':
        variable = ['10m_u_component_of_wind','10m_v_component_of_wind']
        calc = 'sfcWind'
    elif variable == 'relative_humidity':
        variable = ['2m_temperature','2m_dewpoint_temperature']
        calc = 'relative_humidity'
    else:
        calc = ''
        

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
        print('Done')

    # Dates
    i_date = str(year_start) + '-01-01'
    f_date = str(year_end)   + '-12-31'
    
    recent_an = reanalysis.sel(time=slice(i_date, f_date))

    era5_var = recent_an[variable]
    
    lon_min = 267.2
    lon_max = 274
    lat_min = 36
    lat_max = 43.5
    
    illinois_ds = era5_var.where(
    (recent_an.longitude > lon_min) & (recent_an.latitude > lat_min) &
    (recent_an.longitude < lon_max) & (recent_an.latitude < lat_max),
    drop=True)
    
    if dataset == 'raw':
        tri = build_triangulation(illinois_ds.longitude, illinois_ds.latitude)
        longitude = np.linspace(lon_min, lon_max, num=round(lon_max-lon_min)*4+1)
        latitude = np.linspace(lat_min, lat_max, num=round(lat_max-lat_min)*4+1)
            
        mesh = np.stack(np.meshgrid(longitude, latitude, indexing='ij'), axis=-1)
        mesh_int = interpolate(illinois_ds[variable].values, tri, mesh)
        
        fin_array = xr.DataArray(mesh_int, 
                             coords=[('time', illinois_ds.time.data), ('longitude', longitude), ('latitude', latitude)])
    else:
        fin_array = illinois_ds
        
    fin_array = fin_array.rename({'longitude':'lon', 'latitude':'lat'})
    print(fin_array)
    
    # Calculations
    if calc=='vapor_pressure':
        fin_array = vapor_pressure(fin_array) # Calculation
    elif calc=='sfcWind':
        fin_array,_ = wind_tot(fin_array['10m_u_component_of_wind'], fin_array['10m_v_component_of_wind'])
    elif calc=='relative_humidity':
        fin_array = metpy.calc.relative_humidity_from_dewpoint(
                                        fin_array['2m_temperature'] * units.K,
                                        fin_array['2m_dewpoint_temperature'] * units.K)
    
    return fin_array

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variable", required=True, type=str)
    parser.add_argument("--year_start", required=True, type=int)
    parser.add_argument("--year_end", required=True, type=int)
    parser.add_argument("--out_path", required=True, type=str)
    parser.add_argument("--dataset", required=True, type=str)
    args = parser.parse_args()

    variable = args.variable
    year_start = args.year_start
    year_end = args.year_end
    out_path = args.out_path
    dataset = args.dataset
    
    dataarray = era5_processing(variable, year_start, year_end, dataset)
    
    # Saving the dataset
    output_file = (out_path + '/ERA5_IL_' + variable + '_' + str(year_start) + '-' + str(year_end) + '_' + 
                   str(date.today()) + '.nc')
    dataarray.to_netcdf(output_file)
    print('Dataset saved to ' + output_file)