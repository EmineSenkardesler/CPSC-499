import geopandas as gpd
import xarray as xr
import rioxarray
from rasterstats import zonal_stats
import pandas as pd
import gcsfs

# 1) Load IL counties
counties = gpd.read_file("https://www2.census.gov/geo/tiger/TIGER2022/COUNTY/tl_2022_us_county.zip")
#counties_il = counties[counties.STATEFP == "17"].to_crs("EPSG:4326")

# 2) Open the ARCO-ERA5 Zarr
ar_zarr = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
fs = gcsfs.GCSFileSystem(token="anon")
ds = xr.open_zarr(ar_zarr, consolidated=True, storage_options={"token":"anon"})

# 3) Subset to IL bounding box & years 2008–2024
lat_min, lon_min, lat_max, lon_max = counties.total_bounds[1], counties.total_bounds[0], \
                                     counties.total_bounds[3], counties.total_bounds[2]
print(f"Bounding box: lon_min={lon_min}, lon_max={lon_max}, lat_min={lat_min}, lat_max={lat_max}")

# Handle 0–360 longitude
if lon_min < 0:
    lon_min = lon_min + 360
    lon_max = lon_max + 360

# Subset data (lazy evaluation)
da = (
    ds["2m_temperature"]
    .sel(
        time=slice("2008-01-01", "2025-05-31"),
        latitude=slice(lat_max, lat_min),  # Descending latitude
        longitude=slice(lon_min, lon_max)
    )
)

# Verify subset
print(da)

# 4) Set spatial dims and CRS (still lazy)
da = (
    da
    .rio.set_spatial_dims(x_dim="longitude", y_dim="latitude")
    .rio.write_crs("EPSG:4326")
)

# 5) Compute monthly means (lazy computation)
da_monthly = da.resample(time="1MS").mean(keep_attrs=True)

# 6) Zonal-stats for each month (compute in chunks to save memory)
records = []
for ts in da_monthly.time.values:
    # Select and load one month at a time to reduce memory usage
    arr = da_monthly.sel(time=ts).compute()  # Compute only this month's data

    # Check for valid data
    if arr.isnull().all():
        print(f"Skipping {ts} due to no valid data")
        continue

    # Re-tag spatial dims & CRS
    arr = (
        arr
        .rio.set_spatial_dims(x_dim="longitude", y_dim="latitude")
        .rio.write_crs("EPSG:4326")
    )

    try:
        affine = arr.rio.transform()
        stats = zonal_stats(
            counties,
            arr.values,
            affine=affine,
            stats=["mean"],
            nodata=float("nan")
        )
        for ct, st in zip(counties.GEOID, stats):
            records.append({
                "county_fips": ct,
                "time": pd.to_datetime(ts),
                "t2m_mean_K": st["mean"]
            })
    except Exception as e:
        print(f"Error processing {ts}: {e}")
        continue

# 7) Convert to DataFrame and save output
df = pd.DataFrame(records)
df.to_csv("county_monthly_2m_temperature_2008-2024.csv", index=False)
print("Done! → county_monthly_2m_temperature_2008-2024.csv")