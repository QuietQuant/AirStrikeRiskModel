import logging
import math
import os
from typing import Tuple, List, Optional

import rasterio
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import unary_union
import statsmodels.formula.api as smf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import plotly.graph_objects as go


def get_strike_database(datafile_path: str,
                        min_date: str, keep_air_drone_strike: bool, keep_shelling_artilleri_missile: bool) -> pd.DataFrame:
    """
    This function would load the data from the file and return a DataFrame
    """
    data: pd.DataFrame = pd.read_csv(datafile_path)

    # We filter only 'Air/drone strike' and 'Shelling/artillery/missile attack'
    subset = data[((data["sub_event_type"] == "Air/drone strike") & keep_air_drone_strike) | ((data["sub_event_type"] == "Shelling/artillery/missile attack") & keep_shelling_artilleri_missile)]
    # We filter "country" as "Ukraine"
    subset = subset[subset["country"] == "Ukraine"]

    # We filter the data from the min_date
    #subset = subset[subset["event_date"] >= min_date]
    # Problem, subset["event_date"] is a string, we need to convert it to datetime
    subset["event_date"] = pd.to_datetime(subset["event_date"])
    subset = subset[subset["event_date"] >= pd.to_datetime(min_date)]

    return subset



def sample_points_in_polygon(polygon: Polygon, num_points: int):
    """
    Sample random points inside a given polygon using rejection sampling.
    Returns a list of shapely Point objects.
    """
    points = []
    minx, miny, maxx, maxy = polygon.bounds
    attempts = 0
    max_attempts = num_points * 100  # safety threshold
    while len(points) < num_points and attempts < max_attempts:
        rand_x = np.random.uniform(minx, maxx)
        rand_y = np.random.uniform(miny, maxy)
        p = Point(rand_x, rand_y)
        if polygon.contains(p):
            points.append(p)
        attempts += 1
    if len(points) < num_points:
        print(f"Warning: Only sampled {len(points)} out of {num_points} requested points.")
    return points


def generate_negative_points(
        df_events: pd.DataFrame,
        ukraine_shp_path: str,
        admin_shp_path: str,
        sample_factor: float,
        buffer_km: float
) -> pd.DataFrame:
    """
    Generate negative (absence) points for each day.

    Overall, we want a constant number of negative points per day, computed as:
       negatives_per_day = ceil((total_positive_events * sample_factor) / total_days)

    For each day in the period:
      - If there are positive events, we build exclusion zones around them (using the specified buffer in km)
        and subtract these zones from the Ukraine boundary to get the safe zone.
      - If there are no events, the safe zone is the entire Ukraine boundary.
      - Then, we sample negatives_per_day points from the safe zone.
      - Finally, a spatial join with a GADM admin boundaries shapefile assigns admin names.

    Returns a DataFrame with columns: event_date, latitude, longitude, admin2, and admin3.
    """
    logger = logging.getLogger("GenerateNegativePoints")

    # Convert the buffer distance from kilometers to degrees (approximate; 1° ≈ 111 km)
    buffer_deg = buffer_km / 111.0

    # Load the Ukraine boundary shapefile.
    # Make sure the file path is correct and that the CRS is EPSG:4326.
    ukraine_gdf = gpd.read_file(ukraine_shp_path)
    if ukraine_gdf.crs is None or ukraine_gdf.crs.to_string() != "EPSG:4326":
        ukraine_gdf = ukraine_gdf.to_crs("EPSG:4326")

    # Load the GADM admin boundaries shapefile (for admin levels 1 and 2).
    admin_gdf = gpd.read_file(admin_shp_path)
    if admin_gdf.crs is None or admin_gdf.crs.to_string() != "EPSG:4326":
        admin_gdf = admin_gdf.to_crs("EPSG:4326")

    # Ensure event_date is datetime
    df_events['event_date'] = pd.to_datetime(df_events['event_date'])
    min_date = df_events['event_date'].min().date()
    max_date = df_events['event_date'].max().date()
    all_dates = pd.date_range(start=min_date, end=max_date, freq='D')

    total_positive = len(df_events)
    total_days = len(all_dates)
    negatives_per_day = math.ceil(total_positive * sample_factor / total_days)
    logger.info(f"Generating {negatives_per_day} negatives per day (total positives: {total_positive}, days: {total_days}).")

    negative_points_list = []

    # Get the full Ukraine boundary as a single geometry
    ukraine_union = ukraine_gdf.union_all()

    # Loop through each day
    for current_date in all_dates:
        logger.info(f"Processing {current_date}...")

        # Filter events for the current day
        df_day = df_events[df_events['event_date'].dt.date == current_date]
        num_events = len(df_day)

        if num_events == 0:
            # No events on this day: safe zone is the entire Ukraine
            safe_zone = ukraine_union
        else:
            # There are events: create exclusion zones around each event
            exclusion_polygons = []
            for _, row in df_day.iterrows():
                # Create a point (Point(longitude, latitude))
                pt = Point(row['longitude'], row['latitude'])
                circle = pt.buffer(buffer_deg)
                exclusion_polygons.append(circle)

            # Union all exclusion zones for the day
            exclusion_zone = unary_union(exclusion_polygons) if exclusion_polygons else None

            # Compute the safe zone by subtracting the exclusion zone from Ukraine
            if exclusion_zone is not None:
                safe_zone = ukraine_union.difference(exclusion_zone)
            else:
                safe_zone = ukraine_union

        if safe_zone.is_empty:
            logger.info(f"Safe zone is empty on {current_date}; skipping this day.")
            continue

        # Ensure safe_zone is iterable as polygons
        if isinstance(safe_zone, Polygon):
            safe_polygons = [safe_zone]
        elif isinstance(safe_zone, MultiPolygon):
            safe_polygons = list(safe_zone.geoms)
        else:
            logger.info(f"Unexpected geometry type on {current_date}: {type(safe_zone)}; skipping.")
            continue

        # To sample uniformly across multiple polygons, compute areas and weights
        areas = np.array([poly.area for poly in safe_polygons])
        total_area = areas.sum()
        if total_area == 0:
            logger.info(f"Safe zone area is zero on {current_date}; skipping.")
            continue
        weights = areas / total_area

        sampled_points = []
        for i in range(negatives_per_day):
            # Randomly choose one polygon based on its area weight
            chosen_poly = np.random.choice(safe_polygons, p=weights)
            pts = sample_points_in_polygon(chosen_poly, 1)
            if pts:
                sampled_points.append(pts[0])

        # Record the sampled negative points for the current day
        for pt in sampled_points:
            negative_points_list.append({
                'event_date': current_date,
                'latitude': pt.y,
                'longitude': pt.x
            })

    # Create DataFrame from negative points
    df_negatives = pd.DataFrame(negative_points_list)

    if df_negatives.empty:
        #print("No negative points generated.")
        logger.info(f"No negative points for {current_date}.")
        return pd.DataFrame(columns=["event_date", "latitude", "longitude", "admin2", "admin3"])
    else:
        #print("Negative points generated:", df_negatives.shape[0])
        logger.info(f"Negative points generated: {df_negatives.shape[0]}")

    # Convert negative points DataFrame to GeoDataFrame for spatial join
    gdf_neg = gpd.GeoDataFrame(
        df_negatives,
        geometry=gpd.points_from_xy(df_negatives["longitude"], df_negatives["latitude"]),
        crs=ukraine_gdf.crs
    )

    # Rename admin fields from the GADM shapefile if necessary.
    # Assume admin_gdf has columns 'NAME_1' (region) and 'NAME_2' (municipality)
    admin_gdf = admin_gdf.rename(columns={'NAME_1': 'admin2', 'NAME_2': 'admin3'})

    # Spatial join: assign each negative point its admin boundaries
    gdf_joined = gpd.sjoin(gdf_neg, admin_gdf[['admin2', 'admin3', 'geometry']], how='left', predicate='within')
    gdf_joined = gdf_joined.drop(columns=['index_right'])

    # Convert back to a DataFrame (dropping the geometry if not needed)
    df_final = pd.DataFrame(gdf_joined.drop(columns='geometry'))
    return df_final



def get_tile_code(lat: float, lon: float) -> str:
    """
    Given a latitude and longitude, compute the tile identifier based on the
    lower left corner of the 3x3 degree tile.

    Returns a string formatted as e.g. "S48E036".

    For latitude:
      - Compute tile_lat = floor(lat/3)*3.
      - Use "N" if tile_lat is >= 0, or "S" if negative, with two digits (absolute value).
    For longitude:
      - Compute tile_lon = floor(lon/3)*3.
      - Use "E" if tile_lon is >= 0, or "W" if negative, with three digits (absolute value).
    """
    # Compute lower left corner of the tile
    tile_lat = math.floor(lat / 3) * 3
    tile_lon = math.floor(lon / 3) * 3

    # Format latitude: Always two digits (e.g., "N05" or "S48")
    lat_prefix = "N" if tile_lat >= 0 else "S"
    lat_code = f"{abs(tile_lat):02d}"

    # Format longitude: Always three digits (e.g., "E036" or "W120")
    lon_prefix = "E" if tile_lon >= 0 else "W"
    lon_code = f"{abs(tile_lon):03d}"

    return f"{lat_prefix}{lat_code}{lon_prefix}{lon_code}"


def sample_tile_values(df: pd.DataFrame, tile_path: str) -> pd.Series:
    """
    Given a DataFrame of points (with 'latitude' and 'longitude') that belong
    to one tile, open the corresponding tile file and sample the land cover value.

    Parameters:
      df: DataFrame with columns 'latitude' and 'longitude'.
      tile_path: Path to the GeoTIFF file for that tile.

    Returns:
      A Pandas Series of sampled land cover values.
    """
    # Create a list of coordinates in (lon, lat) order.
    coords = list(zip(df['longitude'], df['latitude']))

    with rasterio.open(tile_path) as src:
        # rasterio.sample returns an iterator; each value is an array (for each band).
        samples = list(src.sample(coords))
        # Assuming one band, extract the first element of each sample.
        values = [sample[0] for sample in samples]

    return pd.Series(values, index=df.index)


def sample_landcover(df: pd.DataFrame, base_folder: str, year: int = 2021, version: str = "v200") -> pd.DataFrame:
    """
    For each row in the DataFrame (which must contain 'latitude' and 'longitude'),
    determine its tile and sample the corresponding land cover value.

    Assumes files are named following the pattern:
      ESA_WorldCover_10m_<YEAR>_<VERSION>_<TILE>_Map.tif

    Parameters:
      df: DataFrame with columns 'latitude' and 'longitude'.
      base_folder: Folder where the GeoTIFF files are stored.
      year: Observation year (default 2021).
      version: Product version (default "v200").

    Returns:
      The input DataFrame with an additional column 'landcover'.
    """
    # Create a tile code for each point.
    df = df.copy()
    df['tile'] = df.apply(lambda row: get_tile_code(row['latitude'], row['longitude']), axis=1)

    # Prepare a container for landcover values.
    landcover_values = pd.Series(index=df.index, dtype="int64")

    # Process each unique tile separately.
    for tile in df['tile'].unique():
        # Construct the filename based on the naming convention.
        # For example: "ESA_WorldCover_10m_2021_v200_S48E036_Map.tif"
        filename = f"ESA_WorldCover_10m_{year}_{version}_{tile}_Map.tif"
        tile_path = os.path.join(base_folder, filename)

        # Get indices for points that belong to this tile.
        idx = df[df['tile'] == tile].index

        # Check if file exists.
        if not os.path.exists(tile_path):
            print(f"Tile file {tile_path} does not exist. Skipping {len(idx)} points.")
            continue

        # Sample the tile values for those points.
        sub_df = df.loc[idx]
        sampled = sample_tile_values(sub_df, tile_path)
        landcover_values.loc[idx] = sampled

    # Add the sampled landcover values to the DataFrame.
    df['landcover'] = landcover_values
    return df



def change_land_cover_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Change the land cover label to a more human-readable format.
    """
    # Define a mapping from the original label to a new label
    landcover_mapping = {
        10: "Tree cover",
        20: "Shrubland",
        30: "Grassland",
        40: "Cropland",
        50: "Built-up",
        60: "Bare / sparse vegetation",
        70: "Snow and Ice",
        80: "Permanent water bodies",
        90: "Herbaceous wetland",
        95: "Mangroves",
        100: "Moss and lichen"
    }

    # Apply the mapping to the DataFrame
    df['landcover'] = df['landcover'].map(landcover_mapping)

    return df


def get_conflict_database(datafile_path: str) -> pd.DataFrame:
    """
    Load the conflict data from a CSV file and return it as a DataFrame.

    Parameters:
      datafile_path: Path to the CSV file containing the conflict data.

    Returns:
      A pandas DataFrame with the conflict data.
    """
    data: pd.DataFrame = pd.read_csv(datafile_path)

    # We do another subset, the sub_event_type is 'Government regains territory' 'Armed clash" and 'Non-state actor overtakes territory'
    subset = data[(data["sub_event_type"] == "Government regains territory") | (
            data["sub_event_type"] == "Non-state actor overtakes territory") | (
                          data["sub_event_type"] == "Armed clash")]
    # The countrye is 'Ukraine', Russia or Belarus
    subset = subset[
        (subset["country"] == "Ukraine") | (subset["country"] == "Russia") | (subset["country"] == "Belarus")]

    return subset


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great-circle distance between two points on Earth (in kilometers)
    using the Haversine formula.
    """
    R = 6371.0  # Earth radius in kilometers
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def compute_intensity_for_location_and_date(
        conflict_df: pd.DataFrame,
        location_lat: float,
        location_lon: float,
        current_date: pd.Timestamp,
        time_window: int,
        alpha: float,
        beta: float,
        use_fatalities: bool,
        use_limitation: bool
) -> float:
    """
    Compute the conflict intensity score at a given location and date.

    Each event in the conflict_df within [current_date - time_window, current_date] contributes:
      - Time decay: exp(-alpha * delta_days)
      - Distance decay: exp(-beta * (distance in km)**2)
      - Severity: fatalities (if use_fatalities is True and fatalities > 0) or 1.0 otherwise.

    Parameters:
      conflict_df: DataFrame with columns ['event_date', 'latitude', 'longitude', 'fatalities'].
      location_lat: Latitude of the point.
      location_lon: Longitude of the point.
      current_date: Date (pd.Timestamp) for which intensity is computed.
      time_window: Number of days in the past to include.
      alpha: Time decay parameter.
      beta: Distance decay parameter.
      use_fatalities: If True, multiply by fatalities when available.

    Returns:
      Conflict intensity score as a float.
    """
    # Ensure event_date column is datetime type.
    if conflict_df['event_date'].dtype != 'datetime64[ns]':
        conflict_df = conflict_df.copy()
        conflict_df['event_date'] = pd.to_datetime(conflict_df['event_date'])

    earliest_date: pd.Timestamp = current_date - pd.Timedelta(days=time_window)
    mask = (conflict_df['event_date'] >= earliest_date) & (conflict_df['event_date'] <= current_date)
    recent_events = conflict_df.loc[mask]

    intensity: float = 0.0
    for _, row in recent_events.iterrows():
        event_date: pd.Timestamp = row['event_date']
        event_lat: float = row['latitude']
        event_lon: float = row['longitude']
        fatalities: float = row.get('fatalities', 0)

        delta_days: float = (current_date - event_date).days
        time_weight: float = math.exp(-alpha * delta_days)

        distance: float = haversine_distance(location_lat, location_lon, event_lat, event_lon)
        distance_weight: float = math.exp(-beta * (distance ** 2))

        severity: float = fatalities if use_fatalities and fatalities > 0 else 1.0
        intensity += time_weight * distance_weight * severity

    if use_limitation:
        intensity = math.sqrt(intensity)

    return intensity


def add_intensity(events_df: pd.DataFrame, conflict_df: pd.DataFrame, day_half_life: int, km_half_life: int,
                  use_fatalities: bool, use_limitation: bool, column_name: str) -> None:
    ### WARNING: This function modifies the input DataFrame `events_df` in place

    time_window: int = day_half_life * 5
    # Computation base on half-life
    alpha: float = math.log(2) / day_half_life
    beta: float = math.log(2) / km_half_life

    # We add the "intensity" column
    events_df[column_name] = None

    #print("Conflict database read")

    events_df['event_date'] = pd.to_datetime(events_df['event_date'])

    # We asser taht we have no null latitude or longitude
    assert events_df["latitude"].isnull().sum() == 0
    assert events_df["longitude"].isnull().sum() == 0

    number_lignes: int = len(events_df)

    logger = logging.getLogger("ComputeIntensity")

    for i in range(len(events_df)):
        if i % 1000 == 0:
            #print(f"Compute Intensity - Processing {i} / {len(events_df)}")
            logger.info(f"Compute Intensity - Processing {i} / {len(events_df)}")

        event_date = events_df["event_date"].iloc[i]
        latitude: float = events_df["latitude"].iloc[i]
        longitude: float = events_df["longitude"].iloc[i]

        intensity: float = compute_intensity_for_location_and_date(
            conflict_df, latitude, longitude, event_date,
            time_window=time_window,
            alpha=alpha,
            beta=beta,
            use_fatalities=use_fatalities,
            use_limitation=use_limitation
        )

        #if i % 10 == 0:
        #    print(f"Intensity computed : {intensity}")

        assert not(intensity is None or math.isnan(intensity)), f"Intensity is None or NaN for {latitude}, {longitude}, {event_date}"

        # We assert that intensity is a float
        assert isinstance(intensity, float), f"Intensity is not a float but a {type(intensity)} for {latitude}, {longitude}, {event_date} and value is {intensity}"

        # events_df["intensity"].iloc[i] = intensity
        # Use `df.loc[row_indexer, "col"] = values` instead, to perform the assignment in a single step and ensure this keeps updating the original `df`.
        #events_df.loc[i, "intensity"] = intensity
        # I thin kteh previoul line is shit because it is base o nthe index, and index is not relaible in this df
        idx = events_df.index[i]
        events_df.loc[idx, column_name] = intensity

        #assert events_df["latitude"].isnull().sum() == 0, f"Latitude is null for {latitude}, {longitude}, {event_date}, {intensity}"
        #assert events_df["longitude"].isnull().sum() == 0, f"Longitude is null for {latitude}, {longitude}, {event_date}, {intensity}"
        #assert number_lignes == len(events_df), f"Number of lignes has changed from {number_lignes} to {len(events_df)}  for {latitude}, {longitude}, {event_date}, {intensity}"

    assert events_df["latitude"].isnull().sum() == 0
    assert events_df["longitude"].isnull().sum() == 0
    assert number_lignes == len(events_df)



def build_strike_probability_model(df: pd.DataFrame, with_admin2: bool, with_admin3: bool) -> sm.GLM:
    """
    Builds a logistic regression (binomial GLM) to predict whether a strike occurs
    on a given day-location based on intensity, region, seasonality, etc.
    """
    # Ensure event_date is a datetime
    if pd.api.types.is_datetime64_dtype(df['event_date']) == False:
        df['event_date'] = pd.to_datetime(df['event_date'])

    # Add month or day_of_year for seasonality
    df['month'] = df['event_date'].dt.month.astype('category')
    # or:
    # df['day_of_year'] = df['event_date'].dt.dayofyear
    # df['sin_day'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    # df['cos_day'] = np.cos(2 * np.pi * df['day_of_year'] / 365)

    # We'll assume df has:
    #   - strike_occurred (1 or 0)
    #   - intensity (float)
    #   - admin2 (region), admin3 (municipality)
    #   - month (categorical) or day_of_year columns

    # Example formula: Probability(strike) = logistic( beta0 + beta1*intensity + region effects + month effect + ... )
    formula = "strike ~ conflict_intensity + strike_intensity + C(landcover) + C(month)"
    if with_admin2:
        formula += " + C(admin2)"
    if with_admin3:
        formula += " + C(admin3)"

    # Fit a binomial GLM (logistic regression)
    model = smf.glm(formula=formula, data=df, family=sm.families.Binomial()).fit()

    #print(model.summary())
    return model


def plot_coefficient_summary(model, top_n: int, title: str, with_error: bool) -> plt.Figure:
    """
    Create a bar plot of coefficients (with error bars) for a fitted statsmodels model.

    Parameters:
      model: A fitted statsmodels model (e.g., GLM or Logit).
      top_n: How many coefficients to display, based on absolute size.
             If you have hundreds, you may want to limit to the largest few.
    """
    # Extract coefficients and standard errors
    coefs = model.params
    errs = model.bse

    # Compute 95% confidence intervals
    ci_lower = coefs - 1.96 * errs
    ci_upper = coefs + 1.96 * errs

    # Build a DataFrame to sort/filter
    df_plot = pd.DataFrame({
        "coef": coefs,
        "err": errs,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper
    })

    # The index contains the parameter names
    df_plot["param"] = df_plot.index

    # Sort by absolute value of coefficient and pick top_n
    df_plot = df_plot.reindex(df_plot["coef"].abs().sort_values(ascending=False).index)
    df_plot_top = df_plot.head(top_n).iloc[::-1]  # reverse to plot from smallest to largest in the bar chart

    # Plot
    fig, ax = plt.subplots(figsize=(8, 0.4 * top_n))  # dynamic height
    y_positions = np.arange(len(df_plot_top))

    if with_error:
        # Horizontal bar for each coefficient
        ax.errorbar(
            x=df_plot_top["coef"],
            y=y_positions,
            xerr=[df_plot_top["coef"] - df_plot_top["ci_lower"], df_plot_top["ci_upper"] - df_plot_top["coef"]],
            fmt='o', capsize=3, color='blue'
        )
    else:
        ax.barh(y_positions, df_plot_top["coef"], color='blue')

    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(df_plot_top["param"])
    ax.set_xlabel("Coefficient (95% CI)")
    #ax.set_title("Top {} Coefficients by Absolute Value".format(top_n))
    ax.set_title(title)

    plt.tight_layout()
    plt.show(renderer='browser')

    return fig



def create_grid(
        min_lat: float,
        max_lat: float,
        min_lon: float,
        max_lon: float,
        resolution: float
) -> Tuple[List[float], List[float]]:
    """
    Create a grid of points over the specified latitude and longitude range.

    Parameters:
      min_lat: Minimum latitude.
      max_lat: Maximum latitude.
      min_lon: Minimum longitude.
      max_lon: Maximum longitude.
      resolution: Grid resolution in degrees (e.g., 0.1).

    Returns:
      A tuple of two lists:
        - List of latitudes.
        - List of longitudes.
      Each pair (lat_list[i], lon_list[i]) is a grid point.
    """
    lat_values = np.arange(min_lat, max_lat + resolution, resolution)
    lon_values = np.arange(min_lon, max_lon + resolution, resolution)
    grid_lats: List[float] = []
    grid_lons: List[float] = []

    for lat in lat_values:
        for lon in lon_values:
            grid_lats.append(float(lat))
            grid_lons.append(float(lon))

    return grid_lats, grid_lons


def filter_grid_points_in_ukraine(
        grid_df: pd.DataFrame,
        shapefile_path: str
) -> pd.DataFrame:
    """
    Filter the grid DataFrame to keep only those points located within the
    Ukrainian boundary (provided by the shapefile).

    Parameters:
      - grid_df: a DataFrame with columns 'latitude' and 'longitude'.
      - shapefile_path: path to a shapefile or GeoJSON containing Ukraine's boundary.

    Returns:
      - A filtered DataFrame (subset of grid_df) where each point is in Ukraine.
    """
    # 1) Load the Ukraine polygon(s)
    ukraine_gdf = gpd.read_file(shapefile_path)

    # Ensure the boundary is in WGS84 if your points are lat/lon
    ukraine_gdf = ukraine_gdf.to_crs(epsg=4326)

    # 2) Convert your grid points to a GeoDataFrame
    geometry = gpd.points_from_xy(grid_df["longitude"], grid_df["latitude"])
    grid_gdf = gpd.GeoDataFrame(grid_df.copy(), geometry=geometry, crs="EPSG:4326")

    # 3) Spatial join to keep points that intersect the Ukraine polygon
    #    'inner' join means keep only matching rows (points inside Ukraine).
    #    If the shapefile has multiple polygons, it will handle that automatically.
    filtered_gdf = gpd.sjoin(grid_gdf, ukraine_gdf, how="inner", predicate="intersects")

    # 4) Return a plain DataFrame (dropping the geometry if you want)
    return filtered_gdf.drop(columns=["geometry", "index_right"])


def generate_ukrain_grid(resolution: float, ukr_shapefile: str) -> pd.DataFrame:
    """
    Generate a grid of points covering Ukraine, with a specified resolution.

    Parameters:
      - resolution: the spacing between grid points in degrees.

    Returns:
      - A DataFrame with columns 'latitude' and 'longitude'.
    """
    # Define the bounding box for Ukraine
    min_lat, max_lat = 44, 53
    min_lon, max_lon = 20, 43

    # Create the grid points
    grid_lats, grid_lons = create_grid(min_lat, max_lat, min_lon, max_lon, resolution)

    # Create a DataFrame with the grid points
    data = {"latitude": grid_lats, "longitude": grid_lons}
    grid_df = pd.DataFrame(data)

    # Filter the grid points to keep only those in Ukraine
    filtered_df = filter_grid_points_in_ukraine(grid_df, ukr_shapefile)

    return filtered_df


def randomly_remove(df : pd.DataFrame, percentage: float) -> pd.DataFrame:
    """
    Randomly remove a percentage of the dataframe
    """
    return df.sample(frac=1 - percentage)


def merge_positive_negative(df_pos: pd.DataFrame, df_neg: pd.DataFrame) -> pd.DataFrame:
    """
    Merge positive and negative events
    """
    df_pos['strike'] = 1
    df_neg['strike'] = 0

    result = pd.concat([df_pos, df_neg], ignore_index=True)

    assert result["latitude"].isnull().sum() == 0
    assert result["longitude"].isnull().sum() == 0

    return result


def plot_grid_plotly(
        grid_df: pd.DataFrame,
        value_column: str,
        resolution: float,
        title: str,
        legend_title: str,
        colorscale: str = "Reds",
        lat_range: Optional[Tuple[float, float]] = None,
        lon_range: Optional[Tuple[float, float]] = None,
) -> go.Figure:
    """
    Plots a grid of cells on a map using Plotly. Each row of grid_df is assumed to provide
    the center of a cell. The cell is taken to be a square of size resolution x resolution degrees.

    Parameters:
      - grid_df: DataFrame with columns 'latitude', 'longitude', and the column to color (default "intensity").
      - value_column: Name of the column used for coloring each cell.
      - lat_range: Optional tuple (min_lat, max_lat) for the latitude axis; computed if not provided.
      - lon_range: Optional tuple (min_lon, max_lon) for the longitude axis; computed if not provided.
      - resolution: The size (in degrees) of each grid cell (cell is centered on the provided lat/lon).
      - title: The graph title.
      - legend_title: The title to appear on the colorbar.
      - colorscale: The Plotly colorscale to use (default "Reds").
    """
    # Compute latitude/longitude range if not provided
    if lat_range is None:
        lat_min = grid_df['latitude'].min()
        lat_max = grid_df['latitude'].max()
        margin_lat = (lat_max - lat_min) * 0.1
        lat_range = (lat_min - margin_lat, lat_max + margin_lat)
    if lon_range is None:
        lon_min = grid_df['longitude'].min()
        lon_max = grid_df['longitude'].max()
        margin_lon = (lon_max - lon_min) * 0.1
        lon_range = (lon_min - margin_lon, lon_max + margin_lon)

    # Build GeoJSON features from each grid cell.
    # Each row’s lat/lon is the center of a square cell of side 'resolution'.
    features = []
    for idx, row in grid_df.iterrows():
        lat = row['latitude']
        lon = row['longitude']
        half_res = resolution / 2.0
        # Define the square corners (in [lon, lat] order)
        lon_min_cell = lon - half_res
        lon_max_cell = lon + half_res
        lat_min_cell = lat - half_res
        lat_max_cell = lat + half_res
        polygon = [
            [lon_min_cell, lat_min_cell],
            [lon_min_cell, lat_max_cell],
            [lon_max_cell, lat_max_cell],
            [lon_max_cell, lat_min_cell],
            [lon_min_cell, lat_min_cell]  # close the polygon
        ]
        feature = {
            "type": "Feature",
            "id": str(idx),
            "geometry": {
                "type": "Polygon",
                "coordinates": [polygon]
            },
            "properties": {
                value_column: row[value_column]
            }
        }
        features.append(feature)
    geojson_data = {
        "type": "FeatureCollection",
        "features": features
    }

    # Build the choropleth trace. We use the DataFrame's index (converted to string) as the feature ids.
    fig = go.Figure(go.Choropleth(
        geojson=geojson_data,
        locations=grid_df.index.astype(str),
        z=grid_df[value_column].astype(float),
        colorscale=colorscale,
        colorbar_title=legend_title,
        marker_line_color='black',
        marker_line_width=0.5
    ))

    # Compute map center
    center_lat = (lat_range[0] + lat_range[1]) / 2.0
    center_lon = (lon_range[0] + lon_range[1]) / 2.0

    # Update geographic layout
    fig.update_geos(
        visible=False,
        resolution=50,
        showcountries=True,
        lataxis_range=[lat_range[0], lat_range[1]],
        lonaxis_range=[lon_range[0], lon_range[1]],
        center=dict(lat=center_lat, lon=center_lon),
        projection_type="natural earth"
    )

    fig.update_layout(
        title=title,
        geo=dict(projection_type='natural earth')
    )

    # Show in browser
    fig.show(renderer='browser')

    return fig