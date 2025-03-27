import datetime
import json
import logging
import os

import pandas as pd

from glm_model.Functions import get_strike_database, generate_negative_points, sample_landcover, change_land_cover_label, \
    get_conflict_database, add_intensity, merge_positive_negative, randomly_remove

# Conflict intensity computation parameters
conflict_day_half_life = int(os.environ.get("CONFLICT_DAY_HALF_LIFE", 15))
conflict_km_half_life = int(os.environ.get("CONFLICT_KM_HALF_LIFE", 200))
conflict_use_fatalities = os.environ.get("CONFLICT_USE_FATALITIES", "False").lower() in ("true", "1", "yes")
conflict_use_limitation = os.environ.get("CONFLICT_USE_LIMITATION", "True").lower() in ("true", "1", "yes")

# Strike intensity computation parameters
strike_day_half_life = int(os.environ.get("STRIKE_DAY_HALF_LIFE", 3))
strike_km_half_life = int(os.environ.get("STRIKE_KM_HALF_LIFE", 50))
strike_use_fatalities = os.environ.get("STRIKE_USE_FATALITIES", "False").lower() in ("true", "1", "yes")
strike_use_limitation = os.environ.get("STRIKE_USE_LIMITATION", "True").lower() in ("true", "1", "yes")

# Data selection parameters
min_date = os.environ.get("MIN_DATE", "2022-02-24")
percentage_to_remove = float(os.environ.get("PERCENTAGE_TO_REMOVE", 0.40))
datafile_path: str = os.environ.get("EVENT_DATABASE_PATH", "Input/Europe-Central-Asia_2018-2025_Mar07.csv")
keep_air_drone_strike: bool = os.environ.get("KEEP_STRIKE", "True").lower() in ("true", "1", "yes")
keep_shelling_artilleri_missile: bool = os.environ.get("KEEP_SHELLING", "False").lower() in ("true", "1", "yes")
min_date_for_all_events = os.environ.get("MIN_DATE", "2000-01-01")

# Negative value computation parameters
sample_factor = float(os.environ.get("SAMPLE_FACTOR", 1))
buffer_km = float(os.environ.get("BUFFER_KM", 200))
ukraine_shp_path = os.environ.get("UKRAINE_SHP_PATH", "Input/gadm41_UKR_shp/gadm41_UKR_0.shp")
admin_shp_path = os.environ.get("ADMIN_SHP_PATH", "Input/gadm41_UKR_shp/gadm41_UKR_2.shp")

# Landcover computation parameters
base_folder = os.environ.get("COVER_BASE_FOLDER", "Input/WorldCoverData")
year = int(os.environ.get("YEAR", 2021))
version = os.environ.get("VERSION", "v200")

# Model building parameters
with_admin2 = os.environ.get("WITH_ADMIN2", "False").lower() in ("true", "1", "yes")
with_admin3 = os.environ.get("WITH_ADMIN3", "False").lower() in ("true", "1", "yes")

# Landcover computation parameters
export_base_folder = os.environ.get("EXPORT_BASE_FOLDER", "Output/")

## Start the process
now_str = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
csv_file_path: str = f"{export_base_folder}{now_str} Trainning database export.csv"
model_file_path: str = f"{export_base_folder}{now_str} Model.pickle"
variables_file_path: str = f"{export_base_folder}{now_str} Variables.json"

logger = logging.getLogger("FromDataToModel")
# We define logging level to info
#logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
#logging.basicConfig(level=logging.DEBUG)

#print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Starting")
logger.info("Starting")

varaibles_dict = {
    "conflict_day_half_life": conflict_day_half_life,
    "conflict_km_half_life": conflict_km_half_life,
    "conflict_use_fatalities": conflict_use_fatalities,
    "conflict_use_limitation": conflict_use_limitation,
    "strike_day_half_life": strike_day_half_life,
    "strike_km_half_life": strike_km_half_life,
    "strike_use_fatalities": strike_use_fatalities,
    "strike_use_limitation": strike_use_limitation,
    "min_date": min_date,
    "percentage_to_remove": percentage_to_remove,
    "datafile_path": datafile_path,
    "keep_air_drone_strike": keep_air_drone_strike,
    "keep_shelling_artilleri_missile": keep_shelling_artilleri_missile,
    "min_date_for_all_events": min_date_for_all_events,
    "sample_factor": sample_factor,
    "buffer_km": buffer_km,
    "ukraine_shp_path": ukraine_shp_path,
    "admin_shp_path": admin_shp_path,
    "base_folder": base_folder,
    "year": year,
    "version": version,
    "with_admin2": with_admin2,
    "with_admin3": with_admin3
}

with open(variables_file_path, "w") as f:
    json.dump(varaibles_dict, f)

logger.info("Variables saved")

events_df: pd.DataFrame = get_strike_database(datafile_path=datafile_path, min_date=min_date,
                                              keep_air_drone_strike=keep_air_drone_strike, keep_shelling_artilleri_missile=keep_shelling_artilleri_missile)

logger.info(f"Initial event number: {len(events_df)}")

events_df = randomly_remove(events_df, percentage_to_remove)

logger.info(f"Event number after random: {len(events_df)}")

negatives_events_df = generate_negative_points(events_df, ukraine_shp_path=ukraine_shp_path, admin_shp_path=admin_shp_path,
                                               sample_factor=sample_factor, buffer_km=buffer_km)

logger.info(f"Negative event number: {len(negatives_events_df)}")

df_hist = merge_positive_negative(events_df, negatives_events_df)

logger.info(f"Final event number: {len(df_hist)}")

del events_df
del negatives_events_df

logger.info(f"Cleaning done")

df_hist_cover: pd = sample_landcover(df_hist, base_folder, year=year, version=version)

print(df_hist_cover.head())

assert df_hist_cover["latitude"].isnull().sum() == 0
assert df_hist_cover["longitude"].isnull().sum() == 0

# Fro now we desactiovate this test
#assert df_hist_cover["landcover"].isnull().sum() == 0
# But we display lignes with NaN values
print(df_hist_cover[df_hist_cover["landcover"].isnull()])
# And we remove them
df_hist_cover = df_hist_cover[df_hist_cover["landcover"].notnull()]

logger.info("Land Coverage computed")

del df_hist

logger.info("Cleaning done")

df_hist_cover = change_land_cover_label(df_hist_cover)

logger.info("Land Coverage changed")

conflict_df: pd.DataFrame = get_conflict_database(datafile_path=datafile_path)

logger.info(f"Conflict loaded: {len(conflict_df)}")

add_intensity(df_hist_cover, conflict_df,
              conflict_day_half_life, conflict_km_half_life, conflict_use_fatalities, conflict_use_limitation,
              column_name="conflict_intensity")

print(df_hist_cover.head())

logger.info("Conflict intensity computed")

del conflict_df

logger.info("Cleaning done")

all_events_df: pd.DataFrame = get_strike_database(datafile_path=datafile_path, min_date=min_date_for_all_events,
                                              keep_air_drone_strike=keep_air_drone_strike,
                                              keep_shelling_artilleri_missile=keep_shelling_artilleri_missile)

logger.info(f"All strike loaded: {len(all_events_df)}")

add_intensity(df_hist_cover, all_events_df,
              strike_day_half_life, strike_km_half_life, strike_use_fatalities, strike_use_limitation,
              column_name="strike_intensity")

print(df_hist_cover.head())

logger.info("Strike intensity computed")

del all_events_df

logger.info("Cleaning done")

df_hist_cover.to_csv(csv_file_path, index=False)

logger.info("CSV saved")

logger.info(f"Finished for '{now_str}'")