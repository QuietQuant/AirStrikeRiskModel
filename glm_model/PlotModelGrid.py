import json
import logging
import os

import pandas as pd

import statsmodels.api as sm

from glm_model.Functions import plot_coefficient_summary, generate_ukrain_grid, sample_landcover, \
    change_land_cover_label, get_conflict_database, add_intensity, get_strike_database, plot_grid_plotly

reference_date_str: str = "2025-03-01"
resolution: float = 0.1
number_parameter_plot: int = 20

now_str = "2025-03-18 08-43-17"

export_base_folder = os.environ.get("EXPORT_BASE_FOLDER", "Output/")
grid_export_csv_name: str = f"{export_base_folder}{now_str} Grid probability export.csv"
model_file_path: str = f"{export_base_folder}{now_str} Model.pickle"
variables_file_path: str = f"{export_base_folder}{now_str} Variables.json"
plot_summary_error_path: str = f"{export_base_folder}{now_str} Coefficients summary with error.png"
plot_summary_noerror_path: str = f"{export_base_folder}{now_str} Coefficients summary.png"
plot_conflict_intensity_path: str = f"{export_base_folder}{now_str} Conflict intensity plot.html"
plot_strike_intensity_path: str = f"{export_base_folder}{now_str} strike intensity plot.html"
plot_probability_path: str = f"{export_base_folder}{now_str} Probability plot.html"
plot_probability_and_strike_path: str = f"{export_base_folder}{now_str} Probability and strike plot.html"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PlotModelGrid")

# We load teh varaibels from the file
with open(variables_file_path, "r") as f:
    varaibles_dict = json.load(f)

## Conflict intensity computtion parameter
conflict_day_half_life: int = varaibles_dict["conflict_day_half_life"]
conflict_km_half_life: int = varaibles_dict["conflict_km_half_life"]
conflict_use_fatalities: bool = varaibles_dict["conflict_use_fatalities"]
conflict_use_limitation: bool = varaibles_dict["conflict_use_limitation"]

## Strike intensity computtion parameter
strike_day_half_life: int = varaibles_dict["strike_day_half_life"]
strike_km_half_life: int = varaibles_dict["strike_km_half_life"]
strike_use_fatalities: bool = varaibles_dict["strike_use_fatalities"]
strike_use_limitation: bool = varaibles_dict["strike_use_limitation"]

## Data selection parameter
#datafile_path: str = varaibles_dict["datafile_path"]
min_date_for_all_events: str = varaibles_dict["min_date_for_all_events"]
keep_air_drone_strike: bool = varaibles_dict["keep_air_drone_strike"]
keep_shelling_artilleri_missile: bool = varaibles_dict["keep_shelling_artilleri_missile"]

## Landcover computation parameters
base_folder = base_folder = os.environ.get("COVER_BASE_FOLDER", "Input/WorldCoverData")
year: int = varaibles_dict["year"]
version: str = varaibles_dict["version"]

# Datafile
datafile_path: str = os.environ.get("EVENT_DATABASE_PATH", "Input/Europe-Central-Asia_2018-2025_Mar07.csv")
ukraine_shp_path = os.environ.get("UKRAINE_SHP_PATH", "Input/gadm41_UKR_shp/gadm41_UKR_0.shp")

logger.info("Variable loaded")

model = sm.load(model_file_path)

logger.info("Model loaded")

print(model.summary())

fig = plot_coefficient_summary(model, top_n = number_parameter_plot, title = f"Coeff with err, prefix '{now_str}'", with_error = True)
fig.savefig(plot_summary_error_path)

logger.info(f"Coefficient summary plot saved to {plot_summary_error_path}")

fig = plot_coefficient_summary(model, top_n = number_parameter_plot, title = f"Coeff, prefix '{now_str}'", with_error = False)
fig.savefig(plot_summary_noerror_path)

logger.info(f"Coefficient summary plot saved to {plot_summary_noerror_path}")

grid_df = generate_ukrain_grid(resolution, ukr_shapefile = ukraine_shp_path)

logger.info("Generated grid")

print(grid_df.head())

grid_df = sample_landcover(grid_df, base_folder, year=year, version=version)

logger.info("Landcover sampled")

print(grid_df.head())

# We remove nan values
grid_df = grid_df[grid_df['landcover'].notnull()]

print(grid_df.head())

grid_df = change_land_cover_label(grid_df)

logger.info("Landcover label changed")

print(grid_df.head())

grid_df['event_date'] = pd.to_datetime(reference_date_str)

print(grid_df.head())

conflict_df: pd.DataFrame = get_conflict_database(datafile_path=datafile_path)

logger.info("Conflict database read")

add_intensity(grid_df, conflict_df,
              conflict_day_half_life, conflict_km_half_life, conflict_use_fatalities, use_limitation=conflict_use_limitation,
              column_name="conflict_intensity")

logger.info("Conflict intensity added")

print(grid_df.head())

all_events_df: pd.DataFrame = get_strike_database(datafile_path=datafile_path, min_date=min_date_for_all_events,
                                              keep_air_drone_strike=keep_air_drone_strike,
                                              keep_shelling_artilleri_missile=keep_shelling_artilleri_missile)

logger.info("All events database read")

add_intensity(grid_df, all_events_df,
              strike_day_half_life, strike_km_half_life, strike_use_fatalities, strike_use_limitation,
              column_name="strike_intensity")

logger.info("Strike intensity added")

print(grid_df.head())


grid_df['month'] = grid_df['event_date'].dt.month.astype('category')

grid_df["conflict_intensity"] = pd.to_numeric(grid_df["conflict_intensity"])
grid_df["strike_intensity"] = pd.to_numeric(grid_df["strike_intensity"])
grid_df['landcover'] = grid_df['landcover'].astype('category')

print(grid_df.info())
print(grid_df.head())

logger.info("Prediction start")

predictions = model.predict(grid_df)

grid_df['probabilities'] = predictions

logger.info("Prediction done")

print(grid_df.head())


grid_df.to_csv(grid_export_csv_name, index=False)

logger.info(f"Grid exported to {grid_export_csv_name}")


fig = plot_grid_plotly(
        grid_df,
        value_column = "conflict_intensity",
        resolution = resolution,
        title = f"Conflict intensity graph for {reference_date_str}, resolution {resolution}°, prefix '{now_str}'",
        legend_title = "Intensity",
        colorscale = "Reds"#,
)

logger.info("Plot conflict intensity done")

fig.write_html(plot_conflict_intensity_path)

logger.info(f"Conflict intensity plot saved to {plot_conflict_intensity_path}")

fig = plot_grid_plotly(
        grid_df,
        value_column = "strike_intensity",
        resolution = resolution,
        title = f"Strike intensity graph for {reference_date_str}, resolution {resolution}°, prefix '{now_str}'",
        legend_title = "Intensity",
        colorscale = "Reds"#,
)

logger.info("Plot strike intensity done")

fig.write_html(plot_strike_intensity_path)

logger.info(f"Strike intensity plot saved to {plot_strike_intensity_path}")

fig = plot_grid_plotly(
        grid_df,
        value_column = "probabilities",
        resolution = resolution,
        title = f"Strike probability graph for {reference_date_str}, resolution {resolution}°, prefix '{now_str}'",
        legend_title = "probability",
        colorscale = "Reds"#,
)

logger.info("Plot probability done")

fig.write_html(plot_probability_path)

logger.info(f"Probability plot saved to {plot_probability_path}")


# We take teh last fig, but we add on it all lat/long of strike_df taht are with date higher than reference_date_str
recent_strike_df = all_events_df[all_events_df["event_date"] > reference_date_str]

import plotly.graph_objects as go

fig = plot_grid_plotly(
        grid_df,
        value_column = "probabilities",
        resolution = resolution,
        title = f"Strike prob for {reference_date_str}, prefix '{now_str}', with recent strike",
        legend_title = "probability",
        colorscale = "Reds"#,
)

fig.add_trace(
    go.Scattergeo(
        lat=recent_strike_df["latitude"],
        lon=recent_strike_df["longitude"],
        mode='markers',
        marker=dict(
            size=10,
            color='rgb(0, 0, 255)',
            opacity=0.8
        ),
        name="Recent strike"
    )
)

fig.show(renderer='browser')

logger.info("Probabiluity ansd strike prloted")

fig.write_html(plot_probability_and_strike_path)

logger.info(f"Probability and strike plot saved to {plot_probability_and_strike_path}")

logger.info(f"Finished for '{now_str}'")

