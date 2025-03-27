import json
import logging
import os

import pandas as pd

from gam_model.Functions import build_strike_probability_gam, plot_gam_model, add_month

import pickle

now_str = "2025-03-19 13-45-18"

with_intensity: bool = False

export_base_folder = os.environ.get("EXPORT_BASE_FOLDER", "Output/")
csv_file_path: str = f"{export_base_folder}{now_str} Trainning database export.csv"
model_file_path: str = f"{export_base_folder}{now_str} Model GAM.pickle"
variables_file_path: str = f"{export_base_folder}{now_str} Variables.json"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FromCSVToModel")

# We load teh varaibels from the file
with open(variables_file_path, "r") as f:
    varaibles_dict = json.load(f)

logger.info("Variable loaded")

# We load the data from the file
events_df = pd.read_csv(csv_file_path)

logger.info("Events loaded")

events_df = add_month(events_df)

logger.info("Month added")

model = build_strike_probability_gam(events_df, with_intensity)

logger.info("Model build")

# Save the model
#model.save(model_file_path)
# We save the model
with open(model_file_path, 'wb') as f:
    pickle.dump(model, f)

logger.info("Model saved")

print(model.summary())

plot1, plot2 = plot_gam_model(model, with_intensity)

logger.info("Model printed")