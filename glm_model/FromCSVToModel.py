import json
import logging
import os

import pandas as pd

from glm_model.Functions import build_strike_probability_model, plot_coefficient_summary

now_str = "2025-03-19 13-45-18"

export_base_folder = os.environ.get("EXPORT_BASE_FOLDER", "Output/")
csv_file_path: str = f"{export_base_folder}{now_str} Trainning database export.csv"
model_file_path: str = f"{export_base_folder}{now_str} Model.pickle"
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

#varaibles_dict = {"with_admin2" : False, "with_admin3" : False}
model = build_strike_probability_model(events_df, with_admin2=varaibles_dict["with_admin2"], with_admin3=varaibles_dict["with_admin3"])

logger.info("Model build")

# Save the model
model.save(model_file_path)

logger.info("Model saved")

print(model.summary())

plot_coefficient_summary(model, top_n=20, title = "Model summary", with_error = False)

plot_coefficient_summary(model, top_n=20, title = "Model summary", with_error = True)

logger.info("Model printed")