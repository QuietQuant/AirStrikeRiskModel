# Generalized Linear Model (GLM) Approach

This folder contains a **Generalized Linear Model (GLM)** approach for estimating the probability of an air strike event. GLMs are widely used in insurance industries due to their **high interpretability** and relative simplicity. Here, the model outputs a **binary** (Bernoulli) outcome for each location—i.e., whether a strike occurs or not—together with a corresponding probability between 0 and 1.

## Concept Overview

1. **Binomial / Bernoulli Outcome**  
   - We predict a probability \(*p*\) of an air strike happening.  
   - Although maps show discretized “cells,” the underlying model is continuous; any coordinate can be input to get a predicted probability.

2. **Data Sources**  
   - **ACLED**: Historical air-strike and conflict data ([ACLED website](https://acleddata.com/)).  
   - **ESA WorldCover**: Land coverage info ([WorldCover website](https://esa-worldcover.org/en)).

3. **Intensity Fields**  
   - We build **two “intensity” measures**:
     1. **Strike Intensity** (based on air strikes)  
     2. **Conflict Intensity** (based on broader armed conflicts)  
   - Both intensities come from a **Gaussian kernel** in space/time. Each event is weighted more strongly if it’s close in distance or recent in time, fading out with **two half-life parameters**:
     - A **distance** half-life (km)
     - A **time** half-life (days)
   - We then apply a \(\sqrt{\cdot}\) transform to reduce marginal gains in zones already saturated with conflict or strikes. (No extensive analysis was done on whether \(\log\) or some other transform might be better; this is just our chosen approach.)

4. **Land Cover & Month**  
   - Land cover categories (urban, cropland, forest, etc.) help capture how built-up areas might be more prone to strikes.  
   - Month acts as an indicator of **seasonality** (though we do not strongly calibrate or prove a seasonal effect—this is just a baseline approach).

5. **Generating Negative Data**  
   - Original data has only “positive” (i.e., known strikes). But a logistic model needs 0s and 1s.  
   - **Random Negative Sampling**: We pick random locations/times where no strike was recorded and label them 0. This step is somewhat **stochastic** and can bias results if not done carefully.

## File Structure

1. **`Functions.py`**  
   Contains all helper functions needed to:
   - Load data
   - Compute intensities (kernels)
   - Generate random negatives
   - And more…

2. **`FromDataToCSV.py`**  
   - Processes raw data according to chosen parameters (distance/time half-lives, random negative generation).
   - Enriches records with land cover, intensities, etc.  
   - Outputs a CSV ready for modeling.

3. **`FromCSVToModel.py`**  
   - Reads the CSV produced above.
   - Fits the **GLM** (logistic regression) on the dataset.
   - Saves or prints the fitted model for later use.

4. **`PlotModelGrid.py`**  
   - Uses the fitted model to visualize predicted probabilities across a grid.
   - Some sample plots are stored in an **`example/`** folder to demonstrate the typical outputs.

## How to Use

1. **Edit and Run** `FromDataToCSV.py`  
   - Adjust half-life parameters, negative sampling rate, etc., as needed.  
   - Produces a final CSV with labeled rows (including 0’s from negative sampling).

2. **Train the Model** via `FromCSVToModel.py`  
   - Reads the CSV, fits a logistic GLM with your specified formula (see code for details).

3. **Visualize** with `PlotModelGrid.py`  
   - Generates maps/heatmaps or other visuals to interpret the predictions across lat/lon.

### Potential Improvements

Below is a more thorough look at how the current GLM approach might be enhanced or adapted:

1. **Refinement of Land Cover Categories**  
   - Some categories may be extremely rare or nearly identical in risk behavior. Merging or collapsing similar classes could simplify the model and potentially improve statistical stability.  
   - Conversely, you might further subdivide “urban” (or similar broad categories) if you have more granular data on population density or building types.

2. **Optimization of Kernel Parameters**  
   - The distance and time half-lives used to generate “conflict_intensity” and “strike_intensity” are, at present, only heuristics. Doing a grid search (or another hyperparameter tuning method) on these parameters might produce systematically better fits.  
   - One could also test alternative kernel functions (e.g., exponential vs. Gaussian, or multi-kernel approaches).

3. **Transformation vs. Aggregation**  
   - We currently apply a square root transform to intensity. A logarithmic or other function might better capture diminishing returns for additional localized events.  
   - Alternatively, using a **count-based** approach (e.g., Poisson or Negative Binomial for strike counts within grid cells) might remove the need to generate negative data artificially.

4. **Handling High-Conflict Regions**  
   - When conflict intensity is extremely high, it might dominate the model. You could experiment with capping or thresholding the intensity, or removing “trivially high” conflict zones from the dataset if your focus is on moderate-risk areas.  
   - Another strategy is to build a two-stage model: one for baseline risk in non-conflict zones, and one for conflict zones specifically (which might follow different dynamics).

5. **Temporal Splitting and Validation**  
   - To ensure generalization to future periods, you can adopt a time-based train/test split. For instance, use earlier months for training and later months for validation.  
   - Observing how performance degrades over time can highlight how quickly the conflict environment changes and whether you need more frequent updates.

6. **Extended Features**  
   - Weather data, population density, or additional operational details (e.g. intelligence on troop movement, flight patterns) could refine local risk predictions.  
   - Seasonal effects might be deeper than just “month.” Using sine/cosine transforms or day-of-year patterns could capture subtle cyclical trends.

7. **Regularization / Penalization**  
   - If partial separation occurs or you have many dummy expansions, adding an L2 penalty (ridge regression) or using a penalized likelihood approach in statsmodels can stabilize estimates and reduce overfitting.

8. **Robustness Checks**  
   - Sensitivity tests: how stable are predictions if you remove some fraction of negative points or alter the random seed for negative data generation?  
   - Look at partial dependence or local explanations (like LIME or SHAP) to see whether features behave as expected.

Overall, the current pipeline is a **solid baseline**, but many directions are available for experimentation and fine-tuning. Any combination of these improvements could lead to a richer, more robust model.


## License & Contributing

- Licensed under the **GNU GPL v3**. For details, see [LICENSE](../LICENSE).
- We welcome contributions—feel free to open issues or pull requests for improvements!

Happy modeling!
