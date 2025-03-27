# Generalized Additive Model (GAM) Approach

This folder mirrors the **GLM model** folder in structure and usage, but **replaces** the standard GLM with a **Generalized Additive Model (GAM)**. GAMs allow us to capture **non-linear** effects more flexibly—particularly for spatial location via `(latitude, longitude)` and potentially other features.

## Overview

1. **Same Architecture, Different Core**  
   - Like the GLM approach, we split data processing, model training, and plotting into separate scripts.  
   - But here, a **GAM** (specifically, a logistic GAM for binary outcomes) replaces the linear predictor with smooth functions.

2. **Why GAM?**  
   - A GAM with a 2D smoother (tensor product) on `(latitude, longitude)` can discover **continuous spatial patterns** without manually encoding “zones” or region dummies.  
   - This can remove or reduce the need for a dedicated conflict intensity measure—at least for spatial variation—since the model learns risk “zones” organically.  
   - However, capturing temporal variation remains a challenge; you might explore time-slicing or different data windows to see how historical data depth influences predictions.

3. **Potential Role of Intensity**  
   - If you still want to account for how recent or nearby conflicts might spike risk, you can incorporate “conflict_intensity” or “strike_intensity” as separate smooth terms.  
   - The main difference: with a 2D latitude–longitude smooth, you can potentially skip some aspects of intensity-based distance weighting—**but** that doesn’t necessarily model time-based war evolution the same way.  
   - In practice, a **hybrid** approach (combining intensity features + spatial smooth) may offer the best of both worlds.

## Folder Contents

1. **`Functions.py`**  
   - Utility scripts for data prep, generating negative data if needed, etc., specifically adapted to the GAM approach.

2. **`FromDataToCSV.py`**  
   - Not implemented, we use the same as in the GLM model.

3. **`FromCSVToModel.py`**  
   - Loads the CSV, fits the **GAM**.  
   - Typically uses a logistic family with a 2D tensor product spline for `(latitude, longitude)`—plus any additional smooth or factor terms you define (e.g., `landcover`, `month`).

4. **`PlotModelGrid.py`**  
   - Takes the trained GAM and plots risk maps on a spatial grid.  
   - Let’s you see how the model’s predicted probability varies across the lat/lon domain.

## Usage and Notes

- **Data Preparation**:  
  - Because the GAM approach can handle lat/lon as a smooth, you can skip or reduce “conflict_intensity” and “strike_intensity” if your primary driver is purely geographic.  
  - On the other hand, if time-based recency of conflict matters a great deal, you can keep these intensities or add an explicit time dimension in your GAM formula.

- **Model Tuning**:  
  - GAMs often require carefully setting the basis size (number of spline coefficients) and smoothness penalties. Overly flexible splines can lead to overfitting, while too few basis functions can underfit.  
  - You can cross-validate or hold out data to check performance.

- **Potential Refinements**:  
  - Add or remove kernel-based conflict intensities, or experiment with partial vs. full spatiotemporal smoothing.  
  - Use more advanced partial pooling or hierarchical structures if you incorporate region or landcover as well.  
  - Investigate how different data windows (recent vs. older conflict info) influence the model’s predictions.

## Contributing / License

- As with the rest of this repository, the GAM code here is released under **GNU GPL v3**.  
- Contributions—improved smoothing choices, advanced time modeling, better negative-sampling logic, etc.—are always welcome.

Enjoy exploring the **GAM** approach!
