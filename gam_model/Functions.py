import pandas as pd
from pygam import LogisticGAM, s, te, f

import matplotlib.pyplot as plt

def prepare_data(data: pd.DataFrame, with_intensity: bool) -> pd.DataFrame:
    """
    Prepare the data for the model
    """
    # Select predictors in the order for our model:
    # 0: conflict_intensity, 1: strike_intensity,
    # 2: latitude, 3: longitude, 4: landcover, 5: month.
    if with_intensity:
        X = data[['conflict_intensity', 'strike_intensity', 'latitude', 'longitude', 'landcover', 'month']].copy()
    else:
        X = data[['latitude', 'longitude', 'landcover', 'month']].copy()

    # Convert categorical columns to numeric codes if they aren't already.
    # This is necessary because pyGAM expects numeric arrays.
    if not pd.api.types.is_numeric_dtype(X['landcover']):
        X['landcover'] = X['landcover'].cat.codes
    if not pd.api.types.is_numeric_dtype(X['month']):
        X['month'] = X['month'].cat.codes

    return X

def build_strike_probability_gam(df: pd.DataFrame, with_intensity: bool) -> LogisticGAM:
    """
    Builds a logistic GAM model to predict whether a strike occurs,
    based on conflict_intensity, strike_intensity, latitude, longitude,
    landcover, and month.

    Assumptions:
      - The DataFrame df has the following columns:
          * 'strike': binary outcome (0/1)
          * 'conflict_intensity': numeric
          * 'strike_intensity': numeric
          * 'latitude': numeric
          * 'longitude': numeric
          * 'landcover': categorical (e.g., "Tree cover", "Grassland", etc.)
          * 'month': categorical (e.g., 2, 3, 11, 12)
    Returns:
      A fitted LogisticGAM object.
    """
    X = prepare_data(df, with_intensity)

    # Define the outcome variable (ensure it's integer 0 or 1)
    y = df['strike'].astype(int)

    if with_intensity:
        # Define the model:
        # - s(0) : smooth for conflict_intensity.
        # - s(1) : smooth for strike_intensity.
        # - te(2,3): tensor product smooth for latitude and longitude.
        # - f(4): factor for landcover.
        # - f(5): factor for month.
        gam = LogisticGAM(s(0) + s(1) + te(2, 3) + f(4) + f(5), fit_intercept=True)
    else:
        # Define the model:
        # - te(0,1): tensor product smooth for latitude and longitude.
        # - f(2): factor for landcover.
        # - f(3): factor for month.
        gam = LogisticGAM(te(0, 1) + f(2) + f(3), fit_intercept=True)

    # Fit the model.
    gam.fit(X, y)

    return gam


def plot_gam_model(gam_model: LogisticGAM, with_intensity: bool):
    """
    Plots the fitted GAM model.

    Args:
      gam_model: A fitted LogisticGAM object.
      df: The DataFrame used to fit the model.
    """

    if with_intensity:
        # This must be change if we change the model of course
        # We only put varaibles that are not categorial and not with a tensor product (that are diplay 3D)
        titles = ['conflict_intensity', 'strike_intensity']#, 'latitude', 'longitude']#, 'landcover', 'month']

        limit = len(titles)
        #titles = df.columns[0:(limit + 1)]
        plt.figure()
        fig, axs = plt.subplots(1, limit, figsize=(40, 8))
        for i, ax in enumerate(axs):
            XX = gam_model.generate_X_grid(term=i)
            ax.plot(XX[:, i], gam_model.partial_dependence(term=i, X=XX))
            ax.plot(XX[:, i], gam_model.partial_dependence(term=i, X=XX, width=.95)[1], c='r', ls='--')
            #if i == 0:
            #    ax.set_ylim(-30, 30)
            ax.set_title(titles[i])

        plt.show()

        plt1 = plt
    else:
        plt1 = None

    if with_intensity:
        term_to_plot = 2
    else:
        term_to_plot = 0
    plt.ion()
    plt.rcParams['figure.figsize'] = (12, 8)
    XX = gam_model.generate_X_grid(term=term_to_plot, meshgrid=True)
    Z = gam_model.partial_dependence(term=term_to_plot, X=XX, meshgrid=True)

    ax = plt.axes(projection='3d')
    ax.plot_surface(XX[0], XX[1], Z, cmap='viridis')
    ax.set_xlabel("latitude")
    ax.set_ylabel("longitude")

    plt.show()

    plt2 = plt

    return plt1, plt2


def add_month(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a 'month' column to the DataFrame df, based on the 'event_date' column.

    Args:
      df: A pandas DataFrame with an 'event_date' column.

    Returns:
      A new DataFrame with a 'month' column.
    """
    df['event_date'] = pd.to_datetime(df['event_date'])
    df['month'] = df['event_date'].dt.month.astype('category')

    # We also convert "landcover" to a categorical variable
    df['landcover'] = df['landcover'].astype('category')

    return df




# Example usage:
if __name__ == "__main__":
    # Create a toy example DataFrame
    data = {
        "strike": [0, 1, 0, 1, 0, 1, 0, 1],
        "conflict_intensity": [1.2, 2.5, 1.0, 3.0, 0.8, 2.0, 1.5, 2.8],
        "strike_intensity": [0.5, 1.2, 0.3, 1.5, 0.4, 1.0, 0.6, 1.3],
        "latitude": [44.5, 44.6, 44.7, 44.8, 44.5, 44.6, 44.7, 44.8],
        "longitude": [33.7, 33.8, 33.9, 34.0, 33.7, 33.8, 33.9, 34.0],
        "landcover": pd.Series(
            ["Tree cover", "Tree cover", "Grassland", "Grassland", "Tree cover", "Grassland", "Tree cover",
             "Grassland"]).astype("category"),
        "month": pd.Series([3, 3, 3, 3, 3, 3, 3, 3]).astype("category")
    }
    df_example = pd.DataFrame(data)

    # Build the GAM model
    gam_model = build_strike_probability_gam(df_example)
    print(gam_model.summary())



