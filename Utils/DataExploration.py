import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


def plot_map(df: pd.DataFrame):
    # Calculate boundaries from your data
    lat_min = df['latitude'].min()
    lat_max = df['latitude'].max()
    lon_min = df['longitude'].min()
    lon_max = df['longitude'].max()

    # Determine the center of your data
    center_lat = (lat_min + lat_max) / 2
    center_lon = (lon_min + lon_max) / 2

    # Create boundaries with a ±2 degree margin
    lat_range = [lat_min - 2, lat_max + 2]
    lon_range = [lon_min - 2, lon_max + 2]

    # Create a Scattergeo plot
    fig = go.Figure(go.Scattergeo(
        lat=df['latitude'],
        lon=df['longitude'],
        mode='markers'
    ))

    # Update the geographic layout with the desired boundaries and center
    fig.update_geos(
        visible=False,  # Hides the default geographic features if desired
        resolution=50,  # Resolution of coastlines
        showcountries=True,  # Show country borders
        lataxis_range=lat_range,
        lonaxis_range=lon_range,
        center=dict(lat=center_lat, lon=center_lon)
    )

    fig.update_layout(
        title='Map Plot of Latitude and Longitude',
        geo=dict(projection_type='natural earth')
    )

    fig.show(renderer='browser')


def plot_map_per_category(df: pd.DataFrame, category_column: str):
    # Calculate boundaries from your data
    lat_min = df['latitude'].min()
    lat_max = df['latitude'].max()
    lon_min = df['longitude'].min()
    lon_max = df['longitude'].max()

    # Determine the center of your data
    center_lat = (lat_min + lat_max) / 2
    center_lon = (lon_min + lon_max) / 2

    # Create boundaries with a ±2 degree margin
    lat_range = [lat_min - 2, lat_max + 2]
    lon_range = [lon_min - 2, lon_max + 2]

    # Create a scatter geo plot with color differentiation for 'sub_event_type'
    fig = px.scatter_geo(
        df,
        lat="latitude",
        lon="longitude",
        color=category_column,
        title=f"Map Plot with Different Colors per {category_column}"
    )

    # Update the geographic layout to focus on the area of interest
    fig.update_geos(
        visible=False,
        resolution=50,
        showcountries=True,
        lataxis_range=lat_range,
        lonaxis_range=lon_range,
        center=dict(lat=center_lat, lon=center_lon),
        projection_type="natural earth"
    )

    fig.show(renderer='browser')

    return fig


if __name__ == "__main__":
    datafile_path: str = "Input/Europe-Central-Asia_2018-2025_Mar07.csv"

    data: pd.DataFrame = pd.read_csv(datafile_path)

    print(data.head())
    print(data.columns)
    print(data.dtypes)
    print(data.describe())
    print(data.info())
    print(data.shape)
    print(data.isnull().sum())

    print(data["sub_event_type"].unique())
    print(data["event_date"].unique())

    # We filter only 'Air/drone strike' and 'Shelling/artillery/missile attack'
    subset = data[(data["sub_event_type"] == "Air/drone strike") | (data["sub_event_type"] == "Shelling/artillery/missile attack")]
    # We filter "country" as "Ukraine"
    subset = subset[subset["country"] == "Ukraine"]

    # We plot only the first 1000 points
    #plot_map(subset)
    plot_map(subset[:1000])


    # We create a subset of all except 'Air/drone strike' and 'Shelling/artillery/missile attack'
    subset = data[~((data["sub_event_type"] == "Air/drone strike") | (data["sub_event_type"] == "Shelling/artillery/missile attack"))]
    subset = subset[subset["country"] == "Ukraine"]

    # We plot only the first 1000 points
    plot_map_per_category(subset[:10000], "sub_event_type")


    # We do another subset, the sub_event_type is 'Government regains territory' 'Armed clash" and 'Non-state actor overtakes territory'
    subset = data[(data["sub_event_type"] == "Government regains territory") | (data["sub_event_type"] == "Non-state actor overtakes territory") | (data["sub_event_type"] == "Armed clash")]
    #subset = subset[subset["country"] == "Ukraine"]
    # The countrye is 'Ukraine', Russia or Belarus
    subset = subset[(subset["country"] == "Ukraine") | (subset["country"] == "Russia") | (subset["country"] == "Belarus")]

    # We plot only the first 1000 points
    plot_map(subset[:10000])

