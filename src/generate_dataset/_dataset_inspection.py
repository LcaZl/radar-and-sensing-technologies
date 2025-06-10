import matplotlib.pyplot as plt
import pandas as pd
import logging
from shapely.geometry import GeometryCollection
import folium
from folium.plugins import MarkerCluster
import os
import plotly.express as px
import numpy as np
import geopandas as gpd
import xarray as xr

def generate_maps(
    points: gpd.GeoDataFrame = None, 
    tile_geometry: gpd.GeoDataFrame = None, 
    output_file: str = "map_with_points_and_geometry.html"
) -> folium.Map:
    """
    Generates an interactive map displaying training points and/or tile geometry using Folium.

    Parameters
    ----------
    points : gpd.GeoDataFrame, optional
        GeoDataFrame containing training points with geometries and a 'class' column for labeling.
        If provided, the points will be plotted on the map.
    tile_geometry : gpd.GeoDataFrame, optional
        GeoDataFrame containing the tile geometry to be outlined on the map.
    output_file : str, optional
        Path to save the generated map as an HTML file (default is "map_with_points_and_geometry.html").

    Returns
    -------
    folium.Map
        Generated Folium map object.
    """

    logger = logging.getLogger("map-generation")
    if points is None and tile_geometry is None:
        raise ValueError("At least one of 'points' or 'tile_geometry' must be provided for plotting.")

    if points is not None:
        if points.crs != "EPSG:4326":
            logger.info("Reprojecting points to EPSG:4326")
            points = points.to_crs("EPSG:4326")
    
    if tile_geometry is not None:
        if tile_geometry.crs != "EPSG:4326":
            logger.info("Reprojecting tile_geometry to EPSG:4326")
            tile_geometry = tile_geometry.to_crs("EPSG:4326")

    # Determine map center
    center_lat, center_lon = 0, 0
    if points is not None:
        center_lat, center_lon = points.geometry.y.mean(), points.geometry.x.mean()
    elif tile_geometry is not None:
        tile_polygon = tile_geometry.iloc[0].geoms[0] if isinstance(tile_geometry.iloc[0], GeometryCollection) else tile_geometry.iloc[0]
        centroid = tile_polygon.centroid
        center_lat, center_lon = centroid.y, centroid.x

    # Create Folium map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=6)

    # Add tile geometry
    if tile_geometry is not None:
        tile_polygon = tile_geometry.iloc[0].geoms[0] if isinstance(tile_geometry.iloc[0], GeometryCollection) else tile_geometry.iloc[0]
        folium.GeoJson(
            tile_polygon,
            style_function=lambda x: {'color': 'red', 'weight': 2, 'fill': False}
        ).add_to(m)

    # Add training points
    if points is not None:
        marker_cluster = MarkerCluster().add_to(m)
        for _, row in points.iterrows():
            label = row['class']
            folium.Marker(
                location=[row.geometry.y, row.geometry.x],
                popup=f"Class: {label}"
            ).add_to(marker_cluster)

    m.save(output_file)
    logger.info(f"Map saved to {output_file}")

    return m
    
def inspect_class_distribution(
    samples: pd.DataFrame, 
    class_column: str, 
    title: str = "",  
    plot: bool = False,
    output_path = None
) -> plt.Figure:
    """
    Inspects and visualizes the class distribution of samples using a bar chart.

    Parameters
    ----------
    samples : pd.DataFrame
        DataFrame containing the samples with class labels.
    class_column : str
        Column name representing class labels.
    title : str, optional
        Title for the plot (default is an empty string).
    plot : bool, optional
        If True, displays the plot (default is False).

    Returns
    -------
    plt.Figure
        Matplotlib figure object of the class distribution bar chart.
    """
    
    # Count the number of samples for each class
    class_distribution = samples[class_column].value_counts().sort_index()

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot the bar chart
    class_distribution.plot(kind='bar', color='skyblue', edgecolor='black', ax=ax)

    # Add titles and labels
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Number of Samples", fontsize=12)

    # Rotate x-axis labels for better readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    # Annotate counts on top of each bar
    for bar in ax.patches:
        count = int(bar.get_height())
        ax.annotate(
            f'{count}',
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha='center',
            va='bottom',
            fontsize=10,
            color='black'
        )

    fig.tight_layout()

    if plot:
        plt.show()
        
    if output_path is not None:
        plt.savefig(output_path)
        
    plt.close(fig)
    return fig
    
def pixel_to_latlon(dataset: xr.Dataset, pixel_x: int, pixel_y: int) -> tuple[float, float]:
    """
    Converts pixel coordinates (pixel_x, pixel_y) to geographic coordinates (longitude, latitude)
    based on the GeoTransform metadata of the dataset.

    Parameters
    ----------
    dataset : xarray.Dataset
        Dataset containing spatial_ref attribute with GeoTransform information.
    pixel_x : int
        X coordinate (column) in pixel space.
    pixel_y : int
        Y coordinate (row) in pixel space.

    Returns
    -------
    tuple of float
        (longitude, latitude) of the pixel center.
    """    
    geo_transform_str = dataset.spatial_ref.attrs["GeoTransform"]
    geo_transform = [float(val) for val in geo_transform_str.split()]
    lon = geo_transform[0] + (pixel_x + 0.5) * geo_transform[1] + (pixel_y + 0.5) * geo_transform[2]
    lat = geo_transform[3] + (pixel_x + 0.5) * geo_transform[4] + (pixel_y + 0.5) * geo_transform[5]
    return lon, lat

def verify_coordinates(mapped_points: pd.DataFrame, dataset: xr.Dataset) -> pd.DataFrame:
    """
    Verifies the accuracy of pixel-to-geographic coordinate mapping by recalculating
    (longitude, latitude) from pixel coordinates and comparing with original geometry.

    Parameters
    ----------
    mapped_points : pd.DataFrame or geopandas.GeoDataFrame
        DataFrame containing columns 'x' and 'y' with pixel coordinates and geometry with original coordinates.
    dataset : xarray.Dataset
        Dataset providing spatial reference and GeoTransform.

    Returns
    -------
    pd.DataFrame
        DataFrame with additional columns:
        - 'recalc_lon', 'recalc_lat': recalculated geographic coordinates.
        - 'diff_lon', 'diff_lat': differences between original and recalculated coordinates.
    """
    recalculated_coords = [
        pixel_to_latlon(dataset, px, py)
        for px, py in zip(mapped_points["x"], mapped_points["y"])
    ]
    mapped_points["recalc_lon"], mapped_points["recalc_lat"] = zip(*recalculated_coords)
    mapped_points["diff_lon"] = mapped_points.geometry.x - mapped_points["recalc_lon"]
    mapped_points["diff_lat"] = mapped_points.geometry.y - mapped_points["recalc_lat"]
    return mapped_points


def visual_match_verification_l1(verified_points: pd.DataFrame, save_path: str = None) -> None:
    """
    Visualizes original vs recalculated geographic coordinates using a static Matplotlib scatter plot.

    Parameters
    ----------
    verified_points : pd.DataFrame or geopandas.GeoDataFrame
        DataFrame containing original and recalculated coordinate columns ('geometry', 'recalc_lon', 'recalc_lat').
    save_path : str, optional
        Directory path to save the generated plot as PNG (default is None).

    """
    plt.figure(figsize=(20, 20))
    plt.scatter(
        verified_points.geometry.x, verified_points.geometry.y,
        color='blue', label="Original Points", alpha=0.6, marker='o'
    )
    plt.scatter(
        verified_points["recalc_lon"], verified_points["recalc_lat"],
        color='red', label="Recalculated Points", alpha=0.6, marker='x'
    )
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    plt.title("Original vs Recalculated Coordinates")

    if save_path:
        plt.savefig(os.path.join(save_path, "original_vs_recalculated_static.png"))
    plt.show()

def visual_match_verification_l2(verified_points: pd.DataFrame, save_path: str = None) -> None:
    """
    Visualizes original vs recalculated geographic coordinates using an interactive Plotly scatter plot.

    Parameters
    ----------
    verified_points : pd.DataFrame or geopandas.GeoDataFrame
        DataFrame containing original and recalculated coordinate columns ('geometry', 'recalc_lon', 'recalc_lat').
    save_path : str, optional
        Directory path to save the generated interactive plot as HTML (default is None).
    """
    
    data = pd.DataFrame({
        "Original Longitude": verified_points.geometry.x.values,
        "Original Latitude": verified_points.geometry.y.values,
        "Recalculated Longitude": verified_points["recalc_lon"].values,
        "Recalculated Latitude": verified_points["recalc_lat"].values,
        "Difference Longitude": verified_points["diff_lon"].values,
        "Difference Latitude": verified_points["diff_lat"].values
    })

    fig = px.scatter(
        data,
        x="Original Longitude",
        y="Original Latitude",
        hover_data={
            "Original Longitude": True,
            "Original Latitude": True,
            "Recalculated Longitude": True,
            "Recalculated Latitude": True,
            "Difference Longitude": True,
            "Difference Latitude": True
        },
        title="Original vs Recalculated Coordinates"
    )

    fig.add_scatter(
        x=data["Recalculated Longitude"],
        y=data["Recalculated Latitude"],
        mode="markers",
        name="Recalculated Points",
        marker=dict(color="red", symbol="x")
    )

    fig.update_layout(
        width=1500,
        height=1500,
        legend=dict(title="Legend"),
        xaxis_title="Longitude",
        yaxis_title="Latitude"
    )

    if save_path:
        fig.write_html(os.path.join(save_path, "original_vs_recalculated_interactive.html"))
    fig.show()

def plot_distance_histogram(verified_points, save_path=None):
    verified_points["distance_diff"] = np.sqrt(
        verified_points["diff_lon"]**2 + verified_points["diff_lat"]**2
    )
    verified_points["distance_diff"].hist(bins=100)
    plt.title("Distance Difference (Original vs Recalculated)")
    plt.xlabel("Difference (meters)")
    plt.ylabel("Count")

    if save_path:
        plt.savefig(os.path.join(save_path, "distance_difference_histogram.png"))
    plt.show()
