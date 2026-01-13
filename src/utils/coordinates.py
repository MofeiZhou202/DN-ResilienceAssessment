import numpy as np
from geopy.distance import geodesic


def latlon_to_xy(latitude, longitude, origin_latlon):
    """Convert geographic coordinates to planar offsets (km) relative to origin."""
    origin_lat, origin_lon = origin_latlon

    # East-west distance (positive east of origin)
    east_west = geodesic((latitude, origin_lon), (latitude, longitude)).km
    x = east_west if longitude >= origin_lon else -east_west

    # North-south distance (positive north of origin)
    north_south = geodesic((origin_lat, longitude), (latitude, longitude)).km
    y = north_south if latitude >= origin_lat else -north_south

    return np.array([x, y], dtype=float)