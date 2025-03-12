"""
Utilities for transforming geographical coordinates to Cartesian coordinates.
"""

import numpy as np
from typing import Tuple, Dict, List, Any

# Earth's radius in kilometers
EARTH_RADIUS_KM = 6371.0

def lat_lon_to_cartesian(lat: float, lon: float, ref_lat: float = None, ref_lon: float = None) -> Tuple[float, float]:
    """
    Convert latitude/longitude to approximate Cartesian coordinates (km).

    For small regions, we use a simple approximation that treats the Earth as locally flat.

    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees
        ref_lat: Reference latitude for the origin (if None, use the first point)
        ref_lon: Reference longitude for the origin (if None, use the first point)

    Returns:
        Tuple of (x, y) in kilometers from the reference point
    """
    if ref_lat is None or ref_lon is None:
        ref_lat, ref_lon = lat, lon

    # Convert latitude difference to approximate km (1 degree ~= 111 km)
    y = (lat - ref_lat) * 111.0

    # Convert longitude difference to approximate km (depends on latitude)
    x = (lon - ref_lon) * 111.0 * np.cos(np.radians(ref_lat))

    return x, y

def cartesian_to_lat_lon(x: float, y: float, ref_lat: float, ref_lon: float) -> Tuple[float, float]:
    """
    Convert Cartesian coordinates back to latitude/longitude.

    Args:
        x: x-coordinate in kilometers from the reference point
        y: y-coordinate in kilometers from the reference point
        ref_lat: Reference latitude of the origin
        ref_lon: Reference longitude of the origin

    Returns:
        Tuple of (latitude, longitude) in degrees
    """
    # Convert y change to latitude (1 degree ~= 111 km)
    lat = ref_lat + y / 111.0

    # Convert x change to longitude (depends on latitude)
    lon = ref_lon + x / (111.0 * np.cos(np.radians(ref_lat)))

    return lat, lon

def transform_config_to_cartesian(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform the entire lat/lon configuration to Cartesian coordinates.

    Args:
        config: Dictionary containing lat/lon configuration

    Returns:
        Dictionary with transformed coordinates
    """
    # Find the center point for our projection
    all_lats = []
    all_lons = []

    # Extract all lat/lon points
    for zone in config['adi_zones']:
        all_lats.append(zone['center_lat'])
        all_lons.append(zone['center_lon'])

    for airport in config['airports']:
        all_lats.append(airport['lat'])
        all_lons.append(airport['lon'])

    for zone in config['danger_zones']:
        all_lats.append(zone['center_lat'])
        all_lons.append(zone['center_lon'])

    for point in config['frontline']:
        all_lats.append(point['lat'])
        all_lons.append(point['lon'])

    # Calculate the center point
    ref_lat = np.mean(all_lats)
    ref_lon = np.mean(all_lons)

    # Create a new config with Cartesian coordinates
    cart_config = {
        'reference': {'lat': ref_lat, 'lon': ref_lon},
        'adi_zones': [],
        'airports': [],
        'danger_zones': [],
        'frontline': []
    }

    # Transform ADI zones
    for zone in config['adi_zones']:
        center_x, center_y = lat_lon_to_cartesian(
            zone['center_lat'], zone['center_lon'], ref_lat, ref_lon
        )

        # Convert radii from km to our Cartesian units
        # Assuming radius and epsilon are in km
        cart_zone = {
            'center_x': center_x,
            'center_y': center_y,
            'radius': zone['radius'],
            'epsilon': zone['epsilon'],
            'weapon_type': zone['weapon_type'],
            'original_lat': zone['center_lat'],
            'original_lon': zone['center_lon']
        }
        cart_config['adi_zones'].append(cart_zone)

    # Transform airports
    for airport in config['airports']:
        x, y = lat_lon_to_cartesian(
            airport['lat'], airport['lon'], ref_lat, ref_lon
        )
        cart_airport = {
            'x': x,
            'y': y,
            'capacity': airport['capacity'],
            'original_lat': airport['lat'],
            'original_lon': airport['lon']
        }
        cart_config['airports'].append(cart_airport)

    # Transform danger zones
    for zone in config['danger_zones']:
        center_x, center_y = lat_lon_to_cartesian(
            zone['center_lat'], zone['center_lon'], ref_lat, ref_lon
        )
        cart_zone = {
            'center_x': center_x,
            'center_y': center_y,
            'radius': zone['radius'],
            'threat_level': zone['threat_level'],
            'original_lat': zone['center_lat'],
            'original_lon': zone['center_lon']
        }
        cart_config['danger_zones'].append(cart_zone)

    # Transform frontline points
    for point in config['frontline']:
        x, y = lat_lon_to_cartesian(
            point['lat'], point['lon'], ref_lat, ref_lon
        )
        cart_point = {
            'x': x,
            'y': y,
            'original_lat': point['lat'],
            'original_lon': point['lon']
        }
        cart_config['frontline'].append(cart_point)

    return cart_config