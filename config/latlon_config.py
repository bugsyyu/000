"""
Configuration file for latitude-longitude data of SAM zones, airports,
danger zones, and frontline positions.
"""

latlon_config = {
    'adi_zones': [
        {
            'center_lat': 25.9,
            'center_lon': 119.2,  # Moved west by 0.4 degrees
            'epsilon': 55.0,
            'radius': 20.0,
            'weapon_type': 'SAM'
        },
        {
            'center_lat': 24.5,
            'center_lon': 117.7,  # Moved west by 0.4 degrees
            'epsilon': 55.0,
            'radius': 20.0,
            'weapon_type': 'SAM'
        },
        {
            'center_lat': 25.2,
            'center_lon': 118.4,  # Moved west by 0.4 degrees
            'epsilon': 55.0,
            'radius': 20.0,
            'weapon_type': 'SAM'
        }
    ],
    'airports': [
        {'capacity': 40, 'lat': 25.438002, 'lon': 114.547147},
        {'capacity': 40, 'lat': 28.331053, 'lon': 117.590052},
        {'capacity': 40, 'lat': 26.574531, 'lon': 117.022694},
        {'capacity': 40, 'lat': 23.719297, 'lon': 111.726357},
        {'capacity': 40, 'lat': 27.8334, 'lon': 114.919461},
        {'capacity': 40, 'lat': 26.559179, 'lon': 112.232849},
        {'capacity': 40, 'lat': 28.263258, 'lon': 119.571137},
        {'capacity': 40, 'lat': 23.130664, 'lon': 116.316499},
        {'capacity': 40, 'lat': 25.126433, 'lon': 109.960743}
    ],
    'danger_zones': [
        {'center_lat': 25.15, 'center_lon': 121.5, 'radius': 25, 'threat_level': 0.95},
        {'center_lat': 24.75, 'center_lon': 121.3, 'radius': 20, 'threat_level': 0.9},
        {'center_lat': 24.35, 'center_lon': 121.1, 'radius': 20, 'threat_level': 0.85},
        {'center_lat': 25.3, 'center_lon': 120.8, 'radius': 15, 'threat_level': 0.8},
        {'center_lat': 24.9, 'center_lon': 120.6, 'radius': 15, 'threat_level': 0.75},
        {'center_lat': 24.5, 'center_lon': 120.4, 'radius': 15, 'threat_level': 0.7}
    ],
    'frontline': [
        # Frontline positions adjusted to be closer to the Taiwan Strait
        {'lat': 25.180691, 'lon': 119.884611},  # Moved east by 0.2 degrees
        {'lat': 25.016686, 'lon': 119.788658},
        {'lat': 24.815641, 'lon': 119.561604},
        {'lat': 24.489725, 'lon': 119.242326},
        {'lat': 24.30232, 'lon': 119.12984},
        {'lat': 24.18895, 'lon': 118.98323},
        {'lat': 24.034967, 'lon': 118.850742},
        {'lat': 23.816067, 'lon': 118.648323},
        {'lat': 23.713079, 'lon': 118.368803}
    ]
}