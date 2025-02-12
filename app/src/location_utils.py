# location_utils.py
import math
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class LocationBoundary:
    def __init__(self, bounds: Dict):
        self.ne_lat = float(bounds['northeast']['lat'])
        self.ne_lng = float(bounds['northeast']['lng'])
        self.sw_lat = float(bounds['southwest']['lat'])
        self.sw_lng = float(bounds['southwest']['lng'])
        self.center_lat = (self.ne_lat + self.sw_lat) / 2
        self.center_lng = (self.ne_lng + self.sw_lng) / 2

        # Calculate approximate area
        self.width = calculate_distance(
            self.ne_lat, self.ne_lng,
            self.ne_lat, self.sw_lng
        )
        self.height = calculate_distance(
            self.ne_lat, self.ne_lng,
            self.sw_lat, self.ne_lng
        )
        self.area = self.width * self.height

    def contains(self, lat: float, lng: float) -> bool:
        """Check if a point falls within the boundary."""
        return (self.sw_lat <= lat <= self.ne_lat and
                self.sw_lng <= lng <= self.ne_lng)

    def get_grid_points(self, grid_size: int = 3) -> List[Dict]:
        """Generate a grid of search points within the boundary."""
        points = []
        lat_step = (self.ne_lat - self.sw_lat) / grid_size
        lng_step = (self.ne_lng - self.sw_lng) / grid_size

        for i in range(grid_size):
            for j in range(grid_size):
                lat = self.sw_lat + (i + 0.5) * lat_step
                lng = self.sw_lng + (j + 0.5) * lng_step

                # Calculate appropriate radius for this point
                point_radius = min(
                    25000,  # Maximum 25km radius
                    max(
                        3000,  # Minimum 3km radius
                        math.sqrt(self.area) * 1000 / (grid_size * 2)  # Dynamic radius
                    )
                )

                points.append({
                    'location': {'lat': lat, 'lng': lng},
                    'radius': point_radius
                })

        return points

def calculate_distance(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Calculate distance between two points in kilometers."""
    R = 6371  # Earth's radius in kilometers

    def to_radians(degrees):
        return degrees * math.pi / 180

    lat1, lng1, lat2, lng2 = map(to_radians, [lat1, lng1, lat2, lng2])

    dlat = lat2 - lat1
    dlng = lng2 - lng1

    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    return R * c

def normalize_area(address: str) -> Tuple[str, str, str]:
    """
    Normalize address components into area, city, and country.
    Returns tuple of (area, city, country)
    """
    components = [part.strip() for part in address.split(',')]

    # Remove unit numbers and postal codes
    cleaned_components = []
    for part in components:
        # Remove unit numbers (e.g., #02-10, B1-05)
        if '#' in part or 'B' in part and any(c.isdigit() for c in part):
            continue
        # Remove postal codes (typically 5-6 digits)
        if part.strip().isdigit() and len(part.strip()) in [5, 6]:
            continue
        cleaned_components.append(part.strip())

    # Extract components based on position
    if len(cleaned_components) >= 3:
        area = cleaned_components[0]
        city = cleaned_components[-2]
        country = cleaned_components[-1]
    elif len(cleaned_components) == 2:
        area = cleaned_components[0]
        city = country = cleaned_components[1]
    else:
        area = city = country = cleaned_components[0]

    return area, city, country

def validate_location(location: Dict, boundary: LocationBoundary) -> bool:
    """
    Validate if a location falls within the specified boundary.
    Returns True if valid, False otherwise.
    """
    try:
        lat = float(location['lat'])
        lng = float(location['lng'])
        return boundary.contains(lat, lng)
    except (KeyError, ValueError, TypeError):
        return False

def filter_results_by_boundary(results: List[Dict], boundary: LocationBoundary) -> List[Dict]:
    """Filter results to only include those within the boundary."""
    filtered_results = []

    for result in results:
        try:
            location = result['geometry']['location']
            if validate_location(location, boundary):
                filtered_results.append(result)
            else:
                logger.warning(f"Excluded result outside boundary: {result.get('name', 'Unknown')} "
                             f"at {location.get('lat', 'Unknown')}, {location.get('lng', 'Unknown')}")
        except KeyError:
            logger.warning(f"Skipped result with invalid location data: {result.get('name', 'Unknown')}")

    return filtered_results
