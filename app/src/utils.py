# utils.py
import json
import os
import logging
from datetime import datetime
import math
from typing import Dict, List, Optional
import googlemaps
from dotenv import load_dotenv
from openai import OpenAI
from .location_utils import LocationBoundary, normalize_area, filter_results_by_boundary

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'food_recommender_{datetime.now().strftime("%Y%m%d")}.log')
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize clients
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
gmaps = googlemaps.Client(key=os.getenv("GOOGLE_MAPS_API_KEY"))

# Country code mapping
COUNTRY_TO_ISO = {
    'Singapore': 'SG',
    'Malaysia': 'MY',
    'United States': 'US',
    'USA': 'US',
    'United Kingdom': 'GB',
    'UK': 'GB',
    'Australia': 'AU',
    'Japan': 'JP',
    'China': 'CN',
    'Thailand': 'TH',
    'Indonesia': 'ID',
    'Vietnam': 'VN',
    'South Korea': 'KR',
    'Korea': 'KR',
    'Philippines': 'PH',
    'India': 'IN',
    'Canada': 'CA',
    'New Zealand': 'NZ',
    'Germany': 'DE',
    'France': 'FR',
    'Italy': 'IT',
    'Spain': 'ES',
}

def get_country_iso(country: str) -> str:
    """Convert country name to ISO code."""
    return COUNTRY_TO_ISO.get(country, country)

def get_location_details(location_name: str, country: str) -> Optional[Dict]:
    """Get detailed location information with proper boundary handling."""
    country_iso = get_country_iso(country)
    try:
        # First, get country boundary
        country_result = gmaps.geocode(
            country,
            components={'country': country_iso}
        )[0]
        country_boundary = LocationBoundary(
            country_result['geometry'].get('viewport', country_result['geometry']['bounds'])
        )

        # Then get specific location
        if location_name.lower() != country.lower():
            location_result = gmaps.geocode(
                f"{location_name}, {country}",
                components={'country': country_iso}
            )

            if location_result:
                location = location_result[0]
                location_boundary = LocationBoundary(
                    location['geometry'].get('viewport', location['geometry']['bounds'])
                )

                # Verify location is within country
                center = location['geometry']['location']
                if country_boundary.contains(center['lat'], center['lng']):
                    return {
                        'boundary': location_boundary,
                        'center': center,
                        'address': location['formatted_address'],
                        'types': location['types']
                    }

            logger.warning(f"Location {location_name} not found in {country}, falling back to country search")

        # Return country-level search parameters
        return {
            'boundary': country_boundary,
            'center': country_result['geometry']['location'],
            'address': country_result['formatted_address'],
            'types': country_result['types']
        }

    except Exception as e:
        logger.error(f"Error getting location details: {e}")
        return None

def search_restaurants_by_location(food_item: str, location_data: Dict) -> List[Dict]:
    """Enhanced search with improved boundary handling."""
    logger.info(f"Searching for {food_item} in {location_data['name']}")

    # Search parameters for different location types
    search_params = {
        'country': {'max_results': 20},
        'city': {'max_results': 20},
        'district': {'max_results': 15},
        'area': {'max_results': 10}
    }

    params = search_params.get(location_data['type'])
    all_results = []

    try:
        # Get location details with boundary information
        location_details = get_location_details(
            location_data['name'],
            location_data['country']
        )

        if not location_details:
            raise ValueError(f"Could not find location: {location_data['name']}")

        boundary = location_details['boundary']

        if location_data['type'] == 'country':
            # Get appropriate grid size based on area
            grid_size = 3  # Default
            if boundary.area > 500000:  # Very large country
                grid_size = 4
            elif boundary.area < 5000:  # Small country
                grid_size = 2

            # Generate search points
            search_points = boundary.get_grid_points(grid_size)

            # Search from each point
            for point in search_points:
                try:
                    places_result = gmaps.places_nearby(
                        location=(
                            point['location']['lat'],
                            point['location']['lng']
                        ),
                        radius=point['radius'],
                        keyword=f"{food_item} restaurant",
                        type='restaurant'
                    )

                    if places_result.get('results'):
                        # Filter results to ensure they're within boundary
                        valid_results = filter_results_by_boundary(
                            places_result['results'],
                            boundary
                        )
                        all_results.extend(valid_results)

                except Exception as e:
                    logger.warning(f"Error searching point {point['location']}: {e}")
                    continue

            # Remove duplicates based on place_id
            all_results = list({
                place['place_id']: place for place in all_results
            }.values())

        else:
            # For city/district/area searches
            places_result = gmaps.places_nearby(
                location=(
                    location_details['center']['lat'],
                    location_details['center']['lng']
                ),
                radius=min(
                    5000,  # Maximum 5km for specific locations
                    math.sqrt(boundary.area) * 500  # Dynamic radius based on area
                ),
                keyword=f"{food_item} restaurant",
                type='restaurant'
            )

            if places_result.get('results'):
                # Filter results to ensure they're within boundary
                all_results = filter_results_by_boundary(
                    places_result['results'],
                    boundary
                )

        # Sort by rating and review count
        all_results.sort(
            key=lambda x: (
                x.get('rating', 0) * math.log1p(x.get('user_ratings_total', 0)),
                x.get('user_ratings_total', 0)
            ),
            reverse=True
        )

        # Take top results
        all_results = all_results[:params['max_results']]

        if not all_results:
            return []

        # Process and return results
        return process_place_details(
            all_results,
            food_item,
            location_data
        )

    except Exception as e:
        logger.error(f"Error in location search: {e}")
        return []

def analyze_query_intent(query: str):
    """Analyze if the query is food-related and extract location information."""
    system_msg = """
You are a friendly expert food recommendation assistant. Your task is to:
1. Analyze queries to understand user intent
2. For food-related queries:
   - Identify specific dishes
   - Extract and validate location information
   - Determine search scope based on location specificity
   - For generic "food" or "restaurant" queries without specific dishes, respond as a non-food query
   - For queries mentioning multiple countries, respond as a non-food query
3. For non-food queries:
   - Respond warmly while guiding user towards food recommendations

Response format (JSON):
{
    "is_food_query": boolean,
    "message": string,
    "parsed_data": {
        "food_item": string,
        "location_data": {
            "provided": boolean,
            "type": "country" | "city" | "district" | "area",
            "name": string,
            "country": string,
            "subdivision": string (optional)
        },
        "numeric_price_level": integer (0-4) or null,
        "search_terms": array of strings
    }
}

Examples:
1. Generic food query or multiple locations:
"Food in Singapore and Malaysia"
{
    "is_food_query": false,
    "message": "I see you're interested in multiple locations. To help you better, could you please specify one location at a time?",
    "parsed_data": {
        "food_item": null,
        "location_data": {
            "provided": true,
            "type": null,
            "name": null,
            "country": null
        },
        "search_terms": ["food", "Singapore", "Malaysia"]
    }
}

2. Non-food query:
"What's the weather like?"
{
    "is_food_query": false,
    "message": "I'm your food recommendation assistant! Would you like help finding some great places to eat? Just let me know what cuisine you're interested in and where you'd like to eat.",
    "parsed_data": {
        "food_item": null,
        "location_data": {
            "provided": false,
            "type": null,
            "name": null,
            "country": null
        },
        "search_terms": []
    }
}

3. Valid food query:
"Chicken rice in Singapore"
{
    "is_food_query": true,
    "message": null,
    "parsed_data": {
        "food_item": "chicken rice",
        "location_data": {
            "provided": true,
            "type": "country",
            "name": "Singapore",
            "country": "Singapore"
        },
        "numeric_price_level": null,
        "search_terms": ["chicken rice", "Singapore"]
    }
}

Remember to always set is_food_query to false for:
- Generic "food" or "restaurant" queries without specific dishes
- Queries mentioning multiple locations
- Non-food-related queries
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": f"Analyze this query: '{query}'"}
            ],
            temperature=0.7,
            max_tokens=300
        )

        result = json.loads(response.choices[0].message.content)
        
        # Add these critical error checks back
        if not result["is_food_query"]:
            raise ValueError(result["message"])

        if result["is_food_query"] and not result["parsed_data"]["location_data"]["provided"]:
            raise ValueError(result["message"] or "Please specify a location for your search. For example:\n- 'Best pizza in New York'\n- 'Sushi restaurants in Tokyo, Japan'\n- 'Local food in Singapore'")

        logger.info(f"Query analysis: {json.dumps(result.get('parsed_data', {}), indent=2)}")
        return result

    except Exception as e:
        logger.error(f"Error analyzing query: {e}")
        raise

def get_place_details(place_id: str) -> Optional[Dict]:
    """Get detailed information for a specific place."""
    try:
        place_details_resp = gmaps.place(
            place_id,
            fields=[
                "name", "rating", "user_ratings_total", "formatted_address",
                "price_level", "opening_hours", "vicinity", "type",
                "international_phone_number", "website", "photo",
                "geometry", "reviews"
            ]
        )

        if "result" in place_details_resp:
            return place_details_resp["result"]
    except Exception as e:
        logger.error(f"Error getting place details: {e}")
    return None

def process_reviews(reviews: List[Dict], keyword: str) -> Dict:
    """Process reviews and calculate keyword relevance."""
    processed_reviews = []
    keyword_mentions = 0

    for review in reviews:
        processed_review = {
            "author_name": review.get("author_name", ""),
            "rating": review.get("rating", 0),
            "text": review.get("text", ""),
            "time": review.get("time", ""),
            "relative_time_description": review.get("relative_time_description", "")
        }

        if keyword.lower() in review.get("text", "").lower():
            keyword_mentions += 1

        processed_reviews.append(processed_review)

    return {
        "processed_reviews": processed_reviews,
        "review_count": len(processed_reviews),
        "keyword_relevance": keyword_mentions / len(processed_reviews) if processed_reviews else 0
    }

def process_place_details(places: List[Dict], food_item: str, location_data: Dict) -> List[Dict]:
    """Process and enrich place details with additional context."""
    detailed_results = []

    for place in places:
        try:
            place_details = get_place_details(place['place_id'])
            if place_details:
                enriched_place = place.copy()
                enriched_place.update(place_details)

                # Add location context
                area, city, country = normalize_area(enriched_place['formatted_address'])
                enriched_place['location_context'] = {
                    'search_area': location_data['name'],
                    'location_type': location_data['type'],
                    'country': location_data['country'],
                    'normalized_area': area,
                    'normalized_city': city
                }

                # Process reviews and calculate relevance
                if 'reviews' in place_details:
                    enriched_place.update(
                        process_reviews(place_details['reviews'], food_item)
                    )

                detailed_results.append(enriched_place)
                logger.info(f"Processed: {enriched_place.get('name')} (rating: {enriched_place.get('rating')})")

        except Exception as e:
            logger.error(f"Error processing place {place.get('name')}: {e}")
            continue

    logger.info(f"Successfully processed {len(detailed_results)} restaurants")
    return detailed_results
