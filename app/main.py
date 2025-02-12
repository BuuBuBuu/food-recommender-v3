# main.py
import os
import uvicorn
from fastapi import FastAPI, Query, HTTPException
from typing import List, Optional
import numpy as np
from contextlib import asynccontextmanager
from pydantic import BaseModel
import logging
from app.src.recommendation import FoodRecommenderEnhanced  # Updated import
from app.src.path_config import MODELS_DIR, init_paths  # Updated import

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global recommender instance
recommender = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI application startup and shutdown events"""
    global recommender

    try:
        # Startup: Initialize paths and recommender
        init_paths()
        logger.info("All required files and directories verified")

        recommender = FoodRecommenderEnhanced(str(MODELS_DIR), "production_ranking_model_enhanced.pkl")
        logger.info("Recommender system initialized successfully")

        yield  # Server is running

    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    finally:
        # Cleanup (if needed)
        pass

# Initialize FastAPI with lifespan
app = FastAPI(
    title="Enhanced Food Recommender API",
    description="Restaurant recommendations with location-based search.",
    version="2.1.0",
    lifespan=lifespan
)

# Pydantic models for response
class Location(BaseModel):
    lat: Optional[float]
    lng: Optional[float]

class LocationContext(BaseModel):
    search_area: str
    location_type: str
    country: str

class Review(BaseModel):
    author_name: str
    rating: Optional[float]
    text: str
    time: str
    relative_time_description: str

class Restaurant(BaseModel):
    name: str
    address: str
    rating: Optional[float]
    user_ratings_total: Optional[int]
    price_level: Optional[int]
    place_id: str
    location: Location
    types: List[str]
    website: Optional[str]
    phone: Optional[str]
    photos: List[str]
    predicted_score: float
    adjusted_score: Optional[float] = None
    reviews: List[Review] = []
    opening_hours: Optional[dict] = {}
    location_context: Optional[LocationContext] = None

class RecommendationResponse(BaseModel):
    success: bool
    message: Optional[str] = None
    restaurants: Optional[List[Restaurant]] = None

@app.get("/")
def root():
    return {
        "message": "Welcome to the Enhanced Food Recommender API!",
        "version": "2.1.0",
        "features": [
            "Location-based restaurant search",
            "Enhanced ranking system",
            "Contextual recommendations"
        ]
    }

@app.get("/recommend", response_model=RecommendationResponse)
async def recommend(
    query: str = Query(..., description="Food query with location (e.g., 'Best chicken rice in Tanjong Pagar, Singapore')")
):
    """
    Get restaurant recommendations based on the query and location.
    Handles both food-related queries and friendly chat, with location requirement.
    """
    try:
        if recommender is None:
            raise HTTPException(
                status_code=503,
                detail="Recommender system not initialized"
            )

        # Get recommendations using the enhanced recommender
        results_df = recommender.get_recommendations(query)

        # If it's a non-food query or no location, return the friendly message
        if not results_df.empty and isinstance(results_df, str):
            return RecommendationResponse(
                success=True,
                message=results_df,
                restaurants=[]
            )

        if results_df.empty:
            return RecommendationResponse(
                success=True,
                message="No restaurants found matching your criteria. Try a different location or cuisine.",
                restaurants=[]
            )

        # Replace NaN with None for JSON serialization
        results_df = results_df.replace({np.nan: None})

        # Convert DataFrame to list of Restaurant models
        restaurants = []
        for _, row in results_df.iterrows():
            location_context = None
            if 'location_context' in row and row['location_context']:
                location_context = LocationContext(**row['location_context'])

            reviews = []
            if 'processed_reviews' in row and row['processed_reviews']:
                for review in row['processed_reviews']:
                    reviews.append(Review(
                        author_name=review['author_name'],
                        rating=review.get('rating'),
                        text=review['text'],
                        time=str(review['time']),
                        relative_time_description=review['relative_time_description']
                    ))

            try:
                location = Location(
                    lat=row['geometry']['location']['lat'],
                    lng=row['geometry']['location']['lng']
                )
            except:
                location = Location(lat=0.0, lng=0.0)

            restaurant = Restaurant(
                name=row['name'],
                address=row.get('formatted_address', row.get('vicinity', '')),
                rating=row.get('rating'),
                user_ratings_total=row.get('user_ratings_total'),
                price_level=row.get('price_level'),
                place_id=row['place_id'],
                location=location,
                types=row['types'],
                website=row.get('website', ''),
                phone=row.get('international_phone_number', ''),
                photos=[p.get('photo_reference', '') for p in row.get('photos', [])[:3]],
                predicted_score=float(row['predicted_score']),
                adjusted_score=row.get('adjusted_score'),
                reviews=reviews,
                opening_hours=row.get('opening_hours', {}),
                location_context=location_context
            )
            restaurants.append(restaurant)

        return RecommendationResponse(
            success=True,
            restaurants=restaurants
        )

    except ValueError as e:
        # Return user-friendly messages for common cases
        return RecommendationResponse(
            success=False,
            message=str(e)
        )
    except Exception as e:
        logger.error(f"Error processing recommendation: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
