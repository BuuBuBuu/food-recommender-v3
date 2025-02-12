# recommendation.py
import os
import pickle
import pandas as pd
import numpy as np
from .feature_engineering import extract_candidate_features_improved, predict_scores
from .utils import analyze_query_intent, search_restaurants_by_location

class FoodRecommenderEnhanced:
    def __init__(self, data_folder, model_filename="production_ranking_model_enhanced.pkl"):
        self.data_folder = data_folder
        self.model_data = None
        self._load_models(model_filename)

    def _load_models(self, model_filename):
        model_path = os.path.join(self.data_folder, model_filename)
        try:
            print("Loading model from", model_path)
            with open(model_path, 'rb') as f:
                self.model_data = pickle.load(f)
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def get_recommendations(self, query):
        """
        Get restaurant recommendations based on the query.
        Now handles both food and non-food queries appropriately.
        """
        try:
            # 1) Analyze the query
            query_analysis = analyze_query_intent(query)

            # If it's not a food query, return the friendly message directly
            if not query_analysis["is_food_query"]:
                return query_analysis["message"]

            parsed_data = query_analysis["parsed_data"]
            food_item = parsed_data["food_item"]
            location_data = parsed_data["location_data"]
            price_level = parsed_data.get("numeric_price_level")

            # 2) Search for restaurants based on location
            candidates = search_restaurants_by_location(food_item, location_data)

            if not candidates:
                return pd.DataFrame()

            # 3) Extract features & predict scores
            candidate_features_list = []
            for c in candidates:
                feat_df = extract_candidate_features_improved(c, self.model_data)
                candidate_features_list.append(feat_df)

            candidates_features = pd.concat(candidate_features_list, ignore_index=True)
            predicted_scores = predict_scores(candidates_features, self.model_data)

            # 4) Create results DataFrame
            results = pd.DataFrame([{
                "name": c.get("name", ""),
                "formatted_address": c.get("formatted_address", ""),
                "vicinity": c.get("vicinity", ""),
                "rating": c.get("rating", None),
                "user_ratings_total": c.get("user_ratings_total", 0),
                "price_level": c.get("price_level", None),
                "place_id": c.get("place_id", ""),
                "geometry": c.get("geometry", {}),
                "types": c.get("types", []),
                "website": c.get("website", ""),
                "international_phone_number": c.get("international_phone_number", ""),
                "photos": c.get("photos", []),
                "processed_reviews": c.get("processed_reviews", []),
                "opening_hours": c.get("opening_hours", {}),
                "location_context": c.get("location_context", {})
            } for c in candidates])

            # 5) Add predicted scores
            results["predicted_score"] = predicted_scores

            # 6) Apply price penalty if specified (maintaining original logic)
            if price_level is not None:
                results["price_penalty"] = results["price_level"].apply(
                    lambda pl: self._compute_price_penalty(pl, price_level)
                )
                results["adjusted_score"] = results["predicted_score"] - results["price_penalty"]
                results = results.sort_values("adjusted_score", ascending=False)
            else:
                results = results.sort_values("predicted_score", ascending=False)

            return results.reset_index(drop=True)

        except ValueError as e:
            raise ValueError(str(e))
        except Exception as e:
            print(f"Error getting recommendations: {e}")
            raise ValueError("Sorry, I encountered an error processing your request. Could you try rephrasing your query?")

    def _compute_price_penalty(self, place_price_level, desired_price):
        """Compute penalty for price mismatch"""
        if place_price_level is None or pd.isna(place_price_level):
            place_price_level = 2
        place_price_level = int(place_price_level)
        diff = abs(place_price_level - desired_price)
        return diff * 20  # Basic penalty of 20 points per level difference
