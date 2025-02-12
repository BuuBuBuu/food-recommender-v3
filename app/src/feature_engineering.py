# feature_engineering.py

import os
import time  # Add this import
import pickle
import numpy as np
import pandas as pd
import logging
from .path_config import TFIDF_VECTORIZER_PATH

logger = logging.getLogger(__name__)

def calculate_bayesian_rating(rating, review_count, C=100, m=3.5):
    """
    Calculate weighted rating using Bayesian average.
    """
    if pd.isna(rating) or pd.isna(review_count):
        return m
    weighted_rating = (C * m + review_count * rating) / (C + review_count)
    return weighted_rating

def calculate_temporal_score(reviews, max_age_days=180):
    """
    Calculate score based on review recency.
    """
    if not reviews:
        return 0.5

    try:
        current_time = time.time()
        review_ages = []

        for review in reviews:
            review_time = float(review.get('time', 0))
            age_days = (current_time - review_time) / (24 * 3600)
            review_ages.append(max(0, 1 - (age_days / max_age_days)))

        return np.mean(review_ages) if review_ages else 0.5
    except Exception as e:
        logger.error(f"Error calculating temporal score: {e}")
        return 0.5

def calculate_review_quality_score(reviews):
    """
    Calculate score based on review quality.
    """
    if not reviews:
        return 0.5

    try:
        lengths = [len(review.get('text', '')) for review in reviews]
        avg_length = np.mean(lengths) if lengths else 0
        length_score = min(1.0, avg_length / 500)
        return length_score
    except Exception as e:
        logger.error(f"Error calculating review quality score: {e}")
        return 0.5

def extract_candidate_features_improved(candidate, model_data, debug=False):
    """
    Enhanced feature extraction with improved rating calculations.
    """
    try:
        if debug:
            logger.info(f"Loading TF-IDF vectorizer from: {TFIDF_VECTORIZER_PATH}")

        with open(TFIDF_VECTORIZER_PATH, 'rb') as f:
            tfidf = pickle.load(f)

        # Extract basic information
        rating = candidate.get("rating", 3.5)
        review_count = candidate.get("user_ratings_total", 0)

        # Calculate review scores
        reviews = candidate.get("processed_reviews", [])
        temporal_score = calculate_temporal_score(reviews)
        quality_score = calculate_review_quality_score(reviews)
        weighted_rating = calculate_bayesian_rating(rating, review_count)

        # Base features dictionary
        features = {}

        # Add all original features from model_data
        for feature_name in model_data['feature_names']:
            features[feature_name] = 0.0

        # Update with new values
        features.update({
            "review_count": review_count,
            "price_range": candidate.get("price_level", 2),
            "is_open": 1,
            "latitude": candidate.get("geometry", {}).get("location", {}).get("lat", 0),
            "longitude": candidate.get("geometry", {}).get("location", {}).get("lng", 0),
            "location_cluster": 0
        })

        # Process reviews and types
        review_text = ""
        if reviews:
            review_text = " ".join([review.get("text", "") for review in reviews])

        types_text = " ".join(candidate.get("types", []))
        combined_text = f"{review_text} {types_text}"

        # Add text features
        if combined_text.strip():
            text_features = tfidf.transform([combined_text])
            for i, value in enumerate(text_features.toarray()[0]):
                if f'text_feature_{i}' in model_data['feature_names']:
                    features[f'text_feature_{i}'] = value

        # Create DataFrame with correct column order
        candidate_features = pd.DataFrame([features])

        # Add supplementary features for scoring (not used in model prediction)
        candidate_features['temporal_score'] = temporal_score
        candidate_features['quality_score'] = quality_score
        candidate_features['weighted_rating'] = weighted_rating
        candidate_features['raw_rating'] = rating
        candidate_features['log_review_count'] = np.log1p(review_count)

        # Calculate confidence score
        base_score = weighted_rating * np.log1p(review_count)
        candidate_features['confidence_score'] = base_score * (0.7 + 0.3 * temporal_score)

        if debug:
            logger.info(f"Generated {len(candidate_features.columns)} features")

        return candidate_features

    except Exception as e:
        logger.error(f"Error in feature extraction: {e}")
        return pd.DataFrame([[0] * len(model_data['feature_names'])],
                          columns=model_data['feature_names'])

def predict_scores(candidates_features, model_data, debug=False):
    """
    Enhanced prediction with confidence-weighted scoring.
    """
    if debug:
        logger.info(f"Predicting scores for {len(candidates_features)} candidates")

    # Get base predictions from model
    raw_scores = model_data['model'].predict(candidates_features[model_data['feature_names']])

    # Get supplementary scores
    confidence_scores = candidates_features['confidence_score'].values
    temporal_scores = candidates_features['temporal_score'].values
    quality_scores = candidates_features['quality_score'].values

    # Combine scores with weights
    final_scores = (
        0.5 * raw_scores +  # Base model prediction
        0.3 * confidence_scores +  # Rating confidence
        0.1 * temporal_scores +  # Review recency
        0.1 * quality_scores  # Review quality
    )

    # Scale to desired range (20-80)
    if final_scores.max() > final_scores.min():
        scaled_scores = 20 + (final_scores - final_scores.min()) * 60 / (final_scores.max() - final_scores.min())
    else:
        scaled_scores = np.full_like(final_scores, 50.0)

    if debug:
        logger.info(f"Score range: {scaled_scores.min():.1f} - {scaled_scores.max():.1f}")

    return scaled_scores
