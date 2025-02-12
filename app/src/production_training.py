# production_training.py

"""
This module implements the production model training pipeline for a restaurant ranking system.

Key Components:
    - Data preparation and splitting
    - LightGBM model configuration and training
    - Model evaluation and metrics calculation
    - Feature importance analysis
    - Model persistence with metadata

The model is designed to predict restaurant ranking scores based on various features
including review counts, ratings, text features, and location data.

Technical Details:
    - Uses LightGBM for gradient boosting (faster and more efficient than XGBoost)
    - Implements early stopping to prevent overfitting
    - Includes comprehensive feature importance analysis
    - Saves model with all necessary metadata for production use
    - Handles both numerical and text-based features
"""

#Standard library imports for basic operations
import os # For file and path operations
import pickle # For serializing/deserializing the model and metadata
import time # For tracking training duration

# Numerical and data processing libraries
import numpy as np # For numerical operations
import pandas as pd # For data manipulation and analysis
import lightgbm as lgb # LightGBM for gradient boosting
from sklearn.model_selection import train_test_split # For splitting data into train/test sets
from sklearn.metrics import mean_squared_error, r2_score # For model evaluation
from .path_config import DATA_DIR, MERGED_DATA_PATH, TRAINING_DATA_PATH # Local path configurations


def train_production_model(X, y, data_folder):
    """
    Train a production-ready LightGBM model for resturant ranking.

    We use LightGBM (Light Gradient Boosting Machine) is chosen as the model for this ranking system.
    The reason for choosing this over others is because:
        1. Performance on Tabular Data - LightGBM is optimized for structured/tabular data, which is common in recommendation and ranking problems.
        2. Efficiency and Speed - Compared to other boosting algo like XGBoost, LightGBM is significantly faster due to histogram-based learning and leaf-wise tree growth.
        3. Handles Large Datasets - Our dataset is very large (9gb csv with 5-6 million rows)
        4. Feature Importance Analysis -

    Args:
        X (pd.DataFrame): Feature matrix containing all preprocessed features
        y (pd.Series): Target variable (ranking scores)
        data_folder(str): Directory path where the model will be saved.

    Returns:
        dict: Model data including the trained model, feature names, parameters, and metrics
    """
    print("\nStarting production training...")
    start_time = time.time() # Start timing the training process

    # 1. Split the data into training and testing sets
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42 # 80% training, 20% testing with fixed random seed
    )
    print(f"Training shapes: X={X_train.shape}, y={y_train.shape}")

    # 2. Create LightGBM datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test)

    # 3. Set model parameters
    params = {
        "objective": "regression",  # We choose regression due to the nature of the target variable which is a continuous variable and to better interpret scores
        "metric": "rmse",  # RMSE provides an interpretable error metric in te same units as the target
        "boosting_type": "gbdt",  # Gradient Boosting Decision Trees effectively capture complex non-linear relationships and interactions among features
        "num_leaves": 31,  # Controls model complexity. More leaves increase model flexibility but risk overfitting. Default is 31, a good trade-off.
        "learning_rate": 0.05, # Step size for each iteration (Use smaller value of 0.05 for more robust learning)
        "feature_fraction": 0.9, # Uses 90% of features for training in each iteration, reducing overfitting.
        "bagging_fraction": 0.8, # Uses 80% of training data per boosting round, improving generalization.
        "bagging_freq": 5, # Performs bagging every 5 iterations. Reduces variance while maintaining diversity in training data.
        "verbose": -1,
        "min_data_in_leaf": 100, # Ensures each leaf has at least 100 samples, reducing overfitting.
        "max_bin": 255, # Controls feature discretization for efficient training.
    }

    # 4. Train model with callbacks for early stopping
    print("\nTraining model...")
    callbacks = [
        lgb.early_stopping(
            stopping_rounds=50
        ),  # Halt training if no improvement is observed for 50 consecutive rounds, also provides overfitting prevention.
        lgb.log_evaluation(
            period=100
        ),  # Evaluation metrics are logged every 100 iterations
    ]

    # 5. Train the model
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,  # Maximum number of boosting iterations.
        valid_sets=[valid_data], # Validation dataset
        callbacks=callbacks, # Training callbacks that we set above (early stopping and log eval)
    )

    # 6. Evaluate model performance
    print("\nEvaluating model...")
    y_pred = model.predict(X_test) # Generate predictions
    rmse = np.sqrt(mean_squared_error(y_test, y_pred)) # RMSE Provides a measure of the average prediction error
    r2 = r2_score(y_test, y_pred) # R² Score Indicates the proportion of variance in the target variable that is explained by the model
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test R² score: {r2:.4f}")

    # 6. Feature importance analysis
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importance('gain') # Compute using gain metric to identify which features most significantly reduce prediction error
    })
    importance = importance.sort_values('importance', ascending=False)
    print("\nTop 10 most important features:")
    print(importance.head(10)) # Top 10 features are highlighted

    # 7. Save model and metadata
    print("\nSaving model...")
    model_data = {
        'model': model,
        'feature_names': X.columns.tolist(),
        'model_params': params,
        'feature_importance': importance.to_dict(),
        'metrics': {
            'rmse': rmse,
            'r2': r2
        }
    }

    model_filename = os.path.join(data_folder, "production_ranking_model_enhanced.pkl")
    with open(model_filename, 'wb') as f:
        pickle.dump(model_data, f)

    end_time = time.time()
    print(f"\nTotal training time: {(end_time - start_time)/60:.2f} minutes")
    return model_data

if __name__ == "__main__":
    print("Starting script...")

    # Load existing training data
    data_folder = DATA_DIR
    data_filename = os.path.join(data_folder, "training_data.pkl")

    print(f"\nLoading data from {data_filename}...")
    with open(data_filename, 'rb') as f:
        X, y = pickle.load(f)
    print(f"Data loaded. Shape: {X.shape}")

    # Train improved model
    model_data = train_production_model(X, y, data_folder)
    print("\nProduction training completed successfully!")
