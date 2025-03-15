#!/usr/bin/env python
# run_ml_pipeline.py
import os
import sys
import argparse
import pandas as pd
from datetime import datetime

# Add the project root to the Python path
# This is crucial for imports to work correctly
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def main():
    """Run the machine learning pipeline."""
    parser = argparse.ArgumentParser(description="YouTube Trending ML Pipeline")
    parser.add_argument("--step", type=str, choices=["all", "feature-engineering", "training", "prediction"],
                        default="all", help="Pipeline step to run")
    parser.add_argument("--data-file", type=str, 
                        help="Path to processed data file (default: latest in data/processed)")
    parser.add_argument("--test-video", action="store_true",
                        help="Run prediction on a test video")
    parser.add_argument("--debug", action="store_true",
                        help="Show detailed debug information")
    
    args = parser.parse_args()
    
    # Find data file if not specified
    if args.data_file is None:
        data_dir = os.path.join(os.path.dirname(__file__), "data/processed")
        if os.path.exists(data_dir):
            files = [f for f in os.listdir(data_dir) if f.startswith("all_trending_")]
            if files:
                args.data_file = os.path.join(data_dir, max(files))
    
    if args.data_file is None or not os.path.exists(args.data_file):
        print("No data file found. Run data collection first.")
        return 1
    
    print(f"Using data file: {args.data_file}")
    
    # Create directories if needed
    os.makedirs("data/ml", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Load data
    try:
        df = pd.read_csv(args.data_file)
        print(f"Loaded data with {len(df)} records")
    except Exception as e:
        print(f"Error loading data: {e}")
        return 1
    
    # Feature Engineering
    if args.step in ["all", "feature-engineering"]:
        print("\n=== Running Feature Engineering ===")
        try:
            from src.models.feature_engineering import FeatureEngineer
            engineer = FeatureEngineer()
            
            # Prepare features
            features_df = engineer.prepare_features(df)
            print(f"Created {features_df.shape[1]} features")
            
            # Create target variables
            targets_df = engineer.create_target_variables(df)
            print(f"Created {targets_df.shape[1]} target variables")
            
            # Save feature data
            engineer.save_feature_data(features_df, targets_df)
            print("Feature engineering completed successfully")
        except Exception as e:
            print(f"Error during feature engineering: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()
            return 1
    
    # Model Training
    if args.step in ["all", "training"]:
        print("\n=== Running Model Training ===")
        try:
            from src.models.training import ModelTrainer
            trainer = ModelTrainer()
            
            # Load prepared features and targets
            features_path = os.path.join("data/ml", "features.csv")
            targets_path = os.path.join("data/ml", "targets.csv")
            
            if not os.path.exists(features_path) or not os.path.exists(targets_path):
                print("Feature files not found. Run feature engineering first.")
                return 1
            
            features = pd.read_csv(features_path, index_col=0)
            targets = pd.read_csv(targets_path, index_col=0)
            
            # Make sure features only contain numeric columns
            numeric_features = features.select_dtypes(include=['int64', 'float64', 'bool'])
            if numeric_features.shape[1] < features.shape[1]:
                non_numeric_cols = set(features.columns) - set(numeric_features.columns)
                print(f"Dropping {len(non_numeric_cols)} non-numeric columns")
                if args.debug:
                    print(f"Non-numeric columns: {list(non_numeric_cols)}")
                features = numeric_features
            
            print(f"Loaded {features.shape[1]} features and {targets.shape[1]} targets")
            
            # Train models for available targets
            from sklearn.model_selection import train_test_split
            
            if 'is_viral' in targets.columns:
                print("\nTraining viral prediction model...")
                try:
                    # Make sure target is numeric
                    targets['is_viral'] = pd.to_numeric(targets['is_viral'], errors='coerce').fillna(0).astype(int)
                    
                    X_train, X_test, y_train, y_test = train_test_split(
                        features, targets['is_viral'], test_size=0.2, random_state=42
                    )
                    
                    clf_model = trainer.train_classification_model(
                        X_train, y_train, 'is_viral', model_type='random_forest'
                    )
                    
                    metrics = trainer.evaluate_classification_model(clf_model, X_test, y_test)
                    print(f"Viral prediction metrics: {metrics}")
                    
                    importances = trainer.get_feature_importance(clf_model, features.columns.tolist())
                    
                    trainer.save_model(
                        clf_model, 
                        'viral_prediction_rf', 
                        'is_viral',
                        'random_forest',
                        metrics,
                        importances,
                        features.columns.tolist()
                    )
                except Exception as e:
                    print(f"Error training viral prediction model: {e}")
                    if args.debug:
                        import traceback
                        traceback.print_exc()
            
            if 'engagement_score' in targets.columns:
                print("\nTraining engagement prediction model...")
                try:
                    # Make sure target is numeric
                    targets['engagement_score'] = pd.to_numeric(targets['engagement_score'], errors='coerce').fillna(0)
                    
                    X_train, X_test, y_train, y_test = train_test_split(
                        features, targets['engagement_score'], test_size=0.2, random_state=42
                    )
                    
                    reg_model = trainer.train_regression_model(
                        X_train, y_train, 'engagement_score', model_type='gradient_boosting'
                    )
                    
                    metrics = trainer.evaluate_regression_model(reg_model, X_test, y_test)
                    print(f"Engagement prediction metrics: {metrics}")
                    
                    importances = trainer.get_feature_importance(reg_model, features.columns.tolist())
                    
                    trainer.save_model(
                        reg_model, 
                        'engagement_prediction_gb', 
                        'engagement_score',
                        'gradient_boosting',
                        metrics,
                        importances,
                        features.columns.tolist()
                    )
                except Exception as e:
                    print(f"Error training engagement prediction model: {e}")
                    if args.debug:
                        import traceback
                        traceback.print_exc()
            
            print("Model training completed")
        except Exception as e:
            print(f"Error during model training: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()
            return 1
    
    # Prediction
    if args.step in ["all", "prediction"] or args.test_video:
        print("\n=== Running Prediction Test ===")
        try:
            from src.models.prediction import TrendingPredictor
            predictor = TrendingPredictor()
            
            # Load available models
            loaded_models = predictor.load_available_models()
            
            if not loaded_models:
                print("No models available. Train models first.")
                return 1
            
            print(f"Loaded models: {loaded_models}")
            
            # Sample video data for prediction
            test_videos = [
                {
                    'title': 'Top 10 Machine Learning Projects for Beginners in 2023',
                    'description': 'Learn about the best machine learning projects that you can build to enhance your portfolio and skills.',
                    'duration_seconds': 900,  # 15 minutes
                    'category_id': '28',  # Science & Technology
                    'tags': ['machine learning', 'data science', 'python', 'projects', 'tutorial'],
                    'publish_hour': 16,  # 4pm
                    'publish_weekday': 2  # Wednesday
                },
                {
                    'title': 'EXTREME CHALLENGE: Surviving 24 Hours In A Desert!',
                    'description': 'I spent 24 hours in the desert with no food or water - you won\'t believe what happened!',
                    'duration_seconds': 1200,  # 20 minutes
                    'category_id': '24',  # Entertainment
                    'tags': ['challenge', 'extreme', '24 hours', 'survival', 'desert'],
                    'publish_hour': 18,  # 6pm
                    'publish_weekday': 5  # Saturday
                },
                {
                    'title': 'How I Learned Piano in Just 30 Days - Complete Journey',
                    'description': 'I documented my journey learning piano from scratch in just 30 days. Here\'s everything I learned and how you can do it too.',
                    'duration_seconds': 840,  # 14 minutes
                    'category_id': '26',  # Howto & Style
                    'tags': ['piano', 'music', 'learning', '30 days', 'beginner'],
                    'publish_hour': 14,  # 2pm
                    'publish_weekday': 0  # Monday
                }
            ]
            
            for i, video in enumerate(test_videos, 1):
                print(f"\nTest Video {i}: {video['title']}")
                
                try:
                    # Make prediction
                    prediction = predictor.predict_trending_potential(video)
                    
                    print(f"Trending Score: {prediction['trending_score']}/100")
                    print(f"Engagement Score: {prediction['engagement_score']}/10")
                    print(f"Viral Potential: {prediction.get('viral_potential', 0) * 100:.1f}%")
                    print(f"Views Estimate: {prediction.get('views_estimate', 0):,}")
                    
                    print("\nRecommendations:")
                    for j, rec in enumerate(prediction.get('recommendations', []), 1):
                        print(f"  {j}. {rec}")
                except Exception as e:
                    print(f"Error making prediction for video {i}: {e}")
                    if args.debug:
                        import traceback
                        traceback.print_exc()
        except Exception as e:
            print(f"Error during prediction testing: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()
            return 1
    
    print("\nML pipeline completed successfully")
    return 0

if __name__ == "__main__":
    try:
        # Now import the modules
        from src.models.feature_engineering import FeatureEngineer
        from src.models.training import ModelTrainer
        from src.models.prediction import TrendingPredictor
        sys.exit(main())
    except ImportError as e:
        print(f"Error importing required modules: {e}")
        print("Please make sure you've created all the required Python files and they're in the correct locations.")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)