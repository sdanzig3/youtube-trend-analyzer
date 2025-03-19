#!/usr/bin/env python
# run_ml_pipeline.py

import os
import sys
import logging
import argparse
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"ml_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ml_pipeline")

# Set path to include source directory
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# Import project modules
from src.models.feature_engineering import FeatureEngineer
from src.models.training import ModelTrainer

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run the YouTube Trend Analyzer ML pipeline')
    
    parser.add_argument('--step', type=str, default='all',
                      choices=['all', 'features', 'train', 'evaluate', 'deploy'],
                      help='Which pipeline step to run')
    
    parser.add_argument('--data', type=str, default='latest',
                      help='Data file to use (latest = most recent)')
    
    parser.add_argument('--models', type=str, default='all',
                      choices=['all', 'classification', 'regression', 'viral', 'engagement', 'views'],
                      help='Which models to train')
    
    parser.add_argument('--tune', action='store_true',
                      help='Perform hyperparameter tuning')
    
    parser.add_argument('--cv', type=int, default=5,
                      help='Number of cross-validation folds')
    
    parser.add_argument('--ensemble', action='store_true',
                      help='Create ensemble models')
    
    parser.add_argument('--feature-selection', action='store_true',
                      help='Perform feature selection')
    
    parser.add_argument('--test-size', type=float, default=0.2,
                      help='Proportion of data to use for testing')
    
    parser.add_argument('--output-dir', type=str, default='models',
                      help='Directory to save models')

    return parser.parse_args()

def get_latest_data_file(data_dir: str) -> str:
    """Get the path to the latest data file."""
    # Look for processed data files
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Find the most recent file starting with "all_trending_"
    files = [f for f in os.listdir(data_dir) if f.startswith("all_trending_")]
    if not files:
        raise FileNotFoundError(f"No trending data files found in {data_dir}")
    
    latest_file = max(files)
    file_path = os.path.join(data_dir, latest_file)
    
    return file_path

def run_feature_engineering(data_path: str, output_dir: str = 'data/ml') -> Dict[str, pd.DataFrame]:
    """Run feature engineering on the dataset.
    
    Args:
        data_path: Path to the data file
        output_dir: Directory to save feature data
        
    Returns:
        Dictionary with feature and target DataFrames
    """
    logger.info(f"Starting feature engineering on {data_path}")
    
    # Initialize feature engineer
    engineer = FeatureEngineer()
    
    # Load data
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Check if we have data
    if df.empty:
        raise ValueError(f"Empty DataFrame loaded from {data_path}")
    
    logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    
    # Run basic feature preparation
    logger.info("Preparing basic features")
    features_df = engineer.prepare_features(df)
    
    # Run additional feature engineering steps
    logger.info("Extracting advanced text features")
    text_features = engineer.extract_advanced_text_features(df)
    
    logger.info("Extracting engagement features")
    engagement_features = engineer.extract_engagement_features(df)
    
    logger.info("Extracting temporal patterns")
    temporal_features = engineer.extract_temporal_patterns(df)
    
    logger.info("Extracting content features")
    content_features = engineer.extract_content_features(df)
    
    # Combine all features
    all_features = pd.concat(
        [features_df, text_features, engagement_features, temporal_features, content_features],
        axis=1
    )
    
    # Remove duplicate columns
    all_features = all_features.loc[:, ~all_features.columns.duplicated()]
    
    # Make sure all features are numeric
    numeric_features = all_features.select_dtypes(include=['int64', 'float64', 'bool'])
    if numeric_features.shape[1] < all_features.shape[1]:
        non_numeric_cols = set(all_features.columns) - set(numeric_features.columns)
        logger.warning(f"Dropping {len(non_numeric_cols)} non-numeric columns")
        all_features = numeric_features
    
    # Create target variables
    logger.info("Creating target variables")
    targets_df = engineer.create_target_variables(df)
    
    # Split data
    logger.info("Splitting data into train and test sets")
    X_train, X_test, y_train, y_test = engineer.split_data(all_features, targets_df)
    
    # Save feature data
    logger.info(f"Saving feature data to {output_dir}")
    engineer.save_feature_data(all_features, targets_df, output_dir)
    
    return {
        'features': all_features,
        'targets': targets_df,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }

def train_classification_models(data: Dict[str, pd.DataFrame], 
                              targets: List[str],
                              tune: bool = False,
                              cv: int = 5,
                              ensemble: bool = False,
                              feature_selection: bool = False,
                              output_dir: str = 'models') -> Dict[str, Any]:
    """Train classification models.
    
    Args:
        data: Dictionary with feature and target DataFrames
        targets: List of target column names to model
        tune: Whether to perform hyperparameter tuning
        cv: Number of cross-validation folds
        ensemble: Whether to create ensemble models
        feature_selection: Whether to perform feature selection
        output_dir: Directory to save models
        
    Returns:
        Dictionary with trained models and evaluation metrics
    """
    logger.info(f"Training classification models for targets: {targets}")
    
    # Initialize model trainer
    trainer = ModelTrainer(models_dir=output_dir)
    
    # Get training and test data
    X_train = data['X_train']
    X_test = data['X_test']
    
    # Results dictionary
    results = {}
    
    # Train model for each target
    for target_name in targets:
        logger.info(f"Training models for target: {target_name}")
        
        # Get target data
        if target_name not in data['targets'].columns:
            logger.warning(f"Target {target_name} not found in data")
            continue
        
        y_train = data['y_train'][target_name]
        y_test = data['y_test'][target_name]
        
        # Make sure targets are binary
        y_train = y_train.astype(int)
        y_test = y_test.astype(int)
        
        # Feature selection if requested
        if feature_selection:
            logger.info(f"Performing feature selection for {target_name}")
            X_train_selected, selected_features = trainer.feature_selection(
                X_train, y_train,
                method='importance',
                model_type='classification'
            )
            X_test_selected = X_test[selected_features]
            
            feature_selection_path = os.path.join(output_dir, f"{target_name}_selected_features.txt")
            with open(feature_selection_path, 'w') as f:
                for feature in selected_features:
                    f.write(f"{feature}\n")
            
            logger.info(f"Selected {len(selected_features)} features")
        else:
            X_train_selected = X_train
            X_test_selected = X_test
            selected_features = X_train.columns.tolist()
        
        # Models to train
        model_types = ['random_forest', 'gradient_boosting', 'logistic']
        trained_models = {}
        
        # Train each model type
        for model_type in model_types:
            logger.info(f"Training {model_type} model for {target_name}")
            
            # Hyperparameter tuning if requested
            if tune:
                logger.info(f"Performing hyperparameter tuning for {model_type}")
                tuning_results, best_model = trainer.hyperparameter_tuning(
                    X_train_selected, y_train,
                    model_type='classification',
                    algorithm=model_type,
                    cv=cv
                )
                
                # Save tuning results
                tuning_path = os.path.join(output_dir, f"{target_name}_{model_type}_tuning.json")
                with open(tuning_path, 'w') as f:
                    json.dump(tuning_results, f, indent=2, cls=NumpyEncoder)
                
                logger.info(f"Tuning completed. Best score: {tuning_results['best_score']:.4f}")
                
                # Use the best model
                model = best_model
                params = tuning_results['best_params']
            else:
                # Train with default parameters
                model = trainer.train_classification_model(
                    X_train_selected, y_train,
                    target_name,
                    model_type=model_type
                )
                params = {}
            
            # Evaluate model
            metrics = trainer.evaluate_classification_model(model, X_test_selected, y_test)
            
            # Get feature importances
            importances = trainer.get_feature_importance(model, selected_features)
            
            # Save model
            model_dir = trainer.save_model(
                model,
                f"{target_name}_{model_type}",
                target_name,
                model_type,
                metrics,
                importances,
                selected_features
            )
            
            # Store in results
            trained_models[model_type] = {
                'model': model,
                'metrics': metrics,
                'importances': importances,
                'params': params,
                'model_dir': model_dir
            }
        
        # Create ensemble if requested
        if ensemble and len(trained_models) > 1:
            logger.info(f"Creating ensemble model for {target_name}")
            
            # Get list of models
            models_list = [model_info['model'] for model_info in trained_models.values()]
            
            # Create ensemble
            ensemble_model = trainer.create_ensemble_model(
                models_list,
                X_train_selected, y_train,
                model_type='classification'
            )
            
            # Evaluate ensemble
            ensemble_metrics = trainer.evaluate_classification_model(
                ensemble_model, X_test_selected, y_test
            )
            
            # Save ensemble model
            ensemble_dir = trainer.save_model(
                ensemble_model,
                f"{target_name}_ensemble",
                target_name,
                'ensemble',
                ensemble_metrics,
                {},  # No feature importances for ensemble
                selected_features
            )
            
            # Store in results
            trained_models['ensemble'] = {
                'model': ensemble_model,
                'metrics': ensemble_metrics,
                'model_dir': ensemble_dir
            }
        
        # Store results for this target
        results[target_name] = trained_models
    
    return results

def train_regression_models(data: Dict[str, pd.DataFrame], 
                           targets: List[str],
                           tune: bool = False,
                           cv: int = 5,
                           ensemble: bool = False,
                           feature_selection: bool = False,
                           output_dir: str = 'models') -> Dict[str, Any]:
    """Train regression models.
    
    Args:
        data: Dictionary with feature and target DataFrames
        targets: List of target column names to model
        tune: Whether to perform hyperparameter tuning
        cv: Number of cross-validation folds
        ensemble: Whether to create ensemble models
        feature_selection: Whether to perform feature selection
        output_dir: Directory to save models
        
    Returns:
        Dictionary with trained models and evaluation metrics
    """
    logger.info(f"Training regression models for targets: {targets}")
    
    # Initialize model trainer
    trainer = ModelTrainer(models_dir=output_dir)
    
    # Get training and test data
    X_train = data['X_train']
    X_test = data['X_test']
    
    # Results dictionary
    results = {}
    
    # Train model for each target
    for target_name in targets:
        logger.info(f"Training models for target: {target_name}")
        
        # Get target data
        if target_name not in data['targets'].columns:
            logger.warning(f"Target {target_name} not found in data")
            continue
        
        y_train = data['y_train'][target_name]
        y_test = data['y_test'][target_name]
        
        # Convert to numeric
        y_train = pd.to_numeric(y_train, errors='coerce').fillna(0)
        y_test = pd.to_numeric(y_test, errors='coerce').fillna(0)
        
        # Feature selection if requested
        if feature_selection:
            logger.info(f"Performing feature selection for {target_name}")
            X_train_selected, selected_features = trainer.feature_selection(
                X_train, y_train,
                method='importance',
                model_type='regression'
            )
            X_test_selected = X_test[selected_features]
            
            feature_selection_path = os.path.join(output_dir, f"{target_name}_selected_features.txt")
            with open(feature_selection_path, 'w') as f:
                for feature in selected_features:
                    f.write(f"{feature}\n")
            
            logger.info(f"Selected {len(selected_features)} features")
        else:
            X_train_selected = X_train
            X_test_selected = X_test
            selected_features = X_train.columns.tolist()
        
        # Models to train
        model_types = ['random_forest', 'gradient_boosting', 'linear']
        trained_models = {}
        
        # Train each model type
        for model_type in model_types:
            logger.info(f"Training {model_type} model for {target_name}")
            
            # Hyperparameter tuning if requested
            if tune:
                logger.info(f"Performing hyperparameter tuning for {model_type}")
                tuning_results, best_model = trainer.hyperparameter_tuning(
                    X_train_selected, y_train,
                    model_type='regression',
                    algorithm=model_type,
                    cv=cv
                )
                
                # Save tuning results
                # Save tuning results
                tuning_path = os.path.join(output_dir, f"{target_name}_{model_type}_tuning.json")
                with open(tuning_path, 'w') as f:
                    json.dump(tuning_results, f, indent=2, cls=NumpyEncoder)
                
                logger.info(f"Tuning completed. Best score: {tuning_results['best_score']:.4f}")
                
                # Use the best model
                model = best_model
                params = tuning_results['best_params']
            else:
                # Train with default parameters
                model = trainer.train_regression_model(
                    X_train_selected, y_train,
                    target_name,
                    model_type=model_type
                )
                params = {}
            
            # Evaluate model
            metrics = trainer.evaluate_regression_model(model, X_test_selected, y_test)
            
            # Get feature importances
            importances = trainer.get_feature_importance(model, selected_features)
            
            # Save model
            model_dir = trainer.save_model(
                model,
                f"{target_name}_{model_type}",
                target_name,
                model_type,
                metrics,
                importances,
                selected_features
            )
            
            # Store in results
            trained_models[model_type] = {
                'model': model,
                'metrics': metrics,
                'importances': importances,
                'params': params,
                'model_dir': model_dir
            }
        
        # Create ensemble if requested
        if ensemble and len(trained_models) > 1:
            logger.info(f"Creating ensemble model for {target_name}")
            
            # Get list of models
            models_list = [model_info['model'] for model_info in trained_models.values()]
            
            # Create ensemble
            ensemble_model = trainer.create_ensemble_model(
                models_list,
                X_train_selected, y_train,
                model_type='regression'
            )
            
            # Evaluate ensemble
            ensemble_metrics = trainer.evaluate_regression_model(
                ensemble_model, X_test_selected, y_test
            )
            
            # Save ensemble model
            ensemble_dir = trainer.save_model(
                ensemble_model,
                f"{target_name}_ensemble",
                target_name,
                'ensemble',
                ensemble_metrics,
                {},  # No feature importances for ensemble
                selected_features
            )
            
            # Store in results
            trained_models['ensemble'] = {
                'model': ensemble_model,
                'metrics': ensemble_metrics,
                'model_dir': ensemble_dir
            }
        
        # Store results for this target
        results[target_name] = trained_models
    
    return results


def main():
    """Main function to run the ML pipeline."""
    # Parse arguments
    args = parse_arguments()
    
    try:
        # Get data path
        if args.data == 'latest':
            data_dir = os.path.join(os.path.dirname(__file__), "data/processed")
            data_path = get_latest_data_file(data_dir)
        else:
            data_path = args.data
        
        logger.info(f"Using data from {data_path}")
        
        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Run pipeline steps
        if args.step in ['all', 'features']:
            # Run feature engineering
            data = run_feature_engineering(data_path, os.path.join(args.output_dir, 'features'))
        else:
            # Load existing feature data
            logger.info("Loading existing feature data")
            feature_dir = os.path.join(args.output_dir, 'features')
            
            features = pd.read_csv(os.path.join(feature_dir, 'features.csv'), index_col=0)
            targets = pd.read_csv(os.path.join(feature_dir, 'targets.csv'), index_col=0)
            
            # Initialize feature engineer for splitting
            engineer = FeatureEngineer()
            X_train, X_test, y_train, y_test = engineer.split_data(features, targets, test_size=args.test_size)
            
            data = {
                'features': features,
                'targets': targets,
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test
            }
        
        if args.step in ['all', 'train']:
            # Define targets to model
            classification_targets = []
            regression_targets = []
            
            # Look for available targets
            available_targets = data['targets'].columns
            
            # Classification targets (binary)
            for target in ['is_viral', 'is_highly_engaging', 'is_long_trending']:
                if target in available_targets:
                    classification_targets.append(target)
            
            # Regression targets
            for target in ['engagement_score', 'views_per_hour', 'views_per_hour_log', 'like_view_ratio']:
                if target in available_targets:
                    regression_targets.append(target)
            
            # Filter based on args.models
            if args.models != 'all':
                if args.models == 'classification':
                    regression_targets = []
                elif args.models == 'regression':
                    classification_targets = []
                elif args.models == 'viral':
                    classification_targets = [t for t in classification_targets if 'viral' in t]
                    regression_targets = []
                elif args.models == 'engagement':
                    classification_targets = [t for t in classification_targets if 'engaging' in t]
                    regression_targets = [t for t in regression_targets if 'engagement' in t]
                elif args.models == 'views':
                    classification_targets = []
                    regression_targets = [t for t in regression_targets if 'views' in t]
            
            # Train classification models
            if classification_targets:
                logger.info(f"Training classification models: {classification_targets}")
                classification_results = train_classification_models(
                    data,
                    classification_targets,
                    tune=args.tune,
                    cv=args.cv,
                    ensemble=args.ensemble,
                    feature_selection=args.feature_selection,
                    output_dir=os.path.join(args.output_dir, 'classification')
                )
            
            # Train regression models
            if regression_targets:
                logger.info(f"Training regression models: {regression_targets}")
                regression_results = train_regression_models(
                    data,
                    regression_targets,
                    tune=args.tune,
                    cv=args.cv,
                    ensemble=args.ensemble,
                    feature_selection=args.feature_selection,
                    output_dir=os.path.join(args.output_dir, 'regression')
                )
        
        if args.step in ['all', 'evaluate']:
            # Evaluation is integrated into training for now
            # Could add more comprehensive evaluation here
            logger.info("Evaluation completed during training")
        
        if args.step in ['all', 'deploy']:
            # No deployment step yet
            logger.info("No deployment step implemented yet")
        
        logger.info("ML pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Error in ML pipeline: {e}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())