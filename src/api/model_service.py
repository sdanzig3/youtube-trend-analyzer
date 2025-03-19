# src/api/model_service.py
import os
import pickle
import json
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("model_service")

class ModelService:
    """Service for loading and making predictions with trained models."""
    
    def __init__(self, models_dir: str = 'models'):
        """Initialize the model service.
        
        Args:
            models_dir: Directory containing trained models
        """
        self.models_dir = models_dir
        self.classification_models = {}
        self.regression_models = {}
        self.feature_lists = {}
        
        # Load available models
        self._load_models()
    
    def _load_models(self):
        """Load all available models from the models directory."""
        logger.info(f"Loading models from {self.models_dir}")
        
        # Check classification models
        cls_dir = os.path.join(self.models_dir, 'classification')
        if os.path.exists(cls_dir):
            for model_dir in os.listdir(cls_dir):
                model_path = os.path.join(cls_dir, model_dir)
                if os.path.isdir(model_path):
                    self._load_model(model_path, 'classification')
        
        # Check regression models
        reg_dir = os.path.join(self.models_dir, 'regression')
        if os.path.exists(reg_dir):
            for model_dir in os.listdir(reg_dir):
                model_path = os.path.join(reg_dir, model_dir)
                if os.path.isdir(model_path):
                    self._load_model(model_path, 'regression')
        
        logger.info(f"Loaded {len(self.classification_models)} classification models and {len(self.regression_models)} regression models")
    
    def _load_model(self, model_dir: str, model_category: str):
        """Load a single model from a directory.
        
        Args:
            model_dir: Directory containing the model files
            model_category: 'classification' or 'regression'
        """
        try:
            # Get model name from directory
            model_name = os.path.basename(model_dir)
            
            # Check for model and metadata files
            model_file = os.path.join(model_dir, f"{model_name}.pkl")
            metadata_file = os.path.join(model_dir, f"{model_name}_metadata.json")
            
            if not os.path.exists(model_file) or not os.path.exists(metadata_file):
                logger.warning(f"Skipping {model_name}: Missing files")
                return
            
            # Load model
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
            
            # Load metadata
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Extract target name and feature names
            target_name = metadata.get('target_name', 'unknown')
            features = metadata.get('features', [])
            
            # Store model info
            model_info = {
                'model': model,
                'metadata': metadata,
                'features': features
            }
            
            # Add to appropriate model dictionary
            if model_category == 'classification':
                self.classification_models[target_name] = model_info
            else:
                self.regression_models[target_name] = model_info
            
            # Store feature list
            self.feature_lists[target_name] = features
            
            logger.info(f"Loaded {model_category} model for target: {target_name}")
            
        except Exception as e:
            logger.error(f"Error loading model from {model_dir}: {e}")
    
    def get_available_models(self) -> Dict[str, List[str]]:
        """Get a list of available models.
        
        Returns:
            Dictionary with available classification and regression models
        """
        return {
            'classification': list(self.classification_models.keys()),
            'regression': list(self.regression_models.keys())
        }
    
    def get_required_features(self, target_name: str) -> List[str]:
        """Get the list of features required for a specific model.
        
        Args:
            target_name: Name of the target variable
            
        Returns:
            List of feature names
        """
        return self.feature_lists.get(target_name, [])
    
    def prepare_features(self, data: Dict[str, Any], target_name: str) -> pd.DataFrame:
        """Prepare features for prediction.
        
        Args:
            data: Input data dictionary
            target_name: Name of the target variable
            
        Returns:
            DataFrame with prepared features
        """
        # Get required features
        required_features = self.get_required_features(target_name)
        
        if not required_features:
            raise ValueError(f"No feature information available for {target_name}")
        
        # Create a single-row DataFrame
        df = pd.DataFrame([data])
        
        # Check for missing features
        missing_features = [f for f in required_features if f not in df.columns]
        
        # For any missing features, add them with default value 0
        for feature in missing_features:
            df[feature] = 0
        
        # Select only the required features in the correct order
        features_df = df[required_features]
        
        return features_df
    
    def predict_classification(self, data: Dict[str, Any], target_name: str) -> Dict[str, Any]:
        """Make a classification prediction.
        
        Args:
            data: Input data dictionary
            target_name: Name of the target variable
            
        Returns:
            Dictionary with prediction results
        """
        # Check if model exists
        if target_name not in self.classification_models:
            raise ValueError(f"No classification model available for {target_name}")
        
        # Get model info
        model_info = self.classification_models[target_name]
        model = model_info['model']
        metadata = model_info['metadata']
        
        # Prepare features
        features = self.prepare_features(data, target_name)
        
        # Make prediction
        try:
            # Predict class
            pred_class = model.predict(features)[0]
            
            # Predict probability if available
            pred_prob = None
            if hasattr(model, 'predict_proba'):
                pred_prob = model.predict_proba(features)[0, 1]
            
            # Create result
            result = {
                'target': target_name,
                'prediction': bool(pred_class),
                'probability': float(pred_prob) if pred_prob is not None else None,
                'model_type': metadata.get('model_type', 'unknown')
            }
            
            return result
        
        except Exception as e:
            logger.error(f"Error making prediction for {target_name}: {e}")
            raise
    
    def predict_regression(self, data: Dict[str, Any], target_name: str) -> Dict[str, Any]:
        """Make a regression prediction.
        
        Args:
            data: Input data dictionary
            target_name: Name of the target variable
            
        Returns:
            Dictionary with prediction results
        """
        # Check if model exists
        if target_name not in self.regression_models:
            raise ValueError(f"No regression model available for {target_name}")
        
        # Get model info
        model_info = self.regression_models[target_name]
        model = model_info['model']
        metadata = model_info['metadata']
        
        # Prepare features
        features = self.prepare_features(data, target_name)
        
        # Make prediction
        try:
            # Predict value
            pred_value = model.predict(features)[0]
            
            # Create result
            result = {
                'target': target_name,
                'prediction': float(pred_value),
                'model_type': metadata.get('model_type', 'unknown')
            }
            
            return result
        
        except Exception as e:
            logger.error(f"Error making prediction for {target_name}: {e}")
            raise
    
    def make_all_predictions(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make predictions with all available models.
        
        Args:
            data: Input data dictionary
            
        Returns:
            Dictionary with all prediction results
        """
        results = {
            'classification': {},
            'regression': {}
        }
        
        # Make classification predictions
        for target_name in self.classification_models:
            try:
                pred = self.predict_classification(data, target_name)
                results['classification'][target_name] = pred
            except Exception as e:
                logger.error(f"Error predicting {target_name}: {e}")
                results['classification'][target_name] = {'error': str(e)}
        
        # Make regression predictions
        for target_name in self.regression_models:
            try:
                pred = self.predict_regression(data, target_name)
                results['regression'][target_name] = pred
            except Exception as e:
                logger.error(f"Error predicting {target_name}: {e}")
                results['regression'][target_name] = {'error': str(e)}
        
        return results