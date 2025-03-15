# src/models/training.py
import os
import json
import pickle
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Train and evaluate machine learning models for YouTube trending prediction."""
    
    def __init__(self, models_dir: str = 'models'):
        """Initialize the model trainer.
        
        Args:
            models_dir: Directory to save trained models
        """
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        logger.info(f"ModelTrainer initialized with models directory: {models_dir}")
    
    def train_classification_model(self, 
                                  X_train: pd.DataFrame, 
                                  y_train: pd.Series,
                                  target_name: str,
                                  model_type: str = 'random_forest',
                                  params: Optional[Dict[str, Any]] = None) -> Any:
        """Train a classification model.
        
        Args:
            X_train: Training features
            y_train: Training targets (binary)
            target_name: Name of the target variable (e.g., 'is_viral')
            model_type: Type of model to train ('random_forest', 'gradient_boosting', 'logistic')
            params: Model hyperparameters
            
        Returns:
            Trained model object
        """
        logger.info(f"Training {model_type} classification model for {target_name}")
        
        # Make sure input data is clean
        X_train = X_train.select_dtypes(include=['int64', 'float64', 'bool'])
        y_train = y_train.astype(int)
        
        # Default parameters if none provided
        if params is None:
            params = {}
        
        if model_type == 'random_forest':
            from sklearn.ensemble import RandomForestClassifier
            
            default_params = {
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'max_features': 'sqrt',
                'random_state': 42
            }
            
            # Update defaults with provided params
            model_params = {**default_params, **params}
            model = RandomForestClassifier(**model_params)
            
        elif model_type == 'gradient_boosting':
            from sklearn.ensemble import GradientBoostingClassifier
            
            default_params = {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 3,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'max_features': 'sqrt',  # Changed from 'auto' to 'sqrt'
                'random_state': 42
            }
            
            # Update defaults with provided params
            model_params = {**default_params, **params}
            model = GradientBoostingClassifier(**model_params)
            
        elif model_type == 'logistic':
            from sklearn.linear_model import LogisticRegression
            
            default_params = {
                'penalty': 'l2',
                'C': 1.0,
                'solver': 'lbfgs',
                'max_iter': 1000,
                'random_state': 42
            }
            
            # Update defaults with provided params
            model_params = {**default_params, **params}
            model = LogisticRegression(**model_params)
            
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Train the model
        model.fit(X_train, y_train)
        logger.info(f"Model training completed")
        
        return model
    
    def train_regression_model(self,
                              X_train: pd.DataFrame,
                              y_train: pd.Series,
                              target_name: str,
                              model_type: str = 'random_forest',
                              params: Optional[Dict[str, Any]] = None) -> Any:
        """Train a regression model.
        
        Args:
            X_train: Training features
            y_train: Training targets (continuous)
            target_name: Name of the target variable (e.g., 'views_per_hour')
            model_type: Type of model to train ('random_forest', 'gradient_boosting', 'linear')
            params: Model hyperparameters
            
        Returns:
            Trained model object
        """
        logger.info(f"Training {model_type} regression model for {target_name}")
        
        # Make sure input data is clean
        X_train = X_train.select_dtypes(include=['int64', 'float64', 'bool'])
        y_train = pd.to_numeric(y_train, errors='coerce').fillna(0)
        
        # Default parameters if none provided
        if params is None:
            params = {}
        
        if model_type == 'random_forest':
            from sklearn.ensemble import RandomForestRegressor
            
            default_params = {
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'max_features': 'sqrt',  # Changed from 'auto' to 'sqrt'
                'random_state': 42
            }
            
            # Update defaults with provided params
            model_params = {**default_params, **params}
            model = RandomForestRegressor(**model_params)
            
        elif model_type == 'gradient_boosting':
            from sklearn.ensemble import GradientBoostingRegressor
            
            default_params = {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 3,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'max_features': 'sqrt',  # Changed from 'auto' to 'sqrt'
                'random_state': 42
            }
            
            # Update defaults with provided params
            model_params = {**default_params, **params}
            model = GradientBoostingRegressor(**model_params)
            
        elif model_type == 'linear':
            from sklearn.linear_model import Ridge
            
            default_params = {
                'alpha': 1.0,
                'fit_intercept': True,
                'random_state': 42
            }
            
            # Update defaults with provided params
            model_params = {**default_params, **params}
            model = Ridge(**model_params)
            
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Train the model
        model.fit(X_train, y_train)
        logger.info(f"Model training completed")
        
        return model
    
    def evaluate_classification_model(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate a classification model.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary of evaluation metrics
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        # Make sure data is clean
        X_test = X_test.select_dtypes(include=['int64', 'float64', 'bool'])
        y_test = y_test.astype(int)
        
        try:
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = None
            
            # Check if model can predict probabilities
            if hasattr(model, "predict_proba"):
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0)
            }
            
            # Add AUC if we have probability predictions
            if y_pred_proba is not None:
                metrics['auc'] = roc_auc_score(y_test, y_pred_proba)
            
            # Log the metrics
            logger.info(f"Classification metrics: {metrics}")
            
            return metrics
        except Exception as e:
            logger.error(f"Error evaluating classification model: {e}")
            return {'error': str(e)}
    
    def evaluate_regression_model(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate a regression model.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary of evaluation metrics
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        # Make sure data is clean
        X_test = X_test.select_dtypes(include=['int64', 'float64', 'bool'])
        y_test = pd.to_numeric(y_test, errors='coerce').fillna(0)
        
        try:
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            metrics = {
                'mse': mse,
                'rmse': np.sqrt(mse),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred)
            }
            
            # Log the metrics
            logger.info(f"Regression metrics: {metrics}")
            
            return metrics
        except Exception as e:
            logger.error(f"Error evaluating regression model: {e}")
            return {'error': str(e)}
    
    def get_feature_importance(self, model: Any, feature_names: List[str]) -> Dict[str, float]:
        """Get feature importances from the model.
        
        Args:
            model: Trained model
            feature_names: List of feature names
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        try:
            # Check if model has feature importances
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
            elif hasattr(model, "coef_"):
                importances = np.abs(model.coef_)
                if importances.ndim > 1:
                    importances = importances.mean(axis=0)
            else:
                logger.warning("Model doesn't provide feature importances")
                return {}
            
            # Make sure we have the right number of feature names
            if len(feature_names) != len(importances):
                logger.warning(f"Feature name count ({len(feature_names)}) doesn't match importance count ({len(importances)})")
                # Use generic feature names if mismatch
                feature_names = [f"feature_{i}" for i in range(len(importances))]
            
            # Create dictionary of feature importances
            importance_dict = dict(zip(feature_names, importances))
            
            # Sort by importance (descending)
            importance_dict = {k: v for k, v in sorted(importance_dict.items(), key=lambda item: item[1], reverse=True)}
            
            return importance_dict
        except Exception as e:
            logger.error(f"Error getting feature importances: {e}")
            return {}
    
    def save_model(self, 
                  model: Any, 
                  model_name: str, 
                  target_name: str,
                  model_type: str,
                  metrics: Dict[str, float],
                  feature_importances: Dict[str, float],
                  feature_names: List[str]) -> str:
        """Save the trained model and metadata.
        
        Args:
            model: Trained model
            model_name: Name to save the model under
            target_name: Name of the target variable
            model_type: Type of model
            metrics: Evaluation metrics
            feature_importances: Feature importance scores
            feature_names: List of feature names
            
        Returns:
            Path to saved model
        """
        try:
            # Create metadata
            metadata = {
                'model_name': model_name,
                'target_name': target_name,
                'model_type': model_type,
                'creation_time': datetime.now().isoformat(),
                'metrics': metrics,
                'feature_importances': {k: float(v) for k, v in list(feature_importances.items())[:20]},  # Top 20 features
                'features': feature_names
            }
            
            # Create model directory
            model_dir = os.path.join(self.models_dir, model_name)
            os.makedirs(model_dir, exist_ok=True)
            
            # Save model
            model_path = os.path.join(model_dir, f"{model_name}.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            # Save metadata
            metadata_path = os.path.join(model_dir, f"{model_name}_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            logger.info(f"Model saved to {model_path}")
            logger.info(f"Metadata saved to {metadata_path}")
            
            return model_dir
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return ""
    
    def load_model(self, model_name: str) -> Tuple[Any, Dict[str, Any]]:
        """Load a saved model and its metadata.
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            Tuple of (model, metadata)
        """
        try:
            model_dir = os.path.join(self.models_dir, model_name)
            
            # Check if model directory exists
            if not os.path.exists(model_dir):
                raise FileNotFoundError(f"Model directory not found: {model_dir}")
            
            # Load model
            model_path = os.path.join(model_dir, f"{model_name}.pkl")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            # Load metadata
            metadata_path = os.path.join(model_dir, f"{model_name}_metadata.json")
            if not os.path.exists(metadata_path):
                logger.warning(f"Metadata file not found: {metadata_path}")
                metadata = {}
            else:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            
            logger.info(f"Loaded model from {model_path}")
            
            return model, metadata
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return None, {}


# Example usage
if __name__ == "__main__":
    import os
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    # Initialize model trainer
    trainer = ModelTrainer()
    
    # Load prepared features and targets
    data_dir = os.path.join(os.path.dirname(__file__), "../../data/ml")
    features_path = os.path.join(data_dir, "features.csv")
    targets_path = os.path.join(data_dir, "targets.csv")
    
    if not os.path.exists(features_path) or not os.path.exists(targets_path):
        print("Feature files not found. Run feature engineering first.")
        exit(1)
    
    print(f"Loading features from {features_path}")
    features = pd.read_csv(features_path, index_col=0)
    
    print(f"Loading targets from {targets_path}")
    targets = pd.read_csv(targets_path, index_col=0)
    
    # Check available target columns
    print(f"Available target variables: {list(targets.columns)}")
    
    # Train classification model for 'is_viral' if available
    if 'is_viral' in targets.columns:
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, targets['is_viral'], test_size=0.2, random_state=42
        )
        
        # Train model
        clf_model = trainer.train_classification_model(
            X_train, y_train, 'is_viral', model_type='random_forest'
        )
        
        # Evaluate model
        metrics = trainer.evaluate_classification_model(clf_model, X_test, y_test)
        
        # Get feature importances
        importances = trainer.get_feature_importance(clf_model, features.columns.tolist())
        
        # Save model
        trainer.save_model(
            clf_model, 
            'viral_prediction_rf', 
            'is_viral',
            'random_forest',
            metrics,
            importances,
            features.columns.tolist()
        )
    
    # Train regression model for 'engagement_score' if available
    if 'engagement_score' in targets.columns:
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, targets['engagement_score'], test_size=0.2, random_state=42
        )
        
        # Train model
        reg_model = trainer.train_regression_model(
            X_train, y_train, 'engagement_score', model_type='gradient_boosting'
        )
        
        # Evaluate model
        metrics = trainer.evaluate_regression_model(reg_model, X_test, y_test)
        
        # Get feature importances
        importances = trainer.get_feature_importance(reg_model, features.columns.tolist())
        
        # Save model
        trainer.save_model(
            reg_model, 
            'engagement_prediction_gb', 
            'engagement_score',
            'gradient_boosting',
            metrics,
            importances,
            features.columns.tolist()
        )
    
    print("Model training completed")