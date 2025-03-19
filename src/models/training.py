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
    # Add these methods to your ModelTrainer class

    def hyperparameter_tuning(self, 
                        X_train: pd.DataFrame, 
                        y_train: pd.Series,
                        model_type: str = 'classification',
                        algorithm: str = 'random_forest',
                        cv: int = 5,
                        scoring: str = None,
                        n_iter: int = 20) -> Dict[str, Any]:
        """Perform hyperparameter tuning for a model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            model_type: 'classification' or 'regression'
            algorithm: 'random_forest', 'gradient_boosting', 'xgboost', etc.
            cv: Number of cross-validation folds
            scoring: Scoring metric (if None, use default for model type)
            n_iter: Number of parameter settings to try for randomized search
            
        Returns:
            Dictionary with best parameters and best score
        """
        from sklearn.model_selection import RandomizedSearchCV, KFold
        import numpy as np
        
        logger.info(f"Starting hyperparameter tuning for {algorithm} {model_type}")
        
        # Make sure input data is clean
        X_train = X_train.select_dtypes(include=['int64', 'float64', 'bool'])
        
        if model_type == 'classification':
            y_train = y_train.astype(int)
            if scoring is None:
                scoring = 'roc_auc'
        elif model_type == 'regression':
            y_train = pd.to_numeric(y_train, errors='coerce').fillna(0)
            if scoring is None:
                scoring = 'neg_root_mean_squared_error'
        else:
            raise ValueError(f"Invalid model_type: {model_type}. Must be 'classification' or 'regression'")
        
        # Define parameter spaces for different algorithms
        param_spaces = {}
        
        # Random Forest parameters
        if algorithm == 'random_forest':
            if model_type == 'classification':
                param_spaces = {
                    'n_estimators': np.arange(50, 500, 50),
                    'max_depth': [None, 5, 10, 15, 20, 25, 30],
                    'min_samples_split': [2, 5, 10, 15, 20],
                    'min_samples_leaf': [1, 2, 4, 6, 8],
                    'max_features': ['sqrt', 'log2', None],
                    'class_weight': [None, 'balanced']
                }
            else:  # regression
                param_spaces = {
                    'n_estimators': np.arange(50, 500, 50),
                    'max_depth': [None, 5, 10, 15, 20, 25, 30],
                    'min_samples_split': [2, 5, 10, 15, 20],
                    'min_samples_leaf': [1, 2, 4, 6, 8],
                    'max_features': ['sqrt', 'log2', None]
                }
        
        # Gradient Boosting parameters
        elif algorithm == 'gradient_boosting':
            if model_type == 'classification':
                param_spaces = {
                    'n_estimators': np.arange(50, 500, 50),
                    'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2],
                    'max_depth': [3, 4, 5, 6, 8, 10],
                    'min_samples_split': [2, 5, 10, 15, 20],
                    'min_samples_leaf': [1, 2, 4, 6, 8],
                    'max_features': ['sqrt', 'log2', None],
                    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0]
                }
            else:  # regression
                param_spaces = {
                    'n_estimators': np.arange(50, 500, 50),
                    'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2],
                    'max_depth': [3, 4, 5, 6, 8, 10],
                    'min_samples_split': [2, 5, 10, 15, 20],
                    'min_samples_leaf': [1, 2, 4, 6, 8],
                    'max_features': ['sqrt', 'log2', None],
                    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0]
                }
        
        # XGBoost parameters
        elif algorithm == 'xgboost':
            if model_type == 'classification':
                param_spaces = {
                    'n_estimators': np.arange(50, 500, 50),
                    'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2],
                    'max_depth': [3, 4, 5, 6, 8, 10],
                    'min_child_weight': [1, 3, 5, 7],
                    'gamma': [0, 0.1, 0.2, 0.3, 0.4],
                    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0]
                }
            else:  # regression
                param_spaces = {
                    'n_estimators': np.arange(50, 500, 50),
                    'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2],
                    'max_depth': [3, 4, 5, 6, 8, 10],
                    'min_child_weight': [1, 3, 5, 7],
                    'gamma': [0, 0.1, 0.2, 0.3, 0.4],
                    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0]
                }
        
        # Logistic Regression parameters (classification only)
        elif algorithm == 'logistic':
            if model_type == 'classification':
                param_spaces = {
                    'C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2', 'elasticnet', None],
                    'solver': ['lbfgs', 'liblinear', 'saga'],
                    'class_weight': [None, 'balanced'],
                    'max_iter': [100, 500, 1000, 2000]
                }
            else:
                raise ValueError("Logistic regression is only for classification tasks")
        
        # Linear model parameters (regression only)
        elif algorithm == 'linear':
            if model_type == 'regression':
                param_spaces = {
                    'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
                    'fit_intercept': [True, False],
                    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
                }
            else:
                raise ValueError("Linear models in this context are only for regression tasks")
        
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        # Set up cross-validation
        cv_splitter = KFold(n_splits=cv, shuffle=True, random_state=42)
        
        # Create the appropriate model
        if model_type == 'classification':
            if algorithm == 'random_forest':
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(random_state=42)
            elif algorithm == 'gradient_boosting':
                from sklearn.ensemble import GradientBoostingClassifier
                model = GradientBoostingClassifier(random_state=42)
            elif algorithm == 'xgboost':
                try:
                    import xgboost as xgb
                    model = xgb.XGBClassifier(objective='binary:logistic', random_state=42)
                except ImportError:
                    logger.warning("XGBoost not installed, falling back to GradientBoostingClassifier")
                    from sklearn.ensemble import GradientBoostingClassifier
                    model = GradientBoostingClassifier(random_state=42)
                    algorithm = 'gradient_boosting'
                    # Redefine param_spaces for the new algorithm
                    param_spaces = {
                        'n_estimators': np.arange(50, 500, 50),
                        'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2],
                        'max_depth': [3, 4, 5, 6, 8, 10],
                        'min_samples_split': [2, 5, 10, 15, 20],
                        'min_samples_leaf': [1, 2, 4, 6, 8],
                        'max_features': ['sqrt', 'log2', None],
                        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0]
                    }
            elif algorithm == 'logistic':
                from sklearn.linear_model import LogisticRegression
                model = LogisticRegression(random_state=42)
            else:
                # Default to random forest
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(random_state=42)
                algorithm = 'random_forest'
                # Redefine param_spaces for the new algorithm
                param_spaces = {
                    'n_estimators': np.arange(50, 500, 50),
                    'max_depth': [None, 5, 10, 15, 20, 25, 30],
                    'min_samples_split': [2, 5, 10, 15, 20],
                    'min_samples_leaf': [1, 2, 4, 6, 8],
                    'max_features': ['sqrt', 'log2', None],
                    'class_weight': [None, 'balanced']
                }
        else:  # regression
            if algorithm == 'random_forest':
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor(random_state=42)
            elif algorithm == 'gradient_boosting':
                from sklearn.ensemble import GradientBoostingRegressor
                model = GradientBoostingRegressor(random_state=42)
            elif algorithm == 'xgboost':
                try:
                    import xgboost as xgb
                    model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
                except ImportError:
                    logger.warning("XGBoost not installed, falling back to GradientBoostingRegressor")
                    from sklearn.ensemble import GradientBoostingRegressor
                    model = GradientBoostingRegressor(random_state=42)
                    algorithm = 'gradient_boosting'
                    # Redefine param_spaces for the new algorithm
                    param_spaces = {
                        'n_estimators': np.arange(50, 500, 50),
                        'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2],
                        'max_depth': [3, 4, 5, 6, 8, 10],
                        'min_samples_split': [2, 5, 10, 15, 20],
                        'min_samples_leaf': [1, 2, 4, 6, 8],
                        'max_features': ['sqrt', 'log2', None],
                        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0]
                    }
            elif algorithm == 'linear':
                from sklearn.linear_model import Ridge
                model = Ridge(random_state=42)
            else:
                # Default to random forest
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor(random_state=42)
                algorithm = 'random_forest'
                # Redefine param_spaces for the new algorithm
                param_spaces = {
                    'n_estimators': np.arange(50, 500, 50),
                    'max_depth': [None, 5, 10, 15, 20, 25, 30],
                    'min_samples_split': [2, 5, 10, 15, 20],
                    'min_samples_leaf': [1, 2, 4, 6, 8],
                    'max_features': ['sqrt', 'log2', None]
                }
        
        # Set up randomized search
        search = RandomizedSearchCV(
            model,
            param_distributions=param_spaces,
            n_iter=n_iter,
            cv=cv_splitter,
            scoring=scoring,
            random_state=42,
            n_jobs=-1,  # Use all available cores
            verbose=1,
            return_train_score=True
        )
        
        # Fit the search
        search.fit(X_train, y_train)
        
        # Get best parameters and score
        best_params = search.best_params_
        best_score = search.best_score_
        
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best {scoring} score: {best_score:.4f}")
        
        # Get cross-validation results
        cv_results = {
            'best_params': best_params,
            'best_score': best_score,
            'mean_test_score': search.cv_results_['mean_test_score'],
            'std_test_score': search.cv_results_['std_test_score'],
            'mean_train_score': search.cv_results_['mean_train_score'],
            'params': search.cv_results_['params']
        }
        
        return cv_results, search.best_estimator_
    
    def cross_validate_model(self, model, X, y, cv=5, scoring=None):
        """Perform cross-validation for a model.
        
        Args:
            model: Model to evaluate
            X: Features
            y: Targets
            cv: Number of cross-validation folds
            scoring: Scoring metric or list of metrics
            
        Returns:
            Dictionary with cross-validation results
        """
        from sklearn.model_selection import cross_validate, KFold
        
        # Make sure input data is clean
        X = X.select_dtypes(include=['int64', 'float64', 'bool'])
        
        # Set up cross-validation
        cv_splitter = KFold(n_splits=cv, shuffle=True, random_state=42)
        
        # If no scoring provided, use default based on target type
        if scoring is None:
            if np.issubdtype(y.dtype, np.integer) or y.dtype == bool:
                # Classification
                scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
            else:
                # Regression
                scoring = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']
        
        # Perform cross-validation
        cv_results = cross_validate(
            model, X, y,
            cv=cv_splitter,
            scoring=scoring,
            return_train_score=True,
            n_jobs=-1
        )
        
        # Format results
        results = {}
        for metric in scoring:
            metric_key = f'test_{metric}'
            if metric_key in cv_results:
                results[metric] = {
                    'mean': cv_results[metric_key].mean(),
                    'std': cv_results[metric_key].std(),
                    'values': cv_results[metric_key].tolist()
                }
        
        return results

    def create_ensemble_model(self, 
                            models: List[Any], 
                            X_train: pd.DataFrame, 
                            y_train: pd.Series,
                            weights: Optional[List[float]] = None,
                            model_type: str = 'classification') -> Any:
        """Create an ensemble model from multiple base models.
        
        Args:
            models: List of trained models
            X_train: Training features
            y_train: Training targets
            weights: Optional weights for each model (if None, use uniform weights)
            model_type: 'classification' or 'regression'
            
        Returns:
            Ensemble model
        """
        from sklearn.ensemble import VotingClassifier, VotingRegressor
        
        # Check if we have models
        if not models:
            raise ValueError("No models provided for ensemble")
        
        # Create named estimators
        named_estimators = [
            (f"model_{i}", model) for i, model in enumerate(models)
        ]
        
        # Create appropriate ensemble
        if model_type == 'classification':
            # For classification, we can use probabilities
            ensemble = VotingClassifier(
                estimators=named_estimators,
                voting='soft',  # Use predicted probabilities
                weights=weights,
                n_jobs=-1
            )
        else:  # regression
            ensemble = VotingRegressor(
                estimators=named_estimators,
                weights=weights,
                n_jobs=-1
            )
        
        # Fit the ensemble
        ensemble.fit(X_train, y_train)
        
        return ensemble

    def feature_selection(self, X_train: pd.DataFrame, y_train: pd.Series, 
                        feature_count: Optional[int] = None,
                        method: str = 'importance',
                        model_type: str = 'classification') -> Tuple[pd.DataFrame, List[str]]:
        """Select the most important features.
        
        Args:
            X_train: Training features
            y_train: Training targets
            feature_count: Number of features to select (if None, use automatic selection)
            method: Method to use ('importance', 'recursive', 'univariate')
            model_type: 'classification' or 'regression'
            
        Returns:
            Tuple of (selected features DataFrame, list of selected feature names)
        """
        from sklearn.feature_selection import SelectFromModel, RFECV, SelectKBest, f_classif, f_regression
        
        # Make sure input data is clean
        X_train = X_train.select_dtypes(include=['int64', 'float64', 'bool'])
        
        # Default feature count if not specified
        if feature_count is None:
            feature_count = min(50, X_train.shape[1])
        
        selected_features = None
        
        if method == 'importance':
            # Use a model's feature importances
            if model_type == 'classification':
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            
            model.fit(X_train, y_train)
            
            # Select features based on importance
            selector = SelectFromModel(model, max_features=feature_count, threshold=-np.inf)
            selector.fit(X_train, y_train)
            
        elif method == 'recursive':
            # Use recursive feature elimination with cross-validation
            if model_type == 'classification':
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            
            selector = RFECV(
                estimator=model,
                step=1,
                cv=5,
                scoring='roc_auc' if model_type == 'classification' else 'neg_mean_squared_error',
                min_features_to_select=min(feature_count, X_train.shape[1] // 2)
            )
            selector.fit(X_train, y_train)
            
        elif method == 'univariate':
            # Use univariate feature selection
            if model_type == 'classification':
                selector = SelectKBest(f_classif, k=feature_count)
            else:
                selector = SelectKBest(f_regression, k=feature_count)
            
            selector.fit(X_train, y_train)
        
        else:
            raise ValueError(f"Invalid feature selection method: {method}")
        
        # Get selected features
        selected_indices = selector.get_support()
        selected_feature_names = X_train.columns[selected_indices].tolist()
        
        # Log selected features
        logger.info(f"Selected {len(selected_feature_names)} features using {method} method")
        
        # Create DataFrame with selected features
        X_selected = X_train[selected_feature_names]
        
        return X_selected, selected_feature_names

    def train_stacked_ensemble(self, 
                         X_train: pd.DataFrame, 
                         y_train: pd.Series,
                         X_test: pd.DataFrame,
                         y_test: pd.Series,
                         base_models: List[Dict[str, Any]],
                         meta_model: Dict[str, Any],
                         model_type: str = 'classification',
                         cv: int = 5) -> Dict[str, Any]:
        """Train a stacked ensemble model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Testing features
            y_test: Testing targets
            base_models: List of dictionaries with 'name', 'model', and 'params'
            meta_model: Dictionary with 'name', 'model', and 'params'
            model_type: 'classification' or 'regression'
            cv: Number of cross-validation folds
            
        Returns:
            Dictionary with trained models and evaluation metrics
        """
        from sklearn.model_selection import cross_val_predict, KFold
        import numpy as np
        
        logger.info(f"Training stacked ensemble with {len(base_models)} base models")
        
        # Make sure input data is clean
        X_train = X_train.select_dtypes(include=['int64', 'float64', 'bool'])
        X_test = X_test.select_dtypes(include=['int64', 'float64', 'bool'])
        
        # For classification, convert targets to int
        if model_type == 'classification':
            y_train = y_train.astype(int)
            y_test = y_test.astype(int)
        
        # Set up cross-validation
        cv_splitter = KFold(n_splits=cv, shuffle=True, random_state=42)
        
        # Train base models and generate meta-features
        trained_base_models = []
        meta_features_train = np.zeros((X_train.shape[0], len(base_models)))
        meta_features_test = np.zeros((X_test.shape[0], len(base_models)))
        
        for i, model_config in enumerate(base_models):
            model_name = model_config['name']
            model = model_config['model']
            params = model_config.get('params', {})
            
            logger.info(f"Training base model {i+1}/{len(base_models)}: {model_name}")
            
            # Set parameters if provided
            if params:
                model.set_params(**params)
            
            # Generate out-of-fold predictions for training data
            if model_type == 'classification':
                try:
                    # Try to use predict_proba
                    oof_preds = cross_val_predict(
                        model, X_train, y_train, 
                        cv=cv_splitter, 
                        method='predict_proba',
                        n_jobs=-1
                    )
                    # Use the probability of positive class
                    meta_features_train[:, i] = oof_preds[:, 1]
                except:
                    # Fall back to predict
                    oof_preds = cross_val_predict(
                        model, X_train, y_train, 
                        cv=cv_splitter, 
                        n_jobs=-1
                    )
                    meta_features_train[:, i] = oof_preds
            else:
                # For regression, just use predict
                oof_preds = cross_val_predict(
                    model, X_train, y_train, 
                    cv=cv_splitter, 
                    n_jobs=-1
                )
                meta_features_train[:, i] = oof_preds
            
            # Train the model on the full training data
            model.fit(X_train, y_train)
            trained_base_models.append(model)
            
            # Generate predictions for test data
            if model_type == 'classification' and hasattr(model, 'predict_proba'):
                test_preds = model.predict_proba(X_test)[:, 1]
            else:
                test_preds = model.predict(X_test)
            
            meta_features_test[:, i] = test_preds
        
        # Train meta-model
        meta_model_name = meta_model['name']
        meta_model_instance = meta_model['model']
        meta_params = meta_model.get('params', {})
        
        logger.info(f"Training meta-model: {meta_model_name}")
        
        # Set parameters if provided
        if meta_params:
            meta_model_instance.set_params(**meta_params)
        
        # Train meta-model on base model predictions
        meta_model_instance.fit(meta_features_train, y_train)
        
        # Make final predictions
        if model_type == 'classification' and hasattr(meta_model_instance, 'predict_proba'):
            final_preds = meta_model_instance.predict_proba(meta_features_test)[:, 1]
        else:
            final_preds = meta_model_instance.predict(meta_features_test)
        
        # Evaluate ensemble
        if model_type == 'classification':
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            
            # Convert probabilities to binary predictions for metrics that need them
            if hasattr(meta_model_instance, 'predict_proba'):
                binary_preds = (final_preds >= 0.5).astype(int)
            else:
                binary_preds = final_preds
            
            metrics = {
                'accuracy': accuracy_score(y_test, binary_preds),
                'precision': precision_score(y_test, binary_preds, zero_division=0),
                'recall': recall_score(y_test, binary_preds, zero_division=0),
                'f1': f1_score(y_test, binary_preds, zero_division=0)
            }
            
            # Add AUC if we have probability predictions
            if hasattr(meta_model_instance, 'predict_proba'):
                metrics['auc'] = roc_auc_score(y_test, final_preds)
        else:
            # Regression metrics
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            metrics = {
                'mse': mean_squared_error(y_test, final_preds),
                'rmse': np.sqrt(mean_squared_error(y_test, final_preds)),
                'mae': mean_absolute_error(y_test, final_preds),
                'r2': r2_score(y_test, final_preds)
            }
        
        logger.info(f"Ensemble evaluation metrics: {metrics}")
        
        # Return the trained models and metrics
        return {
            'base_models': trained_base_models,
            'meta_model': meta_model_instance,
            'metrics': metrics,
            'model_type': model_type
        }

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