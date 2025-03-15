# src/models/prediction.py
import os
import json
import pickle
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("prediction.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TrendingPredictor:
    """Make predictions for YouTube trending video potential."""
    
    def __init__(self, models_dir: str = 'models'):
        """Initialize the trending predictor.
        
        Args:
            models_dir: Directory where models are stored
        """
        self.models_dir = models_dir
        self._loaded_models = {}
        self._model_metadata = {}
        logger.info(f"TrendingPredictor initialized with models directory: {models_dir}")
    
    def load_model(self, model_name: str) -> bool:
        """Load a model from disk.
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            True if model loaded successfully, False otherwise
        """
        # Check if model is already loaded
        if model_name in self._loaded_models:
            return True
        
        try:
            model_path = os.path.join(self.models_dir, model_name, f"{model_name}.pkl")
            metadata_path = os.path.join(self.models_dir, model_name, f"{model_name}_metadata.json")
            
            # Check if files exist
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                return False
            
            if not os.path.exists(metadata_path):
                logger.warning(f"Metadata file not found: {metadata_path}")
            
            # Load model
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            # Load metadata if available
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            else:
                metadata = {}
            
            # Store model and metadata
            self._loaded_models[model_name] = model
            self._model_metadata[model_name] = metadata
            
            logger.info(f"Model {model_name} loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return False
    
    def load_available_models(self) -> List[str]:
        """Load all available models in the models directory.
        
        Returns:
            List of successfully loaded model names
        """
        loaded_models = []
        
        try:
            # Check if models directory exists
            if not os.path.exists(self.models_dir):
                logger.warning(f"Models directory not found: {self.models_dir}")
                return loaded_models
            
            # Get subdirectories (model names)
            model_names = [d for d in os.listdir(self.models_dir) 
                          if os.path.isdir(os.path.join(self.models_dir, d))]
            
            # Load each model
            for model_name in model_names:
                if self.load_model(model_name):
                    loaded_models.append(model_name)
            
            logger.info(f"Loaded {len(loaded_models)} models: {loaded_models}")
            
        except Exception as e:
            logger.error(f"Error loading available models: {e}")
        
        return loaded_models
    
    def prepare_features(self, video_data: Dict[str, Any]) -> pd.DataFrame:
        """Prepare features for prediction.
        
        Args:
            video_data: Dictionary with video attributes
            
        Returns:
            DataFrame with features ready for prediction
        """
        import re
        from src.data.youtube_fetcher import CATEGORY_MAPPING
        
        # Create a single-row DataFrame
        features = pd.DataFrame([{}])
        
        # Basic features
        # Extract features we need for prediction
        title = video_data.get('title', '')
        description = video_data.get('description', '')
        duration_seconds = video_data.get('duration_seconds', 0)
        category_id = video_data.get('category_id', '')
        tags = video_data.get('tags', [])
        publish_hour = video_data.get('publish_hour')
        publish_weekday = video_data.get('publish_weekday')
        
        # Prepare categorical features with one-hot encoding
        # Category encoding
        for cat_id in CATEGORY_MAPPING.keys():
            features[f'category_{cat_id}'] = 1 if cat_id == category_id else 0
        
        # Day of week encoding (if available)
        if publish_weekday is not None:
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            for i, day in enumerate(day_names):
                features[f'day_{day}'] = 1 if i == publish_weekday else 0
            
            # Weekend flag
            features['is_weekend'] = 1 if publish_weekday in [5, 6] else 0
        
        # Hour of day encoding (if available)
        if publish_hour is not None:
            # Convert hour to sine and cosine to capture cyclical nature
            features['hour_sin'] = np.sin(2 * np.pi * publish_hour / 24)
            features['hour_cos'] = np.cos(2 * np.pi * publish_hour / 24)
            
            # Time of day
            if 5 <= publish_hour < 12:
                time_of_day = 'Morning'
            elif 12 <= publish_hour < 17:
                time_of_day = 'Afternoon'
            elif 17 <= publish_hour < 21:
                time_of_day = 'Evening'
            else:
                time_of_day = 'Night'
            
            # One-hot encode time of day
            for tod in ['Morning', 'Afternoon', 'Evening', 'Night']:
                features[f'time_{tod}'] = 1 if tod == time_of_day else 0
        
        # Duration features
        features['duration_seconds'] = duration_seconds
        features['duration_seconds_log'] = np.log1p(duration_seconds)
        features['duration_minutes'] = duration_seconds / 60
        
        # Title features
        features['title_length'] = len(title)
        features['title_word_count'] = len(title.split())
        
        # Title special features
        features['title_has_number'] = 1 if re.search(r'\d', title) else 0
        features['title_has_question'] = 1 if '?' in title else 0
        features['title_has_exclamation'] = 1 if '!' in title else 0
        features['title_caps_count'] = sum(1 for word in re.findall(r'\b[A-Z]{2,}\b', title))
        features['title_has_special'] = 1 if re.search(r'[^\w\s\?\!\.,]', title) else 0
        features['title_has_brackets'] = 1 if re.search(r'[\[\]\(\)\{\}]', title) else 0
        
        # Tag features
        features['tag_count'] = len(tags)
        
        # Description features
        features['description_length'] = len(description)
        features['description_word_count'] = len(description.split())
        
        # Normalize selected features
        # Note: In a production system, you'd use the same normalization parameters as during training
        # Here we're using simple log transforms for demonstration
        numerical_cols = ['duration_seconds', 'title_length', 'tag_count', 'description_length']
        
        for col in numerical_cols:
            if col in features.columns:
                features[f'{col}_norm'] = np.log1p(features[col]) / 10
        
        return features
    
    def predict_engagement(self, video_data: Dict[str, Any], model_name: str = 'engagement_prediction_gb') -> float:
        """Predict engagement score for a video.
        
        Args:
            video_data: Dictionary with video attributes
            model_name: Name of the model to use
            
        Returns:
            Predicted engagement score
        """
        # Load model if not already loaded
        if model_name not in self._loaded_models:
            if not self.load_model(model_name):
                logger.error(f"Failed to load model {model_name}")
                return 0.0
        
        # Prepare features
        features = self.prepare_features(video_data)
        
        # Get model and metadata
        model = self._loaded_models[model_name]
        metadata = self._model_metadata.get(model_name, {})
        
        # Check if we have the required features
        model_features = metadata.get('features', [])
        
        if model_features:
            # Keep only the features used by the model
            available_features = [f for f in model_features if f in features.columns]
            features = features[available_features]
            
            # Fill any missing features with zeros
            missing_features = [f for f in model_features if f not in features.columns]
            for feature in missing_features:
                features[feature] = 0
            
            # Ensure features are in the same order as during training
            features = features[model_features]
        
        # Make prediction
        try:
            prediction = model.predict(features)[0]
            logger.info(f"Predicted engagement score: {prediction}")
            return float(prediction)
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return 0.0
    
    def predict_viral_potential(self, video_data: Dict[str, Any], model_name: str = 'viral_prediction_rf') -> float:
        """Predict viral potential (probability) for a video.
        
        Args:
            video_data: Dictionary with video attributes
            model_name: Name of the model to use
            
        Returns:
            Probability of going viral (0-1)
        """
        # Load model if not already loaded
        if model_name not in self._loaded_models:
            if not self.load_model(model_name):
                logger.error(f"Failed to load model {model_name}")
                return 0.0
        
        # Prepare features
        features = self.prepare_features(video_data)
        
        # Get model and metadata
        model = self._loaded_models[model_name]
        metadata = self._model_metadata.get(model_name, {})
        
        # Check if we have the required features
        model_features = metadata.get('features', [])
        
        if model_features:
            # Keep only the features used by the model
            available_features = [f for f in model_features if f in features.columns]
            features = features[available_features]
            
            # Fill any missing features with zeros
            missing_features = [f for f in model_features if f not in features.columns]
            for feature in missing_features:
                features[feature] = 0
            
            # Ensure features are in the same order as during training
            features = features[model_features]
        
        # Make prediction
        try:
            # Check if model can predict probabilities
            if hasattr(model, 'predict_proba'):
                # Return probability of positive class
                prediction = model.predict_proba(features)[0, 1]
            else:
                # Fall back to binary prediction
                prediction = float(model.predict(features)[0])
            
            logger.info(f"Predicted viral potential: {prediction}")
            return float(prediction)
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return 0.0
    
    def get_feature_importance(self, model_name: str) -> Dict[str, float]:
        """Get feature importances for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        # Load model if not already loaded
        if model_name not in self._loaded_models:
            if not self.load_model(model_name):
                logger.error(f"Failed to load model {model_name}")
                return {}
        
        # Get metadata
        metadata = self._model_metadata.get(model_name, {})
        
        # Return feature importances from metadata
        return metadata.get('feature_importances', {})
    
    def generate_recommendations(self, video_data: Dict[str, Any]) -> List[str]:
        """Generate recommendations to improve trending potential.
        
        Args:
            video_data: Dictionary with video attributes
            
        Returns:
            List of recommendation strings
        """
        title = video_data.get('title', '')
        description = video_data.get('description', '')
        duration_seconds = video_data.get('duration_seconds', 0)
        category_id = video_data.get('category_id', '')
        tags = video_data.get('tags', [])
        publish_hour = video_data.get('publish_hour')
        publish_weekday = video_data.get('publish_weekday')
        
        recommendations = []
        
        # Title recommendations
        title_length = len(title)
        if title_length < 30:
            recommendations.append("Make your title longer (30-70 characters is optimal)")
        elif title_length > 100:
            recommendations.append("Consider shortening your title (30-70 characters is optimal)")
        
        if '?' not in title and '!' not in title:
            recommendations.append("Add a question mark or exclamation point to your title to increase engagement")
        
        # Duration recommendations
        duration_minutes = duration_seconds / 60
        if category_id == '10':  # Music
            if duration_minutes > 5:
                recommendations.append("Music videos tend to perform better when they're shorter (under 5 minutes)")
        elif category_id in ['24', '23']:  # Entertainment, Comedy
            if duration_minutes < 8 or duration_minutes > 15:
                recommendations.append("Entertainment videos tend to perform best between 8-15 minutes")
        elif category_id == '20':  # Gaming
            if duration_minutes < 10 or duration_minutes > 25:
                recommendations.append("Gaming videos tend to perform best between 10-25 minutes")
        elif category_id in ['27', '28']:  # Education, Science & Tech
            if duration_minutes < 7 or duration_minutes > 20:
                recommendations.append("Educational videos tend to perform best between 7-20 minutes")
        
        # Tag recommendations
        tag_count = len(tags) if tags else 0
        if tag_count < 5:
            recommendations.append("Add more tags (aim for 8-15 relevant tags)")
        elif tag_count > 20:
            recommendations.append("Too many tags might dilute relevance. Focus on 8-15 highly relevant tags")
        
        # Description recommendations
        description_length = len(description)
        if description_length < 100:
            recommendations.append("Add a more detailed description (200+ characters recommended)")
        
        # Time recommendations
        if publish_hour is not None and publish_weekday is not None:
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_name = day_names[publish_weekday]
            
            # Category-specific time recommendations
            if category_id == '10':  # Music
                if day_name not in ['Friday', 'Saturday']:
                    recommendations.append("Music videos typically perform better when published on Friday or Saturday")
                if not (15 <= publish_hour <= 20):
                    recommendations.append("Music videos typically perform better when published between 3pm-8pm")
            elif category_id == '20':  # Gaming
                if not (14 <= publish_hour <= 22):
                    recommendations.append("Gaming videos typically perform better when published between 2pm-10pm")
            elif category_id == '24':  # Entertainment
                if day_name not in ['Thursday', 'Friday', 'Saturday']:
                    recommendations.append("Entertainment videos often perform better when published Thursday-Saturday")
            elif category_id == '25':  # News & Politics
                if not (7 <= publish_hour <= 11) and not (17 <= publish_hour <= 20):
                    recommendations.append("News videos typically perform better in the morning (7am-11am) or early evening (5pm-8pm)")
        
        return recommendations
    
    def predict_trending_potential(self, video_data: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive trending prediction for a video.
        
        Args:
            video_data: Dictionary with video attributes
            
        Returns:
            Dictionary with prediction results and recommendations
        """
        # Try to predict engagement score
        engagement_score = self.predict_engagement(video_data)
        
        # Try to predict viral potential
        viral_potential = self.predict_viral_potential(video_data)
        
        # Convert to trending score (0-100 scale)
        trending_score = (engagement_score * 0.6 + viral_potential * 10 * 0.4) * 10
        
        # Ensure score is in range 0-100
        trending_score = max(0, min(100, trending_score))
        
        # Generate recommendations
        recommendations = self.generate_recommendations(video_data)
        
        # Create result
        result = {
            'trending_score': round(trending_score, 1),
            'engagement_score': round(engagement_score, 2),
            'viral_potential': round(viral_potential, 3),
            'recommendations': recommendations
        }
        
        # Add views estimate (very rough approximation)
        result['views_estimate'] = int(5000 * (2 ** (trending_score / 20)))
        
        # Add confidence level (placeholder - in a real system this would be model-based)
        result['confidence'] = 0.7
        
        return result


# Example usage
if __name__ == "__main__":
    # Initialize predictor
    predictor = TrendingPredictor()
    
    # Load available models
    loaded_models = predictor.load_available_models()
    
    if not loaded_models:
        print("No models available. Train models first.")
        exit(1)
    
    # Sample video data for prediction
    sample_video = {
        'title': 'Top 10 Machine Learning Projects for Beginners in 2023',
        'description': 'Learn about the best machine learning projects that you can build to enhance your portfolio and skills.',
        'duration_seconds': 900,  # 15 minutes
        'category_id': '28',  # Science & Technology
        'tags': ['machine learning', 'data science', 'python', 'projects', 'tutorial'],
        'publish_hour': 16,  # 4pm
        'publish_weekday': 2  # Wednesday
    }
    
    # Make prediction
    prediction = predictor.predict_trending_potential(sample_video)
    
    print("\nTrending Potential Prediction:")
    print(f"Trending Score: {prediction['trending_score']}/100")
    print(f"Engagement Score: {prediction['engagement_score']}/10")
    print(f"Viral Potential: {prediction['viral_potential'] * 100:.1f}%")
    print(f"Views Estimate: {prediction['views_estimate']:,}")
    print(f"Confidence: {prediction['confidence'] * 100:.1f}%")
    
    print("\nRecommendations:")
    for i, rec in enumerate(prediction['recommendations'], 1):
        print(f"{i}. {rec}")
    
    # Get feature importance
    for model_name in loaded_models:
        importances = predictor.get_feature_importance(model_name)
        if importances:
            print(f"\nTop 5 features for {model_name}:")
            for i, (feature, importance) in enumerate(list(importances.items())[:5], 1):
                print(f"{i}. {feature}: {importance:.4f}")