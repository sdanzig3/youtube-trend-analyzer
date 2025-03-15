# src/models/feature_engineering.py
import os
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Union

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("feature_engineering.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Feature engineering for YouTube trending video prediction."""
    
    def __init__(self):
        """Initialize the feature engineer."""
        logger.info("FeatureEngineer initialized")
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for machine learning.
        
        Args:
            df: DataFrame with processed trending video data
            
        Returns:
            DataFrame with engineered features suitable for ML
        """
        if df.empty:
            logger.warning("Empty DataFrame provided")
            return df
        
        logger.info(f"Preparing features for {len(df)} videos")
        
        # Create a copy to avoid modifying the original
        features_df = df.copy()
        
        # Columns that should be excluded - they're identifiers, not features
        exclude_columns = [
            'video_id', 'title', 'channel_id', 'channel_title', 'publish_time', 
            'fetch_time', 'duration', 'thumbnail_url', 'description', 'tags',
            'region', 'channel_title', 'id', 'fetch_date'  # Additional IDs to exclude
        ]
        
        # Remove excluded columns if they exist
        for col in exclude_columns:
            if col in features_df.columns:
                features_df = features_df.drop(columns=[col])
        
        # 1. Create one-hot encoded features
        categorical_features = self._encode_categorical_features(features_df)
        
        # 2. Create normalized numerical features
        numerical_features = self._normalize_numerical_features(features_df)
        
        # 3. Extract text features from titles
        text_features = self._extract_text_features(features_df)
        
        # 4. Create time-based features
        time_features = self._create_time_features(features_df)
        
        # 5. Combine all features
        all_features = pd.concat(
            [categorical_features, numerical_features, text_features, time_features], 
            axis=1
        )
        
        # 6. Final check - ensure all columns are numeric
        numeric_columns = all_features.select_dtypes(include=['int64', 'float64', 'bool']).columns
        all_features = all_features[numeric_columns]
        
        logger.info(f"Created {all_features.shape[1]} features for ML")
        
        return all_features
    
    def _encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features using one-hot encoding."""
        result = pd.DataFrame(index=df.index)
        
        # Category encoding
        if 'category_id' in df.columns:
            # Convert to string first in case it's numeric
            df['category_id'] = df['category_id'].astype(str)
            category_dummies = pd.get_dummies(df['category_id'], prefix='category')
            result = pd.concat([result, category_dummies], axis=1)
        
        # Region encoding (if available)
        if 'region' in df.columns:
            region_dummies = pd.get_dummies(df['region'], prefix='region')
            result = pd.concat([result, region_dummies], axis=1)
        
        # Publish day encoding
        if 'publish_day' in df.columns:
            day_dummies = pd.get_dummies(df['publish_day'], prefix='day')
            result = pd.concat([result, day_dummies], axis=1)
        
        # Time of day encoding
        if 'time_of_day' in df.columns:
            time_dummies = pd.get_dummies(df['time_of_day'], prefix='time')
            result = pd.concat([result, time_dummies], axis=1)
        
        # Duration category encoding
        if 'duration_category' in df.columns:
            duration_dummies = pd.get_dummies(df['duration_category'], prefix='duration')
            result = pd.concat([result, duration_dummies], axis=1)
        
        return result
    
    def _normalize_numerical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize numerical features for ML."""
        result = pd.DataFrame(index=df.index)
        
        # Select numerical columns to normalize
        numerical_columns = [
            'views', 'likes', 'comments', 'duration_seconds', 
            'tag_count', 'title_length', 'title_word_count'
        ]
        
        # Additional numerical features if available
        additional_columns = [
            'like_view_ratio', 'comment_view_ratio', 'engagement_score',
            'views_per_hour', 'likes_per_hour', 'comments_per_hour',
            'virality_score', 'popularity_score', 'title_caps_count',
            'channel_trending_count', 'hours_since_published'
        ]
        
        # Combine all available numerical columns
        available_columns = [col for col in numerical_columns + additional_columns if col in df.columns]
        
        if not available_columns:
            return result
        
        # Get numerical data
        numerical_data = df[available_columns].copy()
        
        # Ensure all values are numeric 
        for col in numerical_data.columns:
            try:
                numerical_data[col] = pd.to_numeric(numerical_data[col], errors='coerce')
            except:
                # If conversion fails, drop the column
                numerical_data = numerical_data.drop(columns=[col])
                logger.warning(f"Dropped non-numeric column: {col}")
        
        # Fill any NaN values created by coercion
        numerical_data = numerical_data.fillna(0)
        
        # Log transform for count-based features (helps with skewed distributions)
        log_transform_columns = ['views', 'likes', 'comments', 'duration_seconds']
        for col in [c for c in log_transform_columns if c in numerical_data.columns]:
            # Add 1 before log to handle zeros
            result[f'{col}_log'] = np.log1p(numerical_data[col])
        
        # Standard scaling for the rest
        # For simplicity, we'll just do a basic standardization here
        for col in numerical_data.columns:
            if col not in result.columns:  # Avoid duplicates
                mean = numerical_data[col].mean()
                std = numerical_data[col].std()
                if std > 0:
                    result[f'{col}_norm'] = (numerical_data[col] - mean) / std
                else:
                    result[f'{col}_norm'] = 0
        
        return result
    
    def _extract_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from text data like titles."""
        result = pd.DataFrame(index=df.index)
        
        # Basic title features 
        title_features = [
            'title_has_number', 'title_has_question', 'title_has_exclamation',
            'title_has_special', 'title_has_brackets', 'title_caps_count'
        ]
        
        # Add enhanced title features if available
        enhanced_features = [
            'clickbait_score', 'title_positive_words', 'title_negative_words',
            'title_sentiment_score', 'title_is_emotional', 'title_uniqueness'
        ]
        
        # Combine all available title features
        available_features = [col for col in title_features + enhanced_features if col in df.columns]
        
        if available_features:
            for col in available_features:
                try:
                    # Ensure all values are numeric
                    result[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                except:
                    logger.warning(f"Skipped non-numeric text feature: {col}")
        
        return result
    
    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features related to time and dates."""
        result = pd.DataFrame(index=df.index)
        
        # Basic time features
        if 'publish_hour' in df.columns:
            try:
                # Convert hour to sine and cosine to capture cyclical nature
                hours = pd.to_numeric(df['publish_hour'], errors='coerce').fillna(0)
                result['hour_sin'] = np.sin(2 * np.pi * hours / 24)
                result['hour_cos'] = np.cos(2 * np.pi * hours / 24)
            except:
                logger.warning("Could not process publish_hour")
        
        if 'publish_weekday' in df.columns:
            try:
                # Convert weekday to sine and cosine
                weekdays = pd.to_numeric(df['publish_weekday'], errors='coerce').fillna(0)
                result['weekday_sin'] = np.sin(2 * np.pi * weekdays / 7)
                result['weekday_cos'] = np.cos(2 * np.pi * weekdays / 7)
            except:
                logger.warning("Could not process publish_weekday")
        
        if 'publish_month' in df.columns:
            try:
                # Convert month to sine and cosine
                months = pd.to_numeric(df['publish_month'], errors='coerce').fillna(0)
                result['month_sin'] = np.sin(2 * np.pi * months / 12)
                result['month_cos'] = np.cos(2 * np.pi * months / 12)
            except:
                logger.warning("Could not process publish_month")
        
        # Add time since publication if available
        if 'hours_since_published' in df.columns:
            try:
                hours = pd.to_numeric(df['hours_since_published'], errors='coerce').fillna(0)
                result['hours_since_published_log'] = np.log1p(hours)
            except:
                logger.warning("Could not process hours_since_published")
        
        # Weekend flag
        if 'publish_weekend' in df.columns:
            try:
                result['is_weekend'] = pd.to_numeric(df['publish_weekend'], errors='coerce').fillna(0)
            except:
                logger.warning("Could not process publish_weekend")
        
        return result
    
    def create_target_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create target variables for prediction models.
        
        Args:
            df: DataFrame with trending video data
            
        Returns:
            DataFrame with target variables
        """
        targets = pd.DataFrame(index=df.index)
        
        # If we have true engagement metrics, use them
        if 'engagement_score' in df.columns:
            try:
                targets['engagement_score'] = pd.to_numeric(df['engagement_score'], errors='coerce').fillna(0)
            except:
                logger.warning("Could not convert engagement_score to numeric")
        
        if 'popularity_score' in df.columns:
            try:
                targets['popularity_score'] = pd.to_numeric(df['popularity_score'], errors='coerce').fillna(0)
            except:
                logger.warning("Could not convert popularity_score to numeric")
        
        # Binary target: Is video highly engaging?
        if 'engagement_score' in df.columns:
            try:
                # Define highly engaging as top 25%
                engagement = pd.to_numeric(df['engagement_score'], errors='coerce').fillna(0)
                engagement_threshold = engagement.quantile(0.75)
                targets['is_highly_engaging'] = (engagement >= engagement_threshold).astype(int)
            except:
                logger.warning("Could not create is_highly_engaging target")
        
        # Binary target: Is video trending for a long time?
        if 'days_on_trending' in df.columns:
            try:
                # Define long-trending as more than 1 day
                days = pd.to_numeric(df['days_on_trending'], errors='coerce').fillna(0)
                targets['is_long_trending'] = (days > 1).astype(int)
            except:
                logger.warning("Could not create is_long_trending target")
        
        # Binary target: Is video viral?
        if 'virality_score' in df.columns:
            try:
                virality = pd.to_numeric(df['virality_score'], errors='coerce').fillna(0)
                virality_threshold = virality.quantile(0.75)
                targets['is_viral'] = (virality >= virality_threshold).astype(int)
            except:
                logger.warning("Could not create is_viral target")
        
        # Regression target: views per hour
        if 'views_per_hour' in df.columns:
            try:
                views_per_hour = pd.to_numeric(df['views_per_hour'], errors='coerce').fillna(0)
                targets['views_per_hour'] = views_per_hour
                # Log transform for regression
                targets['views_per_hour_log'] = np.log1p(views_per_hour)
            except:
                logger.warning("Could not create views_per_hour target")
        
        # Regression target: like-to-view ratio
        if 'like_view_ratio' in df.columns:
            try:
                targets['like_view_ratio'] = pd.to_numeric(df['like_view_ratio'], errors='coerce').fillna(0)
            except:
                logger.warning("Could not create like_view_ratio target")
        
        return targets
    
    def split_data(self, 
                  features: pd.DataFrame, 
                  targets: pd.DataFrame, 
                  test_size: float = 0.2, 
                  random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into training and testing sets.
        
        Args:
            features: DataFrame with features
            targets: DataFrame with target variables
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        from sklearn.model_selection import train_test_split
        
        # Ensure features and targets have the same index
        features = features.loc[targets.index]
        
        # Double-check that all features are numeric
        features = features.select_dtypes(include=['int64', 'float64', 'bool'])
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            features, targets, test_size=test_size, random_state=random_state
        )
        
        logger.info(f"Split data into train ({len(X_train)} samples) and test ({len(X_test)} samples) sets")
        
        return X_train, X_test, y_train, y_test
    
    def save_feature_data(self, features: pd.DataFrame, targets: pd.DataFrame, directory: str = 'data/ml'):
        """Save feature and target data to CSV files.
        
        Args:
            features: DataFrame with features
            targets: DataFrame with target variables
            directory: Directory to save the files
        """
        import os
        
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # One final check - make sure all feature columns are numeric
        numeric_features = features.select_dtypes(include=['int64', 'float64', 'bool'])
        
        if numeric_features.shape[1] < features.shape[1]:
            non_numeric_cols = set(features.columns) - set(numeric_features.columns)
            logger.warning(f"Dropping {len(non_numeric_cols)} non-numeric columns: {non_numeric_cols}")
            features = numeric_features
        
        # Save features and targets
        features_path = os.path.join(directory, 'features.csv')
        targets_path = os.path.join(directory, 'targets.csv')
        
        features.to_csv(features_path, index=True)
        targets.to_csv(targets_path, index=True)
        
        logger.info(f"Saved {len(features)} samples to {directory}")
        logger.info(f"Features shape: {features.shape}, Targets shape: {targets.shape}")
        
        # Save feature list for reference
        feature_list = list(features.columns)
        feature_list_path = os.path.join(directory, 'feature_list.txt')
        
        with open(feature_list_path, 'w') as f:
            for feature in feature_list:
                f.write(f"{feature}\n")
        
        # Save target list for reference
        target_list = list(targets.columns)
        target_list_path = os.path.join(directory, 'target_list.txt')
        
        with open(target_list_path, 'w') as f:
            for target in target_list:
                f.write(f"{target}\n")


# Example usage
if __name__ == "__main__":
    import os
    import pandas as pd
    
    # Initialize feature engineer
    engineer = FeatureEngineer()
    
    # Look for processed data
    data_dir = os.path.join(os.path.dirname(__file__), "../../data/processed")
    if not os.path.exists(data_dir):
        print("No processed data directory found")
        exit(1)
    
    # Find the most recent file starting with "all_trending_"
    files = [f for f in os.listdir(data_dir) if f.startswith("all_trending_")]
    if not files:
        print("No trending data files found")
        exit(1)
    
    latest_file = max(files)
    file_path = os.path.join(data_dir, latest_file)
    
    print(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    
    # Prepare features
    features_df = engineer.prepare_features(df)
    print(f"Created {features_df.shape[1]} features")
    
    # Create target variables
    targets_df = engineer.create_target_variables(df)
    print(f"Created {targets_df.shape[1]} target variables")
    
    # Split data
    X_train, X_test, y_train, y_test = engineer.split_data(features_df, targets_df)
    
    # Save data
    engineer.save_feature_data(features_df, targets_df)
    
    print("Feature engineering completed")