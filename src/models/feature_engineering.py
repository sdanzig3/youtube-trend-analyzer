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

    # Enhanced feature engineering methods to add to your FeatureEngineer class

    def extract_advanced_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract advanced features from text data like titles and descriptions.
        
        Args:
            df: DataFrame with video data including titles and descriptions
            
        Returns:
            DataFrame with advanced text features
        """
        import re
        from collections import Counter
        
        result = pd.DataFrame(index=df.index)
        
        # Check if title column exists
        if 'title' not in df.columns:
            logger.warning("Title column not found for text feature extraction")
            return result
        
        # Common clickbait phrases and emotion words
        clickbait_phrases = [
            'you won\'t believe', 'mind blowing', 'amazing', 'shocking', 'insane', 
            'unbelievable', 'won\'t believe', 'must see', 'mind-blowing', 'incredible',
            'shocking', 'secret', 'surprising', 'revealed', 'miracle', 'revolutionary'
        ]
        
        # Positive and negative emotional words
        positive_words = [
            'amazing', 'awesome', 'beautiful', 'best', 'brilliant', 'congrats', 
            'excellent', 'exciting', 'fantastic', 'fun', 'glad', 'good', 'great', 
            'happy', 'impressive', 'love', 'perfect', 'remarkable', 'spectacular', 
            'super', 'terrific', 'wonderful'
        ]
        
        negative_words = [
            'angry', 'awful', 'bad', 'disappointing', 'disaster', 'disturbing', 
            'fail', 'horrible', 'negative', 'sad', 'scary', 'shocked', 'terrible', 
            'tragic', 'ugly', 'unfortunate', 'upset', 'worst', 'wrong'
        ]
        
        # Create features based on titles
        titles = df['title'].fillna('').astype(str)
        
        # Basic text cleaning
        def clean_text(text):
            text = text.lower()
            text = re.sub(r'[^\w\s]', ' ', text)  # Replace punctuation with space
            text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
            return text
        
        # Title length stats
        result['title_char_count'] = titles.apply(len)
        result['title_word_count'] = titles.apply(lambda x: len(x.split()))
        
        # Advanced features
        cleaned_titles = titles.apply(clean_text)
        
        # Clickbait score - presence of clickbait phrases
        result['clickbait_score'] = cleaned_titles.apply(
            lambda x: sum(1 for phrase in clickbait_phrases if phrase in x)
        )
        
        # Emotional content
        result['title_positive_words'] = cleaned_titles.apply(
            lambda x: sum(1 for word in positive_words if word in x.split())
        )
        
        result['title_negative_words'] = cleaned_titles.apply(
            lambda x: sum(1 for word in negative_words if word in x.split())
        )
        
        result['title_emotion_ratio'] = (result['title_positive_words'] - result['title_negative_words']).apply(
            lambda x: x if x != 0 else 0
        )
        
        # Title has question
        result['title_has_question'] = titles.apply(lambda x: 1 if '?' in x else 0)
        
        # Title has exclamation
        result['title_has_exclamation'] = titles.apply(lambda x: 1 if '!' in x else 0)
        
        # Title capitalization (ALL CAPS words percentage)
        def caps_percentage(title):
            words = title.split()
            if not words:
                return 0
            caps_words = sum(1 for word in words if word.isupper() and len(word) > 1)
            return caps_words / len(words) * 100
        
        result['title_caps_percentage'] = titles.apply(caps_percentage)
        
        # Number of brackets (often used for clarification or clickbait)
        result['title_brackets_count'] = titles.apply(
            lambda x: x.count('(') + x.count('[') + x.count('{')
        )
        
        # Word diversity (unique words / total words)
        def word_diversity(text):
            words = text.lower().split()
            if not words:
                return 0
            return len(set(words)) / len(words)
        
        result['title_word_diversity'] = cleaned_titles.apply(word_diversity)
        
        # If description column exists, extract features from it too
        if 'description' in df.columns:
            descriptions = df['description'].fillna('').astype(str)
            cleaned_descriptions = descriptions.apply(clean_text)
            
            # Description length
            result['desc_char_count'] = descriptions.apply(len)
            result['desc_word_count'] = descriptions.apply(lambda x: len(x.split()))
            
            # URL count (common in descriptions)
            result['desc_url_count'] = descriptions.apply(
                lambda x: x.count('http')
            )
            
            # Hashtag count
            result['desc_hashtag_count'] = descriptions.apply(
                lambda x: x.count('#')
            )
            
            # Mention count
            result['desc_mention_count'] = descriptions.apply(
                lambda x: x.count('@')
            )
        
        # If tags column exists, extract features from it
        if 'tags' in df.columns:
            # Ensure tags are in a usable format (assuming they might be stored as a string)
            def parse_tags(tag_str):
                if not isinstance(tag_str, str):
                    return []
                # Handle various formats - JSON list, comma-separated, etc.
                tag_str = tag_str.strip()
                if not tag_str:
                    return []
                if tag_str.startswith('[') and tag_str.endswith(']'):
                    try:
                        import json
                        return json.loads(tag_str)
                    except:
                        pass
                return [t.strip() for t in tag_str.split(',')]
            
            # Extract tag features
            tags_list = df['tags'].apply(parse_tags)
            result['tag_count'] = tags_list.apply(len)
            result['tag_avg_length'] = tags_list.apply(
                lambda tags: np.mean([len(t) for t in tags]) if tags else 0
            )
        
        return result

    def extract_engagement_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract engagement-related features from video data.
        
        Args:
            df: DataFrame with video statistics
            
        Returns:
            DataFrame with engagement features
        """
        result = pd.DataFrame(index=df.index)
        
        # Check for required columns
        required_cols = ['views', 'likes', 'comments']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.warning(f"Missing required columns for engagement features: {missing_cols}")
            return result
        
        # Convert to numeric and fill missing values
        views = pd.to_numeric(df['views'], errors='coerce').fillna(0).clip(lower=1)  # Avoid division by zero
        likes = pd.to_numeric(df['likes'], errors='coerce').fillna(0)
        comments = pd.to_numeric(df['comments'], errors='coerce').fillna(0)
        
        # Basic engagement ratios
        result['like_view_ratio'] = (likes / views * 100).clip(upper=100)  # As percentage
        result['comment_view_ratio'] = (comments / views * 100).clip(upper=100)  # As percentage
        
        # Combined engagement score (weighted)
        # Weight likes and comments based on their typical distribution
        result['engagement_score'] = ((likes + comments * 5) / views * 100).clip(upper=100)
        
        # If we have publish time and fetch time, calculate rate metrics
        if 'publish_time' in df.columns and 'fetch_time' in df.columns:
            try:
                # Calculate hours since published
                pub_time = pd.to_datetime(df['publish_time'], errors='coerce')
                fetch_time = pd.to_datetime(df['fetch_time'], errors='coerce')
                
                # Skip rows with invalid datetime
                valid_times = ~(pub_time.isna() | fetch_time.isna())
                hours_diff = (fetch_time[valid_times] - pub_time[valid_times]).dt.total_seconds() / 3600
                
                # Initialize with zeros
                result['hours_since_published'] = 0
                result['views_per_hour'] = 0
                result['likes_per_hour'] = 0
                result['comments_per_hour'] = 0
                
                # Update only rows with valid time data
                result.loc[valid_times, 'hours_since_published'] = hours_diff
                
                # Avoid division by zero by setting a minimum time
                hours_diff = hours_diff.clip(lower=1/60)  # Minimum 1 minute
                
                # Calculate rates
                result.loc[valid_times, 'views_per_hour'] = views[valid_times] / hours_diff
                result.loc[valid_times, 'likes_per_hour'] = likes[valid_times] / hours_diff
                result.loc[valid_times, 'comments_per_hour'] = comments[valid_times] / hours_diff
                
                # Virality score - combination of velocity and total views
                # Emphasized for videos that get lots of engagement in a short time
                result.loc[valid_times, 'virality_score'] = (
                    result.loc[valid_times, 'views_per_hour'] * 
                    np.log1p(views[valid_times]) / 
                    np.sqrt(hours_diff.clip(lower=1))
                )
                
                # Age-adjusted engagement score
                result.loc[valid_times, 'age_adjusted_engagement'] = (
                    result.loc[valid_times, 'engagement_score'] / 
                    np.log1p(hours_diff)
                )
            except Exception as e:
                logger.warning(f"Error calculating time-based features: {e}")
        
        # If we have channel data, create channel-related features
        if 'channel_id' in df.columns:
            try:
                # Count videos by channel
                channel_counts = df['channel_id'].value_counts()
                result['channel_video_count'] = df['channel_id'].map(channel_counts)
                
                # Channel average engagement (if we have enough data)
                if len(df) > 100:  # Only calculate if we have a decent sample
                    channel_avg_engagement = df.groupby('channel_id')['engagement_score'].mean()
                    result['channel_avg_engagement'] = df['channel_id'].map(channel_avg_engagement)
                    
                    # Relative engagement (compared to channel average)
                    result['relative_engagement'] = result['engagement_score'] / result['channel_avg_engagement'].fillna(1)
            except Exception as e:
                logger.warning(f"Error calculating channel features: {e}")
        
        return result

    def extract_temporal_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features related to temporal patterns and trends.
        
        Args:
            df: DataFrame with video data including publishing time
            
        Returns:
            DataFrame with temporal pattern features
        """
        result = pd.DataFrame(index=df.index)
        
        # Check if publish_time column exists
        if 'publish_time' not in df.columns:
            logger.warning("publish_time column not found for temporal pattern extraction")
            return result
        
        try:
            # Convert to datetime
            pub_time = pd.to_datetime(df['publish_time'], errors='coerce')
            valid_times = ~pub_time.isna()
            
            # Basic time components
            result['publish_hour'] = np.nan
            result['publish_day'] = np.nan
            result['publish_month'] = np.nan
            result['publish_year'] = np.nan
            result['publish_dayofweek'] = np.nan
            result['publish_weekend'] = np.nan
            result['publish_quarter'] = np.nan
            
            # Extract for valid times
            result.loc[valid_times, 'publish_hour'] = pub_time[valid_times].dt.hour
            result.loc[valid_times, 'publish_day'] = pub_time[valid_times].dt.day
            result.loc[valid_times, 'publish_month'] = pub_time[valid_times].dt.month
            result.loc[valid_times, 'publish_year'] = pub_time[valid_times].dt.year
            result.loc[valid_times, 'publish_dayofweek'] = pub_time[valid_times].dt.dayofweek
            result.loc[valid_times, 'publish_weekend'] = (pub_time[valid_times].dt.dayofweek >= 5).astype(int)
            result.loc[valid_times, 'publish_quarter'] = pub_time[valid_times].dt.quarter
            
            # Time of day categories
            def time_of_day(hour):
                if 5 <= hour < 12:
                    return 'morning'
                elif 12 <= hour < 17:
                    return 'afternoon'
                elif 17 <= hour < 22:
                    return 'evening'
                else:
                    return 'night'
            
            result.loc[valid_times, 'time_of_day'] = pub_time[valid_times].dt.hour.apply(time_of_day)
            
            # Day part features
            result['publish_morning'] = (result['time_of_day'] == 'morning').astype(int)
            result['publish_afternoon'] = (result['time_of_day'] == 'afternoon').astype(int)
            result['publish_evening'] = (result['time_of_day'] == 'evening').astype(int)
            result['publish_night'] = (result['time_of_day'] == 'night').astype(int)
            
            # Cyclic time features (to preserve cyclical nature)
            result.loc[valid_times, 'hour_sin'] = np.sin(2 * np.pi * pub_time[valid_times].dt.hour / 24)
            result.loc[valid_times, 'hour_cos'] = np.cos(2 * np.pi * pub_time[valid_times].dt.hour / 24)
            
            result.loc[valid_times, 'day_sin'] = np.sin(2 * np.pi * pub_time[valid_times].dt.day / 31)
            result.loc[valid_times, 'day_cos'] = np.cos(2 * np.pi * pub_time[valid_times].dt.day / 31)
            
            result.loc[valid_times, 'month_sin'] = np.sin(2 * np.pi * pub_time[valid_times].dt.month / 12)
            result.loc[valid_times, 'month_cos'] = np.cos(2 * np.pi * pub_time[valid_times].dt.month / 12)
            
            result.loc[valid_times, 'dayofweek_sin'] = np.sin(2 * np.pi * pub_time[valid_times].dt.dayofweek / 7)
            result.loc[valid_times, 'dayofweek_cos'] = np.cos(2 * np.pi * pub_time[valid_times].dt.dayofweek / 7)
            
            # If we have fetch_time, calculate publishing delay features
            if 'fetch_time' in df.columns:
                fetch_time = pd.to_datetime(df['fetch_time'], errors='coerce')
                valid_both = ~(pub_time.isna() | fetch_time.isna())
                
                hours_diff = (fetch_time[valid_both] - pub_time[valid_both]).dt.total_seconds() / 3600
                days_diff = hours_diff / 24
                
                result.loc[valid_both, 'hours_since_published'] = hours_diff
                result.loc[valid_both, 'days_since_published'] = days_diff
                
                # Trending delay categories
                def trending_delay_category(hours):
                    if hours < 24:
                        return 'same_day'
                    elif hours < 48:
                        return 'next_day'
                    elif hours < 168:  # 7 days
                        return 'same_week'
                    else:
                        return 'later'
                
                result.loc[valid_both, 'trending_delay'] = hours_diff.apply(trending_delay_category)
                
                # Convert to dummy variables
                result['trending_same_day'] = (result['trending_delay'] == 'same_day').astype(int)
                result['trending_next_day'] = (result['trending_delay'] == 'next_day').astype(int)
                result['trending_same_week'] = (result['trending_delay'] == 'same_week').astype(int)
                result['trending_later'] = (result['trending_delay'] == 'later').astype(int)
                
        except Exception as e:
            logger.warning(f"Error extracting temporal patterns: {e}")
        
        return result

    def extract_content_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features related to video content and metadata.
        
        Args:
            df: DataFrame with video data
            
        Returns:
            DataFrame with content-related features
        """
        result = pd.DataFrame(index=df.index)
        
        # Duration-related features
        if 'duration' in df.columns or 'duration_seconds' in df.columns:
            try:
                # Get duration in seconds
                if 'duration_seconds' in df.columns:
                    duration_seconds = pd.to_numeric(df['duration_seconds'], errors='coerce').fillna(0)
                else:
                    # Parse ISO 8601 duration format (PT1H2M3S)
                    import re
                    
                    def parse_duration(duration_str):
                        if not isinstance(duration_str, str):
                            return 0
                        
                        # Extract hours, minutes, seconds
                        hours = re.search(r'(\d+)H', duration_str)
                        minutes = re.search(r'(\d+)M', duration_str)
                        seconds = re.search(r'(\d+)S', duration_str)
                        
                        hours = int(hours.group(1)) if hours else 0
                        minutes = int(minutes.group(1)) if minutes else 0
                        seconds = int(seconds.group(1)) if seconds else 0
                        
                        return hours * 3600 + minutes * 60 + seconds
                    
                    duration_seconds = df['duration'].apply(parse_duration)
                
                # Store the duration in seconds
                result['duration_seconds'] = duration_seconds
                
                # Duration in minutes (more interpretable)
                result['duration_minutes'] = duration_seconds / 60
                
                # Log-transformed duration
                result['duration_log'] = np.log1p(duration_seconds)
                
                # Duration categories
                def duration_category(seconds):
                    if seconds < 60:  # < 1 min
                        return 'very_short'
                    elif seconds < 300:  # < 5 min
                        return 'short'
                    elif seconds < 1200:  # < 20 min
                        return 'medium'
                    elif seconds < 3600:  # < 1 hour
                        return 'long'
                    else:  # >= 1 hour
                        return 'very_long'
                
                result['duration_category'] = duration_seconds.apply(duration_category)
                
                # Convert to dummy variables
                result['duration_very_short'] = (result['duration_category'] == 'very_short').astype(int)
                result['duration_short'] = (result['duration_category'] == 'short').astype(int)
                result['duration_medium'] = (result['duration_category'] == 'medium').astype(int)
                result['duration_long'] = (result['duration_category'] == 'long').astype(int)
                result['duration_very_long'] = (result['duration_category'] == 'very_long').astype(int)
                
            except Exception as e:
                logger.warning(f"Error extracting duration features: {e}")
        
        # Title-specific content features
        if 'title' in df.columns:
            try:
                titles = df['title'].fillna('').astype(str)
                
                # Extract patterns from titles
                result['title_has_number'] = titles.str.contains(r'\d').astype(int)
                result['title_has_question'] = titles.str.contains(r'\?').astype(int)
                result['title_has_exclamation'] = titles.str.contains(r'!').astype(int)
                result['title_has_emoji'] = titles.apply(
                    lambda x: bool(re.search(r'[\U00010000-\U0010ffff]', x))
                ).astype(int)
                
                # Common video title patterns
                result['title_has_part_number'] = titles.str.contains(
                    r'part\s*\d+|pt\.?\s*\d+|\(\s*\d+\s*\)|#\s*\d+', 
                    case=False
                ).astype(int)
                
                result['title_has_year'] = titles.str.contains(
                    r'\b20\d{2}\b|\b19\d{2}\b'
                ).astype(int)
                
                result['title_has_tutorial'] = titles.str.contains(
                    r'tutorial|how\s+to|guide|tips|tricks|learn|course', 
                    case=False
                ).astype(int)
                
                result['title_has_review'] = titles.str.contains(
                    r'review|unboxing|vs\.?|versus', 
                    case=False
                ).astype(int)
                
                result['title_has_reaction'] = titles.str.contains(
                    r'react|reaction|watching', 
                    case=False
                ).astype(int)
                
                result['title_is_clickbait'] = titles.str.contains(
                    r'you won\'t believe|mind blowing|amazing|shocking|insane|unbelievable|must see|incredible|shocking|secret|revealed', 
                    case=False
                ).astype(int)
                
                # Title capitalization
                def title_caps_count(title):
                    words = title.split()
                    return sum(1 for word in words if word.isupper() and len(word) > 1)
                
                result['title_caps_count'] = titles.apply(title_caps_count)
                
            except Exception as e:
                logger.warning(f"Error extracting title content features: {e}")
        
        # Description-specific content features
        if 'description' in df.columns:
            try:
                descriptions = df['description'].fillna('').astype(str)
                
                # Length of description
                result['description_length'] = descriptions.apply(len)
                result['description_word_count'] = descriptions.apply(lambda x: len(x.split()))
                
                # Links in description
                result['description_has_links'] = descriptions.str.contains(r'http').astype(int)
                result['description_link_count'] = descriptions.str.count(r'http')
                
                # Social media links
                platforms = ['facebook', 'twitter', 'instagram', 'tiktok', 'linkedin', 'snapchat', 'youtube']
                for platform in platforms:
                    result[f'has_{platform}_link'] = descriptions.str.contains(
                        f'(?:www\.)?{platform}\.com|{platform}(?:\.com)?', 
                        case=False
                    ).astype(int)
                
                # Description has timestamps (common in longer videos)
                result['has_timestamps'] = descriptions.str.contains(
                    r'\d{1,2}:\d{2}'
                ).astype(int)
                
                # Description has music credits
                result['has_music_credits'] = descriptions.str.contains(
                    r'music|song|track|artist|album|composer|producer', 
                    case=False
                ).astype(int)
                
                # Description has partner/sponsor mentions
                result['has_sponsor'] = descriptions.str.contains(
                    r'sponsor|partner|promotion|discount|code|affiliate|deal', 
                    case=False
                ).astype(int)
                
            except Exception as e:
                logger.warning(f"Error extracting description content features: {e}")
        
        return result

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