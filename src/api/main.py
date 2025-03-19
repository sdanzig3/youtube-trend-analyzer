# src/api/main.py
import os
import sys
from typing import List, Dict, Optional, Any
from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
from datetime import datetime
import logging
import json

# Add the project root to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.data.youtube_fetcher import CATEGORY_MAPPING
from .model_service import ModelService

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("api")

# Initialize FastAPI app
app = FastAPI(
    title="YouTube Trending Videos API",
    description="API for analyzing YouTube trending videos and predicting trending potential",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Models directory - can be overridden with environment variable
MODELS_DIR = os.environ.get("MODELS_DIR", "models")

# Global cache for storing data and models
DATA_CACHE = {
    "trending_data": None,
    "last_updated": None,
    "model_service": None
}

# Pydantic models for requests and responses
class VideoDetails(BaseModel):
    video_id: str
    title: str
    channel_title: str
    category_name: str
    views: int
    likes: int
    comments: int
    publish_time: str
    duration_seconds: int
    engagement_score: float
    thumbnail_url: Optional[str] = None
    region: str

class TrendingResponse(BaseModel):
    count: int
    videos: List[VideoDetails]

class CategoryStats(BaseModel):
    category_id: str
    category_name: str
    video_count: int
    avg_views: float
    avg_likes: float
    avg_comments: float
    avg_engagement: float
    optimal_duration_minutes: Optional[float] = None
    optimal_time_of_day: Optional[str] = None

class TimeAnalysis(BaseModel):
    best_day: str
    best_hour: int
    best_time_of_day: str
    weekday_distribution: Dict[str, float]
    hourly_distribution: Dict[int, float]

class PredictionRequest(BaseModel):
    title: str = Field(..., description="Video title")
    description: Optional[str] = Field(None, description="Video description")
    duration_seconds: int = Field(..., description="Video duration in seconds")
    category_id: str = Field(..., description="Video category ID")
    tags: Optional[List[str]] = Field([], description="Video tags")
    thumbnail_url: Optional[str] = Field(None, description="Thumbnail URL")
    
    # Optional fields for better predictions
    publish_hour: Optional[int] = Field(None, description="Hour of publication (0-23)")
    publish_day: Optional[int] = Field(None, description="Day of week (0=Monday, 6=Sunday)")
    publish_month: Optional[int] = Field(None, description="Month of publication (1-12)")
    channel_title: Optional[str] = Field(None, description="Channel title")

class PredictionResponse(BaseModel):
    trending_score: float
    engagement_score: float
    views_estimate: int
    recommendations: List[str]
    model_predictions: Optional[Dict[str, Any]] = None

class AvailableModelsResponse(BaseModel):
    classification: List[str]
    regression: List[str]

# Helper function to load data
def load_data():
    """Load the most recent trending data from the data directory."""
    try:
        # Look for the most recent processed data file
        data_dir = os.path.join(os.path.dirname(__file__), "../../data/processed")
        if not os.path.exists(data_dir):
            return None
        
        # Find the most recent file starting with "all_trending_"
        files = [f for f in os.listdir(data_dir) if f.startswith("all_trending_")]
        if not files:
            return None
        
        latest_file = max(files)
        file_path = os.path.join(data_dir, latest_file)
        
        # Load the data
        df = pd.read_csv(file_path)
        logger.info(f"Loaded trending data from {file_path}: {len(df)} rows")
        return df
    
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None

# Dependency to get data
def get_trending_data():
    """Get trending data from cache or load it."""
    if DATA_CACHE["trending_data"] is None:
        DATA_CACHE["trending_data"] = load_data()
        DATA_CACHE["last_updated"] = datetime.now()
    
    if DATA_CACHE["trending_data"] is None:
        raise HTTPException(status_code=404, detail="No trending data available. Run data collection first.")
    
    return DATA_CACHE["trending_data"]

# Dependency to get model service
def get_model_service():
    """Get or create model service instance."""
    if DATA_CACHE["model_service"] is None:
        DATA_CACHE["model_service"] = ModelService(MODELS_DIR)
    return DATA_CACHE["model_service"]

# Routes
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "YouTube Trending Analysis API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": [
            "/trending",
            "/trending/categories",
            "/trending/time-analysis",
            "/trending/popular-channels",
            "/predict",
            "/api/models",
            "/api/predict/all"
        ]
    }

@app.get("/trending", response_model=TrendingResponse)
async def get_trending(
    region: str = Query("US", description="Region code (e.g., US, GB, CA)"),
    category: Optional[str] = Query(None, description="Category ID"),
    limit: int = Query(50, description="Number of results to return"),
    data: pd.DataFrame = Depends(get_trending_data)
):
    """Get trending videos with optional filters."""
    # Filter by region
    filtered_data = data[data["region"] == region]
    
    # Filter by category if provided
    if category:
        filtered_data = filtered_data[filtered_data["category_id"] == category]
    
    # Sort by popularity score if available, otherwise by views
    if "popularity_score" in filtered_data.columns:
        filtered_data = filtered_data.sort_values("popularity_score", ascending=False)
    else:
        filtered_data = filtered_data.sort_values("views", ascending=False)
    
    # Limit the results
    filtered_data = filtered_data.head(limit)
    
    if filtered_data.empty:
        return TrendingResponse(count=0, videos=[])
    
    # Convert to response model
    videos = []
    for _, row in filtered_data.iterrows():
        videos.append(VideoDetails(
            video_id=row.get("video_id", ""),
            title=row.get("title", ""),
            channel_title=row.get("channel_title", ""),
            category_name=row.get("category_name", CATEGORY_MAPPING.get(row.get("category_id", ""), "Unknown")),
            views=int(row.get("views", 0)),
            likes=int(row.get("likes", 0)),
            comments=int(row.get("comments", 0)),
            publish_time=str(row.get("publish_time", "")),
            duration_seconds=int(row.get("duration_seconds", 0)),
            engagement_score=float(row.get("engagement_score", 0)),
            thumbnail_url=row.get("thumbnail_url", None),
            region=row.get("region", "")
        ))
    
    return TrendingResponse(count=len(videos), videos=videos)

@app.get("/trending/categories")
async def get_category_stats(
    region: str = Query("US", description="Region code (e.g., US, GB, CA)"),
    data: pd.DataFrame = Depends(get_trending_data)
):
    """Get performance statistics by video category."""
    # Filter by region
    filtered_data = data[data["region"] == region]
    
    if filtered_data.empty:
        return {"categories": []}
    
    # Group by category
    category_stats = filtered_data.groupby("category_name").agg({
        "video_id": "count",
        "views": ["mean", "sum"],
        "likes": "mean",
        "comments": "mean",
        "engagement_score": "mean"
    }).reset_index()
    
    # Flatten multi-level columns
    category_stats.columns = [
        "_".join(col).strip("_") for col in category_stats.columns.values
    ]
    
    # Rename columns
    category_stats = category_stats.rename(columns={
        "video_id_count": "video_count",
        "views_mean": "avg_views",
        "views_sum": "total_views",
        "likes_mean": "avg_likes",
        "comments_mean": "avg_comments",
        "engagement_score_mean": "avg_engagement"
    })
    
    # Sort by video count
    category_stats = category_stats.sort_values("video_count", ascending=False)
    
    return {"categories": category_stats.to_dict(orient="records"), "count": len(category_stats)}

@app.get("/trending/time-analysis")
async def get_time_analysis(
    region: str = Query("US", description="Region code (e.g., US, GB, CA)"),
    data: pd.DataFrame = Depends(get_trending_data)
):
    """Get time-based analysis of trending videos."""
    # Filter by region
    filtered_data = data[data["region"] == region]
    
    if filtered_data.empty:
        return {"error": "No data available for the specified region"}
    
    # Check if we have the required columns
    if "publish_hour" not in filtered_data.columns or "publish_weekday" not in filtered_data.columns:
        return {"error": "Time data not available in the dataset"}
    
    # Hour of day distribution
    hour_counts = filtered_data["publish_hour"].value_counts().sort_index()
    total_videos = len(filtered_data)
    hourly_distribution = {int(hour): count / total_videos * 100 for hour, count in hour_counts.items()}
    
    # Day of week distribution
    weekday_counts = filtered_data["publish_weekday"].value_counts().sort_index()
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday_distribution = {day_names[day]: count / total_videos * 100 
                          for day, count in weekday_counts.items() if day < len(day_names)}
    
    # Find best hour (highest average engagement)
    hour_engagement = filtered_data.groupby("publish_hour")["engagement_score"].mean()
    best_hour = int(hour_engagement.idxmax()) if not hour_engagement.empty else 0
    
    # Find best day (highest average engagement)
    day_engagement = filtered_data.groupby("publish_weekday")["engagement_score"].mean()
    best_weekday = int(day_engagement.idxmax()) if not day_engagement.empty else 0
    best_day = day_names[best_weekday] if best_weekday < len(day_names) else "Unknown"
    
    # Determine best time of day
    if 5 <= best_hour < 12:
        best_time_of_day = "Morning"
    elif 12 <= best_hour < 17:
        best_time_of_day = "Afternoon"
    elif 17 <= best_hour < 21:
        best_time_of_day = "Evening"
    else:
        best_time_of_day = "Night"
    
    return {
        "best_hour": best_hour,
        "best_day": best_day,
        "best_time_of_day": best_time_of_day,
        "hourly_distribution": hourly_distribution,
        "weekday_distribution": weekday_distribution
    }

@app.get("/trending/popular-channels")
async def get_popular_channels(
    region: str = Query("US", description="Region code (e.g., US, GB, CA)"),
    limit: int = Query(10, description="Number of channels to return"),
    data: pd.DataFrame = Depends(get_trending_data)
):
    """Get most popular channels based on trending videos."""
    # Filter by region
    filtered_data = data[data["region"] == region]
    
    if filtered_data.empty:
        return {"channels": []}
    
    # Group by channel
    channel_stats = filtered_data.groupby(["channel_id", "channel_title"]).agg({
        "video_id": "count",
        "views": "sum",
        "likes": "sum",
        "comments": "sum",
        "engagement_score": "mean"
    }).reset_index()
    
    # Rename columns
    channel_stats = channel_stats.rename(columns={
        "video_id": "video_count",
        "views": "total_views",
        "likes": "total_likes",
        "comments": "total_comments",
        "engagement_score": "avg_engagement"
    })
    
    # Sort by video count (descending)
    channel_stats = channel_stats.sort_values("video_count", ascending=False)
    
    # Limit results
    channel_stats = channel_stats.head(limit)
    
    return {"channels": channel_stats.to_dict(orient="records"), "count": len(channel_stats)}

@app.post("/predict", response_model=PredictionResponse)
async def predict_trending_potential(
    video: PredictionRequest,
    service: ModelService = Depends(get_model_service)
):
    """Predict the trending potential of a video based on its attributes and ML models."""
    # Convert PredictionRequest to dictionary
    video_data = video.dict()
    
    # Process text features
    process_text_features(video_data)
    
    # Try to get ML model predictions
    model_predictions = {}
    try:
        # Get all model predictions
        ml_predictions = service.make_all_predictions(video_data)
        
        # Add to model_predictions dictionary
        for model_type, predictions in ml_predictions.items():
            model_predictions[model_type] = predictions
    except Exception as e:
        logger.warning(f"Error making ML predictions: {e}")
        # Continue with heuristic-based predictions
    
    # Basic score calculation based on video attributes
    base_score = 5.0  # Middle of 0-10 scale
    
    # Title length factor (titles between 30-70 chars tend to perform better)
    title_length = len(video.title)
    if 30 <= title_length <= 70:
        title_factor = 0.5
    elif title_length < 20 or title_length > 100:
        title_factor = -0.5
    else:
        title_factor = 0
    
    # Duration factor (depends on category)
    duration_minutes = video.duration_seconds / 60
    if video.category_id in ['10', '24']:  # Music, Entertainment
        # Shorter videos tend to perform better in these categories
        duration_factor = 0.5 if duration_minutes < 10 else 0
    elif video.category_id in ['20', '28']:  # Gaming, Science & Tech
        # Medium length videos tend to perform better
        duration_factor = 0.5 if 10 <= duration_minutes <= 20 else 0
    elif video.category_id in ['22', '23', '27']:  # People & Blogs, Comedy, Education
        # Medium-long videos can perform well
        duration_factor = 0.5 if 8 <= duration_minutes <= 18 else 0
    else:
        duration_factor = 0
    
    # Tag factor
    tag_count = len(video.tags) if video.tags else 0
    tag_factor = 0.3 if 5 <= tag_count <= 15 else 0
    
    # Category popularity factor
    category_factors = {
        '10': 0.8,  # Music
        '24': 0.7,  # Entertainment
        '20': 0.6,  # Gaming
        '23': 0.5,  # Comedy
        '22': 0.4,  # People & Blogs
        '28': 0.4,  # Science & Tech
        '27': 0.3,  # Education
        '25': 0.2,  # News & Politics
        '1': 0.2,   # Film & Animation
        '26': 0.2,  # Howto & Style
        '17': 0.1,  # Sports
    }
    category_factor = category_factors.get(video.category_id, 0)
    
    # Calculate trending score (0-10 scale)
    trending_score = base_score + title_factor + duration_factor + tag_factor + category_factor
    
    # If we have ML predictions, incorporate them
    if 'classification' in model_predictions and 'is_viral' in model_predictions['classification']:
        viral_pred = model_predictions['classification']['is_viral']
        if 'probability' in viral_pred and viral_pred['probability'] is not None:
            # Blend ML prediction with heuristic
            ml_score = viral_pred['probability'] * 10  # Convert to 0-10 scale
            trending_score = (trending_score * 0.3) + (ml_score * 0.7)  # Weight ML prediction higher
    
    trending_score = max(0, min(10, trending_score))
    
    # Generate recommendations
    recommendations = []
    
    if title_length < 30:
        recommendations.append("Consider using a longer title (30-70 characters)")
    elif title_length > 70:
        recommendations.append("Consider using a slightly shorter title (30-70 characters)")
    
    if "?" not in video.title and "!" not in video.title:
        recommendations.append("Adding a question mark or exclamation point might increase engagement")
    
    if tag_count < 5:
        recommendations.append("Add more tags (aim for 5-15 relevant tags)")
    elif tag_count > 15:
        recommendations.append("Too many tags might dilute relevance. Focus on 5-15 highly relevant tags")
    
    # Category-specific recommendations
    if video.category_id == '10':  # Music
        if duration_minutes > 10:
            recommendations.append("Music videos tend to perform better when under 10 minutes")
    elif video.category_id == '20':  # Gaming
        if duration_minutes < 10:
            recommendations.append("Gaming videos tend to perform better between 10-20 minutes")
    elif video.category_id == '24':  # Entertainment
        recommendations.append("Entertainment videos with thumbnails featuring faces tend to perform better")
    
    # Estimate views based on trending score
    views_estimate = int(5000 * (2 ** trending_score / 32))
    
    # If we have ML predictions for views, use them
    if 'regression' in model_predictions and 'views_per_hour' in model_predictions['regression']:
        views_pred = model_predictions['regression']['views_per_hour']
        if 'prediction' in views_pred and views_pred['prediction'] is not None:
            # Use ML prediction but adjust with heuristic
            predicted_views_per_hour = views_pred['prediction']
            views_estimate = int(predicted_views_per_hour * 48)  # Estimate for 2 days
    
    # Engagement score estimate (0-10 scale)
    engagement_score = trending_score * 0.8  # Slightly lower than trending score typically
    
    # If we have ML predictions for engagement, use them
    if 'regression' in model_predictions and 'engagement_score' in model_predictions['regression']:
        eng_pred = model_predictions['regression']['engagement_score']
        if 'prediction' in eng_pred and eng_pred['prediction'] is not None:
            # Use ML prediction directly if available
            engagement_score = eng_pred['prediction']
            # Ensure it's in 0-10 range
            engagement_score = max(0, min(10, engagement_score))
    
    return PredictionResponse(
        trending_score=round(trending_score, 1),
        engagement_score=round(engagement_score, 1),
        views_estimate=views_estimate,
        recommendations=recommendations,
        model_predictions=model_predictions
    )

# New endpoints for ML models

@app.get("/api/models", response_model=AvailableModelsResponse, tags=["Models"])
def available_models(service: ModelService = Depends(get_model_service)):
    """Get available prediction models."""
    return service.get_available_models()

@app.get("/api/models/{target_name}/features", response_model=List[str], tags=["Models"])
def model_features(target_name: str, service: ModelService = Depends(get_model_service)):
    """Get required features for a specific model."""
    try:
        features = service.get_required_features(target_name)
        if not features:
            raise HTTPException(status_code=404, detail=f"Model for {target_name} not found")
        return features
    except Exception as e:
        logger.error(f"Error getting features for {target_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict/classification/{target_name}", tags=["Predictions"])
def predict_classification(
    target_name: str, 
    video_data: PredictionRequest,
    service: ModelService = Depends(get_model_service)
):
    """Make a classification prediction."""
    try:
        # Convert VideoData to dict
        data_dict = video_data.dict()
        
        # Process text features
        process_text_features(data_dict)
        
        # Make prediction
        result = service.predict_classification(data_dict, target_name)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error making classification prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict/regression/{target_name}", tags=["Predictions"])
def predict_regression(
    target_name: str, 
    video_data: PredictionRequest,
    service: ModelService = Depends(get_model_service)
):
    """Make a regression prediction."""
    try:
        # Convert VideoData to dict
        data_dict = video_data.dict()
        
        # Process text features
        process_text_features(data_dict)
        
        # Make prediction
        result = service.predict_regression(data_dict, target_name)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error making regression prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict/all", tags=["Predictions"])
def predict_all(
    video_data: PredictionRequest,
    service: ModelService = Depends(get_model_service)
):
    """Make predictions with all available models."""
    try:
        # Convert VideoData to dict
        data_dict = video_data.dict()
        
        # Process text features
        process_text_features(data_dict)
        
        # Make all predictions
        results = service.make_all_predictions(data_dict)
        
        # Add timestamp
        results['timestamp'] = datetime.now().isoformat()
        
        return results
    except Exception as e:
        logger.error(f"Error making predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Helper function to extract text features
def process_text_features(data_dict: Dict[str, Any]):
    """Process and extract features from text fields."""
    import re
    
    # Process title if available
    if 'title' in data_dict and data_dict['title']:
        title = data_dict['title']
        
        # Extract basic title features
        data_dict['title_length'] = len(title)
        data_dict['title_word_count'] = len(title.split())
        data_dict['title_has_number'] = int(bool(re.search(r'\d', title)))
        data_dict['title_has_question'] = int('?' in title)
        data_dict['title_has_exclamation'] = int('!' in title)
        
        # Count capitalized words
        words = title.split()
        data_dict['title_caps_count'] = sum(1 for word in words if word.isupper() and len(word) > 1)
    
    # Process description if available
    if 'description' in data_dict and data_dict['description']:
        desc = data_dict['description']
        
        # Extract basic description features
        data_dict['description_length'] = len(desc)
        data_dict['description_word_count'] = len(desc.split())
        data_dict['description_url_count'] = desc.count('http')
    
    # Process tags if available
    if 'tags' in data_dict and data_dict['tags']:
        tags = data_dict['tags']
        
        # Extract tag features
        data_dict['tag_count'] = len(tags)
        data_dict['tag_avg_length'] = sum(len(tag) for tag in tags) / max(len(tags), 1)
    
    return data_dict

# Run the app with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)