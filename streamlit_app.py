#!/usr/bin/env python
# streamlit_app.py - Main Dashboard for YouTube Trending Video Analysis

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import json
import os
import sys
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from pathlib import Path
import pickle

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

# Try to import project modules
try:
    from src.data.youtube_fetcher import CATEGORY_MAPPING
except ImportError:
    # Fallback if module not found
    CATEGORY_MAPPING = {
        '1': 'Film & Animation',
        '2': 'Autos & Vehicles',
        '10': 'Music',
        '15': 'Pets & Animals',
        '17': 'Sports',
        '18': 'Short Movies',
        '19': 'Travel & Events',
        '20': 'Gaming',
        '21': 'Videoblogging',
        '22': 'People & Blogs',
        '23': 'Comedy',
        '24': 'Entertainment',
        '25': 'News & Politics',
        '26': 'Howto & Style',
        '27': 'Education',
        '28': 'Science & Technology',
        '29': 'Nonprofits & Activism',
        '30': 'Movies',
        '31': 'Anime/Animation',
        '32': 'Action/Adventure',
        '33': 'Classics',
        '34': 'Comedy',
        '35': 'Documentary',
        '36': 'Drama',
        '37': 'Family',
        '38': 'Foreign',
        '39': 'Horror',
        '40': 'Sci-Fi/Fantasy',
        '41': 'Thriller',
        '42': 'Shorts',
        '43': 'Shows',
        '44': 'Trailers'
    }

# Set page configuration
st.set_page_config(
    page_title="YouTube Trending Video Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 1rem 1rem;
    }
    .st-emotion-cache-16idsys p {
        font-size: 16px;
    }
    .header-text {
        font-size: 24px !important;
        font-weight: bold;
        color: #FF0000;
    }
    .subheader-text {
        font-size: 20px !important;
        font-weight: bold;
        color: #606060;
    }
    .card {
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        background-color: white;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-left: 4px solid #FF0000;
    }
    </style>
""", unsafe_allow_html=True)

# Cache function for loading data
@st.cache_data(ttl=3600)
def load_data(data_path=None):
    """Load the trending video data."""
    try:
        if data_path is None:
            # Find the most recent data file
            data_dir = os.path.join(project_root, "data/processed")
            if not os.path.exists(data_dir):
                return None
            
            # Find most recent trending data file
            files = [f for f in os.listdir(data_dir) if f.startswith("all_trending_")]
            if not files:
                return None
            
            latest_file = max(files)
            data_path = os.path.join(data_dir, latest_file)
        
        # Load the data
        df = pd.read_csv(data_path)
        
        # Process datetime columns
        datetime_cols = ['publish_time', 'fetch_time']
        for col in datetime_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Add category names if needed
        if 'category_id' in df.columns and 'category_name' not in df.columns:
            df['category_name'] = df['category_id'].astype(str).map(CATEGORY_MAPPING).fillna('Unknown')
        
        return df
    
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Load model metadata
@st.cache_data(ttl=3600)
def load_model_metadata(models_dir="models"):
    """Load metadata for all trained models."""
    metadata = {
        "classification": {},
        "regression": {}
    }
    
    try:
        # Check classification models
        cls_dir = os.path.join(models_dir, "classification")
        if os.path.exists(cls_dir):
            for model_dir in os.listdir(cls_dir):
                model_path = os.path.join(cls_dir, model_dir)
                if os.path.isdir(model_path):
                    metadata_file = os.path.join(model_path, f"{model_dir}_metadata.json")
                    if os.path.exists(metadata_file):
                        with open(metadata_file, 'r') as f:
                            model_metadata = json.load(f)
                            target = model_metadata.get("target_name", model_dir)
                            metadata["classification"][target] = model_metadata
        
        # Check regression models
        reg_dir = os.path.join(models_dir, "regression")
        if os.path.exists(reg_dir):
            for model_dir in os.listdir(reg_dir):
                model_path = os.path.join(reg_dir, model_dir)
                if os.path.isdir(model_path):
                    metadata_file = os.path.join(model_path, f"{model_dir}_metadata.json")
                    if os.path.exists(metadata_file):
                        with open(metadata_file, 'r') as f:
                            model_metadata = json.load(f)
                            target = model_metadata.get("target_name", model_dir)
                            metadata["regression"][target] = model_metadata
    
    except Exception as e:
        st.error(f"Error loading model metadata: {e}")
    
    return metadata

# Cache function for getting API data
@st.cache_data(ttl=300)
def get_api_data(endpoint, params=None):
    """Get data from the API."""
    api_base_url = st.session_state.get('api_url', 'http://localhost:8000')
    url = f"{api_base_url}/{endpoint}"
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching data from API: {e}")
        return None

# Cache function for making predictions
@st.cache_data(ttl=60)
def make_prediction(video_data):
    """Make a prediction using the API."""
    api_base_url = st.session_state.get('api_url', 'http://localhost:8000')
    url = f"{api_base_url}/api/predict/all"
    
    try:
        response = requests.post(url, json=video_data)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None

# Function to get available models
@st.cache_data(ttl=300)
def get_available_models():
    """Get available models from the API."""
    api_base_url = st.session_state.get('api_url', 'http://localhost:8000')
    url = f"{api_base_url}/api/models"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.warning(f"Could not fetch model information: {e}")
        return {"classification": [], "regression": []}

# Helper function to format metrics
def format_number(num):
    """Format large numbers in a readable way."""
    if num >= 1_000_000:
        return f"{num/1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num/1_000:.1f}K"
    else:
        return str(num)

# Sidebar
with st.sidebar:
    st.image("https://www.gstatic.com/youtube/img/branding/youtubelogo/svg/youtubelogo.svg", width=100)
    st.title("YouTube Trend Analyzer")
    
    st.markdown("---")
    
    # Navigation
    page = st.radio(
        "Navigation",
        ["📊 Dashboard", "📈 Trending Analysis", "🧠 ML Insights", "🔮 Prediction Tool", "⚙️ Settings"],
        key="main_navigation"
    )
    
    st.markdown("---")
    
    # Region selection (global filter)
    regions = ["US", "GB", "CA", "IN", "JP", "KR", "DE", "FR", "RU", "BR", "MX", "AU"]
    selected_region = st.selectbox("Select Region", regions, index=0,key="region_selector")
    
    # Category filter (global)
    if 'data' not in st.session_state:
        st.session_state.data = load_data()
    
    if st.session_state.data is not None:
        categories = st.session_state.data['category_name'].unique().tolist()
        categories = ['All Categories'] + sorted(categories)
        selected_category = st.selectbox("Select Category", categories, key="category_selector")
    else:
        selected_category = "All Categories"
    
    st.markdown("---")
    
    # API configuration
    st.subheader("API Connection")
    api_url = st.text_input("API URL", value=st.session_state.get('api_url', 'http://localhost:8000'), key="sidebar_api_url")
    if st.button("Connect", key="sidebar_connect_button"):
        st.session_state.api_url = api_url
        st.success(f"Connected to API at {api_url}")

# Main content
if page == "📊 Dashboard":
    st.title("YouTube Trending Videos Dashboard")
    st.markdown("Overview of trending video metrics and patterns")
    
    # Load data
    if 'data' not in st.session_state:
        st.session_state.data = load_data()
    
    if st.session_state.data is None:
        st.error("No data available. Please run data collection first.")
        st.stop()
    
    df = st.session_state.data
    
    # Filter by region
    df_region = df[df['region'] == selected_region]
    
    # Filter by category if not "All Categories"
    if selected_category != "All Categories":
        df_region = df_region[df_region['category_name'] == selected_category]
    
    if df_region.empty:
        st.warning(f"No data available for {selected_region} region and {selected_category} category.")
        st.stop()
    
    # Top metrics row
    st.markdown('<div class="header-text">Key Metrics</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="card metric-card">', unsafe_allow_html=True)
        st.metric("Videos Analyzed", format_number(len(df_region)))
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card metric-card">', unsafe_allow_html=True)
        st.metric("Avg. Views", format_number(int(df_region['views'].mean())))
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="card metric-card">', unsafe_allow_html=True)
        st.metric("Avg. Likes", format_number(int(df_region['likes'].mean())))
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="card metric-card">', unsafe_allow_html=True)
        st.metric("Avg. Comments", format_number(int(df_region['comments'].mean())))
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Visualizations row
    st.markdown('<div class="header-text">Trending Patterns</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="subheader-text">Category Distribution</div>', unsafe_allow_html=True)
        
        # Category distribution chart
        category_counts = df_region['category_name'].value_counts().reset_index()
        category_counts.columns = ['category_name', 'count']
        
        fig = px.bar(
            category_counts.sort_values('count', ascending=False).head(10),
            x='category_name',
            y='count',
            title="Top 10 Categories",
            color='count',
            color_continuous_scale='Reds'
        )
        fig.update_layout(xaxis_title="Category", yaxis_title="Number of Videos")
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="subheader-text">Publishing Time Patterns</div>', unsafe_allow_html=True)
        
        # Publishing hour heatmap
        if 'publish_hour' in df_region.columns and 'publish_weekday' in df_region.columns:
            # Create pivot table for hour vs day
            pivot_data = df_region.pivot_table(
                index='publish_weekday',
                columns='publish_hour',
                values='video_id',
                aggfunc='count',
                fill_value=0
            )
            
            # Map weekday numbers to names
            day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            pivot_data.index = [day_names[i] if i < len(day_names) else f"Day {i}" for i in pivot_data.index]
            
            # Create heatmap
            fig = px.imshow(
                pivot_data,
                labels=dict(x="Hour of Day", y="Day of Week", color="Video Count"),
                x=[str(h) for h in range(24)],
                y=pivot_data.index,
                color_continuous_scale='Reds',
                title="Publication Time Heatmap"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Time data not available in the dataset")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Second row of visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="subheader-text">Engagement Metrics</div>', unsafe_allow_html=True)
        
        # Engagement scatter plot
        fig = px.scatter(
            df_region,
            x='views',
            y='likes',
            color='category_name',
            size='comments',
            hover_name='title',
            opacity=0.7,
            title="Views vs. Likes by Category",
            log_x=True,
            log_y=True
        )
        fig.update_layout(xaxis_title="Views (log scale)", yaxis_title="Likes (log scale)")
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="subheader-text">Video Duration Analysis</div>', unsafe_allow_html=True)
        
        # Duration histogram by category
        if 'duration_seconds' in df_region.columns:
            # Convert to minutes for better visualization
            df_region['duration_minutes'] = df_region['duration_seconds'] / 60
            
            # Get top 5 categories
            top_categories = df_region['category_name'].value_counts().head(5).index.tolist()
            df_top_cats = df_region[df_region['category_name'].isin(top_categories)]
            
            fig = px.histogram(
                df_top_cats,
                x='duration_minutes',
                color='category_name',
                marginal="box",
                opacity=0.7,
                title="Video Duration Distribution by Category",
                range_x=[0, 50]  # Limit to 50 minutes for better visualization
            )
            fig.update_layout(xaxis_title="Duration (minutes)", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Duration data not available in the dataset")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Trending videos table
    st.markdown('<div class="header-text">Top Trending Videos</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    # Sort by views
    top_videos = df_region.sort_values("views", ascending=False).head(10)
    
    # Display table
    st.dataframe(
        top_videos[['title', 'channel_title', 'category_name', 'views', 'likes', 'comments']],
        column_config={
            "title": "Video Title",
            "channel_title": "Channel",
            "category_name": "Category",
            "views": st.column_config.NumberColumn("Views", format="%d"),
            "likes": st.column_config.NumberColumn("Likes", format="%d"),
            "comments": st.column_config.NumberColumn("Comments", format="%d")
        },
        hide_index=True,
        use_container_width=True
    )
    
    st.markdown("</div>", unsafe_allow_html=True)

elif page == "📈 Trending Analysis":
    st.title("Detailed Trending Analysis")
    st.markdown("In-depth analysis of trending patterns and metrics")
    
    # Load data
    if 'data' not in st.session_state:
        st.session_state.data = load_data()
    
    if st.session_state.data is None:
        st.error("No data available. Please run data collection first.")
        st.stop()
    
    df = st.session_state.data
    
    # Filter by region
    df_region = df[df['region'] == selected_region]
    
    # Filter by category if not "All Categories"
    if selected_category != "All Categories":
        df_region = df_region[df_region['category_name'] == selected_category]
    
    if df_region.empty:
        st.warning(f"No data available for {selected_region} region and {selected_category} category.")
        st.stop()
    
    # Analysis tabs
    tabs = st.tabs(["Category Analysis", "Time Analysis", "Channel Analysis", "Engagement Analysis"])
    
    # Category Analysis Tab
    with tabs[0]:
        st.markdown("### Category Performance Analysis")
        
        # Group by category
        category_stats = df_region.groupby("category_name").agg({
            "video_id": "count",
            "views": ["mean", "sum"],
            "likes": "mean",
            "comments": "mean",
            "engagement_score": "mean" if "engagement_score" in df_region.columns else "count"
        }).reset_index()
        
        # Flatten multi-level columns
        category_stats.columns = ["_".join(col).strip("_") for col in category_stats.columns.values]
        
        # Rename columns for clarity
        category_stats = category_stats.rename(columns={
            "video_id_count": "video_count",
            "views_mean": "avg_views",
            "views_sum": "total_views",
            "likes_mean": "avg_likes",
            "comments_mean": "avg_comments"
        })
        
        if "engagement_score_mean" in category_stats.columns:
            category_stats = category_stats.rename(columns={"engagement_score_mean": "avg_engagement"})
        else:
            category_stats = category_stats.rename(columns={"engagement_score_count": "video_count2"})
            category_stats["avg_engagement"] = category_stats["avg_likes"] / category_stats["avg_views"] * 100
        
        # Sort by video count
        category_stats = category_stats.sort_values("video_count", ascending=False)
        
        # Display metrics selection
        selected_metric = st.selectbox(
            "Select metric for comparison",
            ["video_count", "avg_views", "total_views", "avg_likes", "avg_comments", "avg_engagement"]
        )
        
        # Create bar chart
        fig = px.bar(
            category_stats.sort_values(selected_metric, ascending=False),
            x="category_name",
            y=selected_metric,
            title=f"Categories by {selected_metric.replace('_', ' ').title()}",
            color=selected_metric,
            color_continuous_scale="Reds"
        )
        fig.update_layout(xaxis_title="Category", yaxis_title=selected_metric.replace("_", " ").title())
        st.plotly_chart(fig, use_container_width=True)
        
        # Show data table
        st.dataframe(
            category_stats,
            column_config={
                "category_name": "Category",
                "video_count": "Videos",
                "avg_views": st.column_config.NumberColumn("Avg. Views", format="%.0f"),
                "total_views": st.column_config.NumberColumn("Total Views", format="%.0f"),
                "avg_likes": st.column_config.NumberColumn("Avg. Likes", format="%.0f"),
                "avg_comments": st.column_config.NumberColumn("Avg. Comments", format="%.0f"),
                "avg_engagement": st.column_config.NumberColumn("Engagement Score", format="%.2f")
            },
            hide_index=True,
            use_container_width=True
        )
    
    # Time Analysis Tab
    with tabs[1]:
        st.markdown("### Temporal Patterns Analysis")
        
        if 'publish_hour' in df_region.columns and 'publish_weekday' in df_region.columns:
            # Get time analysis data from API
            time_analysis = get_api_data("trending/time-analysis", {"region": selected_region})
            
            if time_analysis and "error" not in time_analysis:
                # Display best time metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown('<div class="card metric-card">', unsafe_allow_html=True)
                    st.metric("Best Day", time_analysis.get("best_day", "Unknown"))
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="card metric-card">', unsafe_allow_html=True)
                    st.metric("Best Hour", f"{time_analysis.get('best_hour', 0)}:00")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col3:
                    st.markdown('<div class="card metric-card">', unsafe_allow_html=True)
                    st.metric("Best Time of Day", time_analysis.get("best_time_of_day", "Unknown"))
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Create two columns for visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    # Day of week distribution
                    weekday_data = time_analysis.get("weekday_distribution", {})
                    weekday_df = pd.DataFrame({
                        "Day": list(weekday_data.keys()),
                        "Percentage": list(weekday_data.values())
                    })
                    
                    # Sort by day of week
                    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                    weekday_df["Day"] = pd.Categorical(weekday_df["Day"], categories=day_order, ordered=True)
                    weekday_df = weekday_df.sort_values("Day")
                    
                    fig = px.bar(
                        weekday_df,
                        x="Day",
                        y="Percentage",
                        title="Day of Week Distribution",
                        color="Percentage",
                        color_continuous_scale="Reds"
                    )
                    fig.update_layout(xaxis_title="Day of Week", yaxis_title="Percentage of Videos (%)")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Hour of day distribution
                    hourly_data = time_analysis.get("hourly_distribution", {})
                    hourly_df = pd.DataFrame({
                        "Hour": [f"{h}:00" for h in hourly_data.keys()],
                        "Hour_num": list(map(int, hourly_data.keys())),
                        "Percentage": list(hourly_data.values())
                    })
                    
                    # Sort by hour
                    hourly_df = hourly_df.sort_values("Hour_num")
                    
                    fig = px.line(
                        hourly_df,
                        x="Hour",
                        y="Percentage",
                        title="Hour of Day Distribution",
                        markers=True
                    )
                    fig.update_layout(xaxis_title="Hour of Day", yaxis_title="Percentage of Videos (%)")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Could not fetch time analysis data from the API.")
        else:
            st.info("Time data not available in the dataset")
    
    # Channel Analysis Tab
    with tabs[2]:
        st.markdown("### Channel Performance Analysis")
        
        # Get popular channels from API
        popular_channels = get_api_data("trending/popular-channels", {
            "region": selected_region,
            "limit": 20
        })
        
        if popular_channels and "channels" in popular_channels:
            channels_df = pd.DataFrame(popular_channels["channels"])
            
            # Display top channels table
            st.dataframe(
                channels_df,
                column_config={
                    "channel_title": "Channel",
                    "video_count": "Videos",
                    "total_views": st.column_config.NumberColumn("Total Views", format="%.0f"),
                    "total_likes": st.column_config.NumberColumn("Total Likes", format="%.0f"),
                    "total_comments": st.column_config.NumberColumn("Total Comments", format="%.0f"),
                    "avg_engagement": st.column_config.NumberColumn("Avg. Engagement", format="%.2f")
                },
                hide_index=True,
                use_container_width=True
            )
            
            # Visualization of top 10 channels by video count
            top_10_channels = channels_df.sort_values("video_count", ascending=False).head(10)
            
            fig = px.bar(
                top_10_channels,
                x="channel_title",
                y="video_count",
                title="Top 10 Channels by Number of Trending Videos",
                color="avg_engagement",
                color_continuous_scale="Reds"
            )
            fig.update_layout(xaxis_title="Channel", yaxis_title="Number of Trending Videos")
            st.plotly_chart(fig, use_container_width=True)
            
            # Engagement vs. Video Count
            fig = px.scatter(
                channels_df,
                x="video_count",
                y="avg_engagement",
                size="total_views",
                hover_name="channel_title",
                title="Channel Engagement vs. Trending Video Count",
                log_y=True
            )
            fig.update_layout(xaxis_title="Number of Trending Videos", yaxis_title="Avg. Engagement (log scale)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Could not fetch channel analysis data from the API.")
    
    # Engagement Analysis Tab
    with tabs[3]:
        st.markdown("### Engagement Metrics Analysis")
        
        # Calculate engagement metrics if needed
        if "engagement_score" not in df_region.columns:
            df_region["engagement_score"] = (df_region["likes"] + df_region["comments"] * 5) / df_region["views"] * 100
        
        # Display engagement distribution
        fig = px.histogram(
            df_region,
            x="engagement_score",
            color="category_name",
            title="Engagement Score Distribution",
            opacity=0.7,
            nbins=50
        )
        fig.update_layout(xaxis_title="Engagement Score", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)
        
        # Views vs. likes scatter plot
        fig = px.scatter(
            df_region,
            x="views",
            y="likes",
            color="engagement_score",
            size="comments",
            hover_name="title",
            hover_data=["channel_title", "category_name"],
            log_x=True,
            log_y=True,
            title="Views vs. Likes (colored by engagement score)"
        )
        fig.update_layout(xaxis_title="Views (log scale)", yaxis_title="Likes (log scale)")
        st.plotly_chart(fig, use_container_width=True)
        
        # Average engagement by category
        engagement_by_category = df_region.groupby("category_name")["engagement_score"].mean().reset_index()
        engagement_by_category = engagement_by_category.sort_values("engagement_score", ascending=False)
        
        fig = px.bar(
            engagement_by_category,
            x="category_name",
            y="engagement_score",
            title="Average Engagement Score by Category",
            color="engagement_score",
            color_continuous_scale="Reds"
        )
        fig.update_layout(xaxis_title="Category", yaxis_title="Avg. Engagement Score")
        st.plotly_chart(fig, use_container_width=True)

# ML Insights section of streamlit_app.py

elif page == "🧠 ML Insights":
    st.title("Machine Learning Insights")
    st.markdown("Explore model performance and predictions for trending videos")
    
    # Get available models
    available_models = get_available_models()
    classification_models = available_models.get("classification", [])
    regression_models = available_models.get("regression", [])
    
    # Load model metadata
    model_metadata = load_model_metadata()
    
    if not classification_models and not regression_models:
        st.warning("No trained models available. Please train models first.")
        st.stop()
    
    # Display model overview
    st.markdown("### Available Trained Models")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Classification Models")
        if classification_models:
            for model in classification_models:
                st.markdown(f"- **{model}**: Predicts whether a video will trend as '{model}'")
        else:
            st.info("No classification models available")
    
    with col2:
        st.markdown("#### Regression Models")
        if regression_models:
            for model in regression_models:
                st.markdown(f"- **{model}**: Predicts the value of '{model}' for a video")
        else:
            st.info("No regression models available")
    
    # Model analysis tabs
    tabs = st.tabs(["Model Performance", "Feature Importance", "Prediction Analysis"])
    
    # Model Performance Tab
    with tabs[0]:
        st.markdown("### Model Performance Metrics")
        
        # Choose model type
        model_type = st.radio("Select Model Type", ["Classification", "Regression"], horizontal=True)
        
        if model_type == "Classification" and classification_models:
            # Select model
            selected_model = st.selectbox("Select Model", classification_models)
            
            # Get model metadata
            model_meta = model_metadata["classification"].get(selected_model, {})
            
            if model_meta and "metrics" in model_meta:
                metrics = model_meta["metrics"]
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown('<div class="card metric-card">', unsafe_allow_html=True)
                    st.metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="card metric-card">', unsafe_allow_html=True)
                    st.metric("Precision", f"{metrics.get('precision', 0):.3f}")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col3:
                    st.markdown('<div class="card metric-card">', unsafe_allow_html=True)
                    st.metric("Recall", f"{metrics.get('recall', 0):.3f}")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col4:
                    st.markdown('<div class="card metric-card">', unsafe_allow_html=True)
                    st.metric("F1 Score", f"{metrics.get('f1', 0):.3f}")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # AUC-ROC if available
                if "auc" in metrics:
                    st.markdown("#### ROC-AUC Score")
                    st.markdown('<div class="card metric-card">', unsafe_allow_html=True)
                    st.metric("AUC", f"{metrics.get('auc', 0):.3f}")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Model details
                st.markdown("#### Model Details")
                st.json(model_meta)
            else:
                st.warning(f"No performance metrics available for {selected_model}")
        
        elif model_type == "Regression" and regression_models:
            # Select model
            selected_model = st.selectbox("Select Model", regression_models)
            
            # Get model metadata
            model_meta = model_metadata["regression"].get(selected_model, {})
            
            if model_meta and "metrics" in model_meta:
                metrics = model_meta["metrics"]
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown('<div class="card metric-card">', unsafe_allow_html=True)
                    st.metric("R² Score", f"{metrics.get('r2', 0):.3f}")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="card metric-card">', unsafe_allow_html=True)
                    st.metric("RMSE", f"{metrics.get('rmse', 0):.3f}")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col3:
                    st.markdown('<div class="card metric-card">', unsafe_allow_html=True)
                    st.metric("MAE", f"{metrics.get('mae', 0):.3f}")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col4:
                    st.markdown('<div class="card metric-card">', unsafe_allow_html=True)
                    st.metric("MSE", f"{metrics.get('mse', 0):.3f}")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Model details
                st.markdown("#### Model Details")
                st.json(model_meta)
            else:
                st.warning(f"No performance metrics available for {selected_model}")
        else:
            st.warning(f"No {model_type.lower()} models available")
    
    # Feature Importance Tab
    with tabs[1]:
        st.markdown("### Feature Importance Analysis")
        
        # Choose model type
        model_type = st.radio("Select Model Type", ["Classification", "Regression"], horizontal=True, key="fi_model_type")
        
        if model_type == "Classification" and classification_models:
            # Select model
            selected_model = st.selectbox("Select Model", classification_models, key="fi_class_model")
            
            # Get model metadata
            model_meta = model_metadata["classification"].get(selected_model, {})
            
            if model_meta and "feature_importances" in model_meta:
                importances = model_meta["feature_importances"]
                
                # Convert to DataFrame
                importance_df = pd.DataFrame({
                    "Feature": list(importances.keys()),
                    "Importance": list(importances.values())
                })
                
                # Sort by importance
                importance_df = importance_df.sort_values("Importance", ascending=False)
                
                # Plot top features
                top_n = st.slider("Number of top features to show", 5, 50, 20)
                top_features = importance_df.head(top_n)
                
                fig = px.bar(
                    top_features,
                    x="Importance",
                    y="Feature",
                    orientation='h',
                    title=f"Top {top_n} Features for {selected_model}",
                    color="Importance",
                    color_continuous_scale="Reds"
                )
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
                
                # Display feature table
                st.dataframe(
                    importance_df,
                    column_config={
                        "Feature": "Feature Name",
                        "Importance": st.column_config.NumberColumn("Importance Score", format="%.5f")
                    },
                    hide_index=True,
                    use_container_width=True
                )
            else:
                st.warning(f"No feature importance data available for {selected_model}")
        
        elif model_type == "Regression" and regression_models:
            # Select model
            selected_model = st.selectbox("Select Model", regression_models, key="fi_reg_model")
            
            # Get model metadata
            model_meta = model_metadata["regression"].get(selected_model, {})
            
            if model_meta and "feature_importances" in model_meta:
                importances = model_meta["feature_importances"]
                
                # Convert to DataFrame
                importance_df = pd.DataFrame({
                    "Feature": list(importances.keys()),
                    "Importance": list(importances.values())
                })
                
                # Sort by importance
                importance_df = importance_df.sort_values("Importance", ascending=False)
                
                # Plot top features
                top_n = st.slider("Number of top features to show", 5, 50, 20, key="fi_reg_slider")
                top_features = importance_df.head(top_n)
                
                fig = px.bar(
                    top_features,
                    x="Importance",
                    y="Feature",
                    orientation='h',
                    title=f"Top {top_n} Features for {selected_model}",
                    color="Importance",
                    color_continuous_scale="Blues"
                )
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
                
                # Display feature table
                st.dataframe(
                    importance_df,
                    column_config={
                        "Feature": "Feature Name",
                        "Importance": st.column_config.NumberColumn("Importance Score", format="%.5f")
                    },
                    hide_index=True,
                    use_container_width=True
                )
            else:
                st.warning(f"No feature importance data available for {selected_model}")
        else:
            st.warning(f"No {model_type.lower()} models available")
    
    # Prediction Analysis Tab
    with tabs[2]:
        st.markdown("### Prediction Analysis")
        
        # Load data if not already loaded
        if 'data' not in st.session_state:
            st.session_state.data = load_data()
        
        if st.session_state.data is None:
            st.error("No data available. Please run data collection first.")
            st.stop()
        
        df = st.session_state.data
        
        # Filter by region
        df_region = df[df['region'] == selected_region]
        
        # Filter by category if not "All Categories"
        if selected_category != "All Categories":
            df_region = df_region[df_region['category_name'] == selected_category]
        
        if df_region.empty:
            st.warning(f"No data available for {selected_region} region and {selected_category} category.")
            st.stop()
        
        # Select videos for prediction
        st.markdown("#### Select Videos for Prediction")
        num_videos = st.slider("Number of videos to analyze", 5, 20, 10)
        
        # Sample videos
        sampled_videos = df_region.sample(min(num_videos, len(df_region)))
        
        # Make predictions for selected videos
        predictions = []
        
        for _, video in sampled_videos.iterrows():
            # Prepare video data for prediction
            video_data = {
                "title": video.get("title", ""),
                "description": video.get("description", ""),
                "duration_seconds": int(video.get("duration_seconds", 0)),
                "category_id": str(video.get("category_id", "")),
                "tags": video.get("tags", "").split("|") if isinstance(video.get("tags", ""), str) else [],
                "publish_hour": int(video.get("publish_hour", 0)) if "publish_hour" in video else None,
                "publish_day": int(video.get("publish_weekday", 0)) if "publish_weekday" in video else None
            }
            
            # Make prediction
            try:
                prediction_result = make_prediction(video_data)
                
                # Store prediction with video data
                predictions.append({
                    "video": video,
                    "prediction": prediction_result
                })
            except Exception as e:
                st.error(f"Error making prediction for video {video.get('title', '')}: {e}")
        
        # Display predictions
        if predictions:
            st.markdown("#### Prediction Results")
            
            for idx, pred in enumerate(predictions):
                video = pred["video"]
                prediction = pred["prediction"]
                
                # Create expandable section for each video
                with st.expander(f"{idx+1}. {video.get('title', '')}", expanded=idx == 0):
                    # Video details
                    st.markdown(f"**Channel**: {video.get('channel_title', '')}")
                    st.markdown(f"**Category**: {video.get('category_name', '')}")
                    st.markdown(f"**Duration**: {int(video.get('duration_seconds', 0) / 60)} minutes")
                    st.markdown(f"**Actual Views**: {format_number(video.get('views', 0))}")
                    
                    # Prediction results
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("##### Predicted Trending Score")
                        if "trending_score" in prediction:
                            score = prediction["trending_score"]
                            st.progress(score / 10)
                            st.markdown(f"**Score**: {score:.1f}/10")
                        else:
                            st.info("No trending score available")
                    
                    with col2:
                        st.markdown("##### Predicted Engagement")
                        if "engagement_score" in prediction:
                            score = prediction["engagement_score"]
                            st.progress(score / 10)
                            st.markdown(f"**Score**: {score:.1f}/10")
                        else:
                            st.info("No engagement score available")
                    
                    with col3:
                        st.markdown("##### Predicted Views")
                        if "views_estimate" in prediction:
                            views = prediction["views_estimate"]
                            st.markdown(f"**Views**: {format_number(views)}")
                        else:
                            st.info("No views estimate available")
                    
                    # ML model details
                    if "model_predictions" in prediction:
                        st.markdown("##### ML Model Predictions")
                        
                        # Classification results
                        if "classification" in prediction["model_predictions"]:
                            st.markdown("**Classification Results**")
                            cls_results = prediction["model_predictions"]["classification"]
                            
                            # Convert to DataFrame for display
                            cls_data = []
                            for target, result in cls_results.items():
                                cls_data.append({
                                    "Target": target,
                                    "Prediction": bool(result.get("prediction", False)),
                                    "Probability": result.get("probability", None),
                                    "Model Type": result.get("model_type", "Unknown")
                                })
                            
                            if cls_data:
                                cls_df = pd.DataFrame(cls_data)
                                st.dataframe(cls_df, hide_index=True)
                        
                        # Regression results
                        if "regression" in prediction["model_predictions"]:
                            st.markdown("**Regression Results**")
                            reg_results = prediction["model_predictions"]["regression"]
                            
                            # Convert to DataFrame for display
                            reg_data = []
                            for target, result in reg_results.items():
                                reg_data.append({
                                    "Target": target,
                                    "Prediction": result.get("prediction", 0),
                                    "Model Type": result.get("model_type", "Unknown")
                                })
                            
                            if reg_data:
                                reg_df = pd.DataFrame(reg_data)
                                st.dataframe(reg_df, hide_index=True)
                    
                    # Recommendations
                    if "recommendations" in prediction and prediction["recommendations"]:
                        st.markdown("##### Recommendations")
                        for rec in prediction["recommendations"]:
                            st.markdown(f"- {rec}")
        else:
            st.warning("No prediction results available.")

elif page == "🔮 Prediction Tool":
    st.title("YouTube Trending Prediction Tool")
    st.markdown("Predict the trending potential of a new video")
    
    # Input form for video details
    st.markdown("### Enter Video Details")
    
    with st.form("prediction_form"):
        # Basic information
        title = st.text_input("Video Title", "How to Build a Machine Learning Model - Complete Tutorial 2025",key="prediction_text_title")
        
        # Description
        description = st.text_area("Video Description", "Learn how to build a machine learning model from scratch in this comprehensive tutorial. We'll cover data preprocessing, feature engineering, model selection, and evaluation.")
        
        # Category selection
        categories = list(CATEGORY_MAPPING.items())
        category_options = [f"{cat_id}: {cat_name}" for cat_id, cat_name in categories]

        # Find the index of Education category (or default to 0)
        education_index = 0
        for i, option in enumerate(category_options):
            if "Education" in option:
                education_index = i
                break

        selected_cat = st.selectbox("Video Category", category_options, index=education_index)
        category_id = selected_cat.split(":")[0].strip()
        
        # Duration
        duration_mins = st.number_input("Duration (minutes)", min_value=1, max_value=60, value=15)
        duration_seconds = duration_mins * 60
        
        # Tags
        tags_input = st.text_input("Tags (comma separated)", "machine learning, tutorial, AI, python, data science")
        tags = [tag.strip() for tag in tags_input.split(",") if tag.strip()]
        
        # Publishing details
        col1, col2, col3 = st.columns(3)
        
        with col1:
            publish_hour = st.selectbox("Publishing Hour", list(range(24)), index=14)  # Default to 2 PM
        
        with col2:
            days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            publish_day = st.selectbox("Publishing Day", days, index=2)  # Default to Wednesday
            publish_day_num = days.index(publish_day)
        
        with col3:
            months = ["January", "February", "March", "April", "May", "June", 
                     "July", "August", "September", "October", "November", "December"]
            publish_month = st.selectbox("Publishing Month", months, index=5)  # Default to June
            publish_month_num = months.index(publish_month) + 1
        
        # Submit button
        submit = st.form_submit_button("Predict Trending Potential")
    
    # Make prediction when form is submitted
    if submit:
        # Prepare video data
        video_data = {
            "title": title,
            "description": description,
            "category_id": category_id,
            "duration_seconds": duration_seconds,
            "tags": tags,
            "publish_hour": publish_hour,
            "publish_day": publish_day_num,
            "publish_month": publish_month_num
        }
        
        # Make prediction
        with st.spinner("Making prediction..."):
            prediction = make_prediction(video_data)
        
        if prediction:
            # Display prediction results
            st.markdown("### Prediction Results")
            
            # Create metrics cards
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown('<div class="card metric-card">', unsafe_allow_html=True)
                trending_score = prediction.get("trending_score", 0)
                st.metric("Trending Score", f"{trending_score:.1f}/10")
                # Visual indicator
                st.progress(trending_score / 10)
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="card metric-card">', unsafe_allow_html=True)
                engagement_score = prediction.get("engagement_score", 0)
                st.metric("Engagement Score", f"{engagement_score:.1f}/10")
                # Visual indicator
                st.progress(engagement_score / 10)
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="card metric-card">', unsafe_allow_html=True)
                views_estimate = prediction.get("views_estimate", 0)
                st.metric("Estimated Views", format_number(views_estimate))
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Detailed ML predictions
            st.markdown("### Model Predictions")
            
            # Create tabs for different prediction types
            pred_tabs = st.tabs(["Classification Models", "Regression Models", "Recommendations"])
            
            # Classification tab
            with pred_tabs[0]:
                if "model_predictions" in prediction and "classification" in prediction["model_predictions"]:
                    cls_results = prediction["model_predictions"]["classification"]
                    
                    if cls_results:
                        # Create a DataFrame for display
                        cls_data = []
                        for target, result in cls_results.items():
                            cls_data.append({
                                "Target": target,
                                "Prediction": bool(result.get("prediction", False)),
                                "Probability": result.get("probability", None),
                                "Model Type": result.get("model_type", "Unknown")
                            })
                        
                        cls_df = pd.DataFrame(cls_data)
                        
                        # Display table
                        st.dataframe(
                            cls_df,
                            column_config={
                                "Target": "Prediction Target",
                                "Prediction": "Is Predicted",
                                "Probability": st.column_config.NumberColumn("Probability", format="%.2f"),
                                "Model Type": "Model Type"
                            },
                            hide_index=True,
                            use_container_width=True
                        )
                        
                        # Display probability chart
                        fig = px.bar(
                            cls_df,
                            x="Target",
                            y="Probability",
                            title="Classification Probabilities",
                            color="Probability",
                            color_continuous_scale="Reds"
                        )
                        fig.update_layout(xaxis_title="Prediction Target", yaxis_title="Probability")
                        fig.update_yaxes(range=[0, 1])
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No classification predictions available")
                else:
                    st.info("No classification predictions available")
            
            # Regression tab
            with pred_tabs[1]:
                if "model_predictions" in prediction and "regression" in prediction["model_predictions"]:
                    reg_results = prediction["model_predictions"]["regression"]
                    
                    if reg_results:
                        # Create a DataFrame for display
                        reg_data = []
                        for target, result in reg_results.items():
                            reg_data.append({
                                "Target": target,
                                "Prediction": result.get("prediction", 0),
                                "Model Type": result.get("model_type", "Unknown")
                            })
                        
                        reg_df = pd.DataFrame(reg_data)
                        
                        # Display table
                        st.dataframe(
                            reg_df,
                            column_config={
                                "Target": "Prediction Target",
                                "Prediction": st.column_config.NumberColumn("Predicted Value", format="%.2f"),
                                "Model Type": "Model Type"
                            },
                            hide_index=True,
                            use_container_width=True
                        )
                        
                        # Display prediction chart
                        fig = px.bar(
                            reg_df,
                            x="Target",
                            y="Prediction",
                            title="Regression Predictions",
                            color="Prediction",
                            color_continuous_scale="Blues"
                        )
                        fig.update_layout(xaxis_title="Prediction Target", yaxis_title="Predicted Value")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No regression predictions available")
                else:
                    st.info("No regression predictions available")
            
            # Recommendations tab
            with pred_tabs[2]:
                if "recommendations" in prediction and prediction["recommendations"]:
                    st.markdown("#### Optimization Recommendations")
                    for rec in prediction["recommendations"]:
                        st.markdown(f"- {rec}")
                    
                    # Add general advice based on trending score
                    st.markdown("#### General Advice")
                    
                    if trending_score >= 8:
                        st.success("This video has excellent trending potential! Focus on promotion during the first 24 hours after publishing to maximize its impact.")
                    elif trending_score >= 6:
                        st.info("This video has good trending potential. Consider implementing the recommendations above to further improve its chances.")
                    elif trending_score >= 4:
                        st.warning("This video has moderate trending potential. Follow the recommendations carefully to improve its performance.")
                    else:
                        st.error("This video has low trending potential. Consider revising your content strategy based on the recommendations.")
                else:
                    st.info("No recommendations available")
        else:
            st.error("Failed to make prediction. Please check your API connection and try again.")

# Complete the settings page and add main execution for streamlit_app.py

elif page == "⚙️ Settings":
    st.title("Settings")
    st.markdown("Configure application settings and connections")
    
    # API Configuration
    st.markdown("### API Configuration")
    
    api_url = st.text_input("API URL", value=st.session_state.get('api_url', 'http://localhost:8000'), key="settings_api_url")
    
    if st.button("Test Connection"):
        try:
            response = requests.get(f"{api_url}/")
            if response.status_code == 200:
                st.session_state.api_url = api_url
                st.success(f"Successfully connected to API at {api_url}")
            else:
                st.error(f"Failed to connect to API. Status code: {response.status_code}")
        except Exception as e:
            st.error(f"Error connecting to API: {e}")
    
    # Data settings
    st.markdown("### Data Settings")
    
    # Refresh data
    if st.button("Refresh Data"):
        st.cache_data.clear()
        st.session_state.data = load_data()
        st.success("Data cache cleared and reloaded")
    
    # Model settings
    st.markdown("### Model Settings")
    
    models_dir = st.text_input("Models Directory", value=st.session_state.get('models_dir', 'models'))
    
    if st.button("Scan Models"):
        # Update models directory
        st.session_state.models_dir = models_dir
        
        # Load model metadata
        model_meta = load_model_metadata(models_dir)
        
        # Count models
        cls_count = len(model_meta.get("classification", {}))
        reg_count = len(model_meta.get("regression", {}))
        
        st.success(f"Found {cls_count} classification models and {reg_count} regression models in {models_dir}")
    
    # Display settings
    st.markdown("### Display Settings")
    
    # Theme selection
    theme = st.selectbox(
        "Color Theme",
        ["Default", "Light", "Dark"],
        index=0
    )
    
    if st.button("Apply Theme"):
        if theme == "Light":
            # Light theme CSS
            st.markdown("""
                <style>
                .main {
                    background-color: #ffffff;
                    color: #333333;
                }
                .card {
                    background-color: #f8f9fa;
                }
                </style>
            """, unsafe_allow_html=True)
        elif theme == "Dark":
            # Dark theme CSS
            st.markdown("""
                <style>
                .main {
                    background-color: #1e1e1e;
                    color: #f0f0f0;
                }
                .card {
                    background-color: #2d2d2d;
                }
                </style>
            """, unsafe_allow_html=True)
        
        st.success(f"Applied {theme} theme")
    
    # About section
    st.markdown("### About")
    st.markdown("""
    **YouTube Trending Video Analysis Dashboard**
    
    Version: 1.0.0
    
    This application provides analysis and prediction for YouTube trending videos.
    It uses machine learning models to predict trending potential and engagement
    for videos based on their attributes.
    
    For more information, visit the project GitHub repository.
    """)

# Main execution
if __name__ == "__main__":
    # Initialize session state variables
    if 'api_url' not in st.session_state:
        st.session_state.api_url = 'http://localhost:8000'
    
    if 'models_dir' not in st.session_state:
        st.session_state.models_dir = 'models'
    
    # Load data if not already loaded
    if 'data' not in st.session_state:
        st.session_state.data = load_data()