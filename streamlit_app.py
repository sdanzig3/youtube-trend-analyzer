# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(__file__))

# Import your modules
from src.models.prediction import TrendingPredictor

# Set page configuration
st.set_page_config(
    page_title="YouTube Trending Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add a title
st.title("YouTube Trending Video Analysis")

# Sidebar navigation
page = st.sidebar.selectbox(
    "Choose a page",
    ["Dashboard", "Trending Videos", "Category Analysis", "Prediction Tool", "Advanced Insights"]
)

# Load data
@st.cache_data
def load_data():
    # Find the most recent data file
    data_dir = os.path.join(os.path.dirname(__file__), "data/processed")
    if not os.path.exists(data_dir):
        return None
    
    files = [f for f in os.listdir(data_dir) if f.startswith("all_trending_")]
    if not files:
        return None
    
    latest_file = max(files)
    file_path = os.path.join(data_dir, latest_file)
    
    # Load the data
    return pd.read_csv(file_path)

data = load_data()

if data is None:
    st.error("No data available. Please run data collection first.")
    st.stop()

# Dashboard page
if page == "Dashboard":
    st.header("Dashboard")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Videos", f"{len(data):,}")
    with col2:
        st.metric("Average Views", f"{int(data['views'].mean()):,}")
    with col3:
        st.metric("Average Likes", f"{int(data['likes'].mean()):,}")
    with col4:
        st.metric("Average Comments", f"{int(data['comments'].mean()):,}")
    
    # Category distribution
    st.subheader("Videos by Category")
    category_counts = data['category_name'].value_counts()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=category_counts.values, y=category_counts.index, ax=ax)
    plt.title("Distribution of Trending Videos by Category")
    plt.xlabel("Number of Videos")
    st.pyplot(fig)
    
    # Engagement by category
    st.subheader("Engagement by Category")
    if 'engagement_score' in data.columns:
        engagement_by_category = data.groupby('category_name')['engagement_score'].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=engagement_by_category.values, y=engagement_by_category.index, ax=ax)
        plt.title("Average Engagement Score by Category")
        plt.xlabel("Engagement Score")
        st.pyplot(fig)

# Trending Videos page
elif page == "Trending Videos":
    st.header("Trending Videos")
    
    # Filters
    col1, col2 = st.columns(2)
    with col1:
        region_filter = st.selectbox("Region", data['region'].unique())
    with col2:
        category_filter = st.selectbox("Category", ["All"] + list(data['category_name'].unique()))
    
    # Filter data
    filtered_data = data[data['region'] == region_filter]
    if category_filter != "All":
        filtered_data = filtered_data[filtered_data['category_name'] == category_filter]
    
    # Sort by views
    filtered_data = filtered_data.sort_values("views", ascending=False)
    
    # Display videos
    st.subheader(f"Top Trending Videos ({len(filtered_data)} videos)")
    for _, row in filtered_data.head(10).iterrows():
        with st.container():
            col1, col2 = st.columns([1, 3])
            with col1:
                if 'thumbnail_url' in row and row['thumbnail_url']:
                    st.image(row['thumbnail_url'], width=120)
                else:
                    st.image("https://via.placeholder.com/120x68.png?text=No+Thumbnail", width=120)
            with col2:
                st.subheader(row['title'])
                st.write(f"Channel: {row['channel_title']}")
                st.write(f"Views: {row['views']:,} | Likes: {row['likes']:,} | Comments: {row['comments']:,}")
                if 'engagement_score' in row:
                    st.progress(min(row['engagement_score'] / 10, 1.0))
                    st.write(f"Engagement Score: {row['engagement_score']:.2f}/10")
            st.divider()

# Category Analysis page
elif page == "Category Analysis":
    st.header("Category Analysis")
    
    # Select category
    category = st.selectbox("Select Category", sorted(data['category_name'].unique()))
    
    # Filter data for selected category
    category_data = data[data['category_name'] == category]
    
    # Category metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Video Count", len(category_data))
    with col2:
        st.metric("Average Views", f"{int(category_data['views'].mean()):,}")
    with col3:
        if 'engagement_score' in category_data.columns:
            st.metric("Avg Engagement", f"{category_data['engagement_score'].mean():.2f}")
    
    # Video duration analysis
    st.subheader("Video Duration Analysis")
    if 'duration_seconds' in category_data.columns:
        # Convert to minutes and remove outliers
        duration_data = category_data[category_data['duration_seconds'] < 3600]['duration_seconds'] / 60
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(duration_data, bins=20, kde=True, ax=ax)
        plt.title(f"Distribution of {category} Video Duration")
        plt.xlabel("Duration (minutes)")
        plt.ylabel("Count")
        st.pyplot(fig)
    
    # Publish time analysis
    if 'publish_hour' in category_data.columns:
        st.subheader("Publish Time Analysis")
        hour_counts = category_data['publish_hour'].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=hour_counts.index, y=hour_counts.values, ax=ax)
        plt.title(f"Distribution of {category} Videos by Publish Hour")
        plt.xlabel("Hour of Day")
        plt.ylabel("Number of Videos")
        plt.xticks(range(0, 24, 2))
        st.pyplot(fig)

# Prediction Tool page
elif page == "Prediction Tool":
    st.header("Trending Potential Prediction")
    
    # Load predictor
    predictor = TrendingPredictor()
    loaded_models = predictor.load_available_models()
    
    if not loaded_models:
        st.error("No prediction models available. Please train models first.")
        st.stop()
    
    # Input form
    with st.form("prediction_form"):
        title = st.text_input("Video Title", "How to Build a Machine Learning Project for Your Resume")
        
        col1, col2 = st.columns(2)
        with col1:
            category_id = st.selectbox("Category", 
                                      [(id, name) for id, name in CATEGORY_MAPPING.items()],
                                      format_func=lambda x: x[1])
        with col2:
            duration = st.slider("Duration (minutes)", 1, 60, 15)
        
        description = st.text_area("Description", "Learn how to build an impressive machine learning project for your resume.")
        
        col1, col2 = st.columns(2)
        with col1:
            publish_hour = st.slider("Publish Hour", 0, 23, 16)
        with col2:
            days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            publish_day = st.selectbox("Publish Day", range(7), format_func=lambda x: days[x])
        
        tags = st.text_input("Tags (comma separated)", "machine learning, resume, project, tutorial")
        
        submitted = st.form_submit_button("Predict Trending Potential")
    
    if submitted:
        # Prepare video data
        video_data = {
            'title': title,
            'description': description,
            'duration_seconds': duration * 60,
            'category_id': category_id[0],
            'tags': [tag.strip() for tag in tags.split(',')],
            'publish_hour': publish_hour,
            'publish_weekday': publish_day
        }
        
        # Make prediction
        prediction = predictor.predict_trending_potential(video_data)
        
        # Display results
        st.subheader("Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Trending Score", f"{prediction.get('trending_score', 0):.1f}/100")
        with col2:
            st.metric("Engagement Score", f"{prediction.get('engagement_score', 0):.2f}/10")
        with col3:
            st.metric("Views Estimate", f"{prediction.get('views_estimate', 0):,}")
        
        # Progress bar
        st.progress(min(prediction.get('trending_score', 0) / 100, 1.0))
        
        # Recommendations
        st.subheader("Recommendations")
        for rec in prediction.get('recommendations', []):
            st.write(f"• {rec}")
elif page == "Advanced Insights":
    st.header("Advanced Trending Pattern Analysis")
    
    # 1. Optimal publishing times
    st.subheader("Optimal Publishing Times by Category")
    
    # Create a placeholder for when we have the data
    category = st.selectbox("Select Category", sorted(data['category_name'].unique()))
    category_data = data[data['category_name'] == category]
    
    # Hour analysis for selected category
    if 'publish_hour' in category_data.columns and 'engagement_score' in category_data.columns:
        hour_engagement = category_data.groupby('publish_hour')['engagement_score'].mean().reset_index()
        best_hour = hour_engagement.loc[hour_engagement['engagement_score'].idxmax(), 'publish_hour']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(data=hour_engagement, x='publish_hour', y='engagement_score', marker='o', ax=ax)
        plt.axvline(x=best_hour, color='r', linestyle='--')
        plt.title(f'Engagement by Hour for {category} Videos')
        plt.xlabel('Hour of Day (UTC)')
        plt.ylabel('Average Engagement Score')
        plt.xticks(range(0, 24, 2))
        st.pyplot(fig)
        
        st.info(f"Best publishing hour for {category} videos: {best_hour}:00 UTC")
    
    # 2. Title pattern analysis
    st.subheader("Title Pattern Analysis")
    
    # Word frequency
    if 'title' in data.columns:
        title_words = []
        for title in data['title']:
            words = re.findall(r'\b[a-zA-Z]{3,}\b', title.lower())
            title_words.extend(words)
        
        word_freq = pd.Series(title_words).value_counts()
        top_words = word_freq.head(20)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.barplot(x=top_words.values, y=top_words.index, ax=ax)
        plt.title('Top 20 Words in Trending Video Titles')
        plt.xlabel('Frequency')
        st.pyplot(fig)
        
        # Title length analysis
        if 'title_length' not in data.columns:
            data['title_length'] = data['title'].str.len()
        
        data['title_length_group'] = pd.cut(data['title_length'], 
                                         bins=[0, 20, 40, 60, 80, 100, 120, 140, 160],
                                         labels=['0-20', '21-40', '41-60', '61-80', '81-100', '101-120', '121-140', '141-160'])
        
        if 'engagement_score' in data.columns:
            title_engagement = data.groupby('title_length_group')['engagement_score'].mean().reset_index()
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=title_engagement, x='title_length_group', y='engagement_score', ax=ax)
            plt.title('Engagement by Title Length')
            plt.xlabel('Title Length')
            plt.ylabel('Average Engagement Score')
            st.pyplot(fig)
            
            best_length = title_engagement.loc[title_engagement['engagement_score'].idxmax(), 'title_length_group']
            st.info(f"Optimal title length for maximum engagement: {best_length} characters")
    
    # 3. Regional differences
    st.subheader("Regional Trending Differences")
    
    if 'region' in data.columns:
        region_counts = data['region'].value_counts()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=region_counts.values, y=region_counts.index, ax=ax)
        plt.title('Video Count by Region')
        plt.xlabel('Number of Trending Videos')
        st.pyplot(fig)
        
        # Category distribution by region
        region_cat_counts = data.groupby(['region', 'category_name']).size().reset_index(name='count')
        
        regions = st.multiselect("Select Regions to Compare", sorted(data['region'].unique()), 
                               default=sorted(data['region'].unique())[:3])
        
        if regions:
            filtered_counts = region_cat_counts[region_cat_counts['region'].isin(regions)]
            
            pivot = filtered_counts.pivot(index='category_name', columns='region', values='count')
            
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(pivot, annot=True, cmap='viridis', ax=ax)
            plt.title('Category Distribution by Region')
            plt.tight_layout()
            st.pyplot(fig)
# Run the app
if __name__ == "__main__":
    # The app is run by streamlit automatically
    pass