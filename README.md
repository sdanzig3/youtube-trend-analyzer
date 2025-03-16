# YouTube Trending Video Analysis

A full-stack data science application that analyzes YouTube trending videos, identifies patterns, and predicts trending potential for new videos.

## Project Overview

This application:

1. **Collects data** from YouTube's Trending Videos API across multiple regions
2. **Tracks trending videos** over time to analyze progression patterns
3. **Analyzes patterns** in trending videos to identify key success factors
4. **Trains ML models** to predict trending potential and engagement
5. **Visualizes insights** through an interactive dashboard
6. **Provides recommendations** for optimizing video attributes

## Features

- **Data Collection Pipeline**: Automated collection of trending videos across multiple regions and categories
- **Time-Series Analysis**: Tracking videos over time to understand growth patterns
- **Advanced Analytics**: Processing of video metadata to extract meaningful insights
- **Machine Learning Models**: Prediction of trending potential and engagement scores
- **API Backend**: FastAPI endpoints for accessing data and predictions
- **Interactive Dashboard**: Streamlit-based visualization of trends and patterns
- **Recommendation Engine**: Suggestions for video optimization based on data analysis

## Tech Stack

### Backend
- **Python**: Core language for data processing and ML
- **FastAPI**: API framework for serving data and predictions
- **Pandas & NumPy**: Data manipulation and analysis
- **Scikit-learn**: Machine learning algorithms
- **YouTube Data API**: Data source for trending videos

### Frontend
- **Streamlit**: Interactive data visualization dashboard
- **Matplotlib & Seaborn**: Data visualization libraries

### Data Science
- **Feature Engineering**: Extraction of relevant features from video metadata
- **Classification & Regression Models**: Prediction of trending potential and engagement
- **Time-Series Analysis**: Temporal patterns in video publishing and engagement
- **Advanced Pattern Recognition**: Identifying optimal content strategies across regions and categories

## Project Structure

```
youtube-trend-analyzer/
├── data/                     # Data storage
│   ├── raw/                  # Raw collected data
│   ├── processed/            # Processed data
│   ├── time_series/          # Time-series tracking data
│   └── ml/                   # Machine learning data
├── analysis/                 # Analysis outputs
│   └── advanced/             # Advanced analysis visuals
├── models/                   # Trained ML models
├── notebooks/                # Jupyter notebooks for analysis
│   └── advanced_analysis.ipynb
├── src/                      # Source code
│   ├── data/                 # Data collection and processing
│   │   ├── youtube_fetcher.py
│   │   ├── data_processor.py
│   │   ├── scheduler.py
│   │   └── database.py
│   ├── models/               # ML models
│   │   ├── feature_engineering.py
│   │   ├── training.py
│   │   └── prediction.py
│   └── api/                  # API endpoints
│       └── main.py
├── streamlit_app.py          # Streamlit dashboard
├── initial_collection.py     # Initial data collection script
├── time_series_collection.py # Time-series data collection
├── run_ml_pipeline.py        # ML pipeline script
├── run_api.py                # API runner
└── requirements.txt          # Python dependencies
```

## Setup and Installation

### Prerequisites

- Python 3.9+
- YouTube Data API key

### Installation

1. Clone the repository
   ```bash
   git clone https://github.com/sdanzig3/youtube-trend-analyzer.git
   cd youtube-trend-analyzer
   ```

2. Create a virtual environment and install dependencies
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your YouTube API key
   ```
   YOUTUBE_API_KEY=your_youtube_api_key_here
   ```

### Running the Application

1. Collect trending video data
   ```bash
   python initial_collection.py --regions US,GB,CA,IN,JP,BR --analyze --enhanced
   ```

2. Run time-series data collection (tracks videos over time)
   ```bash
   python time_series_collection.py --hours 3 --iterations 4 --regions US,GB
   ```

3. Run the ML pipeline
   ```bash
   python run_ml_pipeline.py --step all
   ```

4. Start the API (optional)
   ```bash
   python run_api.py
   ```

5. Launch the dashboard
   ```bash
   streamlit run streamlit_app.py
   ```

## Key Insights & Features

### Data Analysis

- Identification of optimal video durations by category
- Publishing time patterns that maximize engagement
- Title and tag characteristics of high-performing videos
- Regional differences in trending content
- View growth rates and velocity patterns
- Performance differences across regions and categories

### Time-Series Analysis

- Tracking trending videos over time to measure:
  - View velocity (views gained per hour)
  - Engagement growth patterns
  - Duration of trending status
  - Category-specific growth rates

### Machine Learning

- Prediction of trending potential with feature importance analysis
- Engagement score estimation based on video attributes
- Recommendations for video optimization

### Dashboard Views

- **Overview**: Summary statistics and trending distributions
- **Trending Videos**: Current top trending videos with metrics
- **Category Analysis**: Deep dive into category-specific patterns
- **Advanced Insights**: Regional comparisons and time-based pattern analysis
- **Prediction Tool**: Interactive prediction of trending potential for new videos

## Advanced Analytics

The project includes in-depth analysis of:

- **Title Pattern Analysis**: Word frequency, optimal title length, and impact of different title styles
- **Optimal Publishing Times**: Best hours and days to publish by category and region
- **Regional Trending Differences**: How trending patterns vary across different countries
- **View Growth Patterns**: How quickly videos accumulate views and engagement
- **Category-Specific Insights**: Tailored recommendations for each content category

## Future Enhancements

- Thumbnail image analysis using computer vision
- Sentiment analysis of video titles and descriptions
- Topic modeling across trending categories
- Content strategy recommendations based on channel performance
- Predictive modeling for view growth trajectory

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- YouTube Data API for providing access to trending video data
- Open-source libraries that made this project possible

---

*Created by Samuel Danziger* 

*This project was developed as a demonstration of full-stack data science capabilities, combining data engineering, machine learning, and web development.*