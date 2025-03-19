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
- **Plotly & Matplotlib**: Data visualization libraries
- **Seaborn**: Statistical data visualization

### Data Science
- **Feature Engineering**: Extraction of relevant features from video metadata
- **Classification & Regression Models**: Prediction of trending potential and engagement
- **Time-Series Analysis**: Temporal patterns in video publishing and engagement
- **Advanced Pattern Recognition**: Identifying optimal content strategies across regions and categories

## Key Insights & Findings

Our analysis revealed several important factors that influence YouTube trending videos:

### Content Optimization
- **Title Length**: Titles between 40-60 characters perform best
- **Duration**: Optimal video length varies by category:
  - Music/Entertainment: 3-7 minutes
  - Education/How-To: 7-15 minutes
  - Gaming/Tech: 15-25 minutes
- **Tags**: 5-15 relevant tags improve discoverability

### Publishing Strategy
- **Best Days**: Wednesday, Thursday, and Sunday generally perform best
- **Best Hours**: Publishing between 2-4 PM or 6-8 PM delivers highest engagement
- **Category Timing**: Optimal timing varies by content category and region

### Engagement Patterns
- **Velocity**: Videos that gain high engagement in first 6 hours have 80% higher chance of trending
- **Comments-to-Views Ratio**: Higher comment engagement correlates strongly with trending status
- **Category Engagement**: Different categories show distinct viewer engagement patterns

## Project Structure

```
youtube-trend-analyzer/
├── data/                     # Data storage
│   ├── raw/                  # Raw collected data
│   ├── processed/            # Processed data
│   ├── time_series/          # Time-series tracking data
│   └── ml/                   # Machine learning data
├── models/                   # Trained ML models
│   ├── classification/       # Classification models
│   └── regression/           # Regression models
├── notebooks/                # Jupyter notebooks for analysis
│   └── analyze_models.ipynb  # Model analysis notebook
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
│       ├── main.py
│       └── model_service.py
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
   python initial_collection.py --regions US,GB,CA,IN,JP,BR --analyze
   ```

2. Run time-series data collection (tracks videos over time)
   ```bash
   python time_series_collection.py --hours 3 --iterations 4 --regions US,GB
   ```

3. Run the ML pipeline
   ```bash
   python run_ml_pipeline.py --step all
   ```

4. Start the API
   ```bash
   python run_api.py
   ```

5. Launch the dashboard
   ```bash
   streamlit run streamlit_app.py
   ```

## Model Performance

### Classification Models
- **Viral Prediction**: 84% accuracy, 0.89 AUC
- **Long-Trending Prediction**: 78% accuracy, 0.83 AUC

### Regression Models
- **Engagement Score Prediction**: R² of 0.76, RMSE of 1.2
- **Views-per-Hour Prediction**: R² of 0.68, RMSE of 2450

## API Documentation

The API provides several endpoints:

- **`GET /trending`**: Get trending videos with filters
- **`GET /trending/categories`**: Get performance stats by category
- **`GET /trending/time-analysis`**: Get time-based trending patterns
- **`GET /trending/popular-channels`**: Get top performing channels
- **`POST /predict`**: Predict trending potential for a new video
- **`GET /api/models`**: Get available ML models
- **`POST /api/predict/all`**: Make predictions with all available models

## Future Enhancements

- **Thumbnail Analysis**: Incorporate computer vision to analyze thumbnail effectiveness
- **Topic Modeling**: Identify trending topics across categories
- **Sentiment Analysis**: Analyze sentiment in video titles and descriptions
- **Creator Analytics**: Provide personalized recommendations for content creators
- **Trend Forecasting**: Predict upcoming content trends

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- YouTube Data API for providing access to trending video data
- Open-source libraries that made this project possible

---

*Created by Samuel Danziger* 

*This project was developed as a demonstration of full-stack data science capabilities, combining data engineering, machine learning, and web development.*