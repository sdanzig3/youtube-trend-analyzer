# YouTube Trending Video Analysis

A full-stack data science application that analyzes YouTube trending videos, identifies patterns, and predicts trending potential for new videos.

## Project Overview

This application:

1. **Collects data** from YouTube's Trending Videos API across multiple regions
2. **Analyzes patterns** in trending videos to identify key success factors
3. **Trains ML models** to predict trending potential and engagement
4. **Visualizes insights** through an interactive dashboard
5. **Provides recommendations** for optimizing video attributes

## Features

- **Data Collection Pipeline**: Automated collection of trending videos across multiple regions and categories
- **Advanced Analysis**: Processing of video metadata to extract meaningful insights
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

## Project Structure

```
youtube-trend-analyzer/
├── data/                     # Data storage
│   ├── raw/                  # Raw collected data
│   ├── processed/            # Processed data
│   └── ml/                   # Machine learning data
├── analysis/                 # Analysis outputs
│   └── enhanced/             # Enhanced analysis visuals
├── models/                   # Trained ML models
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
├── initial_collection.py     # Data collection script
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
   python initial_collection.py --regions US,GB,CA --analyze --enhanced
   ```

2. Run the ML pipeline
   ```bash
   python run_ml_pipeline.py --step all
   ```

3. Start the API (optional)
   ```bash
   python run_api.py
   ```

4. Launch the dashboard
   ```bash
   streamlit run streamlit_app.py
   ```

## Key Insights & Features

### Data Analysis

- Identification of optimal video durations by category
- Publishing time patterns that maximize engagement
- Title and tag characteristics of high-performing videos
- Regional differences in trending content

### Machine Learning

- Prediction of trending potential with feature importance analysis
- Engagement score estimation based on video attributes
- Recommendations for video optimization

### Dashboard Views

- **Overview**: Summary statistics and trending distributions
- **Trending Videos**: Current top trending videos with metrics
- **Category Analysis**: Deep dive into category-specific patterns
- **Prediction Tool**: Interactive prediction of trending potential for new videos

## Future Enhancements

- Thumbnail image analysis using computer vision
- Sentiment analysis of video titles and descriptions
- Topic modeling across trending categories
- Content strategy recommendations based on channel performance

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- YouTube Data API for providing access to trending video data
- Open-source libraries that made this project possible

---

*Created by Samuel Danziger* 

*This project was developed as a demonstration of full-stack data science capabilities, combining data engineering, machine learning, and web development.*