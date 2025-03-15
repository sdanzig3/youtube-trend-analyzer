# YouTube Trending Video Analysis

A full-stack application to analyze YouTube trending videos and predict future trends. This project demonstrates data analysis, machine learning, and full-stack web development skills.

## Project Overview

This application:

1. **Collects data** from YouTube's Trending Videos API
2. **Analyzes patterns** in trending videos
3. **Builds prediction models** to identify potential trending videos
4. **Visualizes insights** through a web interface

## Tech Stack

### Backend
- **Python** - Core language for data processing and ML
- **FastAPI** - API framework
- **Pandas & NumPy** - Data manipulation
- **Matplotlib/Seaborn** - Data visualization
- **Scikit-learn** - Machine learning
- **SQLAlchemy** - ORM for database interactions
- **PostgreSQL** - Database (optional)

### Frontend (Planned)
- **React** - UI framework
- **Chart.js/Recharts** - Interactive visualizations
- **Tailwind CSS** - Styling

## Setup

### Prerequisites

- Python 3.9+
- YouTube Data API key
- Git

### Installation

1. Clone the repository
   ```bash
   git clone https://github.com/yourusername/youtube-trend-analyzer.git
   cd youtube-trend-analyzer
   ```

2. Create and activate a virtual environment
   ```bash
   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   
   # On Windows
   python -m venv venv
   venv\Scripts\activate
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file with your YouTube API key
   ```
   YOUTUBE_API_KEY=your_youtube_api_key_here
   ```

5. Run the setup test
   ```bash
   python test_setup.py
   ```

## Usage

### Collecting Data

Run the initial data collection script:

```bash
python initial_collection.py --regions US --analyze
```

Options:
- `--regions`: Comma-separated list of region codes (e.g., US,GB,CA)
- `--categories`: Collect data by category
- `--analyze`: Perform initial analysis
- `--db`: Store data in database (requires database setup)

### Scheduling Data Collection

For automated data collection at regular intervals:

```bash
python -m src.data.scheduler --regions US --interval 6
```

This will collect trending data every 6 hours for the US region.

## Project Structure

```
youtube-trend-analyzer/
├── data/                  # Data storage
│   ├── raw/               # Raw collected data
│   └── processed/         # Processed data
├── analysis/              # Analysis outputs and visualizations
├── src/
│   ├── data/              # Data collection and processing
│   │   ├── youtube_fetcher.py
│   │   ├── data_processor.py
│   │   ├── scheduler.py
│   │   └── database.py
│   ├── models/            # ML models (future)
│   └── api/               # API endpoints (future)
├── frontend/              # React frontend (future)
├── tests/                 # Unit tests
├── initial_collection.py  # Initial data collection script
├── test_setup.py          # Setup verification script
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

## Features and Components

### Data Collection
- Automated collection of trending videos across multiple regions
- Category-specific trending analysis
- Historical tracking of video performance

### Data Analysis
- Feature extraction from video metadata
- Engagement pattern analysis
- Publishing time optimization
- Title and tag effectiveness analysis

### Prediction Models (Planned)
- Trending potential scoring
- Feature importance analysis
- Content strategy recommendations

### Web Interface (Planned)
- Data visualization dashboard
- Trending prediction tool
- Content optimization recommendations

## Development Roadmap

- [x] Project setup and environment configuration
- [x] Data collection pipeline
- [x] Basic data processing and analysis
- [ ] API endpoint development
- [ ] Initial prediction model
- [ ] Frontend dashboard
- [ ] Advanced analytics features
- [ ] Model optimization
- [ ] User authentication
- [ ] Channel-specific recommendations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- YouTube Data API for providing access to trending video data
- Various open-source libraries used in this project