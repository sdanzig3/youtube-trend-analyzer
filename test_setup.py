# test_setup.py
import os
from dotenv import load_dotenv
import sys

# Load environment variables
load_dotenv()

def test_imports():
    """Test importing all the modules."""
    try:
        from src.data.youtube_fetcher import YouTubeDataFetcher
        from src.data.data_processor import YouTubeDataProcessor
        from src.data.scheduler import DataCollectionScheduler
        from src.data.database import YouTubeDatabase
        
        print("✅ All modules imported successfully.")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_api_connection():
    """Test connection to YouTube API."""
    from src.data.youtube_fetcher import YouTubeDataFetcher
    
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        print("❌ No API key found. Please check your .env file.")
        return False
    
    try:
        fetcher = YouTubeDataFetcher(api_key)
        print("✅ Successfully initialized YouTube Data Fetcher.")
        
        # Try fetching a minimal amount of data
        print("Testing API by fetching 5 trending videos...")
        df = fetcher.fetch_trending_videos(max_results=5)
        
        if not df.empty:
            print(f"✅ Successfully fetched {len(df)} videos.")
            print("\nSample data:")
            print(f"  - First video: {df.iloc[0]['title']}")
            print(f"  - Channel: {df.iloc[0]['channel_title']}")
            print(f"  - Views: {df.iloc[0]['views']}")
            return True
        else:
            print("❌ No videos fetched. API might have an issue.")
            return False
    except Exception as e:
        print(f"❌ YouTube API error: {e}")
        return False

def test_data_processor():
    """Test data processor with sample data."""
    try:
        import pandas as pd
        from src.data.data_processor import YouTubeDataProcessor
        
        # Create a simple sample data frame
        sample_data = {
            'video_id': ['vid1', 'vid2'],
            'title': ['Sample Video 1', 'Sample Video 2 with a question?'],
            'channel_id': ['chan1', 'chan2'],
            'channel_title': ['Channel 1', 'Channel 2'],
            'publish_time': ['2023-01-01T12:00:00Z', '2023-01-02T15:30:00Z'],
            'fetch_time': ['2023-01-03T12:00:00Z', '2023-01-03T12:00:00Z'],
            'category_id': ['1', '10'],
            'views': [1000, 2000],
            'likes': [100, 250],
            'comments': [50, 120],
            'duration': ['PT5M30S', 'PT12M45S'], # ISO 8601 durations
        }
        
        df = pd.DataFrame(sample_data)
        
        # Process the data
        processor = YouTubeDataProcessor()
        processed_df = processor.process_trending_data(df)
        
        if not processed_df.empty:
            print("✅ Data processor working correctly.")
            return True
        else:
            print("❌ Data processor returned empty DataFrame.")
            return False
    except Exception as e:
        print(f"❌ Data processor error: {e}")
        return False

if __name__ == "__main__":
    print("============================================")
    print("YOUTUBE TREND ANALYZER SETUP TEST")
    print("============================================")
    
    # Test imports
    imports_success = test_imports()
    
    if imports_success:
        print("\nTesting YouTube API connection:")
        api_success = test_api_connection()
        
        print("\nTesting data processor:")
        processor_success = test_data_processor()
        
        if api_success and processor_success:
            print("\n✅✅✅ All tests passed! Your environment is correctly set up.")
            print("You can now start collecting data with:")
            print("  python initial_collection.py --regions US --analyze")
        else:
            print("\n⚠️ Some tests failed. Please fix the issues before proceeding.")
    else:
        print("\n⚠️ Import test failed. Please fix the module imports before proceeding.")