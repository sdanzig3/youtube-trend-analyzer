# time_series_collection.py
import os
import time
import logging
import argparse
from datetime import datetime, timedelta
import pandas as pd

# Add the project root to the path for imports
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.youtube_fetcher import YouTubeDataFetcher
from src.data.data_processor import YouTubeDataProcessor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("time_series_collection.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def collect_data(hours=6, iterations=8, regions=None):
    """
    Collect trending data at regular intervals.
    
    Args:
        hours: Hours between collections
        iterations: Number of collections to perform
        regions: List of region codes to collect data from
    """
    if regions is None:
        regions = ['US']
    
    fetcher = YouTubeDataFetcher()
    processor = YouTubeDataProcessor()
    
    # Create directory for time series data
    time_series_dir = os.path.join('data', 'time_series')
    os.makedirs(time_series_dir, exist_ok=True)
    
    # Start timestamp for the series
    start_timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    # Track videos across collections
    all_videos = {}
    
    for i in range(iterations):
        collection_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        logger.info(f"Starting collection {i+1}/{iterations} at {collection_time}")
        
        # Collect data for each region
        for region in regions:
            try:
                # Fetch trending videos
                logger.info(f"Fetching trending videos for region: {region}")
                region_df = fetcher.fetch_trending_videos(region_code=region)
                
                if not region_df.empty:
                    # Add region and collection info
                    region_df['region'] = region
                    region_df['collection_time'] = collection_time
                    region_df['collection_number'] = i+1
                    
                    # Process data
                    processed_df = processor.process_trending_data(region_df)
                    
                    # Save to region-specific file for this collection
                    file_name = f"trend_{region}_{collection_time}.csv"
                    file_path = os.path.join(time_series_dir, file_name)
                    processed_df.to_csv(file_path, index=False)
                    logger.info(f"Saved {len(processed_df)} videos for {region} to {file_path}")
                    
                    # Track videos for time series analysis
                    for _, row in processed_df.iterrows():
                        video_id = row['video_id']
                        if video_id not in all_videos:
                            all_videos[video_id] = []
                        all_videos[video_id].append(row.to_dict())
                
            except Exception as e:
                logger.error(f"Error processing region {region}: {e}")
        
        # If this isn't the last iteration, wait for the next collection
        if i < iterations - 1:
            wait_hours = hours
            logger.info(f"Waiting {wait_hours} hours until next collection...")
            time.sleep(wait_hours * 3600)  # Convert hours to seconds
    
    # After all collections, create time series data for each video
    logger.info(f"Creating time series data for {len(all_videos)} videos")
    
    # Save time series data
    time_series_data = []
    for video_id, snapshots in all_videos.items():
        if len(snapshots) > 1:  # Only include videos with multiple snapshots
            for snapshot in snapshots:
                time_series_data.append(snapshot)
    
    if time_series_data:
        # Save all time series data
        all_series_df = pd.DataFrame(time_series_data)
        series_file = f"time_series_all_{start_timestamp}.csv"
        series_path = os.path.join(time_series_dir, series_file)
        all_series_df.to_csv(series_path, index=False)
        logger.info(f"Saved time series data with {len(all_series_df)} entries to {series_path}")
        
        # Analyze video progression
        progression_data = processor.track_video_progress([pd.DataFrame(snapshots) for snapshots in all_videos.values() if len(snapshots) > 1])
        if not progression_data.empty:
            prog_file = f"video_progression_{start_timestamp}.csv"
            prog_path = os.path.join(time_series_dir, prog_file)
            progression_data.to_csv(prog_path, index=False)
            logger.info(f"Saved video progression data for {len(progression_data)} videos to {prog_path}")
    
    logger.info("Time series collection completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YouTube Trending Time Series Collection")
    parser.add_argument("--hours", type=int, default=6, help="Hours between collections")
    parser.add_argument("--iterations", type=int, default=8, help="Number of collections to perform")
    parser.add_argument("--regions", type=str, default="US", help="Comma-separated list of region codes")
    
    args = parser.parse_args()
    regions = [r.strip() for r in args.regions.split(",")]
    
    collect_data(args.hours, args.iterations, regions)