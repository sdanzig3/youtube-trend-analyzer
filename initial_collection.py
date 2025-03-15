# initial_collection.py
import os
import sys
import time
import logging
import argparse
from datetime import datetime

# Add the src directory to the path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data.youtube_fetcher import YouTubeDataFetcher, CATEGORY_MAPPING
from data.data_processor import YouTubeDataProcessor
from data.database import YouTubeDatabase

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("initial_collection.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Run the initial data collection process."""
    parser = argparse.ArgumentParser(description="YouTube Trending Initial Data Collection")
    parser.add_argument("--regions", type=str, default="US", help="Comma-separated list of region codes")
    parser.add_argument("--categories", action="store_true", help="Collect data by category")
    parser.add_argument("--db", action="store_true", help="Store data in database")
    parser.add_argument("--analyze", action="store_true", help="Perform initial analysis")
    
    args = parser.parse_args()
    
    # Parse regions
    region_list = [r.strip() for r in args.regions.split(",")]
    
    logger.info(f"Starting initial data collection for regions: {', '.join(region_list)}")
    
    # Initialize components
    fetcher = YouTubeDataFetcher()
    processor = YouTubeDataProcessor()
    
    # Initialize database if needed
    db = None
    if args.db:
        try:
            db = YouTubeDatabase()
            db.create_tables()
            
            # Store category mapping
            db.store_video_categories(CATEGORY_MAPPING)
            logger.info("Database initialized and categories stored")
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
            args.db = False  # Disable database storage
    
    # Create timestamp for filenames
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    # Ensure data directories exist
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    # Collect data for each region
    all_data = []
    
    for region in region_list:
        try:
            logger.info(f"Fetching trending videos for region: {region}")
            
            # Fetch trending videos
            region_df = fetcher.fetch_trending_videos(region_code=region)
            
            if not region_df.empty:
                # Add region column
                region_df['region'] = region
                
                # Save raw data
                region_filename = f"trending_{region}_{timestamp}.csv"
                region_path = os.path.join('data/raw', region_filename)
                region_df.to_csv(region_path, index=False)
                logger.info(f"Saved {len(region_df)} videos for {region} to {region_path}")
                
                # Process data
                processed_df = processor.process_trending_data(region_df)
                
                # Add to all data
                all_data.append(processed_df)
                
                # Store in database if enabled
                if args.db and db:
                    try:
                        # Extract channel data
                        channel_df = processor.extract_channel_performance(processed_df)
                        
                        # Store in database
                        db.store_channels(channel_df)
                        db.store_trending_videos(processed_df)
                        logger.info(f"Stored data for {region} in database")
                    except Exception as e:
                        logger.error(f"Database storage error for {region}: {e}")
            
            # Fetch category-specific data if requested
            if args.categories:
                logger.info(f"Fetching category-specific trending videos for {region}")
                
                category_data = fetcher.fetch_trending_by_categories(region_code=region)
                
                for category, cat_df in category_data.items():
                    if not cat_df.empty:
                        # Add region column
                        cat_df['region'] = region
                        
                        # Save category-specific data
                        cat_safe_name = category.replace(' & ', '_').replace(' ', '_')
                        cat_filename = f"trending_{region}_{cat_safe_name}_{timestamp}.csv"
                        cat_path = os.path.join('data/raw', cat_filename)
                        cat_df.to_csv(cat_path, index=False)
                        logger.info(f"Saved {len(cat_df)} videos for {region} - {category}")
                        
                        # Process and store category data
                        if args.db and db:
                            try:
                                processed_cat_df = processor.process_trending_data(cat_df)
                                db.store_trending_videos(processed_cat_df)
                            except Exception as e:
                                logger.error(f"Database storage error for {region} - {category}: {e}")
            
            # Avoid API rate limiting
            time.sleep(2)
            
        except Exception as e:
            logger.error(f"Error processing region {region}: {e}")
    
    # Combine and save all processed data
    if all_data:
        try:
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # Save combined processed data
            combined_filename = f"all_trending_{timestamp}.csv"
            combined_path = os.path.join('data/processed', combined_filename)
            combined_df.to_csv(combined_path, index=False)
            logger.info(f"Saved combined data with {len(combined_df)} entries to {combined_path}")
            
            # Perform initial analysis if requested
            if args.analyze:
                perform_initial_analysis(combined_df, timestamp)
                
        except Exception as e:
            logger.error(f"Error saving combined data: {e}")
    
    logger.info("Initial data collection completed")


def perform_initial_analysis(df, timestamp):
    """Perform initial analysis on collected data.
    
    Args:
        df: DataFrame with processed trending data
        timestamp: Timestamp string for filenames
    """
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        logger.info("Performing initial data analysis")
        
        # Ensure analysis directory exists
        os.makedirs('analysis', exist_ok=True)
        
        # Set plot style
        plt.style.use('seaborn-whitegrid')
        sns.set(font_scale=1.2)
        
        # 1. Category Distribution
        plt.figure(figsize=(12, 8))
        category_counts = df['category_name'].value_counts().sort_values(ascending=False)
        sns.barplot(x=category_counts.values, y=category_counts.index)
        plt.title('Distribution of Trending Videos by Category')
        plt.xlabel('Number of Videos')
        plt.tight_layout()
        plt.savefig(f'analysis/category_distribution_{timestamp}.png')
        
        # 2. Publish Time Analysis
        plt.figure(figsize=(10, 6))
        sns.histplot(df['publish_hour'], bins=24, kde=True)
        plt.title('Distribution of Trending Videos by Publish Hour')
        plt.xlabel('Hour of Day (UTC)')
        plt.ylabel('Number of Videos')
        plt.xticks(range(0, 24, 2))
        plt.tight_layout()
        plt.savefig(f'analysis/publish_hour_distribution_{timestamp}.png')
        
        # 3. Engagement Metrics by Category
        plt.figure(figsize=(14, 8))
        engagement_by_category = df.groupby('category_name')['engagement_score'].mean().sort_values(ascending=False)
        sns.barplot(x=engagement_by_category.values, y=engagement_by_category.index)
        plt.title('Average Engagement Score by Category')
        plt.xlabel('Engagement Score')
        plt.tight_layout()
        plt.savefig(f'analysis/engagement_by_category_{timestamp}.png')
        
        # 4. Video Duration Analysis
        plt.figure(figsize=(10, 6))
        # Filter out extreme outliers for better visualization
        duration_data = df[df['duration_seconds'] < 3600]  # Less than 1 hour
        sns.histplot(duration_data['duration_seconds'] / 60, bins=30, kde=True)  # Convert to minutes
        plt.title('Distribution of Trending Video Duration')
        plt.xlabel('Duration (minutes)')
        plt.ylabel('Number of Videos')
        plt.tight_layout()
        plt.savefig(f'analysis/duration_distribution_{timestamp}.png')
        
        # 5. Correlation Analysis
        plt.figure(figsize=(12, 10))
        correlation_columns = [
            'views', 'likes', 'comments', 'duration_seconds', 
            'tag_count', 'title_length', 'engagement_score',
            'like_view_ratio', 'comment_view_ratio'
        ]
        correlation_df = df[correlation_columns].corr()
        sns.heatmap(correlation_df, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Between Trending Video Features')
        plt.tight_layout()
        plt.savefig(f'analysis/correlation_matrix_{timestamp}.png')
        
        # 6. Top Channels
        plt.figure(figsize=(14, 8))
        channel_counts = df['channel_title'].value_counts().head(20)
        sns.barplot(x=channel_counts.values, y=channel_counts.index)
        plt.title('Top 20 Channels with Most Trending Videos')
        plt.xlabel('Number of Trending Videos')
        plt.tight_layout()
        plt.savefig(f'analysis/top_channels_{timestamp}.png')
        
        # 7. Summary Statistics Report
        summary_stats = df.describe(include='all')
        summary_stats.to_csv(f'analysis/summary_statistics_{timestamp}.csv')
        
        # 8. Title Feature Analysis
        title_features = ['title_has_number', 'title_has_question', 'title_has_exclamation', 
                          'title_has_special', 'title_has_brackets']
        
        plt.figure(figsize=(12, 8))
        title_feature_means = df[title_features].mean().sort_values(ascending=False)
        sns.barplot(x=title_feature_means.values * 100, y=title_feature_means.index)
        plt.title('Percentage of Trending Videos with Title Features')
        plt.xlabel('Percentage (%)')
        plt.tight_layout()
        plt.savefig(f'analysis/title_features_{timestamp}.png')
        
        logger.info("Initial analysis completed, visualizations saved to 'analysis' directory")
        
    except Exception as e:
        logger.error(f"Error performing initial analysis: {e}")


if __name__ == "__main__":
    import pandas as pd
    main()