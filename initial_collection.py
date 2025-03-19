# initial_collection.py
import os
import sys
import time
import logging
import argparse
from datetime import datetime
import json

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
    parser.add_argument("--enhanced", action="store_true", help="Perform enhanced analysis")
    
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
                # Process category data
                processed_cat_df = processor.process_trending_data(cat_df)
                # Add to all data
                all_data.append(processed_cat_df)

                
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
            
            # Perform enhanced analysis if requested
            if args.enhanced:
                perform_enhanced_analysis(combined_df, processor, timestamp)
                
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


def perform_enhanced_analysis(df, processor, timestamp):
    """Perform enhanced analysis on collected data.
    
    Args:
        df: DataFrame with processed trending data
        processor: YouTubeDataProcessor instance
        timestamp: Timestamp string for filenames
    """
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        
        logger.info("Performing enhanced data analysis")
        
        # Ensure enhanced analysis directory exists
        enhanced_dir = os.path.join('analysis', 'enhanced')
        os.makedirs(enhanced_dir, exist_ok=True)
        
        # Set plot style
        plt.style.use('seaborn-whitegrid')
        sns.set(font_scale=1.2)
        
        # 1. Extract channel performance metrics
        channel_stats = processor.extract_channel_performance(df)
        channel_stats.to_csv(os.path.join(enhanced_dir, f'channel_performance_{timestamp}.csv'), index=False)
        
        # Plot top channels by score
        plt.figure(figsize=(14, 8))
        top_channels = channel_stats.sort_values('channel_score', ascending=False).head(15)
        sns.barplot(x='channel_score', y='channel_title', data=top_channels)
        plt.title('Top 15 Channels by Overall Performance Score')
        plt.xlabel('Channel Score (0-100)')
        plt.tight_layout()
        plt.savefig(os.path.join(enhanced_dir, f'top_channel_scores_{timestamp}.png'))
        
        # 2. Extract category performance metrics
        category_stats = processor.analyze_category_performance(df)
        category_stats.to_csv(os.path.join(enhanced_dir, f'category_performance_{timestamp}.csv'), index=False)
        
        # Plot category engagement vs. views
        plt.figure(figsize=(12, 8))
        sns.scatterplot(x='avg_views', y='avg_engagement', size='video_count', 
                        sizes=(100, 1000), alpha=0.7, data=category_stats)
        
        # Add category labels to points
        for i, row in category_stats.iterrows():
            plt.text(row['avg_views']*1.05, row['avg_engagement']*1.02, row['category_name'], 
                    fontsize=10, verticalalignment='center')
        
        plt.title('Category Performance: Engagement vs. Views')
        plt.xlabel('Average Views per Video')
        plt.ylabel('Average Engagement Score')
        plt.tight_layout()
        plt.savefig(os.path.join(enhanced_dir, f'category_engagement_vs_views_{timestamp}.png'))
        
        # Plot optimal duration by category if available
        if 'optimal_duration_minutes' in category_stats.columns:
            plt.figure(figsize=(12, 8))
            duration_data = category_stats.sort_values('optimal_duration_minutes')
            sns.barplot(x='optimal_duration_minutes', y='category_name', data=duration_data)
            plt.title('Optimal Video Duration by Category')
            plt.xlabel('Duration (minutes)')
            plt.tight_layout()
            plt.savefig(os.path.join(enhanced_dir, f'optimal_duration_by_category_{timestamp}.png'))
        
        # 3. Analyze time patterns
        time_patterns = processor.analyze_time_patterns(df)
        
        # Save time pattern data
        for key, pattern_df in time_patterns.items():
            if isinstance(pattern_df, pd.DataFrame):
                pattern_df.to_csv(os.path.join(enhanced_dir, f'{key}_{timestamp}.csv'), index=False)
        
        # Save time recommendations
        if 'time_recommendations' in time_patterns:
            with open(os.path.join(enhanced_dir, f'time_recommendations_{timestamp}.json'), 'w') as f:
                json.dump(time_patterns['time_recommendations'], f, indent=4)
        
        # Plot hourly engagement pattern
        if 'hourly_stats' in time_patterns:
            plt.figure(figsize=(12, 6))
            hourly_data = time_patterns['hourly_stats']
            
            fig, ax1 = plt.subplots(figsize=(12, 6))
            
            # Plot engagement score
            color = 'tab:blue'
            ax1.set_xlabel('Hour of Day')
            ax1.set_ylabel('Engagement Score', color=color)
            ax1.plot(hourly_data['publish_hour'], hourly_data['avg_engagement'], 
                    marker='o', linestyle='-', color=color)
            ax1.tick_params(axis='y', labelcolor=color)
            
            # Create second y-axis for video count
            ax2 = ax1.twinx()
            color = 'tab:red'
            ax2.set_ylabel('Number of Videos', color=color)
            ax2.bar(hourly_data['publish_hour'], hourly_data['video_count'], 
                   alpha=0.3, color=color)
            ax2.tick_params(axis='y', labelcolor=color)
            
            plt.title('Engagement Score and Video Count by Hour of Day')
            plt.xticks(range(0, 24))
            fig.tight_layout()
            plt.savefig(os.path.join(enhanced_dir, f'hourly_engagement_pattern_{timestamp}.png'))
            
        # Plot day of week engagement pattern
        if 'day_stats' in time_patterns:
            plt.figure(figsize=(10, 6))
            day_data = time_patterns['day_stats'].sort_values('publish_weekday')
            
            fig, ax1 = plt.subplots(figsize=(10, 6))
            
            # Plot engagement score
            color = 'tab:green'
            ax1.set_xlabel('Day of Week')
            ax1.set_ylabel('Engagement Score', color=color)
            ax1.plot(day_data['day_name'], day_data['avg_engagement'], 
                    marker='o', linestyle='-', color=color, linewidth=2)
            ax1.tick_params(axis='y', labelcolor=color)
            
            # Create second y-axis for video count
            ax2 = ax1.twinx()
            color = 'tab:orange'
            ax2.set_ylabel('Number of Videos', color=color)
            ax2.bar(day_data['day_name'], day_data['video_count'], 
                   alpha=0.3, color=color)
            ax2.tick_params(axis='y', labelcolor=color)
            
            plt.title('Engagement Score and Video Count by Day of Week')
            plt.xticks(rotation=45)
            fig.tight_layout()
            plt.savefig(os.path.join(enhanced_dir, f'day_engagement_pattern_{timestamp}.png'))
        
        # 4. Advanced title analysis
        if 'clickbait_score' in df.columns:
            # Clickbait score distribution
            plt.figure(figsize=(10, 6))
            sns.histplot(df['clickbait_score'], bins=10, kde=True)
            plt.title('Distribution of Clickbait Score')
            plt.xlabel('Clickbait Score')
            plt.ylabel('Number of Videos')
            plt.tight_layout()
            plt.savefig(os.path.join(enhanced_dir, f'clickbait_score_distribution_{timestamp}.png'))
            
            # Clickbait score vs engagement
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='clickbait_score', y='engagement_score', data=df)
            plt.title('Impact of Clickbait on Engagement')
            plt.xlabel('Clickbait Score')
            plt.ylabel('Engagement Score')
            plt.tight_layout()
            plt.savefig(os.path.join(enhanced_dir, f'clickbait_vs_engagement_{timestamp}.png'))
        
        # 5. Popularity score analysis
        if 'popularity_score' in df.columns:
            # Popular video features
            top_quartile = df[df['popularity_score'] >= df['popularity_score'].quantile(0.75)]
            bottom_quartile = df[df['popularity_score'] <= df['popularity_score'].quantile(0.25)]
            
            # Compare metrics between top and bottom videos
            metrics = ['duration_seconds', 'title_word_count', 'tag_count', 
                      'like_view_ratio', 'comment_view_ratio']
            
            comparison = pd.DataFrame({
                'metric': metrics,
                'top_videos': [top_quartile[m].mean() for m in metrics],
                'bottom_videos': [bottom_quartile[m].mean() for m in metrics]
            })
            
            # Calculate percentage difference
            comparison['difference'] = ((comparison['top_videos'] - comparison['bottom_videos']) / 
                                      comparison['bottom_videos'] * 100).round(1)
            
            comparison.to_csv(os.path.join(enhanced_dir, f'popularity_comparison_{timestamp}.csv'), index=False)
            
            # Plot the comparison
            plt.figure(figsize=(12, 8))
            metrics_display = ['Duration (sec)', 'Title Words', 'Tag Count', 
                              'Like/View Ratio', 'Comment/View Ratio']
            
            x = np.arange(len(metrics_display))
            width = 0.35
            
            fig, ax = plt.subplots(figsize=(12, 8))
            rects1 = ax.bar(x - width/2, comparison['top_videos'], width, label='Top Videos')
            rects2 = ax.bar(x + width/2, comparison['bottom_videos'], width, label='Bottom Videos')
            
            ax.set_title('Comparison of Features: Top vs. Bottom Videos')
            ax.set_xticks(x)
            ax.set_xticklabels(metrics_display)
            ax.legend()
            
            # Add percentage difference labels
            for i, rect in enumerate(rects1):
                height = rect.get_height()
                diff = comparison['difference'][i]
                diff_text = f"+{diff}%" if diff > 0 else f"{diff}%"
                ax.annotate(diff_text,
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom',
                            color='green' if diff > 0 else 'red',
                            fontweight='bold')
            
            fig.tight_layout()
            plt.savefig(os.path.join(enhanced_dir, f'top_vs_bottom_comparison_{timestamp}.png'))
            
        # 6. Create an insights summary file
        with open(os.path.join(enhanced_dir, f'analysis_insights_{timestamp}.txt'), 'w') as f:
            f.write("# YouTube Trending Videos Analysis Insights\n\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Videos Analyzed: {len(df)}\n\n")
            
            # Add category insights
            f.write("## Category Insights\n\n")
            top_category = category_stats.sort_values('category_score', ascending=False).iloc[0]
            f.write(f"Top Performing Category: {top_category['category_name']} (Score: {top_category['category_score']:.1f})\n")
            
            if 'optimal_duration_minutes' in category_stats.columns:
                f.write(f"Optimal Duration: {top_category.get('optimal_duration_minutes', 'N/A'):.1f} minutes\n")
            
            if 'optimal_time_of_day' in category_stats.columns:
                f.write(f"Best Time to Post: {top_category.get('optimal_time_of_day', 'N/A')}\n")
            
            # Add channel insights
            f.write("\n## Channel Insights\n\n")
            top_channel = channel_stats.sort_values('channel_score', ascending=False).iloc[0]
            f.write(f"Top Performing Channel: {top_channel['channel_title']} (Score: {top_channel['channel_score']:.1f})\n")
            f.write(f"Videos in Trending: {top_channel['video_count']}\n")
            f.write(f"Average Views: {top_channel['views_per_video']:,.0f}\n")
            
            # Add time insights
            if 'time_recommendations' in time_patterns:
                rec = time_patterns['time_recommendations']
                f.write("\n## Posting Time Insights\n\n")
                f.write(f"Best Day to Post: {rec.get('best_day_name', 'N/A')}\n")
                f.write(f"Best Hour: {rec.get('best_hour', 'N/A')}:00\n")
                f.write(f"Best Time of Day: {rec.get('best_time_of_day', 'N/A')}\n")
            
            # Add title insights
            f.write("\n## Title Insights\n\n")
            
            # Calculate title feature effectiveness
            if 'title_has_question' in df.columns and 'engagement_score' in df.columns:
                question_impact = df[df['title_has_question'] == 1]['engagement_score'].mean() - df['engagement_score'].mean()
                exclamation_impact = df[df['title_has_exclamation'] == 1]['engagement_score'].mean() - df['engagement_score'].mean()
                number_impact = df[df['title_has_number'] == 1]['engagement_score'].mean() - df['engagement_score'].mean()
                
                f.write("Title Feature Impact on Engagement:\n")
                f.write(f"- Questions: {question_impact:.2f} points\n")
                f.write(f"- Exclamations: {exclamation_impact:.2f} points\n")
                f.write(f"- Numbers: {number_impact:.2f} points\n")
            
            if 'clickbait_score' in df.columns and 'engagement_score' in df.columns:
                clickbait_corr = df[['clickbait_score', 'engagement_score']].corr().iloc[0, 1]
                f.write(f"\nClickbait Correlation with Engagement: {clickbait_corr:.3f}\n")
            
            # Add general stats
            f.write("\n## General Statistics\n\n")
            f.write(f"Average Views: {df['views'].mean():,.0f}\n")
            f.write(f"Average Likes: {df['likes'].mean():,.0f}\n")
            f.write(f"Average Comments: {df['comments'].mean():,.0f}\n")
            
            f.write(f"\nAverage Like/View Ratio: {df['like_view_ratio'].mean():.2f}%\n")
            f.write(f"Average Comment/View Ratio: {df['comment_view_ratio'].mean():.2f}%\n")
            
            avg_duration = df['duration_seconds'].mean() / 60
            f.write(f"Average Duration: {avg_duration:.1f} minutes\n")
        
        logger.info(f"Enhanced analysis completed. Results saved to {enhanced_dir}")
        
    except Exception as e:
        logger.error(f"Error performing enhanced analysis: {e}")


if __name__ == "__main__":
    import pandas as pd
    main()