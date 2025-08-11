"""
Script to precompute video embeddings for faster API responses.

This script fetches video data from external APIs, processes it,
and generates semantic embeddings that are saved to disk for
quick loading by the main application.

Usage:
    python precompute_embeddings.py
"""

import asyncio
import os
import sys
from dotenv import load_dotenv

# Add the current directory to Python path to import app modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services import ExternalAPIService, DataProcessingService
from app.utils import EmbeddingGenerator, EmbeddingStorage
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def main():
    """
    Main function to fetch data and precompute embeddings.
    """
    # Load environment variables
    load_dotenv()
    
    logger.info("Starting embedding precomputation process...")
    
    try:
        # Initialize services
        api_service = ExternalAPIService()
        embedding_generator = EmbeddingGenerator()
        
        # Create data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)
        
        # Fetch posts from external API
        logger.info("Fetching posts from external API...")
        posts_data = await api_service.fetch_posts(page=1, page_size=500)  # Fetch more data
        
        if not posts_data:
            logger.error("No posts fetched from external API. Check your FLIC_TOKEN and internet connection.")
            return
        
        logger.info(f"Fetched {len(posts_data)} posts from external API")
        
        # Transform posts to Video objects
        logger.info("Processing video data...")
        videos = []
        failed_count = 0
        
        for i, post in enumerate(posts_data):
            try:
                video = DataProcessingService.transform_post_to_video(post)
                videos.append(video)
                
                # Log progress every 50 videos
                if (i + 1) % 50 == 0:
                    logger.info(f"Processed {i + 1}/{len(posts_data)} posts...")
                    
            except Exception as e:
                failed_count += 1
                logger.warning(f"Error processing post {post.get('id', 'unknown')}: {str(e)}")
        
        logger.info(f"Successfully processed {len(videos)} videos, {failed_count} failed")
        
        if not videos:
            logger.error("No valid videos processed. Cannot generate embeddings.")
            return
        
        # Save video data
        logger.info("Saving video data...")
        DataProcessingService.save_video_data(videos, "data/video_data.json")
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        embeddings = {}
        failed_embeddings = 0
        
        for i, video in enumerate(videos):
            try:
                video_dict = video.dict()
                embedding = embedding_generator.generate_video_embedding(video_dict)
                embeddings[video.id] = embedding
                
                # Log progress every 25 embeddings
                if (i + 1) % 25 == 0:
                    logger.info(f"Generated {i + 1}/{len(videos)} embeddings...")
                    
            except Exception as e:
                failed_embeddings += 1
                logger.warning(f"Error generating embedding for video {video.id}: {str(e)}")
        
        logger.info(f"Successfully generated {len(embeddings)} embeddings, {failed_embeddings} failed")
        
        # Save embeddings
        if embeddings:
            EmbeddingStorage.save_embeddings(embeddings, "data/embeddings.pkl")
            logger.info("Embeddings saved successfully")
        else:
            logger.error("No embeddings generated. Cannot save.")
            return
        
        logger.info("Embedding precomputation completed successfully!")
        logger.info(f"Total videos processed: {len(videos)}")
        logger.info(f"Total embeddings generated: {len(embeddings)}")
        
    except Exception as e:
        logger.error(f"Error during embedding precomputation: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())

