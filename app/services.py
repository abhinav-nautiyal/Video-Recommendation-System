"""
Business logic services for the video recommendation engine.

This module contains services for:
- Fetching data from external APIs
- Processing video data
- Generating recommendations
"""

import httpx
import json
import os
from typing import List, Dict, Optional, Any
import logging
from .models import Video, RecommendationResponse
from .utils import EmbeddingGenerator, SimilarityCalculator, EmbeddingStorage

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExternalAPIService:
    """
    Service for interacting with external APIs to fetch video data.
    
    This service handles authentication and data fetching from the
    Empowerverse API endpoints.
    """
    
    def __init__(self):
        """Initialize the service with API configuration."""
        self.base_url = os.getenv("API_BASE_URL", "https://api.socialverseapp.com")
        self.flic_token = os.getenv("FLIC_TOKEN")
        
        if not self.flic_token:
            logger.warning("FLIC_TOKEN not found in environment variables")
        
        # Headers for API requests
        self.headers = {
            "Flic-Token": self.flic_token
        } if self.flic_token else {}
    
    async def fetch_posts(self, page: int = 1, page_size: int = 1000) -> List[Dict[str, Any]]:
        """
        Fetch posts from the external API.
        
        Args:
            page (int): Page number for pagination
            page_size (int): Number of posts per page
            
        Returns:
            List[Dict[str, Any]]: List of post data
        """
        url = f"{self.base_url}/posts/summary/get"
        params = {
            "page": page,
            "page_size": page_size
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, headers=self.headers, params=params)
                response.raise_for_status()
                data = response.json()
                
                # Extract posts from the response
                posts = data.get("posts", []) if isinstance(data, dict) else data
                logger.info(f"Fetched {len(posts)} posts from external API")
                return posts
                
        except httpx.HTTPError as e:
            logger.error(f"HTTP error while fetching posts: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error while fetching posts: {str(e)}")
            return []
    
    async def fetch_user_interactions(self, interaction_type: str, page: int = 1, page_size: int = 1000) -> List[Dict[str, Any]]:
        """
        Fetch user interaction data (views, likes, inspires, ratings).
        
        Args:
            interaction_type (str): Type of interaction ('view', 'like', 'inspire', 'rating')
            page (int): Page number for pagination
            page_size (int): Number of interactions per page
            
        Returns:
            List[Dict[str, Any]]: List of interaction data
        """
        resonance_algorithm = "resonance_algorithm_cjsvervb7dbhss8bdrj89s44jfjdbsjd0xnjkbvuire8zcjwerui3njfbvsujc5if"
        url = f"{self.base_url}/posts/{interaction_type}"
        params = {
            "page": page,
            "page_size": page_size,
            "resonance_algorithm": resonance_algorithm
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
                interactions = data.get("data", []) if isinstance(data, dict) else data
                logger.info(f"Fetched {len(interactions)} {interaction_type} interactions")
                return interactions
                
        except httpx.HTTPError as e:
            logger.error(f"HTTP error while fetching {interaction_type} interactions: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error while fetching {interaction_type} interactions: {str(e)}")
            return []


class DataProcessingService:
    """
    Service for processing and transforming raw data into Video objects.
    
    This service handles data cleaning, transformation, and validation.
    """
    
    @staticmethod
    def transform_post_to_video(post_data: Dict[str, Any]) -> Video:
        """
        Transform raw post data from external API into a Video object.
        
        Args:
            post_data (Dict[str, Any]): Raw post data from external API
            
        Returns:
            Video: Validated Video object
        """
        # Extract and clean data fields
        video_id = str(post_data.get("id", ""))
        title = post_data.get("title", "")
        description = post_data.get("description", "")
        
        # Handle tags - they might be a string or list
        tags_raw = post_data.get("tags", [])
        if isinstance(tags_raw, str):
            tags = [tag.strip() for tag in tags_raw.split(",") if tag.strip()]
        elif isinstance(tags_raw, list):
            tags = [str(tag).strip() for tag in tags_raw if str(tag).strip()]
        else:
            tags = []
        
        # Extract transcript or content
        transcript = post_data.get("transcript", post_data.get("content", ""))
        
        # Extract category and project information
        category_raw = post_data.get("category", post_data.get("category_name"))
        if isinstance(category_raw, dict):
            category = category_raw.get("name", category_raw.get("id", ""))
        else:
            category = str(category_raw) if category_raw else ""
        
        project_code_raw = post_data.get("project_code")
        if isinstance(project_code_raw, dict):
            project_code = project_code_raw.get("name", project_code_raw.get("id", ""))
        else:
            project_code = str(project_code_raw) if project_code_raw else None
        
        # Store additional metadata
        metadata = {
            "created_at": post_data.get("created_at"),
            "updated_at": post_data.get("updated_at"),
            "author": post_data.get("author", post_data.get("username")),
            "views": post_data.get("views", 0),
            "likes": post_data.get("likes", 0)
        }
        
        return Video(
            id=video_id,
            title=title,
            description=description,
            tags=tags,
            transcript=transcript,
            category=category,
            project_code=project_code,
            metadata=metadata
        )
    
    @staticmethod
    def save_video_data(videos: List[Video], filepath: str) -> None:
        """
        Save processed video data to a JSON file.
        
        Args:
            videos (List[Video]): List of Video objects to save
            filepath (str): Path where to save the video data
        """
        logger.info(f"Saving {len(videos)} videos to {filepath}")
        
        # Convert Video objects to dictionaries for JSON serialization
        video_dicts = [video.dict() for video in videos]
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(video_dicts, f, indent=2, ensure_ascii=False)
        
        logger.info("Video data saved successfully")
    
    @staticmethod
    def load_video_data(filepath: str) -> List[Video]:
        """
        Load video data from a JSON file.
        
        Args:
            filepath (str): Path to the video data file
            
        Returns:
            List[Video]: List of Video objects
        """
        logger.info(f"Loading video data from {filepath}")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                video_dicts = json.load(f)
            
            videos = [Video(**video_dict) for video_dict in video_dicts]
            logger.info(f"Loaded {len(videos)} videos successfully")
            return videos
            
        except FileNotFoundError:
            logger.warning(f"Video data file not found at {filepath}")
            return []
        except Exception as e:
            logger.error(f"Error loading video data: {str(e)}")
            return []


class RecommendationService:
    """
    Core service for generating video recommendations.
    
    This service combines embedding generation, similarity calculation,
    and filtering to provide personalized video recommendations.
    """
    
    def __init__(self, embeddings_path: str = "data/embeddings.pkl", video_data_path: str = "data/video_data.json"):
        """
        Initialize the recommendation service.
        
        Args:
            embeddings_path (str): Path to the pre-computed embeddings file
            video_data_path (str): Path to the video data file
        """
        self.embeddings_path = embeddings_path
        self.video_data_path = video_data_path
        self.embeddings = {}
        self.videos = []
        self.videos_dict = {}
        
        # Load data
        self._load_data()
    
    def _load_data(self) -> None:
        """Load embeddings and video data from files."""
        # Load embeddings
        self.embeddings = EmbeddingStorage.load_embeddings(self.embeddings_path)
        
        # Load video data
        self.videos = DataProcessingService.load_video_data(self.video_data_path)
        
        # Create a dictionary for quick video lookup
        self.videos_dict = {video.id: video for video in self.videos}
        
        logger.info(f"Loaded {len(self.embeddings)} embeddings and {len(self.videos)} videos")
    
    def get_recommendations_for_user(
        self,
        username: str,
        project_code: Optional[str] = None,
        limit: int = 10
    ) -> RecommendationResponse:
        """
        Generate personalized video recommendations for a user.
        
        For this implementation, we'll use a content-based approach where we
        recommend videos similar to ones the user has previously engaged with.
        In a real-world scenario, this would involve analyzing user interaction
        history from the database.
        
        Args:
            username (str): Username for personalized recommendations
            project_code (Optional[str]): Project code for category-based filtering
            limit (int): Maximum number of recommendations to return
            
        Returns:
            RecommendationResponse: Response containing recommended videos
        """
        logger.info(f"Generating recommendations for user: {username}")
        
        # Filter videos by project_code if specified
        available_videos = self.videos
        if project_code:
            available_videos = [v for v in self.videos if v.project_code == project_code]
            logger.info(f"Filtered to {len(available_videos)} videos for project_code: {project_code}")
        
        # For this demo, we'll recommend the most popular videos
        # In a real implementation, this would be based on user interaction history
        recommendations = available_videos[:limit]
        
        return RecommendationResponse(
            recommendations=recommendations,
            total_count=len(available_videos),
            username=username,
            project_code=project_code
        )
    
    def get_similar_videos(self, video_id: str, limit: int = 10) -> List[Video]:
        """
        Get videos similar to a specific video based on semantic similarity.
        
        Args:
            video_id (str): ID of the target video
            limit (int): Maximum number of similar videos to return
            
        Returns:
            List[Video]: List of similar videos
        """
        if video_id not in self.embeddings:
            logger.warning(f"Video ID {video_id} not found in embeddings")
            return []
        
        target_embedding = self.embeddings[video_id]
        
        # Find similar videos (excluding the target video itself)
        similar_video_ids = SimilarityCalculator.find_most_similar_videos(
            target_embedding,
            {vid: emb for vid, emb in self.embeddings.items() if vid != video_id},
            top_k=limit
        )
        
        # Convert video IDs to Video objects
        similar_videos = []
        for video_id, similarity_score in similar_video_ids:
            if video_id in self.videos_dict:
                video = self.videos_dict[video_id]
                # Add similarity score to metadata
                video_copy = video.copy()
                if video_copy.metadata is None:
                    video_copy.metadata = {}
                video_copy.metadata["similarity_score"] = similarity_score
                similar_videos.append(video_copy)
        
        logger.info(f"Found {len(similar_videos)} similar videos")
        return similar_videos

