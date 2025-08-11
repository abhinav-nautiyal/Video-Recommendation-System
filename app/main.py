"""
FastAPI application entry point for the video recommendation engine.

This module sets up the FastAPI application with all endpoints and middleware.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import logging
import os
from contextlib import asynccontextmanager

from .models import RecommendationResponse, HealthResponse, Video
from .services import RecommendationService, ExternalAPIService, DataProcessingService
from .utils import EmbeddingGenerator, EmbeddingStorage

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for services
recommendation_service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    This function handles startup and shutdown events for the FastAPI application.
    During startup, it initializes services and loads data.
    """
    global recommendation_service
    
    # Startup
    logger.info("Starting up the video recommendation engine...")
    
    try:
        # Initialize the recommendation service
        recommendation_service = RecommendationService(
            embeddings_path="data/embeddings.pkl",
            video_data_path="data/video_data.json"
        )
        
        # If no data is loaded, try to fetch and process data
        if not recommendation_service.videos:
            logger.info("No video data found. Attempting to fetch from external API...")
            await initialize_data()
        
        logger.info("Startup completed successfully")
        
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        # Continue startup even if data loading fails
        recommendation_service = RecommendationService()
    
    yield
    
    # Shutdown
    logger.info("Shutting down the video recommendation engine...")


# Create FastAPI application with lifespan manager
app = FastAPI(
    title="Video Recommendation Engine",
    description="A sophisticated recommendation system that suggests personalized video content based on user preferences and engagement patterns using semantic analysis.",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def initialize_data():
    """
    Initialize data by fetching from external API and generating embeddings.
    
    This function is called during startup if no local data is found.
    """
    global recommendation_service
    
    try:
        # Initialize services
        api_service = ExternalAPIService()
        embedding_generator = EmbeddingGenerator()
        
        # Fetch posts from external API
        logger.info("Fetching posts from external API...")
        posts_data = await api_service.fetch_posts(page=1, page_size=100)  # Limit for demo
        
        if not posts_data:
            logger.warning("No posts fetched from external API")
            return
        
        # Transform posts to Video objects
        logger.info("Processing video data...")
        videos = []
        for post in posts_data:
            try:
                video = DataProcessingService.transform_post_to_video(post)
                videos.append(video)
            except Exception as e:
                logger.warning(f"Error processing post {post.get('id', 'unknown')}: {str(e)}")
        
        if not videos:
            logger.warning("No valid videos processed")
            return
        
        # Save video data
        os.makedirs("data", exist_ok=True)
        DataProcessingService.save_video_data(videos, "data/video_data.json")
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        embeddings = {}
        for video in videos:
            try:
                video_dict = video.dict()
                embedding = embedding_generator.generate_video_embedding(video_dict)
                embeddings[video.id] = embedding
            except Exception as e:
                logger.warning(f"Error generating embedding for video {video.id}: {str(e)}")
        
        # Save embeddings
        if embeddings:
            EmbeddingStorage.save_embeddings(embeddings, "data/embeddings.pkl")
            logger.info(f"Generated and saved {len(embeddings)} embeddings")
        
        # Reinitialize recommendation service with new data
        recommendation_service = RecommendationService()
        
    except Exception as e:
        logger.error(f"Error initializing data: {str(e)}")


@app.get("/", response_model=dict)
async def root():
    """
    Root endpoint providing basic information about the API.
    
    Returns:
        dict: Basic information about the video recommendation engine
    """
    return {
        "message": "Welcome to the Video Recommendation Engine",
        "description": "A sophisticated recommendation system using semantic analysis",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "recommendations": "/feed",
            "documentation": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint to verify the service is running properly.
    
    Returns:
        HealthResponse: Current health status of the service
    """
    global recommendation_service
    
    # Check if recommendation service is initialized and has data
    if recommendation_service and recommendation_service.videos:
        status = "healthy"
        message = f"Service is running. {len(recommendation_service.videos)} videos available."
    elif recommendation_service:
        status = "degraded"
        message = "Service is running but no video data is available."
    else:
        status = "unhealthy"
        message = "Recommendation service is not initialized."
    
    return HealthResponse(status=status, message=message)


@app.get("/feed", response_model=RecommendationResponse)
async def get_recommendations(
    username: str = Query(..., description="Username for personalized recommendations"),
    project_code: Optional[str] = Query(None, description="Project code for category-based filtering"),
    limit: int = Query(10, ge=1, le=100, description="Maximum number of recommendations to return")
):
    """
    Get personalized video recommendations for a user.
    
    This endpoint provides the core functionality of the recommendation engine.
    It returns personalized video suggestions based on the user's preferences
    and optionally filters by project code for category-based recommendations.
    
    Args:
        username (str): Username for personalized recommendations
        project_code (Optional[str]): Project code for category-based filtering
        limit (int): Maximum number of recommendations to return (1-100)
        
    Returns:
        RecommendationResponse: Personalized video recommendations
        
    Raises:
        HTTPException: If the recommendation service is not available or encounters an error
    """
    global recommendation_service
    
    if not recommendation_service:
        raise HTTPException(
            status_code=503,
            detail="Recommendation service is not available. Please try again later."
        )
    
    try:
        logger.info(f"Getting recommendations for user: {username}, project_code: {project_code}, limit: {limit}")
        
        recommendations = recommendation_service.get_recommendations_for_user(
            username=username,
            project_code=project_code,
            limit=limit
        )
        
        logger.info(f"Returning {len(recommendations.recommendations)} recommendations")
        return recommendations
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating recommendations: {str(e)}"
        )


@app.get("/similar/{video_id}", response_model=dict)
async def get_similar_videos(
    video_id: str,
    limit: int = Query(10, ge=1, le=50, description="Maximum number of similar videos to return")
):
    """
    Get videos similar to a specific video based on semantic similarity.
    
    This endpoint uses semantic embeddings to find videos with similar content,
    themes, or topics to the specified video.
    
    Args:
        video_id (str): ID of the target video
        limit (int): Maximum number of similar videos to return (1-50)
        
    Returns:
        dict: Similar videos with similarity scores
        
    Raises:
        HTTPException: If the video is not found or service encounters an error
    """
    global recommendation_service
    
    if not recommendation_service:
        raise HTTPException(
            status_code=503,
            detail="Recommendation service is not available. Please try again later."
        )
    
    try:
        logger.info(f"Getting similar videos for video_id: {video_id}, limit: {limit}")
        
        similar_videos = recommendation_service.get_similar_videos(
            video_id=video_id,
            limit=limit
        )
        
        if not similar_videos:
            raise HTTPException(
                status_code=404,
                detail=f"Video with ID '{video_id}' not found or no similar videos available."
            )
        
        return {
            "target_video_id": video_id,
            "similar_videos": similar_videos,
            "total_count": len(similar_videos)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error finding similar videos: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error finding similar videos: {str(e)}"
        )


@app.get("/videos", response_model=dict)
async def list_videos(
    limit: int = Query(20, ge=1, le=100, description="Maximum number of videos to return"),
    offset: int = Query(0, ge=0, description="Number of videos to skip"),
    project_code: Optional[str] = Query(None, description="Filter by project code")
):
    """
    List available videos with optional filtering and pagination.
    
    This endpoint provides a way to browse the available video catalog
    with support for pagination and filtering by project code.
    
    Args:
        limit (int): Maximum number of videos to return (1-100)
        offset (int): Number of videos to skip for pagination
        project_code (Optional[str]): Filter videos by project code
        
    Returns:
        dict: List of videos with pagination information
        
    Raises:
        HTTPException: If the service encounters an error
    """
    global recommendation_service
    
    if not recommendation_service:
        raise HTTPException(
            status_code=503,
            detail="Recommendation service is not available. Please try again later."
        )
    
    try:
        # Get all videos
        all_videos = recommendation_service.videos
        
        # Filter by project_code if specified
        if project_code:
            filtered_videos = [v for v in all_videos if v.project_code == project_code]
        else:
            filtered_videos = all_videos
        
        # Apply pagination
        total_count = len(filtered_videos)
        paginated_videos = filtered_videos[offset:offset + limit]
        
        return {
            "videos": paginated_videos,
            "total_count": total_count,
            "limit": limit,
            "offset": offset,
            "project_code": project_code
        }
        
    except Exception as e:
        logger.error(f"Error listing videos: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error listing videos: {str(e)}"
        )


@app.post("/sync", response_model=dict)
async def sync_data():
    """
    Manually trigger data synchronization from external API.
    
    This endpoint allows manual refresh of video data and embeddings
    from the external API. Useful for updating the recommendation
    system with new content.
    
    Returns:
        dict: Status of the synchronization process
        
    Raises:
        HTTPException: If synchronization fails
    """
    try:
        logger.info("Manual data synchronization triggered")
        await initialize_data()
        
        global recommendation_service
        video_count = len(recommendation_service.videos) if recommendation_service else 0
        embedding_count = len(recommendation_service.embeddings) if recommendation_service else 0
        
        return {
            "status": "success",
            "message": "Data synchronization completed",
            "videos_loaded": video_count,
            "embeddings_generated": embedding_count
        }
        
    except Exception as e:
        logger.error(f"Error during data synchronization: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Data synchronization failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    
    # Run the application
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

