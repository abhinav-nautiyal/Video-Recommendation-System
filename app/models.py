"""
Data models for the video recommendation engine.

This module defines Pydantic models for data validation and serialization.
These models ensure that our API receives and returns data in the expected format.
"""

from pydantic import BaseModel
from typing import List, Optional, Dict, Any


class Video(BaseModel):
    """
    Represents a video with its metadata.
    
    Attributes:
        id (str): Unique identifier for the video
        title (str): Title of the video
        description (str): Description of the video content
        tags (List[str]): List of tags associated with the video
        transcript (str): Transcript or subtitles of the video
        category (Optional[str]): Category the video belongs to
        project_code (Optional[str]): Project code for category-based filtering
        metadata (Optional[Dict[str, Any]]): Additional metadata
    """
    id: str
    title: str
    description: str
    tags: List[str]
    transcript: str
    category: Optional[str] = None
    project_code: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class RecommendationRequest(BaseModel):
    """
    Request model for getting video recommendations.
    
    Attributes:
        username (str): Username for personalized recommendations
        project_code (Optional[str]): Project code for category-based filtering
        limit (Optional[int]): Maximum number of recommendations to return (default: 10)
    """
    username: str
    project_code: Optional[str] = None
    limit: Optional[int] = 10


class RecommendationResponse(BaseModel):
    """
    Response model for video recommendations.
    
    Attributes:
        recommendations (List[Video]): List of recommended videos
        total_count (int): Total number of available recommendations
        username (str): Username the recommendations were generated for
        project_code (Optional[str]): Project code used for filtering (if any)
    """
    recommendations: List[Video]
    total_count: int
    username: str
    project_code: Optional[str] = None


class HealthResponse(BaseModel):
    """
    Response model for health check endpoint.
    
    Attributes:
        status (str): Health status of the service
        message (str): Additional information about the service status
    """
    status: str
    message: str

