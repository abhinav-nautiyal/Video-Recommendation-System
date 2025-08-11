# Line-by-Line Code Explanation

This document provides a detailed explanation of every line of code in the video recommendation engine project. Each file is broken down with explanations for functions, classes, and important code segments.

## File: `app/models.py`

This file defines the data structures (models) used throughout the application using Pydantic for data validation.

### Imports and Documentation

```python
"""
Data models for the video recommendation engine.

This module defines Pydantic models for data validation and serialization.
These models ensure that our API receives and returns data in the expected format.
"""

from pydantic import BaseModel
from typing import List, Optional, Dict, Any
```

**Explanation**:
- The docstring explains the purpose of this module
- `from pydantic import BaseModel`: Imports the base class for creating data models with automatic validation
- `from typing import List, Optional, Dict, Any`: Imports type hints for better code documentation and IDE support

### Video Model Class

```python
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
```

**Line-by-line explanation**:
- `class Video(BaseModel):`: Creates a new class that inherits from Pydantic's BaseModel, giving us automatic validation and serialization
- The docstring documents what this class represents and lists all its attributes
- `id: str`: Defines a required field that must be a string
- `title: str`: Required string field for the video title
- `description: str`: Required string field for video description
- `tags: List[str]`: Required field that must be a list of strings
- `transcript: str`: Required string field for video transcript/subtitles
- `category: Optional[str] = None`: Optional field that can be a string or None (default is None)
- `project_code: Optional[str] = None`: Optional field for category filtering
- `metadata: Optional[Dict[str, Any]] = None`: Optional field for storing additional data as a dictionary

### RecommendationRequest Model Class

```python
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
```

**Line-by-line explanation**:
- `class RecommendationRequest(BaseModel):`: Defines the structure for incoming recommendation requests
- `username: str`: Required field for the user requesting recommendations
- `project_code: Optional[str] = None`: Optional field to filter recommendations by category
- `limit: Optional[int] = 10`: Optional field to limit the number of results, defaults to 10

### RecommendationResponse Model Class

```python
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
```

**Line-by-line explanation**:
- `class RecommendationResponse(BaseModel):`: Defines the structure for recommendation API responses
- `recommendations: List[Video]`: Required field containing a list of Video objects
- `total_count: int`: Required field showing total available recommendations
- `username: str`: Required field echoing back the username from the request
- `project_code: Optional[str] = None`: Optional field echoing back the project code if used

### HealthResponse Model Class

```python
class HealthResponse(BaseModel):
    """
    Response model for health check endpoint.
    
    Attributes:
        status (str): Health status of the service
        message (str): Additional information about the service status
    """
    status: str
    message: str
```

**Line-by-line explanation**:
- `class HealthResponse(BaseModel):`: Defines the structure for health check responses
- `status: str`: Required field indicating service health (e.g., "healthy", "degraded", "unhealthy")
- `message: str`: Required field providing additional details about the service status

## File: `app/utils.py`

This file contains utility classes and functions for embedding generation, similarity calculation, and data storage.

### Imports and Setup

```python
"""
Utility functions for the video recommendation engine.

This module contains helper functions for:
- Generating semantic embeddings from text
- Calculating similarity between videos
- Processing video data
"""

import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
```

**Line-by-line explanation**:
- The docstring explains the module's purpose
- `import numpy as np`: Imports NumPy for numerical operations with arrays
- `import pickle`: Imports pickle for serializing Python objects to files
- `from sentence_transformers import SentenceTransformer`: Imports the main class for generating embeddings
- `from sklearn.metrics.pairwise import cosine_similarity`: Imports the cosine similarity function
- `from typing import List, Dict, Tuple`: Imports type hints for better code documentation
- `import logging`: Imports logging for tracking application behavior
- `logging.basicConfig(level=logging.INFO)`: Configures logging to show INFO level messages and above
- `logger = logging.getLogger(__name__)`: Creates a logger instance for this module

### EmbeddingGenerator Class

```python
class EmbeddingGenerator:
    """
    Handles the generation of semantic embeddings for video content.
    
    This class uses a pre-trained Sentence-Transformer model to convert
    text (video titles, descriptions, tags, transcripts) into dense vector
    representations that capture semantic meaning.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding generator with a specific model.
        
        Args:
            model_name (str): Name of the Sentence-Transformer model to use.
                             Default is "all-MiniLM-L6-v2" which provides a good
                             balance between performance and speed.
        """
        logger.info(f"Loading Sentence-Transformer model: {model_name}")
        self.model = SentenceTransformer(model_name)
        logger.info("Model loaded successfully")
```

**Line-by-line explanation**:
- `class EmbeddingGenerator:`: Defines a class to handle embedding generation
- The class docstring explains its purpose and functionality
- `def __init__(self, model_name: str = "all-MiniLM-L6-v2"):`: Constructor method with default model name
- The method docstring explains the parameters and purpose
- `logger.info(f"Loading Sentence-Transformer model: {model_name}")`: Logs which model is being loaded
- `self.model = SentenceTransformer(model_name)`: Creates and stores the sentence transformer model instance
- `logger.info("Model loaded successfully")`: Logs successful model loading

### combine_video_text Method

```python
def combine_video_text(self, video_data: Dict) -> str:
    """
    Combine all text fields of a video into a single string for embedding.
    
    This function concatenates the title, description, tags, and transcript
    of a video to create a comprehensive text representation.
    
    Args:
        video_data (Dict): Dictionary containing video information
        
    Returns:
        str: Combined text representation of the video
    """
    # Extract text fields with fallback to empty strings
    title = video_data.get('title', '')
    description = video_data.get('description', '')
    tags = ' '.join(video_data.get('tags', []))
    transcript = video_data.get('transcript', '')
    
    # Combine all text fields with separators
    combined_text = f"Title: {title}. Description: {description}. Tags: {tags}. Transcript: {transcript}"
    
    return combined_text
```

**Line-by-line explanation**:
- `def combine_video_text(self, video_data: Dict) -> str:`: Method that takes a video dictionary and returns a combined string
- The docstring explains the method's purpose, parameters, and return value
- `title = video_data.get('title', '')`: Safely extracts the title, using empty string if not found
- `description = video_data.get('description', '')`: Safely extracts the description
- `tags = ' '.join(video_data.get('tags', []))`: Extracts tags list and joins them with spaces
- `transcript = video_data.get('transcript', '')`: Safely extracts the transcript
- `combined_text = f"Title: {title}. Description: {description}. Tags: {tags}. Transcript: {transcript}"`: Creates a formatted string combining all text fields
- `return combined_text`: Returns the combined text string

### generate_embedding Method

```python
def generate_embedding(self, text: str) -> np.ndarray:
    """
    Generate a semantic embedding for the given text.
    
    Args:
        text (str): Input text to generate embedding for
        
    Returns:
        np.ndarray: Dense vector representation of the text
    """
    # Generate embedding using the Sentence-Transformer model
    embedding = self.model.encode(text, convert_to_numpy=True)
    return embedding
```

**Line-by-line explanation**:
- `def generate_embedding(self, text: str) -> np.ndarray:`: Method that converts text to a numerical vector
- The docstring explains the method's purpose and return type
- `embedding = self.model.encode(text, convert_to_numpy=True)`: Uses the sentence transformer model to convert text to a vector, ensuring the result is a NumPy array
- `return embedding`: Returns the generated embedding vector

### generate_video_embedding Method

```python
def generate_video_embedding(self, video_data: Dict) -> np.ndarray:
    """
    Generate a semantic embedding for a video based on all its text content.
    
    Args:
        video_data (Dict): Dictionary containing video information
        
    Returns:
        np.ndarray: Dense vector representation of the video
    """
    combined_text = self.combine_video_text(video_data)
    return self.generate_embedding(combined_text)
```

**Line-by-line explanation**:
- `def generate_video_embedding(self, video_data: Dict) -> np.ndarray:`: Method that generates an embedding for an entire video
- `combined_text = self.combine_video_text(video_data)`: First combines all video text fields into one string
- `return self.generate_embedding(combined_text)`: Then generates and returns the embedding for the combined text

### SimilarityCalculator Class

```python
class SimilarityCalculator:
    """
    Handles similarity calculations between video embeddings.
    
    This class provides methods to calculate cosine similarity between
    video embeddings and find the most similar videos.
    """
    
    @staticmethod
    def calculate_cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Cosine similarity measures the cosine of the angle between two vectors.
        It ranges from -1 to 1, where 1 means identical, 0 means orthogonal,
        and -1 means opposite.
        
        Args:
            embedding1 (np.ndarray): First embedding vector
            embedding2 (np.ndarray): Second embedding vector
            
        Returns:
            float: Cosine similarity score between the two embeddings
        """
        # Reshape embeddings to 2D arrays for sklearn compatibility
        emb1_2d = embedding1.reshape(1, -1)
        emb2_2d = embedding2.reshape(1, -1)
        
        # Calculate cosine similarity
        similarity = cosine_similarity(emb1_2d, emb2_2d)[0][0]
        return float(similarity)
```

**Line-by-line explanation**:
- `class SimilarityCalculator:`: Defines a class for similarity calculations
- `@staticmethod`: Decorator indicating this method doesn't need class instance (can be called on the class directly)
- `def calculate_cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:`: Method to calculate similarity between two vectors
- The docstring explains cosine similarity and its properties
- `emb1_2d = embedding1.reshape(1, -1)`: Reshapes the first embedding to a 2D array (1 row, all columns) as required by sklearn
- `emb2_2d = embedding2.reshape(1, -1)`: Reshapes the second embedding similarly
- `similarity = cosine_similarity(emb1_2d, emb2_2d)[0][0]`: Calculates cosine similarity and extracts the scalar value
- `return float(similarity)`: Converts to Python float and returns the similarity score

### find_most_similar_videos Method

```python
@staticmethod
def find_most_similar_videos(
    target_embedding: np.ndarray,
    all_embeddings: Dict[str, np.ndarray],
    top_k: int = 10
) -> List[Tuple[str, float]]:
    """
    Find the most similar videos to a target video based on embeddings.
    
    Args:
        target_embedding (np.ndarray): Embedding of the target video
        all_embeddings (Dict[str, np.ndarray]): Dictionary mapping video IDs to embeddings
        top_k (int): Number of most similar videos to return
        
    Returns:
        List[Tuple[str, float]]: List of (video_id, similarity_score) tuples,
                               sorted by similarity score in descending order
    """
    similarities = []
    
    # Calculate similarity with each video
    for video_id, embedding in all_embeddings.items():
        similarity = SimilarityCalculator.calculate_cosine_similarity(
            target_embedding, embedding
        )
        similarities.append((video_id, similarity))
    
    # Sort by similarity score in descending order and return top_k
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]
```

**Line-by-line explanation**:
- `@staticmethod`: Another static method that doesn't require class instance
- The method signature defines parameters for target embedding, all embeddings, and number of results
- `similarities = []`: Initialize empty list to store similarity scores
- `for video_id, embedding in all_embeddings.items():`: Loop through all video embeddings
- `similarity = SimilarityCalculator.calculate_cosine_similarity(target_embedding, embedding)`: Calculate similarity between target and current video
- `similarities.append((video_id, similarity))`: Add video ID and similarity score as a tuple to the list
- `similarities.sort(key=lambda x: x[1], reverse=True)`: Sort the list by similarity score (second element of tuple) in descending order
- `return similarities[:top_k]`: Return only the top k most similar videos

### EmbeddingStorage Class

```python
class EmbeddingStorage:
    """
    Handles saving and loading of pre-computed embeddings.
    
    This class provides methods to persist embeddings to disk and load them
    back, enabling efficient API responses without recomputing embeddings.
    """
    
    @staticmethod
    def save_embeddings(embeddings: Dict[str, np.ndarray], filepath: str) -> None:
        """
        Save embeddings to a pickle file.
        
        Args:
            embeddings (Dict[str, np.ndarray]): Dictionary mapping video IDs to embeddings
            filepath (str): Path where to save the embeddings file
        """
        logger.info(f"Saving embeddings to {filepath}")
        with open(filepath, 'wb') as f:
            pickle.dump(embeddings, f)
        logger.info(f"Embeddings saved successfully. Total videos: {len(embeddings)}")
```

**Line-by-line explanation**:
- `class EmbeddingStorage:`: Class for handling embedding persistence
- `@staticmethod`: Static method for saving embeddings
- `def save_embeddings(embeddings: Dict[str, np.ndarray], filepath: str) -> None:`: Method signature for saving
- `logger.info(f"Saving embeddings to {filepath}")`: Log the save operation
- `with open(filepath, 'wb') as f:`: Open file in write-binary mode using context manager
- `pickle.dump(embeddings, f)`: Serialize the embeddings dictionary to the file
- `logger.info(f"Embeddings saved successfully. Total videos: {len(embeddings)}")`: Log successful completion

### load_embeddings Method

```python
@staticmethod
def load_embeddings(filepath: str) -> Dict[str, np.ndarray]:
    """
    Load embeddings from a pickle file.
    
    Args:
        filepath (str): Path to the embeddings file
        
    Returns:
        Dict[str, np.ndarray]: Dictionary mapping video IDs to embeddings
    """
    logger.info(f"Loading embeddings from {filepath}")
    try:
        with open(filepath, 'rb') as f:
            embeddings = pickle.load(f)
        logger.info(f"Embeddings loaded successfully. Total videos: {len(embeddings)}")
        return embeddings
    except FileNotFoundError:
        logger.warning(f"Embeddings file not found at {filepath}")
        return {}
    except Exception as e:
        logger.error(f"Error loading embeddings: {str(e)}")
        return {}
```

**Line-by-line explanation**:
- `@staticmethod`: Static method for loading embeddings
- `def load_embeddings(filepath: str) -> Dict[str, np.ndarray]:`: Method signature for loading
- `logger.info(f"Loading embeddings from {filepath}")`: Log the load operation
- `try:`: Start exception handling block
- `with open(filepath, 'rb') as f:`: Open file in read-binary mode
- `embeddings = pickle.load(f)`: Deserialize embeddings from the file
- `logger.info(f"Embeddings loaded successfully. Total videos: {len(embeddings)}")`: Log successful loading
- `return embeddings`: Return the loaded embeddings dictionary
- `except FileNotFoundError:`: Handle case where file doesn't exist
- `logger.warning(f"Embeddings file not found at {filepath}")`: Log warning for missing file
- `return {}`: Return empty dictionary if file not found
- `except Exception as e:`: Handle any other exceptions
- `logger.error(f"Error loading embeddings: {str(e)}")`: Log the error
- `return {}`: Return empty dictionary on error

## File: `app/services.py`

This file contains the business logic services for data fetching, processing, and recommendation generation.

### Imports and Setup

```python
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
```

**Line-by-line explanation**:
- The docstring explains the module's business logic purpose
- `import httpx`: Imports httpx for making HTTP requests (async-capable alternative to requests)
- `import json`: Imports JSON handling capabilities
- `import os`: Imports operating system interface for environment variables
- `from typing import List, Dict, Optional, Any`: Imports type hints
- `import logging`: Imports logging functionality
- `from .models import Video, RecommendationResponse`: Imports our custom data models
- `from .utils import EmbeddingGenerator, SimilarityCalculator, EmbeddingStorage`: Imports our utility classes
- The logging setup is the same as in utils.py

### ExternalAPIService Class

```python
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
```

**Line-by-line explanation**:
- `class ExternalAPIService:`: Defines a service class for external API interactions
- `def __init__(self):`: Constructor method
- `self.base_url = os.getenv("API_BASE_URL", "https://api.socialverseapp.com")`: Gets base URL from environment variable with default fallback
- `self.flic_token = os.getenv("FLIC_TOKEN")`: Gets authentication token from environment variable
- `if not self.flic_token:`: Checks if token is missing
- `logger.warning("FLIC_TOKEN not found in environment variables")`: Warns about missing token
- `self.headers = {"Flic-Token": self.flic_token} if self.flic_token else {}`: Creates headers dictionary with token if available, empty otherwise

### fetch_posts Method

```python
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
```

**Line-by-line explanation**:
- `async def fetch_posts(self, page: int = 1, page_size: int = 1000) -> List[Dict[str, Any]]:`: Async method for fetching posts with pagination
- `url = f"{self.base_url}/posts/summary/get"`: Constructs the API endpoint URL
- `params = {"page": page, "page_size": page_size}`: Creates query parameters for pagination
- `try:`: Starts exception handling block
- `async with httpx.AsyncClient() as client:`: Creates an async HTTP client using context manager
- `response = await client.get(url, headers=self.headers, params=params)`: Makes async GET request with headers and parameters
- `response.raise_for_status()`: Raises an exception if the HTTP status indicates an error
- `data = response.json()`: Parses the JSON response
- `posts = data.get("posts", []) if isinstance(data, dict) else data`: Safely extracts posts array from response
- `logger.info(f"Fetched {len(posts)} posts from external API")`: Logs successful fetch
- `return posts`: Returns the list of posts
- `except httpx.HTTPError as e:`: Handles HTTP-specific errors
- `logger.error(f"HTTP error while fetching posts: {str(e)}")`: Logs HTTP errors
- `return []`: Returns empty list on HTTP error
- `except Exception as e:`: Handles any other exceptions
- `logger.error(f"Unexpected error while fetching posts: {str(e)}")`: Logs unexpected errors
- `return []`: Returns empty list on any error

## File: `app/main.py`

This is the main FastAPI application file that sets up the web server and defines all API endpoints.

### Imports and Setup

```python
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
```

**Line-by-line explanation**:
- The docstring explains this is the main application entry point
- `from fastapi import FastAPI, HTTPException, Query`: Imports core FastAPI components
- `from fastapi.middleware.cors import CORSMiddleware`: Imports CORS middleware for cross-origin requests
- `from typing import Optional`: Imports Optional type hint
- `import logging`: Imports logging functionality
- `import os`: Imports OS interface for environment variables
- `from contextlib import asynccontextmanager`: Imports context manager for application lifespan
- The next lines import our custom models, services, and utilities
- `logging.basicConfig(level=logging.INFO)`: Sets up logging configuration
- `logger = logging.getLogger(__name__)`: Creates logger for this module
- `recommendation_service = None`: Global variable to store the recommendation service instance

### Application Lifespan Manager

```python
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
```

**Line-by-line explanation**:
- `@asynccontextmanager`: Decorator that creates an async context manager for application lifespan
- `async def lifespan(app: FastAPI):`: Async function that manages application startup and shutdown
- `global recommendation_service`: Declares we're using the global variable
- `logger.info("Starting up the video recommendation engine...")`: Logs startup beginning
- `try:`: Starts exception handling for startup process
- `recommendation_service = RecommendationService(...)`: Creates recommendation service instance with file paths
- `if not recommendation_service.videos:`: Checks if video data was loaded
- `logger.info("No video data found. Attempting to fetch from external API...")`: Logs attempt to fetch data
- `await initialize_data()`: Calls async function to fetch and process data
- `logger.info("Startup completed successfully")`: Logs successful startup
- `except Exception as e:`: Handles any startup errors
- `logger.error(f"Error during startup: {str(e)}")`: Logs startup errors
- `recommendation_service = RecommendationService()`: Creates service with empty data as fallback
- `yield`: Yields control to the application (startup complete, app running)
- `logger.info("Shutting down the video recommendation engine...")`: Logs shutdown when app stops

### FastAPI Application Creation

```python
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
```

**Line-by-line explanation**:
- `app = FastAPI(...)`: Creates the main FastAPI application instance
- `title="Video Recommendation Engine"`: Sets the API title shown in documentation
- `description="..."`: Sets the API description shown in documentation
- `version="1.0.0"`: Sets the API version
- `lifespan=lifespan`: Connects our lifespan manager to handle startup/shutdown
- `app.add_middleware(CORSMiddleware, ...)`: Adds CORS middleware to the application
- `allow_origins=["*"]`: Allows requests from any origin (should be restricted in production)
- `allow_credentials=True`: Allows credentials in cross-origin requests
- `allow_methods=["*"]`: Allows all HTTP methods
- `allow_headers=["*"]`: Allows all headers

### Root Endpoint

```python
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
```

**Line-by-line explanation**:
- `@app.get("/", response_model=dict)`: Decorator that registers this function as a GET endpoint at the root path
- `async def root():`: Async function that handles root endpoint requests
- The docstring explains the endpoint's purpose
- `return {...}`: Returns a dictionary with API information that FastAPI automatically converts to JSON

### Health Check Endpoint

```python
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
```

**Line-by-line explanation**:
- `@app.get("/health", response_model=HealthResponse)`: Registers GET endpoint at /health with typed response
- `async def health_check():`: Async function for health checking
- `global recommendation_service`: Access the global service instance
- `if recommendation_service and recommendation_service.videos:`: Check if service exists and has video data
- `status = "healthy"`: Set status to healthy if everything is working
- `message = f"Service is running. {len(recommendation_service.videos)} videos available."`: Create informative message
- `elif recommendation_service:`: Check if service exists but has no data
- `status = "degraded"`: Set status to degraded (partially working)
- `message = "Service is running but no video data is available."`: Explain the issue
- `else:`: Handle case where service is not initialized
- `status = "unhealthy"`: Set status to unhealthy
- `message = "Recommendation service is not initialized."`: Explain the problem
- `return HealthResponse(status=status, message=message)`: Return structured health response

### Main Recommendations Endpoint

```python
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
```

**Line-by-line explanation**:
- `@app.get("/feed", response_model=RecommendationResponse)`: Registers GET endpoint with typed response model
- `async def get_recommendations(...)`: Async function with multiple parameters
- `username: str = Query(..., description="...")`: Required query parameter with description for API docs
- `project_code: Optional[str] = Query(None, description="...")`: Optional query parameter
- `limit: int = Query(10, ge=1, le=100, description="...")`: Integer parameter with validation (between 1 and 100)
- The docstring provides comprehensive documentation for the endpoint
- `global recommendation_service`: Access global service instance
- `if not recommendation_service:`: Check if service is available
- `raise HTTPException(status_code=503, detail="...")`: Raise HTTP 503 (Service Unavailable) error
- `try:`: Start exception handling block
- `logger.info(f"Getting recommendations for user: {username}...")`: Log the request details
- `recommendations = recommendation_service.get_recommendations_for_user(...)`: Call service method to get recommendations
- `logger.info(f"Returning {len(recommendations.recommendations)} recommendations")`: Log response details
- `return recommendations`: Return the recommendations response
- `except Exception as e:`: Handle any errors during recommendation generation
- `logger.error(f"Error generating recommendations: {str(e)}")`: Log the error
- `raise HTTPException(status_code=500, detail=f"...")`: Raise HTTP 500 (Internal Server Error)

### Application Runner

```python
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
```

**Line-by-line explanation**:
- `if __name__ == "__main__":`: Checks if this file is being run directly (not imported)
- `import uvicorn`: Imports the ASGI server for running FastAPI
- `uvicorn.run(...)`: Starts the web server
- `"app.main:app"`: Specifies the application location (module.file:variable)
- `host="0.0.0.0"`: Binds to all network interfaces (allows external access)
- `port=8000`: Sets the port number
- `reload=True`: Enables auto-reload when code changes (development feature)
- `log_level="info"`: Sets logging level for the server

## File: `precompute_embeddings.py`

This script handles the pre-computation of video embeddings for improved API performance.

### Script Purpose and Imports

```python
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
```

**Line-by-line explanation**:
- The docstring explains the script's purpose and usage
- `import asyncio`: Imports async programming support
- `import os`: Imports OS interface
- `import sys`: Imports system-specific parameters and functions
- `from dotenv import load_dotenv`: Imports environment variable loading
- `sys.path.append(os.path.dirname(os.path.abspath(__file__)))`: Adds current directory to Python path so we can import our app modules
- The remaining imports bring in our custom services and utilities
- `import logging`: Imports logging functionality

### Main Processing Function

```python
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
```

**Line-by-line explanation**:
- `async def main():`: Defines the main async function
- `load_dotenv()`: Loads environment variables from .env file
- `logger.info("Starting embedding precomputation process...")`: Logs process start
- `try:`: Starts exception handling
- `api_service = ExternalAPIService()`: Creates instance of external API service
- `embedding_generator = EmbeddingGenerator()`: Creates instance of embedding generator
- `os.makedirs("data", exist_ok=True)`: Creates data directory if it doesn't exist
- `logger.info("Fetching posts from external API...")`: Logs data fetching start
- `posts_data = await api_service.fetch_posts(page=1, page_size=500)`: Fetches posts asynchronously
- `if not posts_data:`: Checks if any data was fetched
- `logger.error("No posts fetched...")`: Logs error if no data
- `return`: Exits function early if no data
- `logger.info(f"Fetched {len(posts_data)} posts from external API")`: Logs successful data fetch

### Data Processing Loop

```python
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
```

**Line-by-line explanation**:
- `logger.info("Processing video data...")`: Logs start of data processing
- `videos = []`: Initialize empty list for processed videos
- `failed_count = 0`: Initialize counter for failed processing attempts
- `for i, post in enumerate(posts_data):`: Loop through posts with index
- `try:`: Start exception handling for individual post processing
- `video = DataProcessingService.transform_post_to_video(post)`: Transform raw post to Video object
- `videos.append(video)`: Add successfully processed video to list
- `if (i + 1) % 50 == 0:`: Check if we've processed a multiple of 50 videos
- `logger.info(f"Processed {i + 1}/{len(posts_data)} posts...")`: Log progress every 50 videos
- `except Exception as e:`: Handle processing errors for individual posts
- `failed_count += 1`: Increment failed processing counter
- `logger.warning(f"Error processing post {post.get('id', 'unknown')}: {str(e)}")`: Log processing error
- `logger.info(f"Successfully processed {len(videos)} videos, {failed_count} failed")`: Log final processing results

### Embedding Generation Loop

```python
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
```

**Line-by-line explanation**:
- `logger.info("Generating embeddings...")`: Logs start of embedding generation
- `embeddings = {}`: Initialize empty dictionary for embeddings
- `failed_embeddings = 0`: Initialize counter for failed embedding generation
- `for i, video in enumerate(videos):`: Loop through videos with index
- `try:`: Start exception handling for individual embedding generation
- `video_dict = video.dict()`: Convert Pydantic Video object to dictionary
- `embedding = embedding_generator.generate_video_embedding(video_dict)`: Generate embedding for the video
- `embeddings[video.id] = embedding`: Store embedding with video ID as key
- `if (i + 1) % 25 == 0:`: Check if we've processed a multiple of 25 videos
- `logger.info(f"Generated {i + 1}/{len(videos)} embeddings...")`: Log progress every 25 embeddings
- `except Exception as e:`: Handle embedding generation errors
- `failed_embeddings += 1`: Increment failed embedding counter
- `logger.warning(f"Error generating embedding for video {video.id}: {str(e)}")`: Log embedding error
- `logger.info(f"Successfully generated {len(embeddings)} embeddings, {failed_embeddings} failed")`: Log final embedding results

### Script Execution

```python
if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
```

**Line-by-line explanation**:
- `if __name__ == "__main__":`: Checks if script is being run directly
- `asyncio.run(main())`: Runs the async main function using asyncio event loop

## File: `Dockerfile`

This file defines how to build a Docker container for the application.

```python
# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory in container
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directory
RUN mkdir -p data

# Expose port 8000
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Line-by-line explanation**:
- `FROM python:3.11-slim`: Uses Python 3.11 slim image as the base (smaller than full Python image)
- `WORKDIR /app`: Sets the working directory inside the container to /app
- `ENV PYTHONDONTWRITEBYTECODE=1`: Prevents Python from writing .pyc files
- `ENV PYTHONUNBUFFERED=1`: Ensures Python output is sent directly to terminal without buffering
- `RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*`: Updates package list, installs GCC compiler (needed for some Python packages), and cleans up to reduce image size
- `COPY requirements.txt .`: Copies requirements file first (for better Docker layer caching)
- `RUN pip install --no-cache-dir -r requirements.txt`: Installs Python dependencies without caching to reduce image size
- `COPY . .`: Copies all application code to the container
- `RUN mkdir -p data`: Creates data directory for embeddings and video data
- `EXPOSE 8000`: Declares that the container listens on port 8000
- `CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]`: Default command to run when container starts

This comprehensive code explanation covers every significant line in the project, explaining not just what each line does, but why it's necessary and how it fits into the overall architecture.

