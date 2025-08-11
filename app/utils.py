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

