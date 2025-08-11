# Video Recommendation Engine

The Video Recommendation Engine is a sophisticated system designed to provide personalized video content suggestions. It leverages advanced natural language processing (NLP) and machine learning techniques to understand the semantic meaning of video content, offering more relevant recommendations than traditional keyword-based approaches.

This project implements a high-performance API that can be integrated into various platforms to enhance user engagement by delivering tailored video feeds.

## Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites & Installation](#prerequisites--installation)
- [Usage](#usage)
- [Technical Details](#technical-details)
- [API Endpoints](#api-endpoints)

## Features
- **Semantic Understanding:** Utilizes state-of-the-art Sentence-Transformers to generate embeddings that capture the contextual meaning of video titles, descriptions, tags, and transcripts.
- **Personalized Recommendations:** Delivers tailored video suggestions based on user preferences and content similarity.
- **Category-based Filtering:** Supports filtering recommendations by specific project codes or categories.
- **Efficient Pre-computation:** Video embeddings are pre-computed and stored, ensuring fast, real-time API responses.
- **Scalable Architecture:** Built with FastAPI for high performance and asynchronous operations, suitable for production environments.
- **Containerization:** Includes Docker support for consistent and easy deployment across different systems.
- **Comprehensive Documentation:** Provides detailed explanations of the project, technical methods, code, and launch instructions.

## Project Structure
```bash
video-recommendation-engine/
├── app/
│   ├── __init__.py           # Package initialization
│   ├── main.py               # FastAPI application entry point
│   ├── models.py             # Pydantic data models for API requests/responses
│   ├── services.py           # Business logic for data fetching, processing, and recommendations
│   └── utils.py              # Utility functions for embedding generation, similarity, and storage
├── data/
│   ├── video_data.json       # Processed video metadata (generated after precomputation)
│   └── embeddings.pkl        # Pre-computed video embeddings (generated after precomputation)
├── docs/
│   ├── about_project.md      # Project overview and assignment context
│   ├── tech_methods_algos.md # Explanation of technical methods and algorithms used
│   ├── code_explanation.md   # Line-by-line explanation of the codebase
│   └── how_to_launch.md      # Step-by-step guide to launch the project
├── .env                      # Environment variables (FLIC_TOKEN, API_BASE_URL)
├── .gitignore               # Git ignore rules for version control
├── Dockerfile               # Docker build instructions for containerization
├── requirements.txt         # Python dependencies
├── precompute_embeddings.py # Script to precompute video embeddings
├── how_to_launch.md         # Detailed setup instructions (duplicate for easy access)
└── README.md                # This file
```

## Prerequisites & Installation

### Prerequisites
- Python 3.8 or higher
- `pip` and `venv`
- Docker (optional, for containerized deployment)

### Installation
1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd video-recommendation-engine
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: This will install `torch` and `sentence-transformers` which are large and may take some time. Ensure you have a stable internet connection.*

4.  **Set up your environment variables:**
    Create a `.env` file in the root directory and add the following content:
    ```ini
    # .env

    FLIC_TOKEN=Your flic token here
    API_BASE_URL=https://api.socialverseapp.com
    ```
    *The `FLIC_TOKEN` is provided for the assignment and is free to use.*

## Usage

### 1. Precompute Embeddings (Recommended)
For optimal performance, precompute video embeddings before starting the server. This fetches data from the external API, processes it, and saves the embeddings locally.

```bash
python precompute_embeddings.py
```

### 2. Start the Backend Server
Start the FastAPI application. This will serve the API endpoints.

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The server will start on `http://0.0.0.0:8000`.

### 3. Access the API
Once the server is running, you can access the API and its documentation:
- **API Documentation (Swagger UI)**: `http://localhost:8000/docs`
- **Health Check**: `http://localhost:8000/health`
- **Root Endpoint**: `http://localhost:8000/`

## Technical Details
- **Backend Framework**: **FastAPI** for building robust and high-performance APIs with automatic documentation and data validation.
- **Semantic Embeddings**: **Sentence-Transformers** (based on PyTorch and Hugging Face Transformers) for generating dense vector representations of text.
- **Similarity Calculation**: **Cosine Similarity** from `scikit-learn` for finding the most semantically similar videos.
- **Data Handling**: **Pandas** and **NumPy** for efficient data manipulation and numerical operations.
- **External API Integration**: **httpx** for asynchronous HTTP requests to the Empowerverse API.
- **Environment Management**: **python-dotenv** for loading environment variables.
- **Containerization**: **Docker** for creating isolated and portable application environments.

## API Endpoints

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Root endpoint with basic API information |
| `/health` | GET | Health check and service status |
| `/feed` | GET | Get personalized video recommendations |
| `/similar/{video_id}` | GET | Get videos similar to a specific video |
| `/videos` | GET | List available videos with pagination |
| `/sync` | POST | Manually sync data from external API |

### Example Usage

#### Get Personalized Recommendations

```bash
curl "http://localhost:8000/feed?username=john_doe&limit=5"
```

#### Get Category-based Recommendations

```bash
curl "http://localhost:8000/feed?username=john_doe&project_code=motivation&limit=10"
```

#### Get Similar Videos

```bash
curl "http://localhost:8000/similar/video123?limit=5"
```

## Docker Deployment

### Build and Run with Docker

```bash
# Build the Docker image
docker build -t video-recommendation-engine .

# Run the container
docker run -p 8000:8000 --env-file .env video-recommendation-engine
```

### Using Docker Compose (Optional)

Create a `docker-compose.yml` file:

```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    env_file:
      - .env
    volumes:
      - ./data:/app/data
```

Run with:

```bash
docker-compose up
```
