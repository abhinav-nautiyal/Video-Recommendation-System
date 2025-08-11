# How to Launch the Video Recommendation Engine

This document provides step-by-step instructions for setting up and running the video recommendation engine on your local machine.

## Prerequisites

Before you begin, ensure you have the following installed on your system:

### Required Software

1. **Python 3.8 or higher**
   - Check your Python version: `python --version` or `python3 --version`
   - If you don't have Python, download it from [python.org](https://www.python.org/downloads/)

2. **pip (Python Package Installer)**
   - Usually comes with Python installation
   - Check if installed: `pip --version` or `pip3 --version`

3. **Git (Optional but recommended)**
   - For cloning the repository and version control
   - Download from [git-scm.com](https://git-scm.com/downloads)

4. **Docker (Optional)**
   - For containerized deployment
   - Download from [docker.com](https://www.docker.com/get-started)

## Step 1: Download the Project

### Option A: Extract from ZIP (If you received a ZIP file)

1. Extract the ZIP file to your desired location
2. Open a terminal/command prompt
3. Navigate to the extracted folder:
   ```bash
   cd path/to/video-recommendation-engine
   ```

### Option B: Clone from GitHub (If available)

1. Open a terminal/command prompt
2. Clone the repository:
   ```bash
   git clone <repository-url>
   cd video-recommendation-engine
   ```

## Step 2: Set Up Python Virtual Environment (Recommended)

A virtual environment isolates your project dependencies from other Python projects.

### On Windows:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# You should see (venv) at the beginning of your command prompt
```

### On macOS/Linux:

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# You should see (venv) at the beginning of your terminal prompt
```

### Verify Virtual Environment

After activation, verify you're in the virtual environment:
```bash
which python  # On macOS/Linux
where python   # On Windows
```

This should point to the python executable inside your venv folder.

## Step 3: Install Dependencies

With your virtual environment activated, install the required Python packages:

```bash
pip install -r requirements.txt
```

This will install:
- FastAPI (web framework)
- Uvicorn (ASGI server)
- Pydantic (data validation)
- Sentence-Transformers (for semantic embeddings)
- Scikit-learn (for similarity calculations)
- Pandas (data manipulation)
- NumPy (numerical operations)
- python-dotenv (environment variable management)
- httpx (HTTP client for external APIs)

**Note**: The installation may take several minutes as it downloads machine learning models and dependencies.

## Step 4: Configure Environment Variables

1. **Create a `.env` file** in the project root directory (same level as README.md)

2. **Add the following content** to the `.env` file:
   ```env
   FLIC_TOKEN=flic_11d3da28e403d182c36a3530453e290add87d0b4a40ee50f17611f180d47956f
   API_BASE_URL=https://api.socialverseapp.com
   ```

3. **Important**: Replace the `FLIC_TOKEN` value with your actual token if you have a different one.

### What these variables do:
- `FLIC_TOKEN`: Authentication token for accessing the Empowerverse API
- `API_BASE_URL`: Base URL for the external API endpoints

## Step 5: Precompute Embeddings (Optional but Recommended)

For better performance, precompute video embeddings before starting the server:

```bash
python precompute_embeddings.py
```

**What this does**:
- Fetches video data from the external API
- Processes and cleans the data
- Generates semantic embeddings for all videos
- Saves the data and embeddings to the `data/` folder

**Expected output**:
```
INFO:__main__:Starting embedding precomputation process...
INFO:app.services:Fetching posts from external API...
INFO:app.services:Fetched X posts from external API
INFO:__main__:Processing video data...
INFO:__main__:Processed 50/X posts...
INFO:__main__:Successfully processed X videos, Y failed
INFO:app.utils:Loading Sentence-Transformer model: all-MiniLM-L6-v2
INFO:app.utils:Model loaded successfully
INFO:__main__:Generating embeddings...
INFO:__main__:Generated 25/X embeddings...
INFO:app.utils:Saving embeddings to data/embeddings.pkl
INFO:app.utils:Embeddings saved successfully. Total videos: X
INFO:__main__:Embedding precomputation completed successfully!
```

**If this step fails**:
- Check your internet connection
- Verify your `FLIC_TOKEN` is correct
- The application will still work but may be slower on first requests

## Step 6: Start the Server

Launch the FastAPI server:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**Command breakdown**:
- `uvicorn`: The ASGI server that runs FastAPI applications
- `app.main:app`: Points to the `app` variable in the `app/main.py` file
- `--host 0.0.0.0`: Makes the server accessible from any IP address
- `--port 8000`: Sets the server to run on port 8000
- `--reload`: Automatically restarts the server when code changes (useful for development)

**Expected output**:
```
INFO:     Will watch for changes in these directories: ['/path/to/video-recommendation-engine']
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [XXXXX] using StatReload
INFO:     Started server process [XXXXX]
INFO:     Waiting for application startup.
INFO:app.main:Starting up the video recommendation engine...
INFO:app.services:Loading embeddings from data/embeddings.pkl
INFO:app.services:Embeddings loaded successfully. Total videos: X
INFO:app.services:Loading video data from data/video_data.json
INFO:app.services:Loaded X videos successfully
INFO:app.main:Startup completed successfully
INFO:     Application startup complete.
```

## Step 7: Test the API

Once the server is running, you can test it in several ways:

### Option A: Web Browser

1. **API Documentation**: Open http://localhost:8000/docs in your browser
   - This shows interactive API documentation
   - You can test endpoints directly from this interface

2. **Health Check**: Open http://localhost:8000/health
   - Should show service status and number of available videos

3. **Root Endpoint**: Open http://localhost:8000/
   - Shows basic API information

### Option B: Command Line (curl)

```bash
# Health check
curl http://localhost:8000/health

# Get recommendations for a user
curl "http://localhost:8000/feed?username=testuser&limit=5"

# Get category-based recommendations
curl "http://localhost:8000/feed?username=testuser&project_code=motivation&limit=3"

# List available videos
curl "http://localhost:8000/videos?limit=10"
```

### Option C: Python Script

Create a test script `test_api.py`:

```python
import requests

# Test health endpoint
response = requests.get("http://localhost:8000/health")
print("Health:", response.json())

# Test recommendations
response = requests.get("http://localhost:8000/feed?username=testuser&limit=5")
print("Recommendations:", response.json())
```

Run it:
```bash
python test_api.py
```

## Step 8: Using Docker (Alternative Launch Method)

If you prefer to use Docker:

### Build the Docker Image

```bash
docker build -t video-recommendation-engine .
```

### Run the Docker Container

```bash
docker run -p 8000:8000 --env-file .env video-recommendation-engine
```

**Command breakdown**:
- `docker run`: Runs a Docker container
- `-p 8000:8000`: Maps port 8000 from container to port 8000 on your machine
- `--env-file .env`: Loads environment variables from the .env file
- `video-recommendation-engine`: The name of the Docker image to run

## Troubleshooting

### Common Issues and Solutions

#### 1. "ModuleNotFoundError" when running the application

**Problem**: Python can't find the required modules.

**Solution**:
- Ensure your virtual environment is activated
- Reinstall dependencies: `pip install -r requirements.txt`
- Check that you're in the correct directory

#### 2. "FLIC_TOKEN not found" warning

**Problem**: The application can't find your API token.

**Solution**:
- Verify the `.env` file exists in the project root
- Check that the `.env` file contains the correct `FLIC_TOKEN` value
- Ensure there are no extra spaces or quotes around the token

#### 3. "Port 8000 is already in use"

**Problem**: Another application is using port 8000.

**Solution**:
- Use a different port: `uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload`
- Or stop the other application using port 8000

#### 4. "No video data found" during startup

**Problem**: The application couldn't load or fetch video data.

**Solution**:
- Run the precompute script: `python precompute_embeddings.py`
- Check your internet connection
- Verify your `FLIC_TOKEN` is valid
- The API will still work but may have limited functionality

#### 5. Slow first-time startup

**Problem**: The application takes a long time to start initially.

**Explanation**: This is normal! The sentence transformer model needs to be downloaded (about 23MB) on first use.

**Solution**: Be patient during the first startup. Subsequent startups will be much faster.

#### 6. Memory issues during embedding generation

**Problem**: The system runs out of memory during embedding generation.

**Solution**:
- Reduce the `page_size` in `precompute_embeddings.py` (e.g., from 500 to 100)
- Close other memory-intensive applications
- Consider using a machine with more RAM

## Development Workflow

### Making Changes to the Code

1. **Edit the code** using your preferred text editor or IDE
2. **Save your changes**
3. **The server will automatically reload** (if you used the `--reload` flag)
4. **Test your changes** using the API documentation at http://localhost:8000/docs

### Adding New Dependencies

1. **Install the package**: `pip install package-name`
2. **Update requirements.txt**: `pip freeze > requirements.txt`
3. **Commit the changes** to version control

### Stopping the Server

- **In the terminal where the server is running**: Press `Ctrl+C` (or `Cmd+C` on Mac)
- **The server will shut down gracefully**

### Deactivating the Virtual Environment

When you're done working on the project:

```bash
deactivate
```

This returns you to your system's default Python environment.

## Production Deployment Considerations

### Environment Variables for Production

Create a production `.env` file with:
- Your actual production `FLIC_TOKEN`
- Production `API_BASE_URL` if different
- Additional security configurations

### Performance Optimization

1. **Precompute embeddings** before deployment
2. **Use a production ASGI server** like Gunicorn with Uvicorn workers
3. **Set up reverse proxy** with Nginx for better performance
4. **Configure logging** for production monitoring

### Security Considerations

1. **Never commit `.env` files** to version control
2. **Use environment-specific configurations**
3. **Implement rate limiting** for production APIs
4. **Use HTTPS** in production environments

## Next Steps

After successfully launching the application:

1. **Explore the API documentation** at http://localhost:8000/docs
2. **Test different endpoints** with various parameters
3. **Review the code** to understand the implementation
4. **Experiment with modifications** to learn more about the system
5. **Consider deploying** to a cloud platform for public access

## Getting Help

If you encounter issues not covered in this guide:

1. **Check the application logs** in the terminal where the server is running
2. **Review the error messages** carefully - they often contain helpful information
3. **Verify all prerequisites** are correctly installed
4. **Double-check file paths** and directory structure
5. **Ensure environment variables** are correctly set

The application includes comprehensive logging that will help you diagnose most issues. Pay attention to the log messages as they provide valuable information about what the application is doing and any problems it encounters.

