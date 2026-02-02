# CensorMyPy Docker Setup

This guide will help you set up and run CensorMyPy in a Docker container to avoid dependency conflicts and ensure consistent environments.

## Prerequisites

- Docker installed on your system
- Docker Compose installed
- NVIDIA GPU with CUDA support (recommended for Whisper transcription)
- NVIDIA Docker runtime configured

## Quick Start

1. **Clone or ensure you have the project files**
2. **Build and run the container:**
   ```bash
   docker-compose up --build
   ```

3. **Access the application:**
   - Web interface: http://localhost:8000
   - The container will automatically start the server

## Detailed Setup

### 1. Environment Configuration

The Docker setup includes:
- **NVIDIA CUDA 11.8** base image for GPU acceleration
- **Python 3.9** with all required dependencies
- **FFmpeg** for audio processing
- **Volume mounts** for uploads and project files

### 2. GPU Support

The container is configured for NVIDIA GPU passthrough:
- Uses `nvidia/cuda:11.8-runtime-ubuntu20.04` base image
- GPU devices are automatically detected and mounted
- Whisper transcription will use GPU acceleration if available

### 3. Volume Mounts

- `./uploads:/app/uploads` - Persistent storage for uploaded files
- `.:/app` - Mounts your project directory for development

### 4. Port Mapping

- **8000:8000** - Web server port

## Usage

### Running the Container

```bash
# Build and run in foreground
docker-compose up --build

# Run in background
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop the container
docker-compose down
```

### Accessing the Container

```bash
# Execute commands inside the container
docker-compose exec censormypy bash

# Run specific scripts
docker-compose exec censormypy python async_censormy.py input.mp3 bad_words.txt slurs.txt --method sb
```

## File Structure

```
censorMyPy/
├── Dockerfile              # Container definition
├── docker-compose.yml      # Orchestration configuration
├── requirements_clean.txt  # Python dependencies
├── DOCKER_README.md        # This file
├── uploads/                # Mounted volume for files
└── [your app files]        # Mounted project directory
```

## Troubleshooting

### Common Issues

1. **GPU Not Detected**
   ```bash
   # Check NVIDIA Docker installation
   docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi

   # Ensure NVIDIA runtime is configured in Docker
   ```

2. **Port Already in Use**
   ```bash
   # Change port mapping in docker-compose.yml
   ports:
     - "8001:8000"  # Use different host port
   ```

3. **Permission Issues**
   ```bash
   # Ensure uploads directory exists and is writable
   mkdir -p uploads
   chmod 755 uploads
   ```

4. **Build Failures**
   ```bash
   # Clear Docker cache and rebuild
   docker system prune -a
   docker-compose build --no-cache
   ```

### Logs and Debugging

```bash
# View container logs
docker-compose logs censormypy

# View specific service logs
docker-compose logs -f censormypy

# Debug build process
docker-compose build --progress=plain
```

## Development Workflow

### Making Code Changes

Since the project directory is mounted as a volume, changes to your code are reflected immediately in the container. Simply restart the container to apply changes:

```bash
docker-compose restart
```

### Adding Dependencies

1. Edit `requirements_clean.txt`
2. Rebuild the container:
   ```bash
   docker-compose up --build
   ```

## Production Deployment

For production use, consider:

1. **Use specific image tags** instead of `latest`
2. **Set environment variables** for configuration
3. **Configure proper logging** and monitoring
4. **Use Docker secrets** for sensitive data
5. **Set resource limits** in docker-compose.yml

## Alternative Docker Commands

### Manual Docker Commands (without compose)

```bash
# Build the image
docker build -t censormypy .

# Run with GPU support
docker run --gpus all -p 8000:8000 -v $(pwd):/app -v $(pwd)/uploads:/app/uploads censormypy

# Run without GPU (CPU only)
docker run -p 8000:8000 -v $(pwd):/app -v $(pwd)/uploads:/app/uploads censormypy
```

## Support

If you encounter issues:

1. Check the troubleshooting section above
2. Verify your Docker and NVIDIA driver versions
3. Ensure all prerequisites are installed
4. Check container logs for error messages

The Docker setup provides a clean, isolated environment that eliminates dependency conflicts and ensures consistent behavior across different systems.
