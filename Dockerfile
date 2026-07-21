# Base Image/Blueprint
FROM python:3.11-slim

# Buffer so print() appears immediately
ENV PYTHONUNBUFFERED=1

# Basically ROOT_PATH to then joined with $PATH
ENV PATH="/root/.local/bin:$PATH"

# cd /app or mkdir /app 
    # Every future command is directed to /app
    # copy requirements.txt → (to) /app
WORKDIR /app

# Install system dependencies required by common ML packages.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       gcc \
       libgl1 \
       libglib2.0-0 \
       git \
    && rm -rf /var/lib/apt/lists/*

# Copies req and Host Machine to Container and /app
COPY requirements.txt ./

# Normal pip install for req, no cache
RUN pip install --no-cache-dir -r requirements.txt

# Copy Everything into /app
COPY . ./
# pip install before copying everything into /app
    # because any file change would invalidate the cache 
        # and force pip install to run again.

# Executable, when a container starts, it begins with
    # python -m
ENTRYPOINT ["python", "-m"]

# Defaut Arguments to ENTRYPOINT
    # Docker combines them to create
        # python -m scripts.submission --config configs/inference.yaml
CMD ["scripts.submission", "--config", "configs/inference.yaml"]

# gcc: GNU C Compiler (Many Python Packages run C under the hood)
# libgl1: OpenGL runtime (OpenCV)
# git: Allows pip to install directly from GitHub.

# Running `docker run myimage` equal to running
    # python -m scripts.submission --config configs/inference.yaml

# But if you run `docker run myimage configs/other.yaml`
    # Docker keeps the entrypoint and replaces the CMD:
    # python -m scripts.submission --config configs/other.yaml
        # Similar to argparse

# If we remove the ENTRYPOINT
        # docker run myimage configs/other.yaml
        # -> scripts.submission --config configs/other.yaml
        # So ENTRYPOINT basically force every command to
        # Start with that specific ENTRYPOINT joined together
        
# ENTRYPOINT: the executable that should almost always run.
# CMD: the default arguments, which users can override.
