# 1. Base image: Python 3.11, slim variant
FROM python:3.11-slim

# 2. Working directory inside the container
WORKDIR /app

# 3. Copy ONLY the slim requirements file first (layer-caching trick:
#    deps rarely change, so this layer gets reused when you edit app code)
COPY requirements-inference.txt .

# 4. Install dependencies. The backslash continues the command onto the next
#    line; --extra-index-url pulls the CPU-only torch build, not the giant GPU one.
RUN pip install --no-cache-dir -r requirements-inference.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

# 5. Copy the rest of the project (code + models) into the image.
#    What actually gets copied is governed by .dockerignore.
COPY . .

# 6. Gradio serves on port 7860
EXPOSE 7860

# 7. Command that runs when a container starts
CMD ["python", "app.py"]
