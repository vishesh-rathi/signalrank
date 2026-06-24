FROM python:3.13-slim

# Set environment variables for Python and Streamlit
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860

# Copy uv binary from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory inside container
WORKDIR /app

# Copy project package configurations
COPY pyproject.toml uv.lock ./

# Install project dependencies into the container virtual environment
RUN uv sync --frozen --no-dev

# Copy all source files
COPY . .

# Expose port expected by Hugging Face Spaces for Docker SDK
EXPOSE 7860

# Launch Streamlit app using uv run to invoke the virtual environment
CMD ["uv", "run", "streamlit", "run", "app.py", "--server.port", "7860", "--server.address", "0.0.0.0"]
