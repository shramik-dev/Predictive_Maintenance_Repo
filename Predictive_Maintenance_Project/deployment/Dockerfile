# ----------------------------
# Base Image
# ----------------------------
FROM python:3.9-slim

# ----------------------------
# Set Working Directory
# ----------------------------
WORKDIR /app

# ----------------------------
# Copy Requirements First (for caching)
# ----------------------------
COPY requirements.txt .

# ----------------------------
# Install Dependencies
# ----------------------------
RUN pip install --no-cache-dir -r requirements.txt

# ----------------------------
# Create Non-root User
# ----------------------------
RUN useradd -m -u 1000 user

# Switch to user
USER user

# Set environment variables
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# ----------------------------
# Set User Working Directory
# ----------------------------
WORKDIR $HOME/app

# ----------------------------
# Copy Project Files
# ----------------------------
COPY --chown=user . $HOME/app

# ----------------------------
# Expose Streamlit Port
# ----------------------------
EXPOSE 8501

# ----------------------------
# Run Streamlit App
# ----------------------------
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
