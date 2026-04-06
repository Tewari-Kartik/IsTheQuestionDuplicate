# 1. Use an official, lightweight Python image (Updated to 3.11!)
FROM python:3.11-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Pre-download the NLTK stopwords so the app boots faster
RUN python -m nltk.downloader stopwords

# 5. Copy the rest of your application code and models
COPY . .

# 6. Expose the port FastAPI runs on
EXPOSE 8000

# 7. Command to run the application
# Note: We use 0.0.0.0 so it is accessible outside the container
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]