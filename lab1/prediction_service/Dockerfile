FROM python:3.10-slim

WORKDIR /app

# Copy only requirements first to leverage Docker cache
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port the app runs on (app.py uses port 5000 by default for Flask)
EXPOSE 5000

# Command to run the application 
# app.py is a Flask app. Standard way to run Flask for development:
# CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]
# For production, a Gunicorn server is preferred:
# CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
# Using a simpler CMD for now, assuming app.py handles running the Flask server.
CMD ["python", "app.py"] 