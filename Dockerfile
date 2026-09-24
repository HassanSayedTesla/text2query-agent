FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
COPY scripts ./scripts
COPY .streamlit ./.streamlit

EXPOSE 8501

# Seed the MongoDB database (no-op when data already exists),
# then start the Streamlit UI.
CMD ["sh", "-c", \
     "python scripts/seed_local_mongo.py --uri ${MONGODB_URI:-mongodb://mongodb:27017} --database ${DATABASE_NAME:-sample_mflix} --if-empty \
      && streamlit run app/main.py --server.port 8501 --server.address 0.0.0.0"]