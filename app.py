from fastapi import FastAPI
from pydantic import BaseModel
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Download VADER lexicon
nltk.download("vader_lexicon")

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Initialize FastAPI app
app = FastAPI()

# Define request model
class SentimentRequest(BaseModel):
    text: str

# Sentiment analysis endpoint
@app.post("/sentiment/")
def analyze_sentiment(request: SentimentRequest):
    sentiment_score = sia.polarity_scores(request.text)
    
    # Determine sentiment label
    if sentiment_score["compound"] >= 0.05:
        sentiment = "positive"
    elif sentiment_score["compound"] <= -0.05:
        sentiment = "negative"
    else:
        sentiment = "neutral"
    
    return {"text": request.text, "sentiment": sentiment, "scores": sentiment_score}
