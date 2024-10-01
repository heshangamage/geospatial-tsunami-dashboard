from flask import Flask, jsonify, request
from flask_cors import CORS
import tweepy
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.linear_model import LogisticRegression
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin requests

# Twitter API Setup
consumer_key = 'xney1ZN707VLI9xNrttcPFUam'
consumer_secret = 'xPz3euwA9Nd7q5GU4Fnhv3pu6LzuUb9ekKTcThCP2PI9LiW07Z'
access_token = '1840590319550603264-sFBsjbxlsnccUxTwKC7tm3EPY7HZMJ'
access_token_secret = 'nlu2ThonT0W879fxNbs6luQGNxdhbLZu341LXAwMJMWCV'
auth = tweepy.OAuth1UserHandler(consumer_key, consumer_secret, access_token, access_token_secret)
api = tweepy.API(auth)

# Sentiment Analyzer Setup
analyzer = SentimentIntensityAnalyzer()

# Dummy predictive model (use LogisticRegression for actual predictions)
model = LogisticRegression()

# Route to collect tweets and analyze sentiments
@app.route('/tweets', methods=['GET'])
def get_tweets():
    keyword = request.args.get('keyword', 'tsunami')
    tweets = tweepy.Cursor(api.search_tweets, q=keyword, lang="en", tweet_mode="extended").items(10)
    
    data = []
    for tweet in tweets:
        sentiment_score = analyzer.polarity_scores(tweet.full_text)['compound']
        data.append({
            'text': tweet.full_text,
            'sentiment': sentiment_score,
            'coordinates': tweet.coordinates
        })
    
    return jsonify(data)

# Route for predictive modeling (dummy data)
@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.json  # Expect earthquake magnitude, distance from shore, depth, etc.
    features = np.array([input_data['magnitude'], input_data['distance'], input_data['depth']]).reshape(1, -1)
    
    # Dummy prediction
    prediction = model.predict(features)
    return jsonify({"prediction": prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
