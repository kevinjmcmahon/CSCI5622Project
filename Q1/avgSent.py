import os
import glob
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import csv


file_list = glob.glob("data/*_Transcript.csv")
print(f"Found {len(file_list)} transcript files.")

analyzer = SentimentIntensityAnalyzer()

aggregated_sentiments = []

for file in file_list:
    participant_id = os.path.basename(file).split('_')[0]
    df = pd.read_csv(file, engine='python', quoting=csv.QUOTE_NONE)
    df['sentiment'] = df['Text'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
    avg_sentiment = df['sentiment'].mean()
    aggregated_sentiments.append({'participant_id': participant_id, 'avg_sentiment': avg_sentiment})

df_aggregated = pd.DataFrame(aggregated_sentiments)
print("Aggregated Sentiment DataFrame:")
print(df_aggregated.head())
df_aggregated.to_csv("aggregated_sentiments.csv", index=False)
print("Saved aggregated sentiment file as 'aggregated_sentiments.csv'")
