import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import csv
import glob
import numpy as np


#see if working
sample_file = "data/test.csv"
df = pd.read_csv(sample_file, engine='python', quoting=csv.QUOTE_NONE)
print(df.head())


vectorizer = CountVectorizer()
X_syntactic = vectorizer.fit_transform(df['Text'])
print("Syntactic features shape:", X_syntactic.shape)
# Syntactic features shape: (81, 274)
# 81 is the # of rows in the csv
# 274 is the number of unique words (tokens) found
X_array = X_syntactic.toarray()
print("Sample document vector (first row):", X_array[0])
print(vectorizer.get_feature_names_out())

analyzer = SentimentIntensityAnalyzer()
df['sentiment'] = df['Text'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
print(df[['Text', 'sentiment']].head())
# positive values (closer to +1): positive sentiment
# negative values (closer to -1): negative sentiment
# values near 0: neutral sentiment

plt.hist(df['sentiment'], bins=20, edgecolor='black')
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.title('Distribution of Sentiment Scores')
plt.show()

word_counts = X_syntactic.sum(axis=0)
word_counts = np.array(word_counts).flatten()  
vocab = vectorizer.get_feature_names_out()
df_word_counts = pd.DataFrame({'word': vocab, 'count': word_counts})
top_words = df_word_counts.sort_values(by='count', ascending=False).head(10)

plt.figure(figsize=(6, 4))
plt.bar(top_words['word'], top_words['count'], color='skyblue')
plt.xlabel('Word')
plt.ylabel('Frequency')
plt.title('Top 10 Most Frequent Words')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()