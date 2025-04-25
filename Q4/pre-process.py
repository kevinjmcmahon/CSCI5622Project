import os
import pandas as pd
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

print('Running...')

interviewData = pd.read_excel('Data/DAIC demographic data.xlsx', 
                              sheet_name='Interview_Data', 
                              engine='openpyxl')
interviewData.rename(columns={'Partic#': 'Participant_ID'}, inplace=True) # rename index for consistency

metadataMapping = pd.read_excel('Data/DAIC demographic data.xlsx', 
                                sheet_name='Metadata_mapping', 
                                engine='openpyxl')

# Merge data 
DIAC_demographic_data = interviewData.merge(metadataMapping, how='inner', on='Participant_ID')
DIAC_demographic_data['Condition'] = DIAC_demographic_data['Condition'].map({'WoZ': 0, 'AI': 1})
patient_ids = DIAC_demographic_data['Participant_ID'].tolist()

# Processing text files
text_files_folder = 'Data/E-DAIC_Transcripts'
documents = {}

for filename in os.listdir(text_files_folder):
    if filename.endswith('.csv'):
        patient_id = filename.split('_')[0]
        with open(os.path.join(text_files_folder, filename), 'r', encoding='utf-8') as file:
            text = file.read()
            documents[int(patient_id)] = text

# w2v embeddings using spaCy
nlp = spacy.load("en_core_web_md")
doc_vectors = {}
for patient_id, text in documents.items():
    doc = nlp(text)
    doc_vectors[patient_id] = doc.vector

# convert dict of embeddigns to df
doc_embeddings_df = pd.DataFrame.from_dict(doc_vectors, orient='index')
doc_embeddings_df.index.name = 'Participant_ID'
doc_embeddings_df.index = pd.to_numeric(doc_embeddings_df.index, errors='coerce').astype(int)
doc_embeddings_df.columns = [f'w2v_{i+1}' for i in range(doc_embeddings_df.shape[1])]

# merge embeddings w main df
DIAC_demographic_data = DIAC_demographic_data.merge(doc_embeddings_df, how='inner', on='Participant_ID')
print(f'Shape of DIAC_demographic_data after merging spaCy embeddings: {DIAC_demographic_data.shape}')

# sentiment analysis using VADER
analyzer = SentimentIntensityAnalyzer()
sentiment_scores = {}

for patient_id, text in documents.items():
    sentiment = analyzer.polarity_scores(text)
    sentiment_scores[patient_id] = sentiment

# convert dict of sentiment scores to df
sentiment_df = pd.DataFrame(sentiment_scores).T
sentiment_df.columns = ['neg', 'neu', 'pos', 'compound']
sentiment_df.index.name = 'Participant_ID'

# merge sentiment featuers w main df
DIAC_demographic_data = DIAC_demographic_data.merge(sentiment_df, how='inner', on='Participant_ID')
print(f'Shape of DIAC_demographic_data after adding sentiment features: {DIAC_demographic_data.shape}')

DIAC_demographic_data.to_csv('Data/processedData.csv', index=False)