# a) Extracting langauge features
# Extract several language features from the data. Include at least two of the following types of features, spanning different levels of complexity
import pandas as pd
import nltk
import os 
from sklearn.feature_extraction.text import CountVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Opening excel files/sheets
# No need to open the 'Variable Key' sheets from either .xlsx files. No data in file, just the variable mappings.

# 1. DIAC demographic data.xlsx - Interview_Data
interviewData = pd.read_excel('Data/DAIC demographic data.xlsx', sheet_name='Interview_Data', engine='openpyxl')
# Rename the column to maintain key concurrnecy with all other dataframes, i.e. 'Participant_ID'
interviewData.rename(columns={'Partic#': 'Participant_ID'}, inplace=True)
print(f'Cols in interviewData: {interviewData.columns}')

# 2. DIAC demographic data.xlsx - Metadata_mapping
metadataMapping = pd.read_excel('Data/DAIC demographic data.xlsx', sheet_name='Metadata_mapping', engine='openpyxl')
print(f'Cols in metadataMapping: {metadataMapping.columns}')


## Joining dataframes together to make one large dataframe with all of the data.  
DIAC_demographic_data = interviewData.merge(metadataMapping, how='inner', on='Participant_ID')
patient_ids = DIAC_demographic_data['Participant_ID'].tolist()


## -- PROCESSING TEXT FILES --
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

text_files_folder  = 'Data/E-DAIC_Transcripts'
documents = {}

# Loop through all files in the directory
for filename in os.listdir(text_files_folder):
    print(filename)
    if filename.endswith('.csv'):
        # Extract the patient ID from the filename
        patient_id = filename.split('_')[0]
        # Read the text file
        with open(os.path.join(text_files_folder, filename), 'r') as file:
            text = file.read()
            documents[patient_id] = text

# Lower case & removing stop words to clean the text
documents_lower = [doc.lower() for doc in documents.values()]
count_vectorizer = CountVectorizer(stop_words='english')
X_lexicon = count_vectorizer.fit_transform(documents_lower)

# Convert the sparse matrix to a DataFrame
X_lexicon_df = pd.DataFrame(X_lexicon.toarray(), columns=count_vectorizer.get_feature_names_out())
# Add the patient IDs as a new column
X_lexicon_df['Participant_ID'] = list(map(int, documents.keys()))
X_lexicon_df = X_lexicon_df.set_index('Participant_ID')

# Merge the lexicon features with the demographic data
DIAC_demographic_data = DIAC_demographic_data.merge(X_lexicon_df, how='inner', on='Participant_ID')
print(f'Shape of DIAC_demographic_data: {DIAC_demographic_data.shape}')

## -- SENTIMENT ANALYSIS --
# Initialize the sentiment intensity analyzer
analyzer = SentimentIntensityAnalyzer()
sentiment_scores = {}

for patient_id, text in documents.items(): 
    sentiment = analyzer.polarity_scores(text)
    sentiment_scores[patient_id] = sentiment

# Convert the sentiment scores to a DataFrame
sentiment_df = pd.DataFrame(sentiment_scores).T
sentiment_df.columns = ['neg', 'neu', 'pos', 'compound']
sentiment_df['Participant_ID'] = list(map(int, sentiment_df.index))
sentiment_df = sentiment_df.set_index('Participant_ID')


# Merge the sentiment features with the main dataframe
DIAC_demographic_data = DIAC_demographic_data.merge(sentiment_df, how='inner', on='Participant_ID')
print(f'Shape of DIAC_demographic_data after sentiment analysis: {DIAC_demographic_data.shape}')