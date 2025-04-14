import os
import glob
import pandas as pd
import csv

file_list = glob.glob("data/*_Transcript.csv")
print(f"Found {len(file_list)} transcript files.")

transcript_list = []
for file in file_list:
    participant_id = os.path.basename(file).split('_')[0]
    df_temp = pd.read_csv(file, engine='python', quoting=csv.QUOTE_NONE)
    full_text = " ".join(df_temp['Text'].astype(str).tolist())
    transcript_list.append({'participant_id': participant_id, 'transcript': full_text})
df_transcripts = pd.DataFrame(transcript_list)
df_transcripts.to_csv("aggregated_transcripts.csv", index=False)
print("aggregated_transcripts.csv generated successfully.")
