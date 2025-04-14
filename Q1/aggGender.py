import pandas as pd

df_agg = pd.read_csv("aggregated_sentiments_with_PHQ.csv")
df_agg['participant_id'] = df_agg['participant_id'].astype(str).str.strip()

df_demo = pd.read_excel("DAIC demographic data.xlsx", sheet_name="Interview_Data")
df_demo['participant_id'] = df_demo['Partic#'].astype(str).str.strip()

merged_df = pd.merge(df_agg, df_demo[['participant_id', 'gender']], on='participant_id', how='left')
print("Merged dataset shape:", merged_df.shape)
print(merged_df.head())

merged_df.to_csv("aggregated_sentiments_with_PHQ_and_gender.csv", index=False)

