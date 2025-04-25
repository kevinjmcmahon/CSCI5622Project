import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

plt.figure(figsize=(10, 6))
# Option A — split violin plot
df = pd.read_csv('Data/processedDataKevin.csv')
print(f'Shape: {df.shape}')

race_map = {
    1: 'African American',
    2: 'Asian',
    3: 'White/Causasian',
    4: 'Hispanic',
    5: 'Native American',
    7: 'Other'
}

gender_map = {
    1: 'Male',
    2: 'Female'
}

scores = df['PHQ_Score']
print(scores.min(), scores.max())

df['gender'] = df['gender'].map(gender_map)
df['race']   = df['race'].map(race_map)

sns.violinplot(
    x='race',
    y='PHQ_Score',
    hue='gender',
    data=df,
    split=True,           # puts M/F back‑to‑back
    inner='quartile'      # shows median & IQR
)
plt.title('Distribution of PHQ Scores by Race & Gender')
plt.xlabel('Race')
plt.ylabel('PHQ Score')
plt.legend(title='Gender')
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('Data/processedDataKevin.csv')
gender_map = {1: 'Male', 2: 'Female'}
df['gender'] = df['gender'].map(gender_map)

male_scores   = df.loc[df['gender']=='Male',   'PHQ_Score']
female_scores = df.loc[df['gender']=='Female', 'PHQ_Score']

# Compute overall quartiles once
overall_q = np.percentile(df['PHQ_Score'], [25, 50, 75])

fig, ax = plt.subplots(figsize=(8, 6))

ax.hist(
    male_scores,
    bins=range(0, 24),
    alpha=0.5,
    label='Male',
    color='skyblue',
    edgecolor='black'
)
ax.hist(
    female_scores,
    bins=range(0, 24),
    alpha=0.5,
    label='Female',
    color='salmon',
    edgecolor='black'
)

# Add vertical lines at the overall quartiles
quartile_labels = ['25th pct', 'Median', '75th pct']
for q, lbl in zip(overall_q, quartile_labels):
    ax.axvline(q, color='black', linestyle='--', linewidth=1)
    ax.text(q + 0.2, ax.get_ylim()[1]*0.9, lbl, rotation=90,
            va='top', ha='left', color='black')

# Labels & legend
ax.set_xlabel('PHQ Score')
ax.set_ylabel('Count')
ax.set_title('PHQ Score Distribution by Gender')
ax.legend(title='Gender')
plt.tight_layout()
plt.show()

counts = df['race'].value_counts().sort_index()
plt.figure(figsize=(8, 6))
counts.plot(kind='bar', edgecolor='black')
plt.xlabel('Race')
plt.ylabel('Count')
plt.title('Number of Participants by Race')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()