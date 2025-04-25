# Mitigating bias via reducing socio-demograchic dependencies in features
# Remove the n most informative features of gender
# Remove the m most informative features of race

from Q4.d import xgboost_model
from Q4.d import SimpleNN
import pandas as pd
from scipy.stats import pearsonr

def remove_most_informative_features(numRemove) -> pd.DataFrame:
    data = pd.read_csv('Data/processedDataKevin.csv')
    embedding_cols = [col for col in data.columns if col.startswith('w2v_')]

    # Calculate the correlation between each embedding column and 'gender'
    genderCorr = {}
    for col in embedding_cols:
        correlation, _ = pearsonr(data[col], data['gender'])
        genderCorr[col] = abs(correlation)

    raceCorr = {}
    for col in embedding_cols:
        correlation, _ = pearsonr(data[col], data['race'])
        genderCorr[col] = abs(correlation)

    # Sort features by correlation
    sortedGender = sorted(genderCorr, key=genderCorr.get, reverse=True)[:numRemove]
    sortedRace = sorted(raceCorr, key=raceCorr.get, reverse=True)[:numRemove]

    # Remove the most informative features
    removeFeatures = list(set(sortedGender + sortedRace))
    df = data.drop(columns=removeFeatures)
    return df, len(df.columns)

if __name__ == "__main__":
    for numRemove in [200]:
        print(f"\n\nRemoving {numRemove} most informative race & {numRemove} most informative gender features...")
        df = remove_most_informative_features(numRemove)
        print("\n--Tree-based model...")
        xgboost_model(df[0])
        # print("\n--NN-based model...")
        # model = SimpleNN(input_size=df[1], dataset=df[0]) 
        # model.train_model() 