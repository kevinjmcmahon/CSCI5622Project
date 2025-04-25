# Estimating Depression Severity


# --- FEATURE ENGINEERING - k most informative features ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def top_k_df(k):
    data = pd.read_csv('Data/processedDataKevin.csv')
    embedding_cols = [col for col in data.columns if col.startswith('w2v_')]

    # Calculate the correlation between each embedding column and the 'PHQ-8' score
    feature_correlations = {}
    for col in embedding_cols:
        correlation, _ = pearsonr(data[col], data['PHQ_Score'])
        feature_correlations[col] = abs(correlation)


    top_k_features = sorted(feature_correlations, key=feature_correlations.get, reverse=True)[:k]
    top_k_df = data[['Condition', 'gender', 'race', 'PHQ_Score'] + top_k_features]
    top_k_df            
    return top_k_df

# Function to calc&display performance metrics by gender/race/intersections
def display_performance(results:pd.DataFrame):
    # per‐gender
    gender_map = {
        1: 'Male',
        2: 'Female'
    }

    by_gender = results.groupby('gender')[['true','pred']]\
        .apply(lambda grp: pd.Series({
            'r':   pearsonr(grp['true'], grp['pred'])[0],
            'RE':  np.mean(np.abs(grp['pred']-grp['true'])/grp['true'].max())
        })
    ).rename(index=gender_map)

    print("\nPerformance by Gender:")
    print(by_gender)

    # now per‐race
    race_map = {
        1: 'African American',
        2: 'Asian',
        3: 'White/Causasian',
        4: 'Hispanic',
        5: 'Native American',
        7: 'Other'
    }

    by_race = results.groupby('race')[['true','pred']]\
        .apply(lambda grp: pd.Series({
            'r':   pearsonr(grp['true'], grp['pred'])[0],
            'RE':  np.mean(np.abs(grp['pred']-grp['true'])/grp['true'].max())
        })
    ).rename(index=race_map)

    print("\nPerformance by Race:")
    print(by_race)

    by_race_gender = results.groupby(['race','gender'])[['true','pred']] \
        .apply(lambda grp: pd.Series({
            'r':   pearsonr(grp['true'], grp['pred'])[0],
            'RE':  np.mean(np.abs(grp['pred']-grp['true'])/grp['true'].max())
        })
    ) \
    .rename(index=race_map, level='race') \
    .rename(index=gender_map, level='gender')

    print("\nPerformance by Race & Gender:")
    print(by_race_gender)

    heatmap_data = by_race_gender['r'].unstack()    # rows = race, cols = gender

    plt.figure(figsize=(6, 4))
    plt.imshow(heatmap_data, aspect='auto', cmap='coolwarm')
    plt.colorbar(label='Pearson r')

    plt.xticks(
        range(len(heatmap_data.columns)),
        heatmap_data.columns,        
        rotation=45,
        ha='right'
    )

    plt.yticks(
        range(len(heatmap_data.index)),
        heatmap_data.index          
    )

    plt.title("Pearson r by Gender & Race for Q4")
    plt.tight_layout()
    plt.show()
    
# --- Tree-Based Regression Model - XGBOOST --- 
import xgboost as xgb
from sklearn.model_selection import KFold
from scipy.stats import pearsonr


def xgboost_model(df):
    # Extract for later computation
    races   = df['race'].reset_index(drop=True)
    genders = df['gender'].reset_index(drop=True)

    X = df.drop(columns=['PHQ_Score', 'race', 'gender'])
    y = df['PHQ_Score'].astype('category').cat.codes

    kFold = KFold(n_splits=5, shuffle=True, random_state=42)
    r_scores, rmse_scores = [], []

    # Keep tracks / labels for gender/race calculations later
    all_true, all_pred = [], []
    all_race, all_gender = [], []

    for train_idx, test_idx in kFold.split(X):
        X_train, X_test   = X.iloc[train_idx],   X.iloc[test_idx]
        y_train, y_test   = y.iloc[train_idx],   y.iloc[test_idx]
        race_test         = races.iloc[test_idx]
        gender_test       = genders.iloc[test_idx]

        dTrain = xgb.DMatrix(X_train, label=y_train)
        dTest  = xgb.DMatrix(X_test,  label=y_test)

        params = {
            'objective':   'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth':   5,
            'eta':        0.1,
            'seed':        42,
        }
        num_rounds = 50
        bst = xgb.train(
            params,
            dTrain,
            num_rounds,
            evals=[(dTrain, 'train'), (dTest, 'validation')],
            verbose_eval=False
        )

        # predict & round to nearest int as PHQ scores are ints
        y_pred = np.round(bst.predict(dTest)).astype(int)

        # fold metrics
        r_fold   = pearsonr(y_test, y_pred)[0]
        re_fold  = np.mean(np.abs(y_pred - y_test) / np.max(y_test))
        r_scores.append(r_fold)
        rmse_scores.append(re_fold)

        # Acculumate form df for analysis
        all_true .extend(y_test.tolist())
        all_pred .extend(y_pred.tolist())
        all_race .extend(race_test.tolist())
        all_gender.extend(gender_test.tolist())


    print(f"XGBoost Regression – Mean Pearson r: {np.mean(r_scores):.4f}, Mean RE: {np.mean(rmse_scores):.4f}")

    # build results DataFrame
    results = pd.DataFrame({
        'true':   all_true,
        'pred':   all_pred,
        'race':   all_race,
        'gender': all_gender
    })

    # Print correlation by race/gender and create heatmap viz
    display_performance(results)

# --- DL-Regression Model - PyTorch ---
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold

class dataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
class SimpleNN(nn.Module):
    def __init__(self, input_size, dataset=pd.read_csv('Data/processedDataKevin.csv')): 
        super(SimpleNN, self).__init__()
        self.data = dataset
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 1)  # Regression output
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.relu(self.fc4(x))
        x = self.dropout(x)
        x = self.fc5(x)
        return x
    
    def train_model(self):
        df = self.data
        races  = df['race'].reset_index(drop=True)
        genders = df['gender'].reset_index(drop=True)

        # Drop race/gender from X so we only feed features into the network
        X = df.drop(columns=['PHQ_Score', 'race', 'gender'])
        y = df['PHQ_Score'].astype('category').cat.codes

        kFold = KFold(n_splits=5, shuffle=True, random_state=42)
        r_scores, rmse_scores = [], []

        all_true, all_pred, all_race, all_gender = [], [], [], []

        for train_index, val_index in kFold.split(X):
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]
            race_val   = races.iloc[val_index]
            gender_val = genders.iloc[val_index]

            train_dataset = dataset(X_train, y_train)
            val_dataset   = dataset(X_val, y_val)

            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)

            model     = SimpleNN(input_size=X_train.shape[1])
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            num_epochs = 50

            # --- TRAIN ---
            for epoch in range(num_epochs):
                model.train()
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss    = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

            # --- EVAL (collect predictions) ---
            model.eval()
            preds = []
            with torch.no_grad():
                for batch_X, _ in val_loader:
                    out = model(batch_X)
                    preds.append(out.squeeze().cpu().numpy())
            predictions = np.concatenate(preds)

            # --- APPEND FOLD METRICS ---
            r_fold  = pearsonr(y_val, predictions)[0]
            re_fold = np.mean(np.abs(predictions - y_val) / np.max(y_val))
            r_scores.append(r_fold)        # ← append this fold’s Pearson r
            rmse_scores.append(re_fold)    # ← append this fold’s RE

            # --- COLLECT FOR OVERALL / GROUPED METRICS ---
            all_true.extend( y_val.tolist() )
            all_pred.extend( predictions.tolist() )
            all_race.extend( race_val.tolist() )
            all_gender.extend( gender_val.tolist() )

        # --- PRINT MEAN OF FOLD METRICS ---
        print(f"DL Regression – Mean Pearson r: {np.mean(r_scores):.4f}, Mean RE: {np.mean(rmse_scores):.4f}")


        results = pd.DataFrame({
            'true': all_true,
            'pred': all_pred,
            'race': all_race,
            'gender': all_gender
        })
        
        display_performance(results)

if __name__ == "__main__":
    k_values = [200]  # Number of top features to select
    for k in k_values:
        print(f"\n\n--- Evaluating top {k} features: ---")
        xgboost_model(top_k_df(k))
        # model = SimpleNN(input_size=k, dataset=top_k_df(k)) 
        # model.train_model() 
