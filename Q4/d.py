# Estimating Depression Severity


# --- FEATURE ENGINEERING - k most informative features ---
import pandas as pd
import numpy as np


def top_k_df(k):
    data = pd.read_csv('Data/processedData.csv')
    embedding_cols = [col for col in data.columns if col.startswith('w2v_')]

    # Calculate the correlation between each embedding column and the 'PHQ-8' score
    feature_correlations = {}
    for col in embedding_cols:
        correlation, _ = pearsonr(data[col], data['PHQ_Score'])
        feature_correlations[col] = abs(correlation)


    top_k_features = sorted(feature_correlations, key=feature_correlations.get, reverse=True)[:k]
    top_k_df = data[['Condition', 'gender', 'race', 'PHQ_Score'] + top_k_features]

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

# --- Tree-Based Regression Model - XGBOOST --- 
import xgboost as xgb
from sklearn.model_selection import KFold
from scipy.stats import pearsonr


def xgboost_model(k):
    df = top_k_df(k)
    X = df.drop(columns=['PHQ_Score'])
    y = df['PHQ_Score'].astype('category')
    # Convert categorical labels to numerical values
    y = y.cat.codes

    kFold = KFold(n_splits=5, shuffle=True, random_state=42)
    r_scores, rmse_scores = [], []

    for train_index, test_index in kFold.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        dTrain = xgb.DMatrix(X_train, label=y_train)
        dVal = xgb.DMatrix(X_test, label=y_test)

        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 5,
            'seed': 42,
        }

        num_rounds = 50
        model = xgb.train(params, dTrain, num_rounds, evals=[(dTrain, 'train'), (dVal, 'validation')], verbose_eval=False)
        y_pred = model.predict(dVal)
        y_pred = np.round(y_pred).astype(int)

        # Pearson's correlation
        r_score = pearsonr(y_pred, y_test)[0]
        r_scores.append(r_score)

        # RMSE
        rmse_score = np.mean(np.abs(y_pred - y_test) / np.max(y_test))
        rmse_scores.append(rmse_score)

    print(f"XGBoost Regression - Mean Pearson r: {np.mean(r_scores):.4f}, Mean RMSE: {np.mean(rmse_scores):.4f}")

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
    def __init__(self, input_size):  # Fix the method name
        super(SimpleNN, self).__init__()
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
        df = top_k_df(self.input_size)
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
    # Example usage
    k_values = [50, 100, 150, 200, 250, 300]  # Number of top features to select
    for k in k_values:
        print(f"\n\n--- Evaluating top {k} features: ---")
        xgboost_model(k)
        model = SimpleNN(input_size=k) 
        model.train_model() 
