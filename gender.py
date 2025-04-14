import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

# sklearn and vaderSentiment for feature extraction
from sklearn.feature_extraction.text import CountVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# load file with participant ids, average sentiment, PHQ scores, and gender
df_agg = pd.read_csv("aggregated_sentiments_with_PHQ_and_gender.csv")
df_agg['participant_id'] = df_agg['participant_id'].astype(str)  # make sure IDs are strings
print("Aggregated Data (first 5 rows):")
print(df_agg.head())

# load transcripts (full text) for each participant
df_transcripts = pd.read_csv("aggregated_transcripts.csv")
df_transcripts['participant_id'] = df_transcripts['participant_id'].astype(str)
print("Aggregated Transcripts (first 5 rows):")
print(df_transcripts.head())

# turn transcripts into a bag-of-words model (count how many times each word appears)
vectorizer = CountVectorizer()
X_syntactic = vectorizer.fit_transform(df_transcripts['transcript'])
X_syntactic_array = X_syntactic.toarray()  # convert to regular numpy array
print("Syntactic features shape (participants, unique words):", X_syntactic_array.shape)

# reorder transcript features to match the participant order in the aggregation file
ordered_indices = [list(df_transcripts['participant_id']).index(pid) for pid in df_agg['participant_id']]
X_syn_ordered = X_syntactic_array[ordered_indices, :]

# pull out the average sentiment score from earlier work
sentiment_array = df_agg['avg_sentiment'].values.reshape(-1, 1)

# combine syntactic word counts with the sentiment score
X_combined = np.hstack([X_syn_ordered, sentiment_array])
print("Combined feature matrix shape:", X_combined.shape)

# replace missing values with 0s just in case
X_combined = np.nan_to_num(X_combined, nan=0)

# select 1000 best features using chi-squared feature selection
from sklearn.feature_selection import SelectKBest, chi2
selector = SelectKBest(score_func=chi2, k=1000)
X_selected = selector.fit_transform(X_combined, df_agg['gender'])  # gender is the target
print("Selected features shape:", X_selected.shape)

# map gender values so model understands it
# 2 = male (map to 0), 1 = female (map to 1)
y = df_agg['gender'].map({2: 0, 1: 1}).values
print("Target variable (y) sample:", y[:10])

# use stratified k-fold cross-validation (5 folds) to evaluate model
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score

skf = StratifiedKFold(n_splits=5, shuffle=True)

# random forest model for classification
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100)

# save accuracy scores for each fold
rf_acc = []
rf_bal_acc = []

# train and evaluate random forest across each fold
for train_idx, test_idx in skf.split(X_selected, y):
    # split training and testing data
    X_train, X_test = X_selected[train_idx], X_selected[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # fit model on training data
    rf.fit(X_train, y_train)
    
    # make predictions on test data
    preds_rf = rf.predict(X_test)
    
    # record accuracy and balanced accuracy
    rf_acc.append(accuracy_score(y_test, preds_rf))
    rf_bal_acc.append(balanced_accuracy_score(y_test, preds_rf))

print("Random Forest Average Accuracy:", np.mean(rf_acc))
print("Random Forest Average Balanced Accuracy:", np.mean(rf_bal_acc))

# setup neural network training
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# turn input data into pytorch tensors
X_tensor = torch.tensor(X_selected, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # unsqueeze to make it (N, 1)

# define a basic feedforward neural network
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)  # first hidden layer
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)  # prevent overfitting
        self.fc2 = nn.Linear(64, 32)   # second hidden layer
        self.fc3 = nn.Linear(32, 1)    # output layer
        self.sigmoid = nn.Sigmoid()    # binary output
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

# save accuracy scores for neural network
pt_acc = []
pt_bal_acc = []

# train and evaluate neural network across each fold
for fold, (train_idx, test_idx) in enumerate(skf.split(X_selected, y)):
    print(f"PyTorch Fold {fold+1}")
    
    # split training and testing data
    X_train = X_tensor[train_idx]
    y_train = y_tensor[train_idx]
    X_test = X_tensor[test_idx]
    y_test = y_tensor[test_idx]
    
    # create dataloaders for mini-batch training
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # define model, loss function, and optimizer
    model = SimpleNN(input_dim=X_selected.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # train model for a number of epochs
    model.train()
    num_epochs = 20
    for epoch in range(num_epochs):
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    
    # evaluate model
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            preds = (outputs >= 0.5).float()  # threshold at 0.5
            all_preds.extend(preds.squeeze().cpu().numpy())
            all_labels.extend(batch_y.squeeze().cpu().numpy())
    
    # record accuracy scores
    fold_acc = accuracy_score(all_labels, all_preds)
    fold_bal_acc = balanced_accuracy_score(all_labels, all_preds)
    pt_acc.append(fold_acc)
    pt_bal_acc.append(fold_bal_acc)
    print(f"Fold {fold+1}: Accuracy = {fold_acc:.3f}, Balanced Accuracy = {fold_bal_acc:.3f}\n")

# print final average results
print("Average PyTorch Accuracy:", np.mean(pt_acc))
print("Average PyTorch Balanced Accuracy:", np.mean(pt_bal_acc))
