import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from imblearn.over_sampling import SMOTE
import tkinter as tk
from tkinter import filedialog, messagebox
import pickle
import os

# Вибір CSV 
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(title="Оберіть CSV файл для тренування", filetypes=[("CSV files", "*.csv")])
if not file_path:
    messagebox.showerror("Помилка", "Файл не обрано. Завершення.")
    exit()
df = pd.read_csv(file_path)
print(f"Файл {os.path.basename(file_path)} завантажено. Розмір: {df.shape}")

#Підготовка даних
features = ["amt", "lat", "long", "merch_lat", "merch_long", "city_pop"]
target = "is_fraud"

X = df[features].values
y = df[target].values if target in df.columns else None

# Розділення на train/test/val 
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Масштабування
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_val_scaled = scaler.transform(X_val)

# Обробка дисбалансу (лише train)
if y_train is not None and len(np.unique(y_train)) > 1 and np.mean(y_train) < 0.05:
    smote = SMOTE(sampling_strategy=0.1, random_state=42)
    X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
else:
    print(f"SMOTE пропущено: класи у train={np.unique(y_train)}")

#Isolation Forest
print("Training isolation_forest...")
iso_model = IsolationForest(n_estimators=200, contamination=0.01, random_state=42)
iso_model.fit(X_train_scaled)
with open("isolation_forest_model.pkl", "wb") as f:
    pickle.dump(iso_model, f)

#Autoencoder 
print("Training autoencoder...")
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
dataset = TensorDataset(X_train_tensor)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

class AutoEncoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 32), nn.ReLU(), nn.Linear(32, 8))
        self.decoder = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, input_dim))
    def forward(self, x):
        return self.decoder(self.encoder(x))

ae_model = AutoEncoder(X_train_scaled.shape[1])
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(ae_model.parameters(), lr=1e-3)
epochs = 10

for epoch in range(epochs):
    for batch, in loader:
        optimizer.zero_grad()
        recon = ae_model(batch)
        loss = criterion(recon, batch)
        loss.backward()
        optimizer.step()

torch.save(ae_model.state_dict(), "autoencoder_model.pth")

# One-Class SVM
print("Training one_class_svm...")
subsample_size = min(20000, len(X_train_scaled))
idx = np.random.choice(len(X_train_scaled), subsample_size, replace=False)
X_sub = X_train_scaled[idx]

svm_model = OneClassSVM(nu=0.01, kernel='rbf', gamma='scale')
svm_model.fit(X_sub)
with open("one_class_svm_model.pkl", "wb") as f:
    pickle.dump(svm_model, f)

#LOF
print("Training lof...")
lof_model = LocalOutlierFactor(n_neighbors=20, novelty=True)
lof_model.fit(X_sub)
with open("lof_model.pkl", "wb") as f:
    pickle.dump(lof_model, f)

#Збереження scaler
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Training complete. Моделі та scaler збережено.")
print(f"Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}, Val: {X_val_scaled.shape}")