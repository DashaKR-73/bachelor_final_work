import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score

# Precision/Recall @ Threshold 
def precision_recall_at_k(y_true, scores, threshold=0.95):
    y_pred = (scores >= threshold).astype(int)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    return precision, recall, y_pred

# Логування
class Logger:
    def __init__(self):
        self.logs = []

    def log(self, method, transaction_idx, score, reason=""):
        self.logs.append({
            "method": method,
            "transaction_idx": transaction_idx,
            "score": score,
            "reason": reason
        })

    def get_logs(self):
        return pd.DataFrame(self.logs)

logger = Logger()

#FraudDetector 
class FraudDetector:
    def __init__(self, X_train, y_train, X_val, y_val, threshold=0.95, batch_size=5000):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.threshold = threshold
        self.batch_size = batch_size

        # Масштабування на train
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_val_scaled = self.scaler.transform(self.X_val)

    #  Isolation Forest
    def isolation_forest(self):
        from sklearn.ensemble import IsolationForest
        model = IsolationForest(n_estimators=200, contamination=0.01, random_state=42)
        model.fit(self.X_train_scaled)
        scores = -model.score_samples(self.X_val_scaled)
        scores = (scores - scores.min()) / (scores.max() - scores.min())
        top_idx = np.argsort(scores)[::-1][:5]
        for i in top_idx:
            logger.log("Isolation Forest", i, scores[i], reason="High anomaly score")
        return model, scores

    # Autoencoder
    def autoencoder(self):
        X_tensor = torch.tensor(self.X_train_scaled, dtype=torch.float32)
        val_tensor = torch.tensor(self.X_val_scaled, dtype=torch.float32)
        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)

        class AutoEncoder(nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                self.encoder = nn.Sequential(nn.Linear(input_dim, 32), nn.ReLU(), nn.Linear(32, 8))
                self.decoder = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, input_dim))
            def forward(self, x):
                return self.decoder(self.encoder(x))

        model = AutoEncoder(X_tensor.shape[1])
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        epochs = 10
        for epoch in range(epochs):
            for (batch,) in loader:
                optimizer.zero_grad()
                recon = model(batch)
                loss = criterion(recon, batch)
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            recon_val = model(val_tensor)
            scores = torch.mean((val_tensor - recon_val) ** 2, dim=1).numpy()
            scores = (scores - scores.min()) / (scores.max() - scores.min())
        top_idx = np.argsort(scores)[::-1][:5]
        for i in top_idx:
            logger.log("Autoencoder", i, scores[i], reason="High reconstruction error")
        return model, scores

    #  One-Class SVM 
    def one_class_svm(self):
        from sklearn.svm import OneClassSVM
        subsample_size = min(20000, len(self.X_train_scaled))
        idx = np.random.choice(len(self.X_train_scaled), subsample_size, replace=False)
        X_sub = self.X_train_scaled[idx]
        model = OneClassSVM(nu=0.01, kernel='rbf', gamma='scale')
        model.fit(X_sub)
        scores = -model.decision_function(self.X_val_scaled)
        scores = (scores - scores.min()) / (scores.max() - scores.min())
        top_idx = np.argsort(scores)[::-1][:5]
        for i in top_idx:
            logger.log("One-Class SVM", i, scores[i], reason="High anomaly score")
        return model, scores

    # LOF
    def lof(self):
        from sklearn.neighbors import LocalOutlierFactor
        subsample_size = min(20000, len(self.X_train_scaled))
        idx = np.random.choice(len(self.X_train_scaled), subsample_size, replace=False)
        X_sub = self.X_train_scaled[idx]
        model = LocalOutlierFactor(n_neighbors=20, novelty=True)
        model.fit(X_sub)
        scores = -model.decision_function(self.X_val_scaled)
        scores = (scores - scores.min()) / (scores.max() - scores.min())
        top_idx = np.argsort(scores)[::-1][:5]
        for i in top_idx:
            logger.log("LOF", i, scores[i], reason="High anomaly score")
        return model, scores

#GUI
class App:
    def __init__(self, root):
        self.root = root
        root.title("Система виявлення шахрайських транзакцій")
        root.geometry("900x800")
        self.df = None

        tk.Button(root, text="📂 Завантажити CSV-файл", command=self.load_csv).pack(pady=5)
        self.method = ttk.Combobox(root, values=["Isolation Forest","Autoencoder","One-Class SVM","LOF"], state="readonly", width=30)
        self.method.set("Isolation Forest")
        self.method.pack(pady=5)
        tk.Button(root, text="▶ Запустити аналіз", command=self.run).pack(pady=5)

        self.progress = ttk.Progressbar(root, orient='horizontal', length=600, mode='determinate')
        self.progress.pack(pady=5)

        tk.Label(root, text="Threshold (0-1)").pack()
        self.threshold_var = tk.DoubleVar(value=0.95)
        tk.Entry(root, textvariable=self.threshold_var, width=10).pack()

        self.output = tk.Text(root, height=15, width=105)
        self.output.pack(pady=5)

        self.figure_frame = tk.Frame(root)
        self.figure_frame.pack(fill='both', expand=True)

    def load_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV файли","*.csv")])
        if path:
            self.df = pd.read_csv(path)
            messagebox.showinfo("Готово", "Датасет успішно завантажено")

    def update_progress(self, value):
        self.progress['value'] = value
        self.root.update_idletasks()

    def run(self):
        if self.df is None:
            messagebox.showerror("Помилка", "Спочатку завантажте датасет")
            return

        threshold = self.threshold_var.get()
        features = ["amt"]  # Обираємо відповідні числові колонки
        X = self.df[features].values
        y = self.df["is_fraud"].values if "is_fraud" in self.df.columns else np.zeros(len(self.df))

        # Розділяємо на train/test/val
        from sklearn.model_selection import train_test_split
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y if len(np.unique(y))>1 else None)
        X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp if len(np.unique(y_temp))>1 else None)

        detector = FraudDetector(X_train, y_train, X_val, y_val, threshold=threshold)

        model_name = self.method.get()
        self.output.delete("1.0", tk.END)
        self.output.insert(tk.END, f"Модель: {model_name}\n\n")

        if model_name == "Isolation Forest":
            model, scores = detector.isolation_forest()
        elif model_name == "Autoencoder":
            model, scores = detector.autoencoder()
        elif model_name == "One-Class SVM":
            model, scores = detector.one_class_svm()
        elif model_name == "LOF":
            model, scores = detector.lof()

        # Precision/Recall
        p, r, y_pred = precision_recall_at_k(y_val, scores, threshold=threshold)
        self.output.insert(tk.END, f"Threshold: {threshold:.4f}\nPrecision: {p:.4f}\nRecall: {r:.4f}\n")
        self.output.insert(tk.END, f"Top-підозрілі транзакції: {np.sum(y_pred)}\n\n")

        # Графік
        for widget in self.figure_frame.winfo_children():
            widget.destroy()
        fig, ax = plt.subplots(figsize=(8,5))
        ax.hist(scores[y_val==0], bins=60, alpha=0.6, label="Нормальні")
        ax.hist(scores[y_val==1], bins=60, alpha=0.9, label="Шахрайські")
        # Top anomalies
        top_idx = np.argsort(scores)[::-1][:5]
        ax.hist(scores[top_idx], bins=60, color='red', alpha=0.7, label='Top anomalies')
        # Threshold line
        ax.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.4f})')
        ax.set_yscale("log")
        ax.set_title(f"Аномалійний скор ({model_name})")
        ax.set_xlabel("Score")
        ax.set_ylabel("Кількість (лог)")
        ax.legend()
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.figure_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

        # Лог
        self.output.insert(tk.END, "\nЛог топ-5 підозрілих транзакцій:\n")
        logs_df = logger.get_logs()
        self.output.insert(tk.END, logs_df.head().to_string(index=False))

# Запуск GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
