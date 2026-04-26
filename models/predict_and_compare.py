import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os

DATA_PATH = 'C:/Git projects/ObjectRegconition/data/raw/ecg.csv'
MODEL_PATH = 'C:/Git projects/ObjectRegconition/autoencoder_model.pth'
OUTPUT_PATH = 'C:/Git projects/ObjectRegconition/predictions_results.csv'

class AnomalyDetector(nn.Module):
    def __init__(self, input_dim=140):
        super(AnomalyDetector, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def load_and_preprocess_data(path):
    dataframe = pd.read_csv(path, header=None)
    raw_data = dataframe.values
    labels = raw_data[:, -1]
    data = raw_data[:, 0:-1]
    
    train_data, test_data, train_labels, test_labels = train_test_split(
        data, labels, test_size=0.2, random_state=21
    )
    
    min_val = np.min(train_data)
    max_val = np.max(train_data)
    
    train_data = (train_data - min_val) / (max_val - min_val)
    test_data = (test_data - min_val) / (max_val - min_val)
    
    return train_data.astype(np.float32), test_data.astype(np.float32), train_labels, test_labels

def train_and_save_model(train_data, epochs=2000, batch_size=512, learning_rate=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = AnomalyDetector(input_dim=train_data.shape[1]).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_tensor = torch.tensor(train_data)
    dataset = TensorDataset(train_tensor, train_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"Training autoencoder on {len(train_data)} normal samples...")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_data, batch_target in dataloader:
            batch_data = batch_data.to(device)
            batch_target = batch_target.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(dataloader):.6f}")
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': train_data.shape[1]
    }, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    
    return model, device

def load_model():
    checkpoint = torch.load(MODEL_PATH)
    input_dim = checkpoint['input_dim']
    model = AnomalyDetector(input_dim=input_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    return model, device

def predict(model, test_data, device):
    model.eval()
    test_tensor = torch.tensor(test_data).to(device)
    
    with torch.no_grad():
        reconstructions = model(test_tensor).cpu().numpy()
    
    reconstruction_error = np.mean(np.abs(test_data - reconstructions), axis=1)
    
    return reconstructions, reconstruction_error

def main():
    print("="*60)
    print("ECG Anomaly Detection - Autoencoder (PyTorch)")
    print("="*60)
    
    print("\nLoading data...")
    train_data, test_data, train_labels, test_labels = load_and_preprocess_data(DATA_PATH)
    
    is_normal_train = train_labels == 1.0
    is_normal_test = test_labels == 1.0
    
    train_normal = train_data[is_normal_train]
    
    print(f"Train samples: {len(train_data)} (Normal: {len(train_normal)})")
    print(f"Test samples: {len(test_data)}")
    
    if not os.path.exists(MODEL_PATH):
        print("\nTraining model...")
        model, device = train_and_save_model(train_normal, epochs=50)
    else:
        print("\nLoading trained model...")
        model, device = load_model()
    
    print("\nMaking predictions...")
    reconstructions, errors = predict(model, test_data, device)
    
    normal_errors = errors[is_normal_test]
    abnormal_errors = errors[~is_normal_test]
    
    threshold = np.mean(normal_errors) + 2 * np.std(normal_errors)
    
    predicted_is_abnormal = errors > threshold
    
    true_is_abnormal = ~is_normal_test
    
    correct = (predicted_is_abnormal == true_is_abnormal).astype(int)
    
    results_df = pd.DataFrame({
        'sample_index': range(len(test_data)),
        'true_label': ['Abnormal' if x else 'Normal' for x in true_is_abnormal],
        'predicted_label': ['Abnormal' if x else 'Normal' for x in predicted_is_abnormal],
        'reconstruction_error': errors,
        'threshold': threshold,
        'correct': correct
    })
    
    results_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nResults saved to {OUTPUT_PATH}")
    
    print("\n" + "="*60)
    print("PREDICTIONS vs TRUE RESULTS")
    print("="*60)
    
    print(f"\n{'Sample':<10} {'True':<12} {'Predicted':<12} {'Error':<12} {'Match'}")
    print("-"*55)
    for i in range(min(20, len(results_df))):
        row = results_df.iloc[i]
        print(f" {i:<8} {row['true_label']:<12} {row['predicted_label']:<12} {row['reconstruction_error']:.6f}   {'Yes' if row['correct'] else 'No'}")
    
    print(f"\n... showing first 20 of {len(results_df)} samples")
    
    acc = accuracy_score(true_is_abnormal, predicted_is_abnormal)
    prec = precision_score(true_is_abnormal, predicted_is_abnormal)
    rec = recall_score(true_is_abnormal, predicted_is_abnormal)
    f1 = f1_score(true_is_abnormal, predicted_is_abnormal)
    
    cm = confusion_matrix(true_is_abnormal, predicted_is_abnormal)
    
    print("\n" + "="*60)
    print("METRICS")
    print("="*60)
    
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"                      Predicted")
    print(f"                 Normal    Abnormal")
    print(f"Actual Normal    {cm[0,0]:5d}    {cm[0,1]:5d}")
    print(f"       Abnormal {cm[1,0]:5d}    {cm[1,1]:5d}")
    
    print("\n" + "="*60)
    print("ERROR COMPARISON")
    print("="*60)
    print(f"\nNormal (true):   Mean={np.mean(normal_errors):.6f}, Std={np.std(normal_errors):.6f}")
    print(f"Abnormal (true): Mean={np.mean(abnormal_errors):.6f}, Std={np.std(abnormal_errors):.6f}")
    print(f"\nThreshold: {threshold:.6f}")

if __name__ == "__main__":
    main()