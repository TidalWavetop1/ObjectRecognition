import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

DATA_PATH = 'C:/Git projects/ObjectRegconition/data/raw/ecg.csv'
MODEL_PATH = 'C:/Git projects/ObjectRegconition/autoencoder_model.pth'

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

def load_data():
    dataframe = pd.read_csv(DATA_PATH, header=None)
    raw_data = dataframe.values
    labels = raw_data[:, -1]
    data = raw_data[:, 0:-1]
    
    _, test_data, _, test_labels = train_test_split(
        data, labels, test_size=0.2, random_state=21
    )
    
    min_val = np.min(test_data) if len(test_data) > 0 else 0
    max_val = np.max(test_data) if len(test_data) > 0 else 1
    test_data = (test_data - min_val) / (max_val - min_val)
    
    return test_data.astype(np.float32), test_labels

def load_model():
    checkpoint = torch.load(MODEL_PATH)
    input_dim = checkpoint['input_dim']
    model = AnomalyDetector(input_dim=input_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    return model, device

def visualize_ecg_comparison(test_data, test_labels):
    model, device = load_model()
    
    test_tensor = torch.tensor(test_data).to(device)
    
    with torch.no_grad():
        reconstructions = model(test_tensor).cpu().numpy()
    
    test_labels = test_labels == 1.0
    
    normal_indices = np.where(test_labels)[0][:3]
    abnormal_indices = np.where(~test_labels)[0][:3]
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle('ECG Signal Comparison: Original vs Reconstructed (Autoencoder)', fontsize=14, fontweight='bold')
    
    colors_original = ['green', 'blue', 'purple']
    colors_recon = ['lightgreen', 'lightskyblue', 'violet']
    
    for i, idx in enumerate(normal_indices):
        ax = axes[i, 0]
        original = test_data[idx]
        reconstructed = reconstructions[idx]
        
        ax.plot(original, color='green', alpha=0.7, linewidth=1.5, label='Original')
        ax.plot(reconstructed, color='red', alpha=0.7, linewidth=1.5, linestyle='--', label='Reconstructed')
        ax.set_title(f'Normal Sample #{idx} (Error: {np.mean(np.abs(original - reconstructed)):.6f})', fontsize=10)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Amplitude')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    for i, idx in enumerate(abnormal_indices):
        ax = axes[i, 1]
        original = test_data[idx]
        reconstructed = reconstructions[idx]
        
        ax.plot(original, color='blue', alpha=0.7, linewidth=1.5, label='Original')
        ax.plot(reconstructed, color='red', alpha=0.7, linewidth=1.5, linestyle='--', label='Reconstructed')
        ax.set_title(f'Abnormal Sample #{idx} (Error: {np.mean(np.abs(original - reconstructed)):.6f})', fontsize=10)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Amplitude')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('C:/Git projects/ObjectRegconition/ecg_comparison.png', dpi=150, bbox_inches='tight')
    print(f"ECG comparison saved to ecg_comparison.png")
    plt.close()

def visualize_reconstruction_errors(test_data, test_labels):
    model, device = load_model()
    
    test_tensor = torch.tensor(test_data).to(device)
    
    with torch.no_grad():
        reconstructions = model(test_tensor).cpu().numpy()
    
    errors = np.mean(np.abs(test_data - reconstructions), axis=1)
    test_labels = test_labels == 1.0
    
    threshold = np.mean(errors[test_labels]) + 2 * np.std(errors[test_labels])
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Reconstruction Error Analysis', fontsize=14, fontweight='bold')
    
    ax1 = axes[0, 0]
    normal_errors = errors[test_labels]
    abnormal_errors = errors[~test_labels]
    
    ax1.hist(normal_errors, bins=30, alpha=0.7, label='Normal', color='green', edgecolor='darkgreen')
    ax1.hist(abnormal_errors, bins=30, alpha=0.7, label='Abnormal', color='red', edgecolor='darkred')
    ax1.axvline(threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold={threshold:.4f}')
    ax1.set_xlabel('Reconstruction Error')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Error Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[0, 1]
    indices = range(100)
    colors = ['green' if l else 'red' for l in test_labels[:100]]
    ax2.bar(indices, errors[:100], color=colors, alpha=0.7)
    ax2.axhline(threshold, color='black', linestyle='--', linewidth=2, label='Threshold')
    ax2.set_xlabel('Sample Index')
    ax2.set_ylabel('Reconstruction Error')
    ax2.set_title('Error per Sample (First 100)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[1, 0]
    box_data = [normal_errors, abnormal_errors]
    bp = ax3.boxplot(box_data, tick_labels=['Normal', 'Abnormal'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightgreen')
    bp['boxes'][1].set_facecolor('lightcoral')
    ax3.axhline(threshold, color='black', linestyle='--', linewidth=2, label='Threshold')
    ax3.set_ylabel('Reconstruction Error')
    ax3.set_title('Error Box Plot')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    ax4 = axes[1, 1]
    pred_abnormal = errors > threshold
    true_abnormal = ~test_labels
    
    correct_normal = np.sum((~pred_abnormal) & (~true_abnormal))
    wrong_normal = np.sum(pred_abnormal & (~true_abnormal))
    correct_abnormal = np.sum(pred_abnormal & true_abnormal)
    wrong_abnormal = np.sum((~pred_abnormal) & true_abnormal)
    
    categories = ['Correct\nNormal', 'Correct\nAbnormal', 'Wrong\n(Norm→Abn)', 'Wrong\n(Abn→Norm)']
    values = [correct_normal, correct_abnormal, wrong_normal, wrong_abnormal]
    colors_bar = ['green', 'orange', 'red', 'darkred']
    
    bars = ax4.bar(categories, values, color=colors_bar, edgecolor='black', alpha=0.8)
    ax4.set_ylabel('Count')
    ax4.set_title('Prediction Breakdown')
    
    for bar, val in zip(bars, values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3, str(val), 
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('C:/Git projects/ObjectRegconition/error_analysis.png', dpi=150, bbox_inches='tight')
    print(f"Error analysis saved to error_analysis.png")
    plt.close()

def main():
    print("Loading ECG data and model...")
    test_data, test_labels = load_data()
    print(f"Test samples: {len(test_data)}")
    
    print("\nGenerating ECG comparison plots...")
    visualize_ecg_comparison(test_data, test_labels)
    
    print("\nGenerating error analysis plots...")
    visualize_reconstruction_errors(test_data, test_labels)
    
    print("\nDone!")

if __name__ == "__main__":
    main()