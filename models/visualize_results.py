import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

RESULTS_PATH = 'C:/Git projects/ObjectRegconition/predictions_results.csv'

def visualize_results():
    df = pd.read_csv(RESULTS_PATH)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('ECG Autoencoder - Prediction Results Visualization', fontsize=14, fontweight='bold')
    
    ax1 = axes[0, 0]
    normal_errors = df[df['true_label'] == 'Normal']['reconstruction_error']
    abnormal_errors = df[df['true_label'] == 'Abnormal']['reconstruction_error']
    
    ax1.hist(normal_errors, bins=40, alpha=0.7, label='Normal (True)', color='green', edgecolor='darkgreen')
    ax1.hist(abnormal_errors, bins=40, alpha=0.7, label='Abnormal (True)', color='red', edgecolor='darkred')
    threshold = df['threshold'].iloc[0]
    ax1.axvline(threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold={threshold:.4f}')
    ax1.set_xlabel('Reconstruction Error', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title('Distribution of Reconstruction Errors', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[0, 1]
    cm = confusion_matrix(df['true_label'], df['predicted_label'], labels=['Normal', 'Abnormal'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Abnormal'])
    disp.plot(ax=ax2, cmap='Blues', values_format='d')
    ax2.set_title('Confusion Matrix', fontsize=12)
    
    ax3 = axes[1, 0]
    correct_normal = ((df['true_label'] == 'Normal') & (df['predicted_label'] == 'Normal')).sum()
    correct_abnormal = ((df['true_label'] == 'Abnormal') & (df['predicted_label'] == 'Abnormal')).sum()
    wrong_normal_wrong = ((df['true_label'] == 'Normal') & (df['predicted_label'] == 'Abnormal')).sum()
    wrong_abnormal_wrong = ((df['true_label'] == 'Abnormal') & (df['predicted_label'] == 'Normal')).sum()
    
    categories = ['Correct\nNormal', 'Correct\nAbnormal', 'Wrong\n(Normal→Abn)', 'Wrong\n(Abn→Normal)']
    values = [correct_normal, correct_abnormal, wrong_normal_wrong, wrong_abnormal_wrong]
    colors_bar = ['green', 'orange', 'red', 'darkred']
    
    bars = ax3.bar(categories, values, color=colors_bar, edgecolor='black', alpha=0.8)
    ax3.set_ylabel('Count', fontsize=11)
    ax3.set_title('Prediction Results Breakdown', fontsize=12)
    
    for bar, val in zip(bars, values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, str(val), 
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax4 = axes[1, 1]
    samples = range(50)
    threshold = df['threshold'].iloc[0]
    
    for i, (_, row) in enumerate(df.head(50).iterrows()):
        color = 'green' if row['true_label'] == 'Normal' else 'red'
        ax4.bar(i, row['reconstruction_error'], color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    ax4.axhline(threshold, color='blue', linestyle='--', linewidth=2, label=f'Threshold')
    ax4.set_xlabel('Sample Index', fontsize=11)
    ax4.set_ylabel('Reconstruction Error', fontsize=11)
    ax4.set_title('Error by Sample (First 50)\nGreen=Actual Normal, Red=Actual Abnormal', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('C:/Git projects/ObjectRegconition/predictions_visualization.png', dpi=150, bbox_inches='tight')
    print(f"Visualization saved to predictions_visualization.png")
    plt.close()

def main():
    df = pd.read_csv(RESULTS_PATH)
    
    total = len(df)
    correct_normal = ((df['true_label'] == 'Normal') & (df['predicted_label'] == 'Normal')).sum()
    correct_abnormal = ((df['true_label'] == 'Abnormal') & (df['predicted_label'] == 'Abnormal')).sum()
    wrong_normal = ((df['true_label'] == 'Normal') & (df['predicted_label'] == 'Abnormal')).sum()
    wrong_abnormal = ((df['true_label'] == 'Abnormal') & (df['predicted_label'] == 'Normal')).sum()
    
    accuracy = (correct_normal + correct_abnormal) / total * 100
    
    print("="*60)
    print("PREDICTION RESULTS SUMMARY")
    print("="*60)
    print(f"\nTotal samples: {total}")
    print(f"Correct predictions: {correct_normal + correct_abnormal}")
    print(f"Incorrect predictions: {wrong_normal + wrong_abnormal}")
    print(f"Accuracy: {accuracy:.2f}%")
    
    print("\n" + "-"*50)
    print("CONFUSION MATRIX")
    print("-"*50)
    print(f"\n                 Predicted")
    print(f"                 Normal    Abnormal")
    print(f"Actual Normal    {correct_normal:5d}    {wrong_normal:5d}")
    print(f"       Abnormal {wrong_abnormal:5d}    {correct_abnormal:5d}")
    
    print("\n" + "-"*50)
    print("ERROR STATISTICS")
    print("-"*50)
    normal_errors = df[df['true_label'] == 'Normal']['reconstruction_error']
    abnormal_errors = df[df['true_label'] == 'Abnormal']['reconstruction_error']
    
    print(f"Normal samples:   Mean={normal_errors.mean():.6f}, Std={normal_errors.std():.6f}")
    print(f"Abnormal samples: Mean={abnormal_errors.mean():.6f}, Std={abnormal_errors.std():.6f}")
    print(f"\nThreshold used: {df['threshold'].iloc[0]:.6f}")
    
    visualize_results()

if __name__ == "__main__":
    main()