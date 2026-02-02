"""
Calculate average metrics from experimental results.
"""

import pandas as pd

# Read CSV files
df_ver1 = pd.read_csv('results/metrics_ver1.csv')
df_ver2 = pd.read_csv('results/metrics_ver2.csv')

# Calculate averages
print("=" * 60)
print("実験結果の平均値")
print("=" * 60)

print("\nnonpara30w_ver1 (疑似ささやき声 Ver.1):")
print(f"  MCD:     {df_ver1['MCD_dB'].mean():.2f} dB")
print(f"  Mel-MSE: {df_ver1['Mel_MSE'].mean():.4f}")

print("\nnonpara30w_ver2 (疑似ささやき声 Ver.2):")
print(f"  MCD:     {df_ver2['MCD_dB'].mean():.2f} dB")
print(f"  Mel-MSE: {df_ver2['Mel_MSE'].mean():.4f}")

# Calculate improvement
mcd_improvement = ((df_ver1['MCD_dB'].mean() - df_ver2['MCD_dB'].mean()) / 
                   df_ver1['MCD_dB'].mean() * 100)
mse_improvement = ((df_ver1['Mel_MSE'].mean() - df_ver2['Mel_MSE'].mean()) / 
                   df_ver1['Mel_MSE'].mean() * 100)

print("\n" + "=" * 60)
print("Ver.2 の改善率 (Ver.1 と比較):")
print(f"  MCD改善:     {mcd_improvement:.2f}%")
print(f"  Mel-MSE改善: {mse_improvement:.2f}%")
print("=" * 60)
