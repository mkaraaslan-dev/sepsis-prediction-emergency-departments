import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer, KNNImputer

import warnings
warnings.filterwarnings('ignore')

# --- MGP LIBRARY CHECK ---
try:
    from mgp import MGPImputer
    USING_MGP = True
    print("✅ MGP Library Loaded.")
except ImportError:
    USING_MGP = False
    print("⚠️ WARNING: MGP Library not found! MGP results will be 0 in simulation.")

# --- 1. DATA LOADING ---
# Filenames remain the same to match your local files
FILE_POS = "sepsis pozitif grup_e.xlsx"
FILE_NEG = "sepsis negatif grup_e.xlsx"

try:
    df_pos = pd.read_excel(FILE_POS)
    df_neg = pd.read_excel(FILE_NEG)
    df = pd.concat([df_pos, df_neg], ignore_index=True)
except FileNotFoundError:
    print("❌ Error: Excel files not found. Please check file paths.")
    exit()

# Column names must match the Excel headers (Turkish headers kept for reading)
target_cols = [
    'YAŞ', 'NABIZ', 'SKB', 'DKB', 'OAB', 'SOLUNUM SAYISI', 'ATEŞ',
    'SATURASYON', 'GKS', 'LAKTAT', 'PH', 'PCO2', 'BE', 'WBC',
    'PLT', 'GLUKOZ', 'KRE', 'ÜRE', 'T.BİL', 'D.BİL', 'CİNSİYET_KODLU'
]

# --- 2. DATA CLEANING ---
if 'CİNSİYET' in df.columns:
    # E/K (Turkish) -> M/F (English) mapping logic included
    df['CİNSİYET_KODLU'] = df['CİNSİYET'].astype(str).str.upper().map({'E': 0, 'K': 1, 'M': 0, 'F': 1})

def strict_clean(val):
    if pd.isna(val):
        return np.nan
    # Replace comma with dot for decimals
    val_str = str(val).strip().replace(',', '.')
    try:
        return float(val_str)
    except ValueError:
        return np.nan

df_clean = df.copy()
valid_cols = [c for c in target_cols if c in df_clean.columns]

for col in valid_cols:
    df_clean[col] = df_clean[col].apply(strict_clean)

# --- 3. GOLD STANDARD ---
df_gold = df_clean[valid_cols].dropna()
print(f"Total Rows: {len(df)}")
print(f"Gold Standard (Clean) Rows: {len(df_gold)}")

if len(df_gold) < 20:
    raise ValueError("❌ Not enough clean data for Gold Standard testing.")

# --- 4. MASKING (Creating Artificial Missingness) ---
X_true = df_gold.values
np.random.seed(42)
# 20% Missing Data
mask = np.random.rand(*X_true.shape) < 0.20
X_corrupted = X_true.copy()
X_corrupted[mask] = np.nan

# --- 5. MODELS (MICE REMOVED) ---
imputers = {
    'Mean': SimpleImputer(strategy='mean'),
    'Median': SimpleImputer(strategy='median'),
    'KNN': KNNImputer(n_neighbors=5)
}

if USING_MGP:
    # Renamed to English
    imputers['MGP (Proposed)'] = MGPImputer(
        n_inducing_points=50,
        n_iterations=300,
        verbose=False
    )

scores_rmse = {}
scores_mae = {}
scores_r2 = {}

print("\n" + "="*60)
print(f"{'MODEL':<20} | {'RMSE':<10} | {'MAE':<10} | {'R²':<8}")
print("="*60)

# --- 6. EXECUTION LOOP ---
for name, model in imputers.items():
    try:
        if "MGP" in name:
            X_imputed, _ = model.fit_transform(X_corrupted)
        else:
            X_imputed = model.fit_transform(X_corrupted)

        # Evaluate only on the masked (missing) parts
        y_true = X_true[mask]
        y_pred = X_imputed[mask]

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        scores_rmse[name] = rmse
        scores_mae[name] = mae
        scores_r2[name] = r2

        print(f"{name:<20} | {rmse:<10.4f} | {mae:<10.4f} | {r2:<8.4f}")

    except Exception as e:
        print(f"{name:<20} | ERROR: {e}")

# --- 7. RESULTS TABLE ---
df_results = pd.DataFrame({
    'Model': list(scores_rmse.keys()),
    'RMSE': list(scores_rmse.values()),
    'MAE': list(scores_mae.values()),
    'R2': list(scores_r2.values())
}).sort_values(by='RMSE')

print("\n--- COMPARATIVE RESULTS ---")
print(df_results)

best_model = df_results.iloc[0]['Model']
best_rmse = df_results.iloc[0]['RMSE']
worst_rmse = df_results.iloc[-1]['RMSE']

improvement = ((worst_rmse - best_rmse) / worst_rmse) * 100

print(f"\n🏆 BEST MODEL: {best_model}")
print(f"📉 RMSE Improvement over worst model: {improvement:.2f}%")

# --- 8. VISUALIZATION (ENGLISH) ---
plt.figure(figsize=(10, 6))

bar_width = 0.25
index = np.arange(len(df_results))

# Plotting bars
plt.bar(index, df_results['RMSE'], bar_width, label='RMSE', color='#4c72b0')
plt.bar(index + bar_width, df_results['MAE'], bar_width, label='MAE', color='#55a868')
plt.bar(index + 2 * bar_width, df_results['R2'], bar_width, label='R²', color='#c44e52')

# Labels and Title (English)
plt.xlabel('Imputation Methods', fontsize=12)
plt.ylabel('Score Value', fontsize=12)
plt.title('Comparison of Missing Data Imputation Methods', fontsize=14)
plt.xticks(index + bar_width, df_results['Model'], rotation=15)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()
output_img = "Imputation_Comparison_English_NoMICE.png"
plt.savefig(output_img, dpi=300)
print(f"\n📊 Chart saved: {output_img}")

# --- 9. EXPORT ---
output_excel = "Imputation_Results_English.xlsx"
df_results.to_excel(output_excel, index=False)
print(f"📁 Results table saved: {output_excel}")