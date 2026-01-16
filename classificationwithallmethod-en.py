import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import json
import warnings
import xgboost as xgb
import os

# Sklearn Tools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

# Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, LSTM, Multiply, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# ============================================================
# 1. SETTINGS
# ============================================================
print("--- STEP 5: COMPREHENSIVE SET-BASED PERFORMANCE ANALYSIS ---")

# File Paths
RAW_FILE_POS = "sepsis pozitif grup_e.xlsx"
RAW_FILE_NEG = "sepsis negatif grup_e.xlsx"
MGP_FILE_POS = "sepsis_pozitif_MGP_dolu.xlsx"
MGP_FILE_NEG = "sepsis_negatif_MGP_dolu.xlsx"
OUTPUT_FILE = "Set_Based_Detailed_Comparison.xlsx"

# Load Feature Sets
try:
    with open("feature_sets.json", "r") as f:
        FEATURE_SETS = json.load(f)
    print("Feature sets loaded successfully.")
except:
    print("WARNING: feature_sets.json not found. Please run the Feature Selection step first.")
    exit()

# Numeric Columns (Turkish names in file, mapped later if needed, kept as logic for extraction)
NUMERIC_COLS = [
    'YAŞ', 'NABIZ', 'SKB', 'DKB', 'OAB', 'SOLUNUM SAYISI', 
    'ATEŞ', 'SATURASYON', 'GKS', 'LAKTAT', 'PH', 'PCO2', 'BE', 
    'WBC', 'PLT', 'GLUKOZ', 'KRE', 'ÜRE', 'T.BİL', 'D.BİL', 
    'CİNSİYET_KODLU'
]

# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def load_data(is_mgp=False):
    """Loads data. Returns filled files if MGP is True, else raw files."""
    try:
        if is_mgp:
            f_pos, f_neg = MGP_FILE_POS, MGP_FILE_NEG
        else:
            f_pos, f_neg = RAW_FILE_POS, RAW_FILE_NEG
            
        df_p = pd.read_excel(f_pos)
        df_n = pd.read_excel(f_neg)
        df_p['TARGET'] = 1
        df_n['TARGET'] = 0
        df = pd.concat([df_p, df_n], ignore_index=True)
        
        # Cleaning & Preprocessing
        if 'CİNSİYET' in df.columns:
            # Encoding Gender: Male/Erkek=0, Female/Kadın=1
            df['CİNSİYET_KODLU'] = df['CİNSİYET'].astype(str).str.upper().map({'E': 0, 'K': 1, 'M': 0, 'F': 1})
        
        for col in NUMERIC_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
        
        # If MGP, fill NaNs with 0 (files should be full, this is a safety net)
        if is_mgp:
            df = df.fillna(0)
            
        return df
    except Exception as e:
        print(f"Data loading error (MGP={is_mgp}): {e}")
        return None

def get_imputed_data(method):
    """Prepares the dataframe based on the selected imputation method."""
    if method == 'MGP':
        return load_data(is_mgp=True)
    
    # Load raw data for other methods
    df = load_data(is_mgp=False)
    if df is None: return None
    
    # Select only numeric columns present
    cols = [c for c in NUMERIC_COLS if c in df.columns]
    X = df[cols].values
    
    if method == 'Mean': imp = SimpleImputer(strategy='mean')
    elif method == 'Median': imp = SimpleImputer(strategy='median')
    elif method == 'KNN': imp = KNNImputer(n_neighbors=5)
    # MICE removed as requested
    
    X_filled = imp.fit_transform(X)
    
    df_filled = pd.DataFrame(X_filled, columns=cols)
    df_filled['TARGET'] = df['TARGET'].values # Add target back
    return df_filled

def create_dl_model(model_type, input_shape):
    """Deep Learning Model Architectures"""
    inputs = Input(shape=input_shape)
    if model_type == 'CNN':
        x = Conv1D(32, 2, activation='relu', padding='same')(inputs)
        x = MaxPooling1D(2)(x)
        x = Flatten()(x)
        x = Dense(32, activation='relu')(x)
    elif model_type == 'LSTM':
        x = LSTM(32)(inputs)
        x = Dropout(0.2)(x)
    elif model_type == 'Attention':
        # Self-Attention Mechanism
        features = Dense(32, activation='relu')(inputs)
        att_probs = Dense(input_shape[1], activation='softmax')(Flatten()(inputs))
        att_probs = Reshape((input_shape[1], 1))(att_probs)
        att_mul = Multiply()([inputs, att_probs])
        x = Flatten()(att_mul)
        x = Dense(32, activation='relu')(x)
        
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ============================================================
# 3. MAIN LOOP (SET -> IMPUTATION -> MODEL)
# ============================================================

# MICE removed from list
IMPUTATION_METHODS = ['Mean', 'Median', 'KNN', 'MGP']
MODEL_LIST = ['Random Forest', 'SVM', 'XGBoost', 'CNN', 'LSTM', 'Attention']

writer = pd.ExcelWriter(OUTPUT_FILE, engine='xlsxwriter')

# 1. Feature Set Loop
for set_name, features in FEATURE_SETS.items():
    print(f"\n################################################")
    print(f"### PROCESSING SET: {set_name}")
    print(f"################################################")
    
    set_results = []
    
    # 2. Imputation Loop
    for imp_method in IMPUTATION_METHODS:
        print(f"  > Imputation Method: {imp_method}")
        
        df_curr = get_imputed_data(imp_method)
        if df_curr is None: continue
        
        valid_features = [f for f in features if f in df_curr.columns]
        if not valid_features:
            print("    ! No valid columns found for this set.")
            continue
            
        X = df_curr[valid_features].values
        y = df_curr['TARGET'].values
        
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Reshape for DL models
        X_train_dl = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test_dl = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        
        # 3. Model Loop
        for model_name in MODEL_LIST:
            try:
                y_pred = []
                y_prob = []
                
                if model_name in ['CNN', 'LSTM', 'Attention']:
                    model = create_dl_model(model_name, (X_train_dl.shape[1], 1))
                    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
                    model.fit(X_train_dl, y_train, epochs=30, batch_size=16, validation_split=0.1, verbose=0, callbacks=[es])
                    y_prob = model.predict(X_test_dl, verbose=0).flatten()
                    y_pred = (y_prob > 0.5).astype(int)
                else:
                    if model_name == 'Random Forest': clf = RandomForestClassifier(n_estimators=100)
                    elif model_name == 'SVM': clf = SVC(probability=True)
                    elif model_name == 'XGBoost': clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', verbosity=0)
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)
                    y_prob = clf.predict_proba(X_test)[:, 1]

                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                try: auroc = roc_auc_score(y_test, y_prob)
                except: auroc = 0.5
                tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
                sens = tp / (tp+fn) if (tp+fn) > 0 else 0
                spec = tn / (tn+fp) if (tn+fp) > 0 else 0
                
                set_results.append({
                    "Imputation": imp_method,
                    "Model": model_name,
                    "AUROC": auroc,
                    "Accuracy": acc,
                    "F1-Score": f1,
                    "Sensitivity": sens,
                    "Specificity": spec
                })
            except Exception as e:
                print(f"    ERROR ({model_name}): {e}")

    df_set = pd.DataFrame(set_results)
    sheet_name = set_name[:30].replace(":", "").replace("/", "_")
    df_set.to_excel(writer, sheet_name=sheet_name, index=False)
    print(f"   -> Sheet '{sheet_name}' added.")

writer.close()
print(f"\nANALYSIS COMPLETE! Results saved to '{OUTPUT_FILE}'.")

# ============================================================
# 6. VISUALIZATION
# ============================================================
print("\n" + "="*50)
print("PREPARING VISUAL COMPARISON CHARTS...")
print("="*50)

# 6.1. Read back the Excel file
all_sheets_dict = pd.read_excel(OUTPUT_FILE, sheet_name=None)
df_list = []
for sheet_name, df_sheet in all_sheets_dict.items():
    df_sheet['Feature Set Source'] = sheet_name
    df_list.append(df_sheet)
df_final_results = pd.concat(df_list, ignore_index=True)

sns.set_theme(style="whitegrid")
METRIC_TO_PLOT = 'AUROC'

# 6.2. Heatmaps (Per Feature Set)
for set_source in df_final_results['Feature Set Source'].unique():
    plt.figure(figsize=(12, 8))
    
    df_subset = df_final_results[df_final_results['Feature Set Source'] == set_source]
    try:
        pivot_table = df_subset.pivot(index='Imputation', columns='Model', values=METRIC_TO_PLOT)
        
        sns.heatmap(pivot_table, annot=True, fmt=".3f", cmap="viridis", linewidths=.5, cbar_kws={'label': METRIC_TO_PLOT})
        plt.title(f'Performance Heatmap ({METRIC_TO_PLOT})\nFeature Set: {set_source}', fontsize=14)
        plt.ylabel('Imputation Method', fontsize=12)
        plt.xlabel('Classification Model', fontsize=12)
        plt.tight_layout()
        
        filename = f"Heatmap_{set_source}.png"
        plt.savefig(filename, dpi=300)
        print(f"Graph saved: {filename}")
        plt.close()
    except Exception as e:
        print(f"Could not create heatmap for ({set_source}): {e}")

# 6.3. Summary Plot
plt.figure(figsize=(14, 8))
sns.boxplot(data=df_final_results, x='Imputation', y=METRIC_TO_PLOT, palette='Set2')
sns.swarmplot(data=df_final_results, x='Imputation', y=METRIC_TO_PLOT, color='black', alpha=0.3, size=4)

plt.title(f"General Performance Distribution by Imputation Method ({METRIC_TO_PLOT})", fontsize=16)
plt.ylabel(METRIC_TO_PLOT, fontsize=14)
plt.xlabel("Imputation Method", fontsize=14)
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

summary_filename = "General_Comparison_Summary_Chart.png"
plt.savefig(summary_filename, dpi=300)
print(f"\nSummary Chart saved: {summary_filename}")

print("\n--- ALL PROCESSES COMPLETED SUCCESSFULLY! ---")