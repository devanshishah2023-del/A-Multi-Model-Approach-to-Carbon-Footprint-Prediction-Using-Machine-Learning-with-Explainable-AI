"""
Carbon Footprint Model Training - MULTI-MODEL ARCHITECTURE COMPARISON

This script tries multiple model architectures and selects the best:
1. Random Forest (tree-based, often best for tabular data)
2. XGBoost (gradient boosting, state-of-the-art for tabular)
3. Simple Neural Network (2 layers)
4. Deep Neural Network (4 layers)
5. Linear Regression (baseline)

Automatically selects and saves the best performing model.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
import kagglehub
import os
import glob
import pickle
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost (install if needed)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Install with: pip install xgboost")

import shap
import matplotlib.pyplot as plt

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device('cpu')

print("=" * 80)
print("MULTI-MODEL ARCHITECTURE COMPARISON")
print("Testing 5 different models to find the best for your data")
print("=" * 80)

# Step 1: Load and prepare data
print("\nStep 1: Loading dataset...")
path = kagglehub.dataset_download("sanjaybora143/carbon-footprint-regression")
csv_file = glob.glob(os.path.join(path, '*.csv'))[0]
df = pd.read_csv(csv_file)
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

print("\nStep 2: Cleaning data...")
df = df.drop_duplicates()
for col in df.select_dtypes(include=[np.number]).columns:
    df[col].fillna(df[col].median(), inplace=True)
for col in df.select_dtypes(include=['object']).columns:
    df[col].fillna(df[col].mode()[0], inplace=True)
print(f"After cleaning: {df.shape[0]} rows")

print("\nStep 3: Identifying target variable...")
target_keywords = ['emission', 'carbon', 'co2']
target_col = None
for col in df.columns:
    if any(kw in col.lower() for kw in target_keywords):
        if df[col].dtype in [np.float64, np.int64]:
            target_col = col
            break
if target_col is None:
    target_col = df.select_dtypes(include=[np.number]).columns[-1]
print(f"Target variable: {target_col}")

print("\nStep 4: Preparing features...")
y = df[target_col].values
X_df = df.drop(columns=[target_col])

# Data validation
print("\nData Statistics:")
print(f"Target range: {y.min():.2f} to {y.max():.2f}")
print(f"Target mean: {y.mean():.2f}, std: {y.std():.2f}")
print(f"Target median: {np.median(y):.2f}")

valid_mask = ~(np.isnan(y) | np.isinf(y))
if not valid_mask.all():
    print(f"Removing {(~valid_mask).sum()} invalid values")
    y = y[valid_mask]
    X_df = X_df[valid_mask]

cat_cols = X_df.select_dtypes(include=['object']).columns.tolist()
num_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
print(f"Categorical features: {len(cat_cols)}")
print(f"Numerical features: {len(num_cols)}")

print("\nStep 5: Encoding categorical features...")
encoders = {}
X = X_df.copy()
for col in cat_cols:
    le = LabelEncoder()
    X[f'{col}_enc'] = le.fit_transform(X_df[col])
    encoders[col] = le

encoded_cats = [f'{col}_enc' for col in cat_cols]
all_features = num_cols + encoded_cats
X_array = X[all_features].values
feature_names = all_features

print("\nStep 6: Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_array)

print("\nStep 7: Splitting data...")
X_train, X_temp, y_train, y_temp = train_test_split(
    X_scaled, y, test_size=0.3, random_state=SEED
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=SEED
)
print(f"Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")

# Metrics calculation functions
def calculate_metrics(y_true, y_pred):
    """Calculate all regression metrics"""
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    # MAPE with protection
    epsilon = np.mean(np.abs(y_true)) * 0.01
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100
    
    # Regression accuracy
    percentage_errors = np.abs((y_pred - y_true) / (np.abs(y_true) + 1e-8)) * 100
    absolute_threshold = 0.15 * np.mean(np.abs(y_true))
    absolute_errors = np.abs(y_pred - y_true)
    within_tolerance = (percentage_errors <= 15) | (absolute_errors <= absolute_threshold)
    accuracy = (within_tolerance.sum() / len(y_pred)) * 100
    
    return {
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'accuracy': accuracy
    }

# Store results
results = {}

print("\n" + "=" * 80)
print("TESTING DIFFERENT MODEL ARCHITECTURES")
print("=" * 80)

# ============================================================================
# MODEL 1: RANDOM FOREST (Best for small tabular data)
# ============================================================================
print("\n[1/5] Training Random Forest Regressor...")
print("-" * 80)

rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=SEED,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
rf_val_pred = rf_model.predict(X_val)
rf_test_pred = rf_model.predict(X_test)

rf_val_metrics = calculate_metrics(y_val, rf_val_pred)
rf_test_metrics = calculate_metrics(y_test, rf_test_pred)

print(f"Validation - R²: {rf_val_metrics['r2']:.4f}, RMSE: {rf_val_metrics['rmse']:.2f}, Accuracy: {rf_val_metrics['accuracy']:.1f}%")
print(f"Test       - R²: {rf_test_metrics['r2']:.4f}, RMSE: {rf_test_metrics['rmse']:.2f}, Accuracy: {rf_test_metrics['accuracy']:.1f}%")

results['Random Forest'] = {
    'model': rf_model,
    'val_metrics': rf_val_metrics,
    'test_metrics': rf_test_metrics,
    'predictions': rf_test_pred
}

# ============================================================================
# MODEL 2: XGBOOST (State-of-the-art for tabular data)
# ============================================================================
if XGBOOST_AVAILABLE:
    print("\n[2/5] Training XGBoost Regressor...")
    print("-" * 80)
    
    xgb_model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        n_jobs=-1
    )
    
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    xgb_val_pred = xgb_model.predict(X_val)
    xgb_test_pred = xgb_model.predict(X_test)
    
    xgb_val_metrics = calculate_metrics(y_val, xgb_val_pred)
    xgb_test_metrics = calculate_metrics(y_test, xgb_test_pred)
    
    print(f"Validation - R²: {xgb_val_metrics['r2']:.4f}, RMSE: {xgb_val_metrics['rmse']:.2f}, Accuracy: {xgb_val_metrics['accuracy']:.1f}%")
    print(f"Test       - R²: {xgb_test_metrics['r2']:.4f}, RMSE: {xgb_test_metrics['rmse']:.2f}, Accuracy: {xgb_test_metrics['accuracy']:.1f}%")
    
    results['XGBoost'] = {
        'model': xgb_model,
        'val_metrics': xgb_val_metrics,
        'test_metrics': xgb_test_metrics,
        'predictions': xgb_test_pred
    }
else:
    print("\n[2/5] XGBoost not available - skipping")

# ============================================================================
# MODEL 3: SIMPLE NEURAL NETWORK (2 layers)
# ============================================================================
print("\n[3/5] Training Simple Neural Network (2 layers)...")
print("-" * 80)

class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.network(x)

def train_pytorch_model(model, train_loader, val_loader, epochs=100, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        for features, targets in train_loader:
            optimizer.zero_grad()
            predictions = model(features).squeeze()
            loss = criterion(predictions, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        model.eval()
        val_losses = []
        with torch.no_grad():
            for features, targets in val_loader:
                predictions = model(features).squeeze()
                loss = criterion(predictions, targets)
                val_losses.append(loss.item())
        
        val_loss = np.mean(val_losses)
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= 20:
                break
    
    model.load_state_dict(best_state)
    return model

class CarbonDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

train_dataset = CarbonDataset(X_train, y_train)
val_dataset = CarbonDataset(X_val, y_val)
test_dataset = CarbonDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

simple_nn = SimpleNN(X_scaled.shape[1]).to(device)
simple_nn = train_pytorch_model(simple_nn, train_loader, val_loader, epochs=100)

simple_nn.eval()
with torch.no_grad():
    simple_val_pred = simple_nn(torch.FloatTensor(X_val)).squeeze().numpy()
    simple_test_pred = simple_nn(torch.FloatTensor(X_test)).squeeze().numpy()

simple_val_metrics = calculate_metrics(y_val, simple_val_pred)
simple_test_metrics = calculate_metrics(y_test, simple_test_pred)

print(f"Validation - R²: {simple_val_metrics['r2']:.4f}, RMSE: {simple_val_metrics['rmse']:.2f}, Accuracy: {simple_val_metrics['accuracy']:.1f}%")
print(f"Test       - R²: {simple_test_metrics['r2']:.4f}, RMSE: {simple_test_metrics['rmse']:.2f}, Accuracy: {simple_test_metrics['accuracy']:.1f}%")

results['Simple NN'] = {
    'model': simple_nn,
    'val_metrics': simple_val_metrics,
    'test_metrics': simple_test_metrics,
    'predictions': simple_test_pred
}

# ============================================================================
# MODEL 4: DEEP NEURAL NETWORK (4 layers - original)
# ============================================================================
print("\n[4/5] Training Deep Neural Network (4 layers)...")
print("-" * 80)

class DeepNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.network(x)

deep_nn = DeepNN(X_scaled.shape[1]).to(device)
deep_nn = train_pytorch_model(deep_nn, train_loader, val_loader, epochs=100)

deep_nn.eval()
with torch.no_grad():
    deep_val_pred = deep_nn(torch.FloatTensor(X_val)).squeeze().numpy()
    deep_test_pred = deep_nn(torch.FloatTensor(X_test)).squeeze().numpy()

deep_val_metrics = calculate_metrics(y_val, deep_val_pred)
deep_test_metrics = calculate_metrics(y_test, deep_test_pred)

print(f"Validation - R²: {deep_val_metrics['r2']:.4f}, RMSE: {deep_val_metrics['rmse']:.2f}, Accuracy: {deep_val_metrics['accuracy']:.1f}%")
print(f"Test       - R²: {deep_test_metrics['r2']:.4f}, RMSE: {deep_test_metrics['rmse']:.2f}, Accuracy: {deep_test_metrics['accuracy']:.1f}%")

results['Deep NN'] = {
    'model': deep_nn,
    'val_metrics': deep_val_metrics,
    'test_metrics': deep_test_metrics,
    'predictions': deep_test_pred
}

# ============================================================================
# MODEL 5: LINEAR REGRESSION (Baseline)
# ============================================================================
print("\n[5/5] Training Ridge Regression (Baseline)...")
print("-" * 80)

ridge_model = Ridge(alpha=1.0, random_state=SEED)
ridge_model.fit(X_train, y_train)
ridge_val_pred = ridge_model.predict(X_val)
ridge_test_pred = ridge_model.predict(X_test)

ridge_val_metrics = calculate_metrics(y_val, ridge_val_pred)
ridge_test_metrics = calculate_metrics(y_test, ridge_test_pred)

print(f"Validation - R²: {ridge_val_metrics['r2']:.4f}, RMSE: {ridge_val_metrics['rmse']:.2f}, Accuracy: {ridge_val_metrics['accuracy']:.1f}%")
print(f"Test       - R²: {ridge_test_metrics['r2']:.4f}, RMSE: {ridge_test_metrics['rmse']:.2f}, Accuracy: {ridge_test_metrics['accuracy']:.1f}%")

results['Ridge'] = {
    'model': ridge_model,
    'val_metrics': ridge_val_metrics,
    'test_metrics': ridge_test_metrics,
    'predictions': ridge_test_pred
}

# ============================================================================
# COMPARE ALL MODELS AND SELECT BEST
# ============================================================================
print("\n" + "=" * 80)
print("MODEL COMPARISON - VALIDATION SET")
print("=" * 80)
print(f"{'Model':<20} | {'R²':>8} | {'RMSE':>8} | {'MAE':>8} | {'MAPE':>8} | {'Acc%':>8}")
print("-" * 80)

for model_name, data in results.items():
    m = data['val_metrics']
    print(f"{model_name:<20} | {m['r2']:8.4f} | {m['rmse']:8.2f} | {m['mae']:8.2f} | {m['mape']:7.1f}% | {m['accuracy']:7.1f}%")

# Select best model based on validation R²
best_model_name = max(results.items(), key=lambda x: x[1]['val_metrics']['r2'])[0]
best_model_data = results[best_model_name]

print("\n" + "=" * 80)
print(f"BEST MODEL: {best_model_name}")
print("=" * 80)

print("\nTEST SET PERFORMANCE:")
print("-" * 80)
test_m = best_model_data['test_metrics']
print(f"R² Score:       {test_m['r2']:.4f}")
print(f"RMSE:           {test_m['rmse']:.2f} kg CO2e")
print(f"MAE:            {test_m['mae']:.2f} kg CO2e")
print(f"MAPE:           {test_m['mape']:.2f}%")
print(f"Accuracy (±15%): {test_m['accuracy']:.1f}%")

# Generate SHAP for best model
print("\nGenerating SHAP explanations for best model...")

best_model = best_model_data['model']

if best_model_name in ['Random Forest', 'XGBoost', 'Ridge']:
    explainer = shap.Explainer(best_model, X_train[:100])
    shap_values = explainer(X_test[:100])
    if len(shap_values.shape) > 2:
        shap_vals = shap_values.values[:, :, 0]
    else:
        shap_vals = shap_values.values
else:
    def predict_fn(X):
        best_model.eval()
        with torch.no_grad():
            return best_model(torch.FloatTensor(X)).squeeze().numpy()
    explainer = shap.KernelExplainer(predict_fn, X_train[:100])
    shap_vals = explainer.shap_values(X_test[:100], nsamples=100)

feature_importance = np.abs(shap_vals).mean(axis=0)
top_features_idx = np.argsort(feature_importance)[::-1]

print("\nTop 10 Most Important Features:")
print("=" * 80)
for i, idx in enumerate(top_features_idx[:10], 1):
    print(f"{i:2d}. {feature_names[idx]:30s} - Importance: {feature_importance[idx]:.4f}")
print("=" * 80)

# Save best model
shap_data = {
    'shap_values': shap_vals,
    'feature_importance': feature_importance,
    'feature_names': feature_names,
    'X_shap_background': X_train[:100],
    'top_features': [feature_names[idx] for idx in top_features_idx[:10]],
    'top_importance': [feature_importance[idx] for idx in top_features_idx[:10]]
}

model_package = {
    'model': best_model,
    'model_type': best_model_name,
    'scaler': scaler,
    'encoders': encoders,
    'feature_names': feature_names,
    'num_cols': num_cols,
    'cat_cols': cat_cols,
    'target_col': target_col,
    'input_dim': X_scaled.shape[1],
    'metrics': test_m,
    'shap_data': shap_data,
    'all_results': {name: data['test_metrics'] for name, data in results.items()}
}

with open('carbon_model.pkl', 'wb') as f:
    pickle.dump(model_package, f)

print("\nModel saved to: carbon_model.pkl")
print(f"\nBest architecture: {best_model_name}")
print("=" * 80)