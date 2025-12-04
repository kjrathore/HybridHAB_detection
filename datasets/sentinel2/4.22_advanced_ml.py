#!/usr/bin/env python3
"""
Advanced Alexandrium Prediction - Following Best Practices from Literature
Implements: XGBoost, Gradient Boosting, Random Forest, SHAP explainability,
spectral indices, and proper feature engineering.
"""

import subprocess
import sys


import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

sns.set_style('whitegrid')


OUTPUT_DIR = Path('regr_outputs')


def add_spectral_indices(df, stat_type='median'):
    """
    Add spectral indices commonly used for algae detection.
    Based on best practices from HAB prediction literature.
    """
    df = df.copy()
    
    # Extract band values
    b1 = df[f'rrs_B1_{stat_type}']
    b2 = df[f'rrs_B2_{stat_type}']
    b3 = df[f'rrs_B3_{stat_type}']
    b4 = df[f'rrs_B4_{stat_type}']
    b5 = df[f'rrs_B5_{stat_type}']
    b6 = df[f'rrs_B6_{stat_type}']
    b7 = df[f'rrs_B7_{stat_type}']
    b8a = df[f'rrs_B8A_{stat_type}']
    
    eps = 1e-10  # Small value to avoid division by zero
    
    # Normalized Difference Chlorophyll Index (NDCI) - Key for algae
    df[f'NDCI_{stat_type}'] = (b5 - b4) / (b5 + b4 + eps)
    
    # Normalized Difference Vegetation Index (NDVI) - Modified for water
    df[f'NDVI_{stat_type}'] = (b8a - b4) / (b8a + b4 + eps)
    
    # Red/Blue ratio - Indicates phytoplankton
    df[f'RedBlue_{stat_type}'] = b4 / (b2 + eps)
    
    # NIR/Red ratio - Sensitive to algae biomass
    df[f'NIRRed_{stat_type}'] = b5 / (b4 + eps)
    
    # Green/Red ratio
    df[f'GreenRed_{stat_type}'] = b3 / (b4 + eps)
    
    # Red Edge Position (average of red edge bands)
    df[f'RedEdge_{stat_type}'] = (b5 + b6 + b7) / 3
    
    # Blue-Green ratio
    df[f'BlueGreen_{stat_type}'] = b2 / (b3 + eps)
    
    # Floating Algae Index (FAI) approximation
    df[f'FAI_{stat_type}'] = b8a - (b4 + (b7 - b4) * (865 - 665) / (783 - 665))
    
    print(f"Added {8} spectral indices for {stat_type} features")
    
    return df


def prepare_advanced_features(df, stat_type='median', use_indices=True):
    """
    Prepare features with spectral indices and standardization.
    """
    df_work = df.copy()
    
    # Add spectral indices
    if use_indices:
        df_work = add_spectral_indices(df_work, stat_type)
    
    # Select all features
    if use_indices:
        feature_cols = [c for c in df_work.columns 
                       if stat_type in c and ('rrs_B' in c or 'NDCI' in c or 
                          'NDVI' in c or 'Red' in c or 'NIR' in c or 
                          'Green' in c or 'Blue' in c or 'FAI' in c)]
    else:
        feature_cols = [c for c in df_work.columns if f'rrs_B' in c and stat_type in c]
    
    # Remove inf and NaN
    df_work = df_work.replace([np.inf, -np.inf], np.nan)
    df_work = df_work.dropna(subset=feature_cols + ['rel_abundance'])
    
    X = df_work[feature_cols].values
    y = df_work['rel_abundance'].values
    
    # Log transform target (handles zeros with log1p)
    y_log = np.log1p(y)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"\n{stat_type.upper()} features prepared:")
    print(f"  Original features: {len([c for c in feature_cols if 'rrs_B' in c])}")
    print(f"  Spectral indices: {len([c for c in feature_cols if 'rrs_B' not in c])}")
    print(f"  Total features: {len(feature_cols)}")
    print(f"  Samples: {len(X_scaled)}")
    print(f"  Target: log-transformed abundance")
    
    return X_scaled, y, y_log, feature_cols, scaler, df_work


def train_ensemble_models(X, y_log, y_original, feature_names):
    """
    Train multiple ensemble models (best practices from literature).
    """
    # Split data
    X_train, X_test, y_train_log, y_test_log = train_test_split(
        X, y_log, test_size=0.25, random_state=42
    )
    
    # Get original scale targets for test set
    y_train_orig = np.expm1(y_train_log)
    y_test_orig = np.expm1(y_test_log)
    
    # Define models (based on HAB literature)
    models = {
        'Random Forest': RandomForestRegressor(
            n_estimators=200, max_depth=15, min_samples_split=5,
            min_samples_leaf=2, random_state=42, n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=200, max_depth=8, learning_rate=0.05,
            min_samples_split=5, random_state=42
        ),
        'XGBoost': XGBRegressor(
            n_estimators=200, max_depth=8, learning_rate=0.05,
            min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1
        )
    }
    
    results = []
    trained_models = {}
    
    print("\n" + "="*80)
    print("ENSEMBLE MODEL TRAINING")
    print("="*80)
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train model (in log space)
        model.fit(X_train, y_train_log)
        
        # Predictions in log space
        y_pred_train_log = model.predict(X_train)
        y_pred_test_log = model.predict(X_test)
        
        # Back-transform to original scale
        y_pred_train_orig = np.expm1(y_pred_train_log)
        y_pred_test_orig = np.expm1(y_pred_test_log)
        
        # Cross-validation (5-fold)
        cv_scores = cross_val_score(model, X_train, y_train_log, cv=5, 
                                     scoring='r2', n_jobs=-1)
        
        # Metrics in original scale
        train_r2 = r2_score(y_train_orig, y_pred_train_orig)
        test_r2 = r2_score(y_test_orig, y_pred_test_orig)
        test_rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_test_orig))
        test_mae = mean_absolute_error(y_test_orig, y_pred_test_orig)
        
        print(f"  Train R²:   {train_r2:.4f}")
        print(f"  Test R²:    {test_r2:.4f}")
        print(f"  CV R² (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"  Test RMSE:  {test_rmse:.4f}")
        print(f"  Test MAE:   {test_mae:.4f}")
        
        results.append({
            'model': name,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'cv_r2_mean': cv_scores.mean(),
            'cv_r2_std': cv_scores.std(),
            'test_rmse': test_rmse,
            'test_mae': test_mae
        })
        
        trained_models[name] = {
            'model': model,
            'predictions_train': y_pred_train_orig,
            'predictions_test': y_pred_test_orig,
            'y_train': y_train_orig,
            'y_test': y_test_orig
        }
    
    # Find best model
    results_df = pd.DataFrame(results)
    best_idx = results_df['test_r2'].idxmax()
    best_model_name = results_df.loc[best_idx, 'model']
    
    print(f"\n{'='*80}")
    print(f"BEST MODEL: {best_model_name} (Test R² = {results_df.loc[best_idx, 'test_r2']:.4f})")
    print(f"{'='*80}")
    
    return results_df, trained_models, best_model_name, feature_names


def explain_with_shap(model, X_test, feature_names, model_name, stat_type):
    """
    Use SHAP for explainable AI (best practice from literature).
    """
    print(f"\nGenerating SHAP explanations for {model_name}...")
    
    # Create SHAP explainer
    if model_name == 'XGBoost':
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(X_test)
    
    # Create summary plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Feature importance bar plot
    shap_importance = np.abs(shap_values).mean(axis=0)
    top_n = min(15, len(feature_names))
    top_indices = np.argsort(shap_importance)[-top_n:]
    
    ax1.barh(range(top_n), shap_importance[top_indices], color='steelblue', alpha=0.8)
    ax1.set_yticks(range(top_n))
    ax1.set_yticklabels([feature_names[i] for i in top_indices], fontsize=9)
    ax1.set_xlabel('Mean |SHAP value|', fontsize=11)
    ax1.set_title(f'Top {top_n} Feature Importance (SHAP)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # SHAP summary plot (beeswarm)
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, 
                     show=False, max_display=top_n, plot_type='dot')
    plt.gcf().axes[0].set_position([0.55, 0.1, 0.4, 0.8])
    plt.gcf().axes[-1].set_position([0.96, 0.1, 0.02, 0.8])
    
    plt.suptitle(f'SHAP Analysis: {model_name} - {stat_type.upper()} Features',
                fontsize=14, fontweight='bold')
    
    # Save plot
    shap_file = OUTPUT_DIR / f'shap_analysis_{stat_type}_{model_name.replace(" ", "_")}.png'
    plt.savefig(shap_file, dpi=150, bbox_inches='tight')
    print(f"SHAP plot saved to: {shap_file}")
    plt.close()
    
    return shap_values, shap_importance


def create_comparison_plots(results_df, trained_models, best_model_name, stat_type):
    """
    Create comprehensive comparison plots.
    """
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Plot 1: Model Comparison (R² scores)
    ax1 = fig.add_subplot(gs[0, 0])
    models_sorted = results_df.sort_values('test_r2')
    colors = ['gold' if m == best_model_name else 'steelblue' for m in models_sorted['model']]
    ax1.barh(range(len(models_sorted)), models_sorted['test_r2'], color=colors, alpha=0.8)
    ax1.set_yticks(range(len(models_sorted)))
    ax1.set_yticklabels(models_sorted['model'])
    ax1.set_xlabel('Test R²', fontsize=11)
    ax1.set_title('Model Comparison (Test R²)', fontsize=12, fontweight='bold')
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Plot 2: CV scores with error bars
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.errorbar(range(len(results_df)), results_df['cv_r2_mean'], 
                yerr=results_df['cv_r2_std'], fmt='o', capsize=5, capthick=2,
                color='darkblue', ecolor='gray', markersize=8)
    ax2.set_xticks(range(len(results_df)))
    ax2.set_xticklabels(results_df['model'], rotation=45, ha='right')
    ax2.set_ylabel('CV R² Score', fontsize=11)
    ax2.set_title('Cross-Validation Performance (5-fold)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: RMSE Comparison
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.bar(range(len(results_df)), results_df['test_rmse'], color='coral', alpha=0.8)
    ax3.set_xticks(range(len(results_df)))
    ax3.set_xticklabels(results_df['model'], rotation=45, ha='right')
    ax3.set_ylabel('RMSE', fontsize=11)
    ax3.set_title('Prediction Error (RMSE)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plots 4-6: Predictions vs Actual for each model
    for idx, (name, data) in enumerate(trained_models.items()):
        ax = fig.add_subplot(gs[1, idx])
        y_test = data['y_test']
        y_pred = data['predictions_test']
        
        ax.scatter(y_test, y_pred, alpha=0.5, s=30, edgecolors='k', linewidth=0.5)
        lim = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        ax.plot(lim, lim, 'r--', lw=2, label='Perfect Prediction')
        ax.set_xlabel('Actual Abundance', fontsize=10)
        ax.set_ylabel('Predicted Abundance', fontsize=10)
        r2 = r2_score(y_test, y_pred)
        title_suffix = " ⭐" if name == best_model_name else ""
        ax.set_title(f'{name} (R²={r2:.4f}){title_suffix}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Ensemble Models Comparison - {stat_type.upper()} Features',
                fontsize=15, fontweight='bold', y=0.995)
    
    plot_file = OUTPUT_DIR / f'ensemble_comparison_{stat_type}.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"\nComparison plot saved to: {plot_file}")
    plt.close()


def main():
    print("="*80)
    print("ADVANCED HAB PREDICTION - Following Best Practices from Literature")
    print("="*80)
    
    # Load features
    df_features = pd.read_csv(OUTPUT_DIR / 'alexandrium_features.csv')
    df_features['date'] = pd.to_datetime(df_features['date'])
    
    print(f"Loaded {len(df_features)} observations")
    
    # Process MEDIAN features
    print("\n" + "="*80)
    print("MEDIAN BAND FEATURES + SPECTRAL INDICES")
    print("="*80)
    
    X_med, y_med, y_med_log, features_med, scaler_med, df_med = prepare_advanced_features(
        df_features, stat_type='median', use_indices=True
    )
    
    results_med, models_med, best_med, _ = train_ensemble_models(
        X_med, y_med_log, y_med, features_med
    )
    
    # SHAP explanation for best model
    best_model_med = models_med[best_med]['model']
    X_test_idx = int(len(X_med) * 0.75)
    X_test_med = X_med[X_test_idx:]
    
    shap_values_med, shap_importance_med = explain_with_shap(
        best_model_med, X_test_med, features_med, best_med, 'median'
    )
    
    # Create comparison plots
    create_comparison_plots(results_med, models_med, best_med, 'median')
    
    # Save results
    results_med.to_csv(OUTPUT_DIR / 'ensemble_results_median.csv', index=False)
    
    # Process MAXIMUM features
    print("\n" + "="*80)
    print("MAXIMUM BAND FEATURES + SPECTRAL INDICES")
    print("="*80)
    
    X_max, y_max, y_max_log, features_max, scaler_max, df_max = prepare_advanced_features(
        df_features, stat_type='max', use_indices=True
    )
    
    results_max, models_max, best_max, _ = train_ensemble_models(
        X_max, y_max_log, y_max, features_max
    )
    
    # SHAP explanation for best model
    best_model_max = models_max[best_max]['model']
    X_test_max = X_max[X_test_idx:]
    
    shap_values_max, shap_importance_max = explain_with_shap(
        best_model_max, X_test_max, features_max, best_max, 'max'
    )
    
    # Create comparison plots
    create_comparison_plots(results_max, models_max, best_max, 'max')
    
    # Save results
    results_max.to_csv(OUTPUT_DIR / 'ensemble_results_max.csv', index=False)
    
    # Final Summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    best_med_r2 = results_med['test_r2'].max()
    best_max_r2 = results_max['test_r2'].max()
    
    print(f"\nMEDIAN Features:")
    print(f"  Best Model: {best_med}")
    print(f"  Test R²: {best_med_r2:.4f}")
    
    print(f"\nMAXIMUM Features:")
    print(f"  Best Model: {best_max}")
    print(f"  Test R²: {best_max_r2:.4f}")
    
    if best_med_r2 > best_max_r2:
        print(f"\n→ MEDIAN features with {best_med} perform best overall!")
        print(f"  Improvement over simple linear: {best_med_r2:.4f}")
    else:
        print(f"\n→ MAXIMUM features with {best_max} perform best overall!")
        print(f"  Improvement over simple linear: {best_max_r2:.4f}")
    
    print("\n" + "="*80)
    print("OUTPUT FILES")
    print("="*80)
    print(f"Results: ensemble_results_median.csv, ensemble_results_max.csv")
    print(f"Plots: ensemble_comparison_*.png, shap_analysis_*.png")
    print("\n" + "="*80)


if __name__ == '__main__':
    main()