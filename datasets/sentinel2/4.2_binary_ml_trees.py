#!/usr/bin/env python3
"""
Binary Classification for Alexandrium Prediction with Threshold Tuning
Tests multiple thresholds to find optimal performance for HAB risk prediction.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve)
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

sns.set_style('whitegrid')

OUTPUT_DIR = Path('binary_clf_outputs')


def add_spectral_indices(df, stat_type='median'):
    """
    Add spectral indices commonly used for algae detection.
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
    
    eps = 1e-10
    
    # Spectral indices
    df[f'NDCI_{stat_type}'] = (b5 - b4) / (b5 + b4 + eps)
    df[f'NDVI_{stat_type}'] = (b8a - b4) / (b8a + b4 + eps)
    df[f'RedBlue_{stat_type}'] = b4 / (b2 + eps)
    df[f'NIRRed_{stat_type}'] = b5 / (b4 + eps)
    df[f'GreenRed_{stat_type}'] = b3 / (b4 + eps)
    df[f'RedEdge_{stat_type}'] = (b5 + b6 + b7) / 3
    df[f'BlueGreen_{stat_type}'] = b2 / (b3 + eps)
    df[f'FAI_{stat_type}'] = b8a - (b4 + (b7 - b4) * (865 - 665) / (783 - 665))
    
    return df


def find_optimal_threshold(y_true, y_proba, thresholds_to_test=None):
    """
    Test multiple thresholds and find the optimal one based on F1 score.
    Returns threshold analysis results.
    """
    if thresholds_to_test is None:
        # Test thresholds from 10th to 90th percentile of data
        thresholds_to_test = np.percentile(y_true, np.arange(10, 91, 5))
    
    results = []
    for threshold in thresholds_to_test:
        y_binary = (y_true >= threshold).astype(int)
        
        # Skip if too imbalanced (less than 5% positive or negative)
        pos_rate = y_binary.mean()
        if pos_rate < 0.05 or pos_rate > 0.95:
            continue
        
        # Predict based on threshold
        y_pred = (y_proba >= 0.5).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_binary, y_pred)
        precision = precision_score(y_binary, y_pred, zero_division=0)
        recall = recall_score(y_binary, y_pred, zero_division=0)
        f1 = f1_score(y_binary, y_pred, zero_division=0)
        
        try:
            roc_auc = roc_auc_score(y_binary, y_proba)
        except:
            roc_auc = 0.5
        
        results.append({
            'threshold': threshold,
            'positive_rate': pos_rate,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc
        })
    
    return pd.DataFrame(results)


def prepare_binary_features(df, stat_type='median', threshold=None, use_indices=True):
    """
    Prepare features with binary target based on threshold.
    """
    df_work = df.copy()
    
    # Add spectral indices
    if use_indices:
        df_work = add_spectral_indices(df_work, stat_type)
    
    # Select features
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
    y_continuous = df_work['rel_abundance'].values
    
    # If no threshold provided, use median
    if threshold is None:
        threshold = np.median(y_continuous[y_continuous > 0])
    
    # Create binary target
    y_binary = (y_continuous >= threshold).astype(int)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pos_rate = y_binary.mean()
    print(f"\n{stat_type.upper()} features prepared:")
    print(f"  Threshold: {threshold:.4f}")
    print(f"  Total features: {len(feature_cols)}")
    print(f"  Samples: {len(X_scaled)}")
    print(f"  Positive class rate: {pos_rate:.2%} ({y_binary.sum()}/{len(y_binary)})")
    
    return X_scaled, y_binary, y_continuous, feature_cols, scaler, df_work, threshold


def train_binary_models(X, y_binary, feature_names):
    """
    Train binary classification ensemble models.
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=0.25, random_state=42, stratify=y_binary
    )
    
    # Define classification models
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=15, min_samples_split=5,
            min_samples_leaf=2, random_state=42, n_jobs=-1,
            class_weight='balanced'
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=200, max_depth=8, learning_rate=0.05,
            min_samples_split=5, random_state=42
        ),
        'XGBoost': XGBClassifier(
            n_estimators=200, max_depth=8, learning_rate=0.05,
            min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1, scale_pos_weight=1
        )
    }
    
    results = []
    trained_models = {}
    
    print("\n" + "="*80)
    print("BINARY CLASSIFICATION MODEL TRAINING")
    print("="*80)
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Prediction probabilities for ROC
        y_proba_train = model.predict_proba(X_train)[:, 1]
        y_proba_test = model.predict_proba(X_test)[:, 1]
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                     scoring='f1', n_jobs=-1)
        
        # Classification metrics
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        test_precision = precision_score(y_test, y_pred_test)
        test_recall = recall_score(y_test, y_pred_test)
        test_f1 = f1_score(y_test, y_pred_test)
        test_roc_auc = roc_auc_score(y_test, y_proba_test)
        
        print(f"  Train Accuracy: {train_acc:.4f}")
        print(f"  Test Accuracy:  {test_acc:.4f}")
        print(f"  Test Precision: {test_precision:.4f}")
        print(f"  Test Recall:    {test_recall:.4f}")
        print(f"  Test F1 Score:  {test_f1:.4f}")
        print(f"  Test ROC-AUC:   {test_roc_auc:.4f}")
        print(f"  CV F1 (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        results.append({
            'model': name,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1,
            'test_roc_auc': test_roc_auc,
            'cv_f1_mean': cv_scores.mean(),
            'cv_f1_std': cv_scores.std()
        })
        
        trained_models[name] = {
            'model': model,
            'predictions_train': y_pred_train,
            'predictions_test': y_pred_test,
            'probabilities_train': y_proba_train,
            'probabilities_test': y_proba_test,
            'y_train': y_train,
            'y_test': y_test
        }
    
    # Find best model
    results_df = pd.DataFrame(results)
    best_idx = results_df['test_f1'].idxmax()
    best_model_name = results_df.loc[best_idx, 'model']
    
    print(f"\n{'='*80}")
    print(f"BEST MODEL: {best_model_name} (F1 = {results_df.loc[best_idx, 'test_f1']:.4f})")
    print(f"{'='*80}")
    
    return results_df, trained_models, best_model_name


def plot_feature_importance(model, feature_names, model_name, stat_type, top_n=20):
    """
    Plot feature importance from Random Forest or tree-based models.
    """
    print(f"\nGenerating feature importance plot for {model_name}...")
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Sort by importance
    indices = np.argsort(importances)[::-1]
    
    # Select top N features
    top_n = min(top_n, len(feature_names))
    top_indices = indices[:top_n]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot 1: Bar plot of top features
    ax1.barh(range(top_n), importances[top_indices][::-1], color='steelblue', alpha=0.8)
    ax1.set_yticks(range(top_n))
    ax1.set_yticklabels([feature_names[i] for i in top_indices[::-1]], fontsize=9)
    ax1.set_xlabel('Feature Importance', fontsize=11)
    ax1.set_title(f'Top {top_n} Features by Importance', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Plot 2: Cumulative importance
    cumsum = np.cumsum(importances[indices])
    ax2.plot(range(1, len(cumsum)+1), cumsum, 'o-', linewidth=2, markersize=4, color='darkgreen')
    ax2.axhline(y=0.95, color='red', linestyle='--', linewidth=2, label='95% threshold')
    ax2.axhline(y=0.90, color='orange', linestyle='--', linewidth=2, label='90% threshold')
    ax2.set_xlabel('Number of Features', fontsize=11)
    ax2.set_ylabel('Cumulative Importance', fontsize=11)
    ax2.set_title('Cumulative Feature Importance', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Find number of features for 90% and 95% importance
    n_90 = np.argmax(cumsum >= 0.90) + 1
    n_95 = np.argmax(cumsum >= 0.95) + 1
    ax2.axvline(x=n_90, color='orange', linestyle=':', alpha=0.5)
    ax2.axvline(x=n_95, color='red', linestyle=':', alpha=0.5)
    ax2.text(n_90, 0.85, f'{n_90} features', fontsize=9, ha='center')
    ax2.text(n_95, 0.80, f'{n_95} features', fontsize=9, ha='center')
    
    plt.suptitle(f'Feature Importance Analysis: {model_name} - {stat_type.upper()} Features',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    plot_file = OUTPUT_DIR / f'feature_importance_{stat_type}_{model_name.replace(" ", "_")}.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"Feature importance plot saved to: {plot_file}")
    plt.close()
    
    # Print top 10 features
    print(f"\nTop 10 Most Important Features:")
    for i, idx in enumerate(top_indices[:10], 1):
        print(f"  {i:2d}. {feature_names[idx]:30s} : {importances[idx]:.4f}")
    
    return importances, indices


def create_threshold_analysis_plot(threshold_results, optimal_threshold, stat_type):
    """
    Plot threshold analysis results.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    # Plot each metric
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'positive_rate']
    titles = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC', 'Positive Class Rate']
    colors = ['steelblue', 'green', 'orange', 'red', 'purple', 'brown']
    
    for idx, (metric, title, color) in enumerate(zip(metrics, titles, colors)):
        ax = axes[idx]
        ax.plot(threshold_results['threshold'], threshold_results[metric], 
               'o-', color=color, linewidth=2, markersize=6, alpha=0.7)
        ax.axvline(optimal_threshold, color='red', linestyle='--', 
                  linewidth=2, label=f'Optimal: {optimal_threshold:.4f}')
        ax.set_xlabel('Threshold', fontsize=10)
        ax.set_ylabel(title, fontsize=10)
        ax.set_title(f'{title} vs Threshold', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.suptitle(f'Threshold Analysis - {stat_type.upper()} Features',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plot_file = OUTPUT_DIR / f'threshold_analysis_{stat_type}.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"\nThreshold analysis plot saved to: {plot_file}")
    plt.close()


def create_classification_plots(results_df, trained_models, best_model_name, 
                                stat_type, threshold):
    """
    Create comprehensive classification plots.
    """
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    # Plot 1: Model Comparison (F1 scores)
    ax1 = fig.add_subplot(gs[0, 0])
    models_sorted = results_df.sort_values('test_f1')
    colors = ['gold' if m == best_model_name else 'steelblue' for m in models_sorted['model']]
    ax1.barh(range(len(models_sorted)), models_sorted['test_f1'], color=colors, alpha=0.8)
    ax1.set_yticks(range(len(models_sorted)))
    ax1.set_yticklabels(models_sorted['model'])
    ax1.set_xlabel('Test F1 Score', fontsize=11)
    ax1.set_title('Model Comparison (F1)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Plot 2: Multi-metric comparison
    ax2 = fig.add_subplot(gs[0, 1])
    metrics = ['test_acc', 'test_precision', 'test_recall', 'test_f1', 'test_roc_auc']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']
    x = np.arange(len(metrics))
    width = 0.25
    
    for idx, (_, row) in enumerate(results_df.iterrows()):
        values = [row[m] for m in metrics]
        ax2.bar(x + idx*width, values, width, label=row['model'], alpha=0.8)
    
    ax2.set_xlabel('Metrics', fontsize=11)
    ax2.set_ylabel('Score', fontsize=11)
    ax2.set_title('Multi-Metric Comparison', fontsize=12, fontweight='bold')
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(metric_labels, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: ROC Curves
    ax3 = fig.add_subplot(gs[0, 2])
    for name, data in trained_models.items():
        fpr, tpr, _ = roc_curve(data['y_test'], data['probabilities_test'])
        auc = roc_auc_score(data['y_test'], data['probabilities_test'])
        linestyle = '-' if name == best_model_name else '--'
        linewidth = 2.5 if name == best_model_name else 1.5
        ax3.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})', 
                linestyle=linestyle, linewidth=linewidth)
    
    ax3.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
    ax3.set_xlabel('False Positive Rate', fontsize=11)
    ax3.set_ylabel('True Positive Rate', fontsize=11)
    ax3.set_title('ROC Curves', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # Plots 4-6: Confusion Matrices
    for idx, (name, data) in enumerate(trained_models.items()):
        ax = fig.add_subplot(gs[1, idx])
        cm = confusion_matrix(data['y_test'], data['predictions_test'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                   cbar_kws={'label': 'Count'})
        ax.set_xlabel('Predicted', fontsize=10)
        ax.set_ylabel('Actual', fontsize=10)
        title_suffix = " ⭐" if name == best_model_name else ""
        ax.set_title(f'{name}{title_suffix}', fontsize=11, fontweight='bold')
    
    # Plots 7-9: Probability distributions
    for idx, (name, data) in enumerate(trained_models.items()):
        ax = fig.add_subplot(gs[2, idx])
        
        # Plot probability distributions for each class
        proba_class0 = data['probabilities_test'][data['y_test'] == 0]
        proba_class1 = data['probabilities_test'][data['y_test'] == 1]
        
        ax.hist(proba_class0, bins=20, alpha=0.6, label='Low Risk (0)', color='blue')
        ax.hist(proba_class1, bins=20, alpha=0.6, label='High Risk (1)', color='red')
        ax.axvline(0.5, color='black', linestyle='--', linewidth=2, label='Threshold=0.5')
        ax.set_xlabel('Predicted Probability', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title(f'{name} - Probability Distribution', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'Binary Classification Results - {stat_type.upper()} Features\n' + 
                f'Threshold: {threshold:.4f}',
                fontsize=15, fontweight='bold', y=0.998)
    
    plot_file = OUTPUT_DIR / f'binary_classification_{stat_type}.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"Classification plot saved to: {plot_file}")
    plt.close()


def test_multiple_thresholds(df_features, stat_type='median'):
    """
    Test multiple thresholds and find the optimal one.
    """
    print("\n" + "="*80)
    print(f"THRESHOLD OPTIMIZATION - {stat_type.upper()} FEATURES")
    print("="*80)
    
    # Get abundance data
    y_continuous = df_features['rel_abundance'].values
    y_continuous = y_continuous[~np.isnan(y_continuous)]
    
    # Define thresholds to test (percentiles of non-zero values)
    non_zero_vals = y_continuous[y_continuous > 0]
    thresholds_to_test = np.percentile(non_zero_vals, np.arange(10, 91, 5))
    
    print(f"\nTesting {len(thresholds_to_test)} thresholds...")
    print(f"Range: {thresholds_to_test.min():.4f} to {thresholds_to_test.max():.4f}")
    
    # Test each threshold
    threshold_results = []
    
    for threshold in thresholds_to_test:
        try:
            # Prepare features with this threshold
            X, y_binary, y_cont, features, scaler, df_work, _ = prepare_binary_features(
                df_features, stat_type=stat_type, threshold=threshold, use_indices=True
            )
            
            # Quick train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_binary, test_size=0.25, random_state=42, stratify=y_binary
            )
            
            # Train a simple RF model for evaluation
            model = RandomForestClassifier(n_estimators=100, random_state=42, 
                                          class_weight='balanced', n_jobs=-1)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            pos_rate = y_binary.mean()
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            roc_auc = roc_auc_score(y_test, y_proba)
            
            threshold_results.append({
                'threshold': threshold,
                'positive_rate': pos_rate,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc
            })
            
        except Exception as e:
            print(f"  Skipping threshold {threshold:.4f}: {e}")
            continue
    
    results_df = pd.DataFrame(threshold_results)
    
    # Find optimal threshold (maximize F1 score)
    optimal_idx = results_df['f1_score'].idxmax()
    optimal_threshold = results_df.loc[optimal_idx, 'threshold']
    optimal_f1 = results_df.loc[optimal_idx, 'f1_score']
    
    print(f"\n{'='*80}")
    print(f"OPTIMAL THRESHOLD FOUND: {optimal_threshold:.4f}")
    print(f"  F1 Score: {optimal_f1:.4f}")
    print(f"  Accuracy: {results_df.loc[optimal_idx, 'accuracy']:.4f}")
    print(f"  Precision: {results_df.loc[optimal_idx, 'precision']:.4f}")
    print(f"  Recall: {results_df.loc[optimal_idx, 'recall']:.4f}")
    print(f"  ROC-AUC: {results_df.loc[optimal_idx, 'roc_auc']:.4f}")
    print(f"  Positive Rate: {results_df.loc[optimal_idx, 'positive_rate']:.2%}")
    print(f"{'='*80}")
    
    # Save results
    results_df.to_csv(OUTPUT_DIR / f'threshold_optimization_{stat_type}.csv', index=False)
    
    # Create threshold analysis plot
    create_threshold_analysis_plot(results_df, optimal_threshold, stat_type)
    
    return optimal_threshold, results_df


def main():
    print("="*80)
    print("BINARY CLASSIFICATION WITH THRESHOLD OPTIMIZATION")
    print("="*80)
    
    # Load features
    df_features = pd.read_csv(OUTPUT_DIR / 'alexandrium_features.csv')
    df_features['date'] = pd.to_datetime(df_features['date'])
    
    print(f"Loaded {len(df_features)} observations")
    print(f"Abundance range: {df_features['rel_abundance'].min():.4f} to " + 
          f"{df_features['rel_abundance'].max():.4f}")
    
    # Process MEDIAN features
    # Step 1: Find optimal threshold
    optimal_threshold_med, threshold_results_med = test_multiple_thresholds(
        df_features, stat_type='median'
    )
    
    # Step 2: Train models with optimal threshold
    print("\n" + "="*80)
    print(f"TRAINING WITH OPTIMAL THRESHOLD: {optimal_threshold_med:.4f}")
    print("="*80)
    
    X_med, y_med_binary, y_med_cont, features_med, scaler_med, df_med, _ = prepare_binary_features(
        df_features, stat_type='median', threshold=optimal_threshold_med, use_indices=True
    )
    
    results_med, models_med, best_med = train_binary_models(
        X_med, y_med_binary, features_med
    )
    
    # Print detailed classification report for best model
    print(f"\nDetailed Classification Report - {best_med}:")
    print("="*80)
    best_model_data = models_med[best_med]
    print(classification_report(best_model_data['y_test'], 
                               best_model_data['predictions_test'],
                               target_names=['Low Risk', 'High Risk']))
    
    # Feature importance for Random Forest
    if 'Random Forest' in models_med:
        rf_model = models_med['Random Forest']['model']
        plot_feature_importance(rf_model, features_med, 'Random Forest', 'median', top_n=20)
    
    # Create plots
    create_classification_plots(results_med, models_med, best_med, 
                               'median', optimal_threshold_med)
    
    # Save results
    results_med.to_csv(OUTPUT_DIR / 'binary_classification_results_median.csv', index=False)
    
    # Process MAXIMUM features
    optimal_threshold_max, threshold_results_max = test_multiple_thresholds(
        df_features, stat_type='max'
    )
    
    print("\n" + "="*80)
    print(f"TRAINING WITH OPTIMAL THRESHOLD: {optimal_threshold_max:.4f}")
    print("="*80)
    
    X_max, y_max_binary, y_max_cont, features_max, scaler_max, df_max, _ = prepare_binary_features(
        df_features, stat_type='max', threshold=optimal_threshold_max, use_indices=True
    )
    
    results_max, models_max, best_max = train_binary_models(
        X_max, y_max_binary, features_max
    )
    
    print(f"\nDetailed Classification Report - {best_max}:")
    print("="*80)
    best_model_data = models_max[best_max]
    print(classification_report(best_model_data['y_test'], 
                               best_model_data['predictions_test'],
                               target_names=['Low Risk', 'High Risk']))
    
    # Feature importance for Random Forest
    if 'Random Forest' in models_max:
        rf_model = models_max['Random Forest']['model']
        plot_feature_importance(rf_model, features_max, 'Random Forest', 'max', top_n=20)
    
    create_classification_plots(results_max, models_max, best_max, 
                               'max', optimal_threshold_max)
    
    results_max.to_csv(OUTPUT_DIR / 'binary_classification_results_max.csv', index=False)
    
    # Final Summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    best_med_f1 = results_med['test_f1'].max()
    best_max_f1 = results_max['test_f1'].max()
    
    print(f"\nMEDIAN Features:")
    print(f"  Optimal Threshold: {optimal_threshold_med:.4f}")
    print(f"  Best Model: {best_med}")
    print(f"  Test F1 Score: {best_med_f1:.4f}")
    
    print(f"\nMAXIMUM Features:")
    print(f"  Optimal Threshold: {optimal_threshold_max:.4f}")
    print(f"  Best Model: {best_max}")
    print(f"  Test F1 Score: {best_max_f1:.4f}")
    
    if best_med_f1 > best_max_f1:
        print(f"\n→ MEDIAN features with {best_med} perform best overall!")
        print(f"  F1 Score: {best_med_f1:.4f} at threshold {optimal_threshold_med:.4f}")
    else:
        print(f"\n→ MAXIMUM features with {best_max} perform best overall!")
        print(f"  F1 Score: {best_max_f1:.4f} at threshold {optimal_threshold_max:.4f}")
    
    print("\n" + "="*80)
    print("OUTPUT FILES")
    print("="*80)
    print("Threshold Analysis:")
    print("  - threshold_optimization_median.csv")
    print("  - threshold_optimization_max.csv")
    print("  - threshold_analysis_*.png")
    print("\nClassification Results:")
    print("  - binary_classification_results_median.csv")
    print("  - binary_classification_results_max.csv")
    print("  - binary_classification_*.png")
    print("\nFeature Importance:")
    print("  - feature_importance_median_Random_Forest.png")
    print("  - feature_importance_max_Random_Forest.png")
    print("\n" + "="*80)


if __name__ == '__main__':
    main()