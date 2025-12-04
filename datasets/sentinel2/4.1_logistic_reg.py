#!/usr/bin/env python3
"""
Simplified Logistic Regression for Alexandrium Binary Classification
Focus: Single model approach with threshold optimization and coefficient interpretation
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

sns.set_style('whitegrid')

OUTPUT_DIR = Path('logistic_reg_outputs')


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


def prepare_features(df, stat_type='median', threshold=None, use_indices=True):
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
    
    # If no threshold provided, use median of non-zero values
    if threshold is None:
        threshold = np.median(y_continuous[y_continuous > 0])
    
    # Create binary target
    y_binary = (y_continuous >= threshold).astype(int)
    
    # Standardize features (important for logistic regression!)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pos_rate = y_binary.mean()
    print(f"\n{stat_type.upper()} features prepared:")
    print(f"  Threshold: {threshold:.4f}")
    print(f"  Total features: {len(feature_cols)}")
    print(f"  Samples: {len(X_scaled)}")
    print(f"  Positive class rate: {pos_rate:.2%} ({y_binary.sum()}/{len(y_binary)})")
    
    return X_scaled, y_binary, y_continuous, feature_cols, scaler, df_work, threshold


def optimize_threshold(df, stat_type='median'):
    """
    Find optimal threshold by testing multiple values.
    """
    print("\n" + "="*80)
    print(f"THRESHOLD OPTIMIZATION - {stat_type.upper()}")
    print("="*80)
    
    y_continuous = df['rel_abundance'].values
    y_continuous = y_continuous[~np.isnan(y_continuous)]
    
    # Test thresholds at percentiles
    non_zero_vals = y_continuous[y_continuous > 0]
    thresholds_to_test = np.percentile(non_zero_vals, np.arange(10, 91, 5))
    
    print(f"\nTesting {len(thresholds_to_test)} thresholds...")
    print(f"Range: {thresholds_to_test.min():.4f} to {thresholds_to_test.max():.4f}")
    
    results = []
    
    for threshold in thresholds_to_test:
        try:
            # Prepare data
            X, y_binary, _, features, scaler, _, _ = prepare_features(
                df, stat_type=stat_type, threshold=threshold, use_indices=True
            )
            
            # Skip if class imbalance is too extreme
            pos_rate = y_binary.mean()
            if pos_rate < 0.05 or pos_rate > 0.95:
                continue
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_binary, test_size=0.25, random_state=42, stratify=y_binary
            )
            
            # Simple logistic regression for threshold testing
            model = LogisticRegression(max_iter=1000, random_state=42, 
                                      class_weight='balanced')
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            
            # Metrics
            results.append({
                'threshold': threshold,
                'positive_rate': pos_rate,
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1_score': f1_score(y_test, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_test, y_proba)
            })
            
        except Exception as e:
            print(f"  Skipping threshold {threshold:.4f}: {e}")
            continue
    
    results_df = pd.DataFrame(results)
    
    # Find optimal (maximize F1)
    optimal_idx = results_df['f1_score'].idxmax()
    optimal_threshold = results_df.loc[optimal_idx, 'threshold']
    
    print(f"\n{'='*80}")
    print(f"OPTIMAL THRESHOLD: {optimal_threshold:.4f}")
    print(f"  F1 Score: {results_df.loc[optimal_idx, 'f1_score']:.4f}")
    print(f"  Accuracy: {results_df.loc[optimal_idx, 'accuracy']:.4f}")
    print(f"  Precision: {results_df.loc[optimal_idx, 'precision']:.4f}")
    print(f"  Recall: {results_df.loc[optimal_idx, 'recall']:.4f}")
    print(f"  ROC-AUC: {results_df.loc[optimal_idx, 'roc_auc']:.4f}")
    print(f"{'='*80}")
    
    # Save results
    results_df.to_csv(OUTPUT_DIR / f'logistic_threshold_optimization_{stat_type}.csv', 
                     index=False)
    
    return optimal_threshold, results_df


def train_logistic_regression(X, y_binary, feature_names, stat_type):
    """
    Train logistic regression with different regularization approaches.
    """
    print("\n" + "="*80)
    print("LOGISTIC REGRESSION TRAINING")
    print("="*80)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=0.25, random_state=42, stratify=y_binary
    )
    
    # Test different regularization strengths
    models = {
        'L2 (C=1.0)': LogisticRegression(penalty='l2', C=1.0, max_iter=1000, 
                                          random_state=42, class_weight='balanced'),
        'L2 (C=0.1)': LogisticRegression(penalty='l2', C=0.1, max_iter=1000, 
                                          random_state=42, class_weight='balanced'),
        'L1 (C=1.0)': LogisticRegression(penalty='l1', C=1.0, solver='liblinear',
                                          max_iter=1000, random_state=42, 
                                          class_weight='balanced')
    }
    
    results = []
    trained_models = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        y_proba_test = model.predict_proba(X_test)[:, 1]
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                    scoring='f1', n_jobs=-1)
        
        # Metrics
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
            'y_pred_test': y_pred_test,
            'y_proba_test': y_proba_test,
            'y_test': y_test
        }
    
    results_df = pd.DataFrame(results)
    best_idx = results_df['test_f1'].idxmax()
    best_model_name = results_df.loc[best_idx, 'model']
    
    print(f"\n{'='*80}")
    print(f"BEST MODEL: {best_model_name} (F1 = {results_df.loc[best_idx, 'test_f1']:.4f})")
    print(f"{'='*80}")
    
    # Save results
    results_df.to_csv(OUTPUT_DIR / f'logistic_regression_results_{stat_type}.csv', 
                     index=False)
    
    return trained_models[best_model_name]['model'], trained_models, results_df, best_model_name


def plot_coefficients(model, feature_names, stat_type, top_n=20):
    """
    Plot logistic regression coefficients (feature importance for linear models).
    """
    print(f"\nGenerating coefficient plot...")
    
    # Get coefficients
    coefficients = model.coef_[0]
    
    # Sort by absolute value
    abs_coef = np.abs(coefficients)
    indices = np.argsort(abs_coef)[::-1]
    
    # Select top N
    top_n = min(top_n, len(feature_names))
    top_indices = indices[:top_n]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot 1: Top coefficients (with sign)
    colors = ['red' if c < 0 else 'green' for c in coefficients[top_indices][::-1]]
    ax1.barh(range(top_n), coefficients[top_indices][::-1], color=colors, alpha=0.7)
    ax1.set_yticks(range(top_n))
    ax1.set_yticklabels([feature_names[i] for i in top_indices[::-1]], fontsize=9)
    ax1.set_xlabel('Coefficient Value', fontsize=11)
    ax1.set_title(f'Top {top_n} Features by |Coefficient|', fontsize=12, fontweight='bold')
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.text(0.02, 0.98, 'Green = increases HAB risk\nRed = decreases HAB risk',
            transform=ax1.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 2: Coefficient magnitude
    ax2.barh(range(top_n), abs_coef[top_indices][::-1], color='steelblue', alpha=0.8)
    ax2.set_yticks(range(top_n))
    ax2.set_yticklabels([feature_names[i] for i in top_indices[::-1]], fontsize=9)
    ax2.set_xlabel('|Coefficient| (Absolute Value)', fontsize=11)
    ax2.set_title(f'Feature Importance by |Coefficient|', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.suptitle(f'Logistic Regression Coefficients - {stat_type.upper()} Features',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save
    plot_file = OUTPUT_DIR / f'logistic_coefficients_{stat_type}.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"Coefficient plot saved to: {plot_file}")
    plt.close()
    
    # Print top features
    print(f"\nTop 10 Features by |Coefficient|:")
    for i, idx in enumerate(top_indices[:10], 1):
        sign = '+' if coefficients[idx] > 0 else '-'
        print(f"  {i:2d}. {feature_names[idx]:30s} : {sign}{abs_coef[idx]:.4f}")


def create_comprehensive_plot(trained_models, results_df, best_model_name, 
                              stat_type, threshold):
    """
    Create comprehensive visualization for logistic regression results.
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Plot 1: Model comparison
    ax1 = fig.add_subplot(gs[0, 0])
    models_sorted = results_df.sort_values('test_f1')
    colors = ['gold' if m == best_model_name else 'steelblue' for m in models_sorted['model']]
    ax1.barh(range(len(models_sorted)), models_sorted['test_f1'], color=colors, alpha=0.8)
    ax1.set_yticks(range(len(models_sorted)))
    ax1.set_yticklabels(models_sorted['model'], fontsize=9)
    ax1.set_xlabel('Test F1 Score', fontsize=11)
    ax1.set_title('Model Comparison', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Plot 2: Multi-metric comparison for best model
    ax2 = fig.add_subplot(gs[0, 1])
    best_row = results_df[results_df['model'] == best_model_name].iloc[0]
    metrics = ['test_acc', 'test_precision', 'test_recall', 'test_f1', 'test_roc_auc']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']
    values = [best_row[m] for m in metrics]
    colors_metrics = ['steelblue', 'green', 'orange', 'red', 'purple']
    ax2.bar(range(len(metrics)), values, color=colors_metrics, alpha=0.8)
    ax2.set_xticks(range(len(metrics)))
    ax2.set_xticklabels(metric_labels, rotation=45, ha='right')
    ax2.set_ylabel('Score', fontsize=11)
    ax2.set_title(f'Performance Metrics - {best_model_name}', fontsize=12, fontweight='bold')
    ax2.set_ylim([0, 1.0])
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: ROC Curve
    ax3 = fig.add_subplot(gs[0, 2])
    best_data = trained_models[best_model_name]
    fpr, tpr, _ = roc_curve(best_data['y_test'], best_data['y_proba_test'])
    auc = roc_auc_score(best_data['y_test'], best_data['y_proba_test'])
    ax3.plot(fpr, tpr, linewidth=2.5, label=f'ROC (AUC={auc:.3f})', color='darkblue')
    ax3.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
    ax3.set_xlabel('False Positive Rate', fontsize=11)
    ax3.set_ylabel('True Positive Rate', fontsize=11)
    ax3.set_title('ROC Curve', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Confusion Matrix
    ax4 = fig.add_subplot(gs[1, 0])
    cm = confusion_matrix(best_data['y_test'], best_data['y_pred_test'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4,
               xticklabels=['Low Risk', 'High Risk'],
               yticklabels=['Low Risk', 'High Risk'],
               cbar_kws={'label': 'Count'})
    ax4.set_xlabel('Predicted', fontsize=11)
    ax4.set_ylabel('Actual', fontsize=11)
    ax4.set_title('Confusion Matrix', fontsize=12, fontweight='bold')
    
    # Plot 5: Probability Distribution
    ax5 = fig.add_subplot(gs[1, 1])
    proba_class0 = best_data['y_proba_test'][best_data['y_test'] == 0]
    proba_class1 = best_data['y_proba_test'][best_data['y_test'] == 1]
    ax5.hist(proba_class0, bins=20, alpha=0.6, label='Low Risk (0)', color='blue')
    ax5.hist(proba_class1, bins=20, alpha=0.6, label='High Risk (1)', color='red')
    ax5.axvline(0.5, color='black', linestyle='--', linewidth=2, label='Threshold=0.5')
    ax5.set_xlabel('Predicted Probability', fontsize=11)
    ax5.set_ylabel('Frequency', fontsize=11)
    ax5.set_title('Probability Distribution', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Plot 6: Prediction vs Actual scatter
    ax6 = fig.add_subplot(gs[1, 2])
    # Add jitter for visualization
    jitter = 0.05
    y_test_jitter = best_data['y_test'] + np.random.normal(0, jitter, len(best_data['y_test']))
    y_pred_jitter = best_data['y_pred_test'] + np.random.normal(0, jitter, len(best_data['y_pred_test']))
    ax6.scatter(y_test_jitter, best_data['y_proba_test'], alpha=0.5, s=30, 
               c=best_data['y_test'], cmap='coolwarm', edgecolors='k', linewidth=0.5)
    ax6.axhline(y=0.5, color='black', linestyle='--', linewidth=2, label='Decision Boundary')
    ax6.set_xlabel('Actual Class (with jitter)', fontsize=11)
    ax6.set_ylabel('Predicted Probability', fontsize=11)
    ax6.set_title('Predictions vs Actual', fontsize=12, fontweight='bold')
    ax6.set_xticks([0, 1])
    ax6.set_xticklabels(['Low Risk', 'High Risk'])
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle(f'Logistic Regression Results - {stat_type.upper()} Features\n' +
                f'Threshold: {threshold:.4f}',
                fontsize=15, fontweight='bold', y=0.998)
    
    plot_file = OUTPUT_DIR / f'logistic_regression_{stat_type}.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"\nComprehensive plot saved to: {plot_file}")
    plt.close()


def main():
    print("="*80)
    print("SIMPLIFIED LOGISTIC REGRESSION - BINARY HAB CLASSIFICATION")
    print("="*80)
    
    # Load data
    df_features = pd.read_csv(OUTPUT_DIR / 'alexandrium_features.csv')
    df_features['date'] = pd.to_datetime(df_features['date'])
    
    print(f"\nLoaded {len(df_features)} observations")
    print(f"Abundance range: {df_features['rel_abundance'].min():.4f} to " +
          f"{df_features['rel_abundance'].max():.4f}")
    
    # Process MEDIAN features
    print("\n" + "="*80)
    print("PROCESSING MEDIAN FEATURES")
    print("="*80)
    
    # Step 1: Optimize threshold
    optimal_threshold_med, threshold_results_med = optimize_threshold(
        df_features, stat_type='median'
    )
    
    # Step 2: Prepare data with optimal threshold
    X_med, y_med, _, features_med, scaler_med, _, _ = prepare_features(
        df_features, stat_type='median', threshold=optimal_threshold_med, use_indices=True
    )
    
    # Step 3: Train logistic regression
    best_model_med, models_med, results_med, best_name_med = train_logistic_regression(
        X_med, y_med, features_med, 'median'
    )
    
    # Step 4: Detailed classification report
    print(f"\nDetailed Classification Report - {best_name_med}:")
    print("="*80)
    best_data_med = models_med[best_name_med]
    print(classification_report(best_data_med['y_test'], 
                               best_data_med['y_pred_test'],
                               target_names=['Low Risk', 'High Risk']))
    
    # Step 5: Plot coefficients
    plot_coefficients(best_model_med, features_med, 'median', top_n=20)
    
    # Step 6: Create comprehensive plot
    create_comprehensive_plot(models_med, results_med, best_name_med,
                             'median', optimal_threshold_med)
    
    # Process MAXIMUM features
    print("\n" + "="*80)
    print("PROCESSING MAXIMUM FEATURES")
    print("="*80)
    
    optimal_threshold_max, threshold_results_max = optimize_threshold(
        df_features, stat_type='max'
    )
    
    X_max, y_max, _, features_max, scaler_max, _, _ = prepare_features(
        df_features, stat_type='max', threshold=optimal_threshold_max, use_indices=True
    )
    
    best_model_max, models_max, results_max, best_name_max = train_logistic_regression(
        X_max, y_max, features_max, 'max'
    )
    
    print(f"\nDetailed Classification Report - {best_name_max}:")
    print("="*80)
    best_data_max = models_max[best_name_max]
    print(classification_report(best_data_max['y_test'], 
                               best_data_max['y_pred_test'],
                               target_names=['Low Risk', 'High Risk']))
    
    plot_coefficients(best_model_max, features_max, 'max', top_n=20)
    
    create_comprehensive_plot(models_max, results_max, best_name_max,
                             'max', optimal_threshold_max)
    
    # Final Summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    best_med_f1 = results_med['test_f1'].max()
    best_max_f1 = results_max['test_f1'].max()
    
    print(f"\nMEDIAN Features:")
    print(f"  Optimal Threshold: {optimal_threshold_med:.4f}")
    print(f"  Best Model: {best_name_med}")
    print(f"  Test F1 Score: {best_med_f1:.4f}")
    
    print(f"\nMAXIMUM Features:")
    print(f"  Optimal Threshold: {optimal_threshold_max:.4f}")
    print(f"  Best Model: {best_name_max}")
    print(f"  Test F1 Score: {best_max_f1:.4f}")
    
    if best_med_f1 > best_max_f1:
        print(f"\n→ MEDIAN features perform best!")
        print(f"  F1 Score: {best_med_f1:.4f} at threshold {optimal_threshold_med:.4f}")
    else:
        print(f"\n→ MAXIMUM features perform best!")
        print(f"  F1 Score: {best_max_f1:.4f} at threshold {optimal_threshold_max:.4f}")
    
    print("\n" + "="*80)
    print("OUTPUT FILES")
    print("="*80)
    print("Threshold Optimization:")
    print("  - logistic_threshold_optimization_median.csv")
    print("  - logistic_threshold_optimization_max.csv")
    print("\nModel Results:")
    print("  - logistic_regression_results_median.csv")
    print("  - logistic_regression_results_max.csv")
    print("\nVisualizations:")
    print("  - logistic_coefficients_median.png")
    print("  - logistic_coefficients_max.png")
    print("  - logistic_regression_median.png")
    print("  - logistic_regression_max.png")
    print("\n" + "="*80)


if __name__ == '__main__':
    main()