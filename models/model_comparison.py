"""
모델 비교 및 하이퍼파라미터 튜닝
- Autoencoder vs Transformer 성능 비교
- 하이퍼파라미터 튜닝
- 결과 시각화
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.data_loader import load_data, add_attack_type, add_binary_label
from preprocessing.preprocessor import prepare_data
from models.autoencoder import AnomalyDetectorAE
from models.transformer import AnomalyDetectorTransformer

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, confusion_matrix

def train_and_evaluate(model, X_train, y_train, X_test, y_test, model_name: str) -> dict:
    """모델 학습 및 평가"""
    print(f"\n{'='*50}")
    print(f"Training {model_name}...")
    print(f"{'='*50}")
    
    # 학습 시간 측정
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # 예측
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)
    
    # 평가 지표
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_score)
    cm = confusion_matrix(y_test, y_pred)
    
    results = {
        'model_name': model_name,
        'accuracy': accuracy,
        'f1_score': f1,
        'auc_score': auc,
        'train_time': train_time,
        'confusion_matrix': cm
    }
    
    print(f"\n[{model_name} Results]")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC Score: {auc:.4f}")
    print(f"Train Time: {train_time:.2f}s")
    
    return results


def tune_autoencoder(X_train, y_train, X_test, y_test) -> dict:
    """Autoencoder 하이퍼파라미터 튜닝"""
    print("\n[Autoencoder Hyperparameter Tuning]")
    
    param_grid = {
        'hidden_dims': [[64, 32, 16], [128, 64, 32], [64, 32]],
        'epochs': [30, 50],
        'learning_rate': [0.001, 0.0001]
    }
    
    best_f1 = 0
    best_params = None
    results = []
    
    for hidden_dims in param_grid['hidden_dims']:
        for epochs in param_grid['epochs']:
            for lr in param_grid['learning_rate']:
                print(f"\nTrying: hidden={hidden_dims}, epochs={epochs}, lr={lr}")
                
                model = AnomalyDetectorAE(
                    input_dim=X_train.shape[1],
                    hidden_dims=hidden_dims,
                    epochs=epochs,
                    learning_rate=lr,
                    batch_size=256
                )
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_test)
                f1 = f1_score(y_test, y_pred)
                
                results.append({
                    'hidden_dims': str(hidden_dims),
                    'epochs': epochs,
                    'learning_rate': lr,
                    'f1_score': f1
                })
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_params = {
                        'hidden_dims': hidden_dims,
                        'epochs': epochs,
                        'learning_rate': lr
                    }
                    print(f"New best F1: {f1:.4f}")
    
    print(f"\nBest Autoencoder Params: {best_params}")
    print(f"Best F1 Score: {best_f1:.4f}")
    
    return {'best_params': best_params, 'best_f1': best_f1, 'all_results': results}


def tune_transformer(X_train, y_train, X_test, y_test) -> dict:
    """Transformer 하이퍼파라미터 튜닝"""
    print("\n[Transformer Hyperparameter Tuning]")
    
    param_grid = {
        'embed_dim': [32, 64],
        'num_heads': [2, 4],
        'num_layers': [1, 2],
        'epochs': [20, 30]
    }
    
    best_f1 = 0
    best_params = None
    results = []
    
    for embed_dim in param_grid['embed_dim']:
        for num_heads in param_grid['num_heads']:
            for num_layers in param_grid['num_layers']:
                for epochs in param_grid['epochs']:
                    print(f"\nTrying: embed={embed_dim}, heads={num_heads}, layers={num_layers}, epochs={epochs}")
                    
                    model = AnomalyDetectorTransformer(
                        input_dim=X_train.shape[1],
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        num_layers=num_layers,
                        ff_dim=embed_dim * 2,
                        epochs=epochs,
                        batch_size=256
                    )
                    model.fit(X_train, y_train)
                    
                    y_pred = model.predict(X_test)
                    f1 = f1_score(y_test, y_pred)
                    
                    results.append({
                        'embed_dim': embed_dim,
                        'num_heads': num_heads,
                        'num_layers': num_layers,
                        'epochs': epochs,
                        'f1_score': f1
                    })
                    
                    if f1 > best_f1:
                        best_f1 = f1
                        best_params = {
                            'embed_dim': embed_dim,
                            'num_heads': num_heads,
                            'num_layers': num_layers,
                            'epochs': epochs
                        }
                        print(f"New best F1: {f1:.4f}")
    
    print(f"\nBest Transformer Params: {best_params}")
    print(f"Best F1 Score: {best_f1:.4f}")
    
    return {'best_params': best_params, 'best_f1': best_f1, 'all_results': results}


def compare_models(X_train, y_train, X_test, y_test) -> pd.DataFrame:
    """모델 성능 비교"""
    results = []
    
    # Autoencoder
    ae_model = AnomalyDetectorAE(
        input_dim=X_train.shape[1],
        hidden_dims=[64, 32, 16],
        epochs=50,
        batch_size=256
    )
    ae_result = train_and_evaluate(ae_model, X_train, y_train, X_test, y_test, "Autoencoder")
    results.append(ae_result)
    
    # Transformer
    tf_model = AnomalyDetectorTransformer(
        input_dim=X_train.shape[1],
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        epochs=30,
        batch_size=256
    )
    tf_result = train_and_evaluate(tf_model, X_train, y_train, X_test, y_test, "Transformer")
    results.append(tf_result)
    
    # 결과 정리
    comparison_df = pd.DataFrame([
        {
            'Model': r['model_name'],
            'Accuracy': f"{r['accuracy']:.4f}",
            'F1 Score': f"{r['f1_score']:.4f}",
            'AUC Score': f"{r['auc_score']:.4f}",
            'Train Time (s)': f"{r['train_time']:.2f}"
        }
        for r in results
    ])
    
    print("\n" + "="*60)
    print("MODEL COMPARISON RESULTS")
    print("="*60)
    print(comparison_df.to_string(index=False))
    
    return comparison_df, results


def save_results(comparison_df: pd.DataFrame, filename: str = "model_comparison.csv"):
    """결과 저장"""
    comparison_df.to_csv(filename, index=False)
    print(f"\nResults saved to {filename}")


if __name__ == "__main__":
    # 데이터 로드 및 전처리
    print("Loading data...")
    train_df, test_df = load_data("data/KDDTrain+.txt", "data/KDDTest+.txt")
    train_df = add_binary_label(add_attack_type(train_df))
    test_df = add_binary_label(add_attack_type(test_df))
    data = prepare_data(train_df, test_df)
    
    X_train, y_train = data['X_train'], data['y_train']
    X_test, y_test = data['X_test'], data['y_test']
    
    # 모델 비교
    comparison_df, results = compare_models(X_train, y_train, X_test, y_test)
    
    # 결과 저장
    save_results(comparison_df)
    
    # 하이퍼파라미터 튜닝
    # ae_tuning = tune_autoencoder(X_train, y_train, X_test, y_test)
    # tf_tuning = tune_transformer(X_train, y_train, X_test, y_test)