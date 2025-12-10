"""
Autoencoder 기반 이상 탐지 모델
- 정상 데이터로 학습
- 복원 오차로 이상 판별
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.data_loader import load_data, add_attack_type, add_binary_label
from preprocessing.preprocessor import prepare_data

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, f1_score, roc_auc_score


class Autoencoder(nn.Module):
    """Autoencoder 모델 정의"""
    
    def __init__(self, input_dim: int, hidden_dims: list = [64, 32, 16]):
        super().__init__()
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.BatchNorm1d(h_dim)
            ])
            prev_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        hidden_dims_rev = hidden_dims[::-1]
        for i, h_dim in enumerate(hidden_dims_rev[1:]):
            decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.BatchNorm1d(h_dim)
            ])
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def get_reconstruction_error(self, x):
        """복원 오차 계산"""
        with torch.no_grad():
            reconstructed = self.forward(x)
            error = torch.mean((x - reconstructed) ** 2, dim=1)
        return error.numpy()


class AnomalyDetectorAE:
    """Autoencoder 기반 이상 탐지기"""
    
    def __init__(self, input_dim: int, hidden_dims: list = [64, 32, 16],
                 learning_rate: float = 0.001, epochs: int = 50, 
                 batch_size: int = 256):
        self.model = Autoencoder(input_dim, hidden_dims)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.threshold = None
        self.history = {'train_loss': []}
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray = None):
        """모델 학습 (정상 데이터만 사용)"""
        # 정상 데이터만 선택
        if y_train is not None:
            X_normal = X_train[y_train == 0]
        else:
            X_normal = X_train
        
        # 데이터 준비
        X_tensor = torch.FloatTensor(X_normal)
        dataset = TensorDataset(X_tensor, X_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # 학습 설정
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # 학습 루프
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_x, _ in dataloader:
                optimizer.zero_grad()
                output = self.model(batch_x)
                loss = criterion(output, batch_x)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            self.history['train_loss'].append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{self.epochs}] Loss: {avg_loss:.6f}")
        
        # 임계값 설정 (정상 데이터의 복원 오차 기준)
        self.model.eval()
        errors = self.model.get_reconstruction_error(X_tensor)
        self.threshold = np.percentile(errors, 95)
        print(f"Threshold set to: {self.threshold:.6f}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """이상 탐지 예측 (0=정상, 1=이상)"""
        self.model.eval()
        X_tensor = torch.FloatTensor(X)
        errors = self.model.get_reconstruction_error(X_tensor)
        predictions = (errors > self.threshold).astype(int)
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """복원 오차 점수 반환"""
        self.model.eval()
        X_tensor = torch.FloatTensor(X)
        errors = self.model.get_reconstruction_error(X_tensor)
        return errors
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """모델 평가"""
        y_pred = self.predict(X_test)
        y_score = self.predict_proba(X_test)
        
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_score)
        
        print("\n[Autoencoder Evaluation]")
        print(classification_report(y_test, y_pred, target_names=['Normal', 'Attack']))
        print(f"F1 Score: {f1:.4f}")
        print(f"AUC Score: {auc:.4f}")
        
        return {'f1': f1, 'auc': auc, 'predictions': y_pred}


if __name__ == "__main__":
    # 데이터 로드 및 전처리
    train_df, test_df = load_data("data/KDDTrain+.txt", "data/KDDTest+.txt")
    train_df = add_binary_label(add_attack_type(train_df))
    test_df = add_binary_label(add_attack_type(test_df))
    data = prepare_data(train_df, test_df)
    
    # 모델 학습
    detector = AnomalyDetectorAE(
        input_dim=data['X_train'].shape[1],
        hidden_dims=[64, 32, 16],
        epochs=50,
        batch_size=256
    )
    detector.fit(data['X_train'], data['y_train'])
    
    # 평가
    results = detector.evaluate(data['X_test'], data['y_test'])

'''
Transformer 구조
[입력 41개]
     ↓
[Input Embedding] → 64차원
     ↓
[Transformer Block x 2]
  - Multi-Head Attention (4 heads)
  - Feed Forward (128)
  - LayerNorm + Dropout
     ↓
[Classifier]
  - Linear(64→32) → ReLU
  - Linear(32→1) → Sigmoid
     ↓
[출력: 0~1 확률]
'''