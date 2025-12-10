"""
NSL-KDD 데이터 전처리
- 범주형 변수 인코딩
- 수치형 변수 스케일링
- 학습/테스트 데이터 분리
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


# 범주형 컬럼
CATEGORICAL_COLS = ['protocol_type', 'service', 'flag']

# 제외할 컬럼
DROP_COLS = ['label', 'difficulty', 'attack_type']


class Preprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = None
    
    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """학습 데이터 전처리 (fit + transform)"""
        df = df.copy()
        
        # 1. 범주형 인코딩
        for col in CATEGORICAL_COLS:
            self.label_encoders[col] = LabelEncoder()
            df[col] = self.label_encoders[col].fit_transform(df[col])
        
        # 2. 불필요한 컬럼 제거
        drop_cols = [c for c in DROP_COLS if c in df.columns]
        target = df['is_attack'].values if 'is_attack' in df.columns else None
        df = df.drop(columns=drop_cols + ['is_attack'], errors='ignore')
        
        # 3. 피처 컬럼 저장
        self.feature_columns = df.columns.tolist()
        
        # 4. 스케일링
        X = self.scaler.fit_transform(df.values)
        
        return X, target
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """테스트 데이터 전처리 (transform only)"""
        df = df.copy()
        
        # 1. 범주형 인코딩 (학습 데이터 기준)
        for col in CATEGORICAL_COLS:
            # 학습에 없던 카테고리 처리
            df[col] = df[col].apply(
                lambda x: x if x in self.label_encoders[col].classes_ else 'unknown'
            )
            # unknown 클래스 추가
            if 'unknown' not in self.label_encoders[col].classes_:
                self.label_encoders[col].classes_ = np.append(
                    self.label_encoders[col].classes_, 'unknown'
                )
            df[col] = self.label_encoders[col].transform(df[col])
        
        # 2. 불필요한 컬럼 제거
        drop_cols = [c for c in DROP_COLS if c in df.columns]
        target = df['is_attack'].values if 'is_attack' in df.columns else None
        df = df.drop(columns=drop_cols + ['is_attack'], errors='ignore')
        
        # 3. 피처 컬럼 순서 맞추기
        df = df[self.feature_columns]
        
        # 4. 스케일링
        X = self.scaler.transform(df.values)
        
        return X, target
    
    def get_feature_names(self) -> list:
        """피처 컬럼명 반환"""
        return self.feature_columns


def prepare_data(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """전체 전처리 파이프라인"""
    preprocessor = Preprocessor()
    
    # 전처리 수행
    X_train, y_train = preprocessor.fit_transform(train_df)
    X_test, y_test = preprocessor.transform(test_df)
    
    # 결과 출력
    print(f"[Preprocessing Complete]")
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"Features: {len(preprocessor.get_feature_names())}")
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'preprocessor': preprocessor
    }


if __name__ == "__main__":
    from data_loader import load_data, add_attack_type, add_binary_label
    
    # 데이터 로드
    train_df, test_df = load_data("data/KDDTrain+.txt", "data/KDDTest+.txt")
    train_df = add_binary_label(add_attack_type(train_df))
    test_df = add_binary_label(add_attack_type(test_df))
    
    # 전처리
    data = prepare_data(train_df, test_df)
    
    print(f"\nX_train sample:\n{data['X_train'][:3, :5]}")
    print(f"\ny_train sample: {data['y_train'][:10]}")