"""
NSL-KDD 데이터 로더
- 데이터 로드 및 컬럼 지정
- 공격 유형 분류
- 이진 라벨 생성
"""

import pandas as pd
import numpy as np

# NSL-KDD 컬럼명 (41개 특성 + 라벨 + 난이도)
COLUMN_NAMES = [
    # 기본 특성 (9개)
    'duration', 'protocol_type', 'service', 'flag',
    'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent',
    
    # 콘텐츠 특성 (13개)
    'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
    'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds',
    'is_host_login', 'is_guest_login',
    
    # 시간 기반 트래픽 특성 (9개)
    'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate',
    
    # 호스트 기반 트래픽 특성 (10개)
    'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate',
    
    # 라벨
    'label', 'difficulty'
]

# 공격 유형 분류
ATTACK_TYPES = {
    'normal': 'normal',
    # DoS
    'back': 'DoS', 'land': 'DoS', 'neptune': 'DoS', 'pod': 'DoS',
    'smurf': 'DoS', 'teardrop': 'DoS', 'mailbomb': 'DoS',
    'apache2': 'DoS', 'processtable': 'DoS', 'udpstorm': 'DoS',
    # Probe
    'ipsweep': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe',
    'satan': 'Probe', 'mscan': 'Probe', 'saint': 'Probe',
    # R2L
    'ftp_write': 'R2L', 'guess_passwd': 'R2L', 'imap': 'R2L',
    'multihop': 'R2L', 'phf': 'R2L', 'spy': 'R2L',
    'warezclient': 'R2L', 'warezmaster': 'R2L', 'sendmail': 'R2L',
    'named': 'R2L', 'snmpgetattack': 'R2L', 'snmpguess': 'R2L',
    'xlock': 'R2L', 'xsnoop': 'R2L', 'worm': 'R2L',
    # U2R
    'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'perl': 'U2R',
    'rootkit': 'U2R', 'httptunnel': 'U2R', 'ps': 'U2R',
    'sqlattack': 'U2R', 'xterm': 'U2R'
}


def load_data(train_path: str, test_path: str) -> tuple:
    """NSL-KDD 데이터 로드"""
    train_df = pd.read_csv(train_path, names=COLUMN_NAMES, header=None)
    test_df = pd.read_csv(test_path, names=COLUMN_NAMES, header=None)
    return train_df, test_df


def add_attack_type(df: pd.DataFrame) -> pd.DataFrame:
    """공격 유형 컬럼 추가 (DoS, Probe, R2L, U2R)"""
    df = df.copy()
    df['attack_type'] = df['label'].map(ATTACK_TYPES).fillna('Unknown')
    return df


def add_binary_label(df: pd.DataFrame) -> pd.DataFrame:
    """이진 라벨 추가 (0=정상, 1=공격)"""
    df = df.copy()
    df['is_attack'] = (df['label'] != 'normal').astype(int)
    return df


def get_data_info(df: pd.DataFrame, name: str = "Data"):
    """데이터 기본 정보 출력"""
    print(f"\n[{name}]")
    print(f"Total: {len(df):,} samples")
    print(f"Columns: {len(df.columns)}")
    
    if 'is_attack' in df.columns:
        counts = df['is_attack'].value_counts()
        print(f"Normal: {counts.get(0, 0):,} ({counts.get(0, 0)/len(df)*100:.1f}%)")
        print(f"Attack: {counts.get(1, 0):,} ({counts.get(1, 0)/len(df)*100:.1f}%)")


if __name__ == "__main__":
    train_df, test_df = load_data("data/KDDTrain+.txt", "data/KDDTest+.txt")
    
    train_df = add_attack_type(train_df)
    train_df = add_binary_label(train_df)
    test_df = add_attack_type(test_df)
    test_df = add_binary_label(test_df)
    
    get_data_info(train_df, "Train Data")
    get_data_info(test_df, "Test Data")