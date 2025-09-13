# =================================================================
# 하이브리드 CTR 모델: baseline.py + 고급 피처 엔지니어링
# PyTorch 딥러닝 + LSTM + 피처 엔지니어링
# =================================================================
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import os
import random
import gc
import warnings
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings('ignore')

# =================================================================
# 설정
# =================================================================
CFG = {
    'BATCH_SIZE': 2048,
    'EPOCHS': 15,
    'LEARNING_RATE': 1e-3,
    'SEED': 42,
    'SAMPLE_SIZE': 2000000,  # 200만개
}

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(CFG['SEED'])

# =================================================================
# 안전한 데이터 로딩
# =================================================================
def safe_load_data(path, sample_size=None):
    print(f"데이터 로딩: {path}")
    
    try:
        # 청크 기반 로딩
        parquet_file = pq.ParquetFile(path)
        total_rows = parquet_file.metadata.num_rows
        print(f"전체 데이터: {total_rows:,}행")
        
        if sample_size and sample_size < total_rows:
            print(f"샘플링: {sample_size:,}개")
            chunks = []
            processed = 0
            target_per_chunk = sample_size // 10
            
            for batch in parquet_file.iter_batches(batch_size=50000):
                chunk_df = batch.to_pandas()
                
                if len(chunk_df) > target_per_chunk:
                    sampled_chunk = chunk_df.sample(n=target_per_chunk, random_state=CFG['SEED'])
                else:
                    sampled_chunk = chunk_df
                
                chunks.append(sampled_chunk)
                processed += len(sampled_chunk)
                
                if processed >= sample_size:
                    break
                
                del chunk_df
                gc.collect()
            
            df = pd.concat(chunks, ignore_index=True)[:sample_size]
            del chunks
            gc.collect()
        else:
            df = pd.read_parquet(path)
        
        print(f"로딩 완료: {df.shape}")
        return df
        
    except Exception as e:
        print(f"청크 로딩 실패: {e}, 전체 로딩 시도...")
        df = pd.read_parquet(path)
        if sample_size and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=CFG['SEED']).reset_index(drop=True)
        return df

# =================================================================
# 고급 피처 엔지니어링
# =================================================================
def create_advanced_features(df, label_encoders=None, target_encodings=None, is_train=True):
    print("고급 피처 엔지니어링 시작...")
    
    df = df.copy()
    
    # 1. 기본 통계 피처
    print("1. 기본 통계 피처 생성...")
    hist_b_cols = [col for col in df.columns if col.startswith('history_b_')]
    if hist_b_cols:
        df['history_b_mean'] = df[hist_b_cols].mean(axis=1).astype(np.float32)
        df['history_b_std'] = df[hist_b_cols].std(axis=1).astype(np.float32)
        df['history_b_max'] = df[hist_b_cols].max(axis=1).astype(np.float32)
        df['history_b_min'] = df[hist_b_cols].min(axis=1).astype(np.float32)
        df['history_b_range'] = (df['history_b_max'] - df['history_b_min']).astype(np.float32)
    
    hist_a_cols = [col for col in df.columns if col.startswith('history_a_')]
    if hist_a_cols:
        df['history_a_mean'] = df[hist_a_cols].mean(axis=1).astype(np.float32)
        df['history_a_std'] = df[hist_a_cols].std(axis=1).astype(np.float32)
        df['history_a_max'] = df[hist_a_cols].max(axis=1).astype(np.float32)
        df['history_a_min'] = df[hist_a_cols].min(axis=1).astype(np.float32)
    
    # 2. 시간 피처
    print("2. 시간 피처 생성...")
    df['hour'] = pd.to_numeric(df['hour'], errors='coerce').fillna(0)
    df['day_of_week'] = pd.to_numeric(df['day_of_week'], errors='coerce').fillna(0)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24).astype(np.float32)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24).astype(np.float32)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7).astype(np.float32)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7).astype(np.float32)
    
    # 3. 교차 피처
    print("3. 교차 피처 생성...")
    df['day_hour'] = df['day_of_week'].astype(str) + '_' + df['hour'].astype(str)
    df['gender_age'] = df['gender'].astype(str) + '_' + df['age_group'].astype(str)
    df['inv_age'] = df['inventory_id'].astype(str) + '_' + df['age_group'].astype(str)
    
    # 4. 시퀀스 피처 (간단버전)
    print("4. 시퀀스 기본 피처...")
    df['seq_length'] = df['seq'].fillna('').apply(lambda x: len(x.split(',')) if x else 0).astype(np.int16)
    df['seq_is_empty'] = (df['seq_length'] == 0).astype(np.int8)
    
    # 5. Label Encoding (카테고리 → 숫자)
    print("5. 카테고리 인코딩...")
    categorical_cols = ['gender', 'age_group', 'day_hour', 'gender_age', 'inv_age']
    
    if is_train:
        label_encoders = {}
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                label_encoders[col] = le
    else:
        for col in categorical_cols:
            if col in df.columns and col in label_encoders:
                le = label_encoders[col]
                # 새로운 카테고리는 0으로 처리
                df[f'{col}_encoded'] = df[col].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else 0
                )
    
    # 6. 간단한 타겟 인코딩 (훈련 데이터만)
    if is_train and 'clicked' in df.columns:
        print("6. 타겟 인코딩...")
        target_encodings = {}
        TARGET = 'clicked'
        
        # K-fold 타겟 인코딩
        kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=CFG['SEED'])
        
        for col in ['inventory_id', 'day_hour']:
            if col in df.columns:
                df[f'{col}_target'] = 0.0
                overall_mean = df[TARGET].mean()
                
                for train_idx, val_idx in kf.split(df, df[TARGET]):
                    train_mean = df.iloc[train_idx].groupby(col)[TARGET].mean()
                    df.iloc[val_idx, df.columns.get_loc(f'{col}_target')] = df.iloc[val_idx][col].map(train_mean)
                
                df[f'{col}_target'].fillna(overall_mean, inplace=True)
                
                # 전체 인코딩 저장
                full_encoding = df.groupby(col)[TARGET].mean()
                target_encodings[col] = (full_encoding, overall_mean)
    
    elif not is_train and target_encodings:
        print("6. 테스트 데이터 타겟 인코딩...")
        for col in ['inventory_id', 'day_hour']:
            if col in target_encodings:
                encoding_map, overall_mean = target_encodings[col]
                df[f'{col}_target'] = df[col].map(encoding_map).fillna(overall_mean)
    
    print(f"피처 엔지니어링 완료. 최종 shape: {df.shape}")
    
    if is_train:
        return df, label_encoders, target_encodings
    else:
        return df

# =================================================================
# Dataset 클래스
# =================================================================
class AdvancedClickDataset(Dataset):
    def __init__(self, df, feature_cols, seq_col, target_col=None, has_target=True):
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.seq_col = seq_col
        self.target_col = target_col
        self.has_target = has_target
        
        # 비-시퀀스 피처: 수치형으로 변환
        self.X = self.df[self.feature_cols].fillna(0).astype(np.float32).values
        
        # 시퀀스 데이터
        self.seq_strings = self.df[self.seq_col].astype(str).fillna('').values
        
        if self.has_target:
            self.y = self.df[self.target_col].astype(np.float32).values
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        
        # 시퀀스 처리 (최대 50개 아이템만)
        s = self.seq_strings[idx]
        if s and s != 'nan':
            try:
                items = s.split(',')[:50]  # 최대 50개만
                arr = np.array([float(item.strip()) for item in items if item.strip()], dtype=np.float32)
            except:
                arr = np.array([], dtype=np.float32)
        else:
            arr = np.array([], dtype=np.float32)
        
        if arr.size == 0:
            arr = np.array([0.0], dtype=np.float32)  # 빈 시퀀스 방어
        
        seq = torch.from_numpy(arr)
        
        if self.has_target:
            y = torch.tensor(self.y[idx], dtype=torch.float32)
            return x, seq, y
        else:
            return x, seq

def collate_fn_train(batch):
    xs, seqs, ys = zip(*batch)
    xs = torch.stack(xs)
    ys = torch.stack(ys)
    seqs_padded = nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0.0)
    seq_lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    seq_lengths = torch.clamp(seq_lengths, min=1)
    return xs, seqs_padded, seq_lengths, ys

def collate_fn_infer(batch):
    xs, seqs = zip(*batch)
    xs = torch.stack(xs)
    seqs_padded = nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0.0)
    seq_lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    seq_lengths = torch.clamp(seq_lengths, min=1)
    return xs, seqs_padded, seq_lengths

# =================================================================
# 고급 모델 아키텍처
# =================================================================
class AdvancedCTRModel(nn.Module):
    def __init__(self, d_features, lstm_hidden=64, lstm_layers=2, 
                 hidden_units=[512, 256, 128], dropout=0.3):
        super().__init__()
        
        # 피처 정규화
        self.bn_x = nn.BatchNorm1d(d_features)
        
        # 시퀀스 처리: Bi-LSTM + Attention
        self.lstm = nn.LSTM(
            input_size=1, 
            hidden_size=lstm_hidden, 
            num_layers=lstm_layers,
            batch_first=True, 
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(lstm_hidden * 2, lstm_hidden),
            nn.Tanh(),
            nn.Linear(lstm_hidden, 1),
            nn.Softmax(dim=1)
        )
        
        # MLP with residual connections
        input_dim = d_features + lstm_hidden * 2
        layers = []
        prev_dim = input_dim
        
        for i, h in enumerate(hidden_units):
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = h
        
        layers.append(nn.Linear(prev_dim, 1))
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x_feats, x_seq, seq_lengths):
        batch_size = x_feats.size(0)
        
        # 피처 정규화
        x = self.bn_x(x_feats)
        
        # 시퀀스 처리
        x_seq = x_seq.unsqueeze(-1)  # (B, L, 1)
        
        # LSTM
        packed = nn.utils.rnn.pack_padded_sequence(
            x_seq, seq_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        
        # Attention pooling
        attn_weights = self.attention(lstm_out)  # (B, L, 1)
        seq_repr = torch.sum(lstm_out * attn_weights, dim=1)  # (B, lstm_hidden*2)
        
        # Concatenate features
        combined = torch.cat([x, seq_repr], dim=1)
        
        # MLP
        output = self.mlp(combined)
        return output.squeeze(1)

# =================================================================
# 훈련 함수
# =================================================================
def train_advanced_model(train_df, feature_cols, seq_col, target_col, 
                        batch_size=2048, epochs=15, lr=1e-3, device="cpu"):
    
    print("모델 훈련 시작...")
    
    # 1. Train/Validation Split
    tr_df, va_df = train_test_split(train_df, test_size=0.2, random_state=CFG['SEED'], 
                                    shuffle=True, stratify=train_df[target_col])
    
    print(f"Train: {tr_df.shape}, Validation: {va_df.shape}")
    
    # 2. Dataset & DataLoader
    train_dataset = AdvancedClickDataset(tr_df, feature_cols, seq_col, target_col, has_target=True)
    val_dataset = AdvancedClickDataset(va_df, feature_cols, seq_col, target_col, has_target=True)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_train)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_train)
    
    # 3. 모델 초기화
    d_features = len(feature_cols)
    model = AdvancedCTRModel(
        d_features=d_features, 
        lstm_hidden=64, 
        lstm_layers=2,
        hidden_units=[512, 256, 128], 
        dropout=0.3
    ).to(device)
    
    # 4. 손실함수 & 옵티마이저
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    best_val_auc = 0.0  # AUC는 높을수록 좋으므로 0으로 초기화
    patience = 5
    patience_counter = 0
    
    # 5. 훈련 루프
    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for xs, seqs, seq_lens, ys in tqdm(train_loader, desc=f"Train Epoch {epoch}"):
            xs, seqs, seq_lens, ys = xs.to(device), seqs.to(device), seq_lens.to(device), ys.to(device)
            
            optimizer.zero_grad()
            logits = model(xs, seqs, seq_lens)
            loss = criterion(logits, ys)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
            train_batches += 1
        
        train_loss /= train_batches
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for xs, seqs, seq_lens, ys in tqdm(val_loader, desc=f"Val Epoch {epoch}"):
                xs, seqs, seq_lens, ys = xs.to(device), seqs.to(device), seq_lens.to(device), ys.to(device)
                logits = model(xs, seqs, seq_lens)
                loss = criterion(logits, ys)
                val_loss += loss.item()
                val_batches += 1
                
                # AUC 계산용 데이터 수집
                probs = torch.sigmoid(logits)
                all_preds.append(probs.cpu().numpy())
                all_targets.append(ys.cpu().numpy())
        
        val_loss /= val_batches
        
        # AUC 계산
        from sklearn.metrics import roc_auc_score
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        val_auc = roc_auc_score(all_targets, all_preds)
        
        print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if val_auc > best_val_auc:  # AUC 기준으로 변경 (높을수록 좋음)
            best_val_auc = val_auc
            patience_counter = 0
            # 완전한 모델 저장 (구조 + 가중치)
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_class': model.__class__,
                'model_args': {
                    'd_features': d_features,
                    'lstm_hidden': 64,
                    'lstm_layers': 2,
                    'hidden_units': [512, 256, 128],
                    'dropout': 0.3
                },
                'epoch': epoch,
                'best_auc': val_auc,
                'optimizer_state_dict': optimizer.state_dict()
            }, 'C:/Users/82107/Desktop/open/best_hybrid_model.pth')
            print(f"새로운 최고 모델 저장! AUC: {val_auc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    # 최고 모델 안전하게 로드
    checkpoint = torch.load('C:/Users/82107/Desktop/open/best_hybrid_model.pth', 
                           map_location=device, weights_only=False)  # 보안 경고 해결
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"최고 성능 모델 로드 완료! 최고 AUC: {checkpoint['best_auc']:.4f}")
    
    return model

# =================================================================
# 메인 실행
# =================================================================
def main():
    print("=== 하이브리드 CTR 모델 ===")
    
    # 1. 데이터 로딩
    print("\n1. 데이터 로딩...")
    all_train = safe_load_data('C:/Users/82107/Desktop/open/train.parquet', CFG['SAMPLE_SIZE'])
    test = safe_load_data('C:/Users/82107/Desktop/open/test.parquet')
    
    print(f"Train shape: {all_train.shape}")
    print(f"Test shape: {test.shape}")
    
    # 2. 클래스 균형 샘플링 (baseline 방식)
    print("\n2. 클래스 균형 샘플링...")
    clicked_1 = all_train[all_train['clicked'] == 1]
    clicked_0 = all_train[all_train['clicked'] == 0].sample(n=len(clicked_1)*2, random_state=CFG['SEED'])
    train = pd.concat([clicked_1, clicked_0], axis=0).sample(frac=1, random_state=CFG['SEED']).reset_index(drop=True)
    
    print(f"Final train shape: {train.shape}")
    print(f"Clicked 0: {len(train[train['clicked']==0]):,}")
    print(f"Clicked 1: {len(train[train['clicked']==1]):,}")
    
    # 3. 피처 엔지니어링
    print("\n3. 고급 피처 엔지니어링...")
    train_processed, label_encoders, target_encodings = create_advanced_features(train, is_train=True)
    
    test_ids = test['ID'].copy()
    test = test.drop(columns=['ID'])
    test_processed = create_advanced_features(test, label_encoders, target_encodings, is_train=False)
    
    # 4. 피처 선택
    TARGET = 'clicked'
    SEQ_COL = 'seq'
    EXCLUDE_COLS = {TARGET, SEQ_COL, 'ID', 'day_hour', 'gender_age', 'inv_age'}  # 원본 카테고리 제외
    feature_cols = [c for c in train_processed.columns if c not in EXCLUDE_COLS]
    
    print(f"사용할 피처 수: {len(feature_cols)}")
    
    # 5. 모델 훈련
    print("\n4. 모델 훈련...")
    model = train_advanced_model(
        train_df=train_processed,
        feature_cols=feature_cols,
        seq_col=SEQ_COL,
        target_col=TARGET,
        batch_size=CFG['BATCH_SIZE'],
        epochs=CFG['EPOCHS'],
        lr=CFG['LEARNING_RATE'],
        device=device
    )
    
    # 6. 예측
    print("\n5. 테스트 예측...")
    test_dataset = AdvancedClickDataset(test_processed, feature_cols, SEQ_COL, has_target=False)
    test_loader = DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, collate_fn=collate_fn_infer)
    
    model.eval()
    predictions = []
    with torch.no_grad():
        for xs, seqs, lens in tqdm(test_loader, desc="Inference"):
            xs, seqs, lens = xs.to(device), seqs.to(device), lens.to(device)
            outputs = torch.sigmoid(model(xs, seqs, lens))
            predictions.append(outputs.cpu())
    
    test_preds = torch.cat(predictions).numpy()
    
    # 7. 제출 파일
    submission = pd.DataFrame({
        'ID': test_ids,
        'clicked': test_preds
    })
    submission.to_csv('C:/Users/82107/Desktop/open/hybrid_submission.csv', index=False)
    
    print(f"\n=== 완료 ===")
    print(f"예측값 범위: {test_preds.min():.4f} ~ {test_preds.max():.4f}")
    print(f"제출 파일: hybrid_submission.csv")

if __name__ == "__main__":
    main()