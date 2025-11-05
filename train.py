import pandas as pd
import polars as pl
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
import warnings
import gc
import joblib
import os

warnings.filterwarnings('ignore')

# ==================================
# GPU/CPU 자동 감지
# ==================================
def check_gpu_available():
    """GPU 사용 가능 여부를 확인"""
    try:
        # XGBoost GPU 체크
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("GPU 감지됨 - GPU 모드로 학습합니다")
            return True
    except:
        pass

    print("GPU 미감지 - CPU 모드로 학습합니다")
    return False

USE_GPU = check_gpu_available()

# ==================================
# 설정
# ==================================
TRAIN_PATH = "../data/train_undersampled_1_to_5.parquet"
MODELS_DIR = "../models/"

# 모델 및 재현성 설정
N_SPLITS = 3
RANDOM_STATE = 42

# 폴더 생성
os.makedirs(MODELS_DIR, exist_ok=True)

# ==================================
# 데이터 로드 및 피처 엔지니어링
# ==================================
print("데이터 로드 시작...")
if not os.path.exists(TRAIN_PATH):
    print(f"오류: 훈련 데이터 파일이 없습니다. 다음 경로를 확인해주세요: {os.path.abspath(TRAIN_PATH)}")
    print("먼저 python src/create_undersampled_dataset.py 를 실행했는지 확인해주세요.")
    exit()

train_df = pl.read_parquet(TRAIN_PATH)
print("데이터 로드 완료!")

print("피처엔지니어링 시작...!")

seq_as_list = pl.col("seq").str.split(",").cast(pl.List(pl.Float64))
history_b_cols = [f'history_b_{i}' for i in range(1,31)]
history_a_cols = [f'history_a_{i}' for i in range(1,8)]

categorical_features = ['gender', 'age_group', 'inventory_id', 'day_of_week', 'hour']

train_df = (
    train_df
    .with_columns([
        seq_as_list.list.mean().alias('seq_mean'),
        seq_as_list.list.std().alias('seq_std'),
        seq_as_list.list.max().alias('seq_max'),
        seq_as_list.list.min().alias('seq_min')
    ])
    .drop('seq')
    .with_columns([
        pl.mean_horizontal(history_b_cols).alias('history_b_mean'),
        pl.concat_list(history_b_cols).list.std().alias('history_b_std'),
        pl.mean_horizontal(history_a_cols).alias('history_a_mean'),
        pl.concat_list(history_a_cols).list.std().alias('history_a_std'),
        (pl.col('hour').cast(pl.Utf8) + "_" + pl.col('day_of_week').cast(pl.Utf8)).alias('day_hour'),
        (pl.col('inventory_id').cast(pl.Utf8) + "_" + pl.col('age_group').cast(pl.Utf8)).alias('inv_id_age_group'),
        pl.col("feat_e_3").is_null().cast(pl.Int8).alias("is_feat_e_3_missing")
    ])
)

#Pandas로 전환
train_df = train_df.to_pandas()

# 타입 변환
train_df['hour'] = pd.to_numeric(train_df['hour'], errors='coerce')
train_df['day_of_week'] = pd.to_numeric(train_df['day_of_week'], errors='coerce')

# 주기성 피처
train_df['hour_sin'] = np.sin(2 * np.pi * train_df['hour'] / 24)
train_df['hour_cos'] = np.cos(2 * np.pi * train_df['hour'] / 24)
train_df['day_of_week_sin'] = np.sin(2 * np.pi * train_df['day_of_week'] / 7)
train_df['day_of_week_cos'] = np.cos(2 * np.pi * train_df['day_of_week'] / 7)

# 타겟 인코딩
encoding_features = ['inventory_id', 'day_hour', 'age_group', 'inv_id_age_group']
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

for col in encoding_features:
    train_df[f'{col}_target_encoded'] = 0.0
    for train_idx, val_idx in skf.split(train_df, train_df['clicked']):
        train_encoding = train_df.iloc[train_idx].groupby(col)['clicked'].mean()
        train_df.loc[val_idx, f'{col}_target_encoded'] = train_df.iloc[val_idx][col].map(train_encoding)

target_encoding_maps = {}
for col in encoding_features:
    target_encoding_maps[col] = train_df.groupby(col)['clicked'].mean()

train_df = train_df.drop(columns=['day_hour', 'inv_id_age_group'])

# 이상치 처리
print("이상치 처리 중...")
numeric_features = (
    [f'l_feat_{i}' for i in range(1, 28)] + [f'feat_a_{i}' for i in range(1, 19)] +
    [f'feat_b_{i}' for i in range(1, 7)] + [f'feat_c_{i}' for i in range(1, 9)] +
    [f'feat_d_{i}' for i in range(1, 7)] + [f'feat_e_{i}' for i in range(1, 11)] +
    [f'history_a_{i}' for i in range(1, 8)] + [f'history_b_{i}' for i in range(1, 31)] +
    ['history_b_mean', 'history_b_std', 'history_a_mean', 'history_a_std',
     'seq_mean', 'seq_std', 'seq_max', 'seq_min']
)

clip_bounds = {}
for col in numeric_features:
    if col in train_df.columns:
        lower = train_df[col].quantile(0.01)
        upper = train_df[col].quantile(0.99)
        clip_bounds[col] = (lower, upper)
        train_df[col] = train_df[col].clip(lower, upper)

print("피처엔지니어링 완료!")

# ==================================
# 모델 학습
# ==================================
X = train_df.drop(columns=['clicked'])
y = train_df['clicked']

# 추론 시 컬럼 순서 일치를 위해 저장
train_columns = X.columns.tolist()

for col in X.select_dtypes(include='object').columns:
    X[col] = X[col].astype('category').cat.codes

scale_pos_weight_value = y.value_counts()[0] / y.value_counts()[1]

# --- LightGBM --- #
print("\n[LightGBM] 모델 학습 시작...")
lgbm_params = {
    'objective': 'binary', 'metric': 'auc',
    'scale_pos_weight': scale_pos_weight_value, 'random_state': RANDOM_STATE, 'verbose': -1,
    'learning_rate': 0.027357360985606104, 'num_leaves': 94, 'max_depth': 12,
    'min_child_samples': 56, 'subsample': 0.7355028193733041, 'colsample_bytree': 0.5020780377426634,
    'reg_alpha': 1.3450748035440614, 'reg_lambda': 0.8851408162033914
}

# GPU/CPU 설정
if USE_GPU:
    lgbm_params.update({'device': 'gpu', 'gpu_platform_id': 0, 'gpu_device_id': 0})
else:
    lgbm_params.update({'device': 'cpu'})

train_data_full = lgb.Dataset(X, label=y)
final_model = lgb.train(lgbm_params, train_data_full, num_boost_round=1308, callbacks=[lgb.log_evaluation(100)])
print("LightGBM 모델 학습 완료!")

# --- XGBoost --- #
print("\n[XGBoost] 모델 학습 시작...")
xgb_params = {
    'objective': 'binary:logistic', 'eval_metric': 'auc',
    'scale_pos_weight': scale_pos_weight_value, 'random_state': RANDOM_STATE, 'verbosity': 1,
    'max_depth': 10, 'learning_rate': 0.006229765764527009, 'subsample': 0.7765182233835292,
    'colsample_bytree': 0.573556750803278, 'min_child_weight': 45, 'gamma': 0.685398998466496,
    'reg_alpha': 0.38488409751539265, 'reg_lambda': 0.5173731295762696
}

# GPU/CPU 설정
if USE_GPU:
    xgb_params.update({'tree_method': 'gpu_hist'})
else:
    xgb_params.update({'tree_method': 'hist'})

dtrain = xgb.DMatrix(X, label=y)
xgb_model = xgb.train(xgb_params, dtrain, num_boost_round=2751, verbose_eval=100)
print("XGBoost 모델 학습 완료!")

# --- CatBoost --- #
print("\n[CatBoost] 모델 학습 시작...")
catboost_params = {
    'loss_function': 'Logloss', 'eval_metric': 'AUC', 'depth': 9, 'learning_rate': 0.015887402428557403,
    'l2_leaf_reg': 1.5283126467878267, 'bagging_temperature': 0.7572280338208215,
    'subsample': 0.5651908745011828, 'min_child_samples': 92, 'scale_pos_weight': scale_pos_weight_value,
    'random_state': RANDOM_STATE, 'iterations': 2990, 'verbose': 100
}

# CatBoost는 bagging_temperature 사용으로 인해 CPU 모드로 고정
# (bagging_temperature는 GPU에서 지원되지 않음)
catboost_params.update({'task_type': 'CPU'})
print("CatBoost는 bagging_temperature 파라미터 사용으로 CPU 모드로 학습합니다")

catboost_model = CatBoostClassifier(**catboost_params)
catboost_model.fit(X, y)
print("CatBoost 모델 학습 완료!")

# ==================================
# 모델 및 추론용 객체 저장
# ==================================
print("\n모델 및 추론용 객체 저장 시작...")

# 모델 저장
final_model.save_model(os.path.join(MODELS_DIR, "lightgbm_model.txt"))
xgb_model.save_model(os.path.join(MODELS_DIR, "xgboost_model.json"))
catboost_model.save_model(os.path.join(MODELS_DIR, "catboost_model.cbm"))
print(f"  - 모델 3개 저장 완료: {os.path.abspath(MODELS_DIR)}")

# 추론에 필요한 객체 저장
inference_artifacts = {
    "target_encoding_maps": target_encoding_maps,
    "clip_bounds": clip_bounds,
    "train_columns": train_columns,
    "encoding_features": encoding_features,
    "numeric_features": numeric_features
}
artifact_path = os.path.join(MODELS_DIR, "inference_artifacts.joblib")
joblib.dump(inference_artifacts, artifact_path)
print(f"  - 추론용 객체 저장 완료: {os.path.abspath(artifact_path)}")

print("\n모든 학습 및 저장 작업 완료!")
