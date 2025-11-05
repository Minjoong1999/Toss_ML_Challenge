import pandas as pd
import polars as pl
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import warnings
import gc
import joblib
import os

warnings.filterwarnings('ignore')

# ==================================
# 설정
# ==================================
TEST_PATH = "../data/test.parquet"
MODELS_DIR = "../models/"
SUBMISSION_PATH = "../submission.csv"

# ==================================
# 모델 및 추론용 객체 로드
# ==================================
print("모델 및 추론용 객체 로드 시작...")

# 모델 로드
lgbm_model = lgb.Booster(model_file=os.path.join(MODELS_DIR, "lightgbm_model.txt"))
xgb_model = xgb.Booster()
xgb_model.load_model(os.path.join(MODELS_DIR, "xgboost_model.json"))
catboost_model = CatBoostClassifier()
catboost_model.load_model(os.path.join(MODELS_DIR, "catboost_model.cbm"))
print("  - 모델 3개 로드 완료")

# 추론용 객체 로드
artifact_path = os.path.join(MODELS_DIR, "inference_artifacts.joblib")
inference_artifacts = joblib.load(artifact_path)
target_encoding_maps = inference_artifacts["target_encoding_maps"]
clip_bounds = inference_artifacts["clip_bounds"]
train_columns = inference_artifacts["train_columns"]
encoding_features = inference_artifacts["encoding_features"]
numeric_features = inference_artifacts["numeric_features"]
print(f"  - 추론용 객체 로드 완료: {os.path.abspath(artifact_path)}")

# ==================================
# 테스트 데이터 추론 (청크 처리)
# ==================================
print("\n테스트 데이터 처리 시작 (청크 방식)...")

if not os.path.exists(TEST_PATH):
    print(f"오류: 테스트 데이터 파일이 없습니다. 다음 경로를 확인해주세요: {os.path.abspath(TEST_PATH)}")
    exit()

# 테스트 데이터 정보 미리 읽기
test_df_info = pl.read_parquet(TEST_PATH)
total_rows = len(test_df_info)
test_ids_all = test_df_info['ID'].to_pandas()
del test_df_info
gc.collect()

print(f"총 테스트 데이터: {total_rows:,}개")

# 청크 크기 설정
CHUNK_SIZE = 500000
num_chunks = (total_rows + CHUNK_SIZE - 1) // CHUNK_SIZE
print(f"청크 크기: {CHUNK_SIZE:,}, 총 {num_chunks}개 청크로 처리")

# 예측 결과 저장 리스트
lgb_predictions_list = []
xgb_predictions_list = []
catboost_predictions_list = []

history_b_cols = [f'history_b_{i}' for i in range(1,31)]
history_a_cols = [f'history_a_{i}' for i in range(1,8)]

# 청크별 처리
for chunk_idx in range(num_chunks):
    start_idx = chunk_idx * CHUNK_SIZE
    end_idx = min((chunk_idx + 1) * CHUNK_SIZE, total_rows)

    print(f"\n[청크 {chunk_idx+1}/{num_chunks}] 처리 중... (행 {start_idx:,} ~ {end_idx:,})")

    # 청크 데이터 로드
    test_chunk = pl.read_parquet(TEST_PATH).slice(start_idx, end_idx - start_idx)

    # 피처 엔지니어링
    seq_as_list_test = pl.col('seq').str.split(",").cast(pl.List(pl.Float64))
    test_chunk = (
        test_chunk
        .with_columns([
            seq_as_list_test.list.mean().alias('seq_mean'),
            seq_as_list_test.list.std().alias('seq_std'),
            seq_as_list_test.list.max().alias('seq_max'),
            seq_as_list_test.list.min().alias('seq_min')
        ])
        .drop('seq')
        .with_columns([
            pl.mean_horizontal(history_b_cols).alias('history_b_mean'),
            pl.concat_list(history_b_cols).list.std().alias('history_b_std'),
            pl.mean_horizontal(history_a_cols).alias('history_a_mean'),
            pl.concat_list(history_a_cols).list.std().alias('history_a_std'),
            (pl.col('hour').cast(pl.Utf8) + "_" + pl.col("day_of_week").cast(pl.Utf8)).alias('day_hour'),
            (pl.col('inventory_id').cast(pl.Utf8) + "_" + pl.col('age_group').cast(pl.Utf8)).alias('inv_id_age_group'),
            pl.col("feat_e_3").is_null().cast(pl.Int8).alias("is_feat_e_3_missing")
        ])
    )
    test_chunk = test_chunk.to_pandas()

    # 타입 변환 및 주기성 피처
    test_chunk['hour'] = pd.to_numeric(test_chunk['hour'], errors='coerce')
    test_chunk['day_of_week'] = pd.to_numeric(test_chunk['day_of_week'], errors='coerce')
    test_chunk['hour_sin'] = np.sin(2 * np.pi * test_chunk['hour'] / 24)
    test_chunk['hour_cos'] = np.cos(2 * np.pi * test_chunk['hour'] / 24)
    test_chunk['day_of_week_sin'] = np.sin(2 * np.pi * test_chunk['day_of_week'] / 7)
    test_chunk['day_of_week_cos'] = np.cos(2 * np.pi * test_chunk['day_of_week'] / 7)

    # 타겟 인코딩 적용
    for col in encoding_features:
        encoding_map = target_encoding_maps[col]
        test_chunk[f'{col}_target_encoded'] = test_chunk[col].map(encoding_map)

    # 이상치 클리핑 적용
    for col in numeric_features:
        if col in test_chunk.columns and col in clip_bounds:
            lower, upper = clip_bounds[col]
            test_chunk[col] = test_chunk[col].clip(lower, upper)

    # 불필요한 컬럼 제거
    test_chunk = test_chunk.drop(columns=['ID', 'day_hour', 'inv_id_age_group'])

    # 카테고리 인코딩
    for col in test_chunk.select_dtypes(include='object').columns:
        test_chunk[col] = test_chunk[col].astype('category').cat.codes

    # train과 컬럼 순서 맞추기
    test_chunk = test_chunk[train_columns]

    # 예측
    print("  - LightGBM 예측...")
    lgb_pred = lgbm_model.predict(test_chunk)
    lgb_predictions_list.append(lgb_pred)

    print("  - XGBoost 예측...")
    dtest = xgb.DMatrix(test_chunk)
    xgb_pred = xgb_model.predict(dtest)
    xgb_predictions_list.append(xgb_pred)

    print("  - CatBoost 예측...")
    catboost_pred = catboost_model.predict_proba(test_chunk)[:, 1]
    catboost_predictions_list.append(catboost_pred)

    del test_chunk, dtest
    gc.collect()
    print(f"  ✅ 청크 {chunk_idx+1} 완료")

# ==================================
# 앙상블 및 제출 파일 생성
# ==================================
print("\n모든 청크 예측 결과 결합 중...")
lgb_predictions = np.concatenate(lgb_predictions_list)
xgb_predictions = np.concatenate(xgb_predictions_list)
catboost_predictions = np.concatenate(catboost_predictions_list)

print("\n3-way 앙상블 (가중 평균: LGB 0.2 + XGB 0.5 + Cat 0.3)...")
ensemble_predictions = (0.2 * lgb_predictions + 0.5 * xgb_predictions + 0.3 * catboost_predictions)

print("제출 파일 생성 중...")
submission = pd.DataFrame({'ID': test_ids_all, 'clicked': ensemble_predictions})
submission.to_csv(SUBMISSION_PATH, index=False)

print(f"\n 제출 파일 저장 완료: {os.path.abspath(SUBMISSION_PATH)}")
print("모든 작업 완료!")
