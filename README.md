# Toss NEXT ML CHALLENGE : 광고 클릭 예측(CTR) 모델 개발

이 리포지토리는 Toss NEXT ML CHALLENGE 본선 제출용 코드입니다.

## 1. 프로젝트 구조

```
toss_ml_submission/
├── data/
│   ├── train.parquet         # (필수) 원본 훈련 데이터
│   └── test.parquet          # (필수) 테스트 데이터
├── models/                   # 학습된 모델 가중치 및 객체가 저장되는 폴더
│   ├── lightgbm_model.txt
│   ├── xgboost_model.json
│   └── catboost_model.cbm
│   └── inference_artifacts.joblib
├── src/
│   ├── create_undersampled_dataset.py  # 1. 데이터 전처리(언더샘플링) 스크립트
│   ├── train.py                        # 2. 모델 학습 스크립트
│   └── inference.py                    # 3. 추론 및 제출 파일 생성 스크립트
├── requirements.txt          # 라이브러리 및 버전 정보
├── README.md                 # 실행 가이드
└── submission.csv            # (생성 파일) 최종 제출 파일
```

## 2. 환경 설정

### 2.1 요구사항

**필수 환경**:
- **Python**: 3.9 이상 권장 (3.9, 3.10, 3.11, 3.12 테스트 완료)
- **GPU**: NVIDIA GPU (CUDA 지원) - 선택사항이지만 학습 속도 향상
  - CUDA 11.0 이상 권장
  - GPU가 없는 경우 CPU로도 실행 가능 (단, 학습 시간 증가)
- **메모리**: 최소 16GB RAM 권장
- **저장공간**: 최소 10GB 이상

### 2.2 설치 방법

1.  **데이터 준비**
    - `data/` 폴더에 대회에서 제공된 원본 `train.parquet` 파일과 `test.parquet` 파일을 위치시킵니다.

2.  **Python 가상환경 생성 (권장)**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Windows: venv\Scripts\activate
    ```

3.  **라이브러리 설치**
    ```bash
    pip install -r requirements.txt
    ```

### 2.3 라이브러리 버전 (requirements.txt)

본 프로젝트는 다음 라이브러리를 사용합니다:
- pandas==2.2.2
- polars==1.25.2
- numpy==2.0.2
- lightgbm==4.6.0
- xgboost==3.0.5
- scikit-learn==1.6.1
- joblib==1.5.2
- catboost==1.2.8

**중요**: 위 버전을 정확히 사용해야 완벽한 재현이 가능합니다.

## 3. 실행 순서

- 모든 스크립트는 `toss_ml_submission` 폴더 최상위에서 실행하는 것을 기준으로 작성되었습니다.
- 아래 3단계 순서대로 스크립트를 실행해주세요.

**1단계: 언더샘플링 데이터 생성**

```bash
python src/create_undersampled_dataset.py
```
- `data/train.parquet` 파일을 읽어 1:5 비율로 언더샘플링을 수행하고, `data/train_undersampled_1_to_5.parquet` 파일을 생성합니다.

**2단계: 모델 학습**

```bash
python src/train.py
```
- `data/train_undersampled_1_to_5.parquet` 파일을 사용하여 LightGBM, XGBoost, CatBoost 모델을 학습합니다.
- 학습된 모델 가중치와 추론에 필요한 객체(`target_encoding_maps` 등)를 `models/` 폴더에 저장합니다.

**3단계: 추론 및 제출 파일 생성**

```bash
python src/inference.py
```
- `data/test.parquet` 파일을 읽어 추론을 수행합니다.
- `models/` 폴더에 저장된 모델과 객체를 로드하여 예측을 진행합니다.
- 최종 예측 결과를 앙상블하여 `submission.csv` 파일을 프로젝트 최상위 경로에 생성합니다.

## 4. 최종 산출물

- 모든 과정이 성공적으로 완료되면, `toss_ml_submission` 폴더에 `submission.csv` 파일이 생성됩니다.

## 5. 재현성 관련 중요 사항

### 5.1 GPU 학습의 비결정성

본 모델은 GPU를 사용하여 학습되었습니다. LightGBM과 XGBoost의 GPU 버전은 부동소수점 연산 순서의 비결정성으로 인해, **동일한 데이터와 동일한 random seed를 사용해도 매번 약간씩 다른 모델이 생성됩니다.**

#### 원인
- GPU의 병렬 연산에서 atomic operation의 실행 순서가 매번 다를 수 있음
- 부동소수점 연산에서 (a + b) + c ≠ a + (b + c)로 인한 미묘한 차이 누적
- GPU 스레드 스케줄링 순서의 변동

#### 영향
- 같은 코드를 재실행 시 예측값의 평균 절대 차이(MAE): 약 0.001 ~ 0.005
- 이는 LightGBM, XGBoost 공식 문서에서도 명시된 GPU 학습의 특성입니다

### 5.2 제출한 모델의 재현성

본 제출 패키지는 다음을 보장합니다:

✅ **코드 재현성 (100% 보장)**
- 제출한 `models/` 폴더의 모델 파일을 사용하여 `inference.py`를 실행하면
- 제출한 `submission.csv`를 정확히 재현할 수 있습니다

⚠️ **모델 재학습 시 주의사항**
- `train.py`를 재실행하여 모델을 재학습하면
- GPU 비결정성으로 인해 기존 모델과 약간 다른 모델이 생성됩니다
- 결과적으로 예측값도 약간 달라질 수 있습니다 (MAE 0.001~0.005 수준)

### 5.3 모델 학습 환경

- **GPU**: NVIDIA GPU (CUDA 지원)
- **Random Seed**: 42 (모든 랜덤 연산에 고정)
- **라이브러리 버전**: requirements.txt 참조

### 5.4 참고 문서

- [LightGBM GPU Performance](https://lightgbm.readthedocs.io/en/latest/GPU-Performance.html)
- [XGBoost GPU Support](https://xgboost.readthedocs.io/en/stable/gpu/index.html)

### 5.5 완벽한 재현을 원하는 경우

CPU로 학습하면 완벽한 재현성(determinism)을 보장할 수 있습니다:

```python
# train.py의 USE_GPU 설정 변경
USE_GPU = False  # CPU 강제 사용
```

단, CPU 학습은 GPU 대비 10배 이상 오래 걸릴 수 있습니다.
