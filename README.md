# NLP 문서요약 대회

## 프로젝트 구조

```
nlp_project/
├── data/                    # 데이터
│   └── sample/             # 샘플 데이터
│       ├── train_sample/
│       └── val_sample/
│
├── notebooks/              # EDA 및 실험용 노트북
│   └── eda.ipynb
│
├── src/                    # 핵심 코드
│   ├── analyze_sample_data.py
│   └── create_sample_dataset.py
│
├── outputs/               # 모든 결과물
│   ├── models/           # 체크포인트
│   └── submissions/      # 제출 파일
│
├── docs/                  # 문서
│   ├── EDA_Report.md
│   └── EDA_Report_Sample.md
│
├── train_small.py        # 학습 스크립트
├── test_model.py         # 테스트/예측 스크립트
├── environment.yml       # conda 환경 설정
└── README.md
```

## 사용 방법

### 환경 설정
```bash
conda env create -f environment.yml
conda activate nlp_project
```

### 학습
```bash
python train_small.py
```

### 예측
```bash
python test_model.py
```

## 폴더별 설명

- **data/**: 학습/테스트 데이터를 여기에 저장
- **notebooks/**: EDA 및 프로토타입 실험용
- **src/**: 재사용 가능한 코드 (데이터 처리, 모델, 유틸 등)
- **outputs/**: 학습된 모델과 제출 파일 저장
- **docs/**: 프로젝트 문서 (EDA 리포트, 실험 기록 등)
