### step 1) 데이터셋 생성 : `generate_dataset.py`

- (IN) 
  - 메타(테이블) 데이터
  - 로그 데이터 1)

- (OUT)
  - 전처리 후 메타 데이터 : `doll_merged.csv`
  - FA를 수행한 메타 데이터 : `FA_doll_merged.csv`
  - FA를 수행한 로그 데이터 1) :  `FA_log_action.csv`
  - FA를 수행한 로그 데이터 2) :  `FA_log_program.csv`



예시

```bash
$ python generate_dataset.py --doll=6 --action=3 --program=4
```

- FA의 요소(component) 개수 : 메타데이터=6, 로그데이터1=3, 로그데이터2=4

<br>

### step 2) 군집화 `clustering.py`

- (IN)
  - FA를 수행한 메타 데이터 : `FA_doll_merged.csv`
  - FA를 수행한 로그 데이터 1) :  `FA_log_action.csv`
  - FA를 수행한 로그 데이터 2) :  `FA_log_program.csv`
- (OUT)
  - 군집화된 고객 데이터 : `FA_with_cluster.csv`
- 주요 과정 순서
  - (1) 데이터 병합
  - (2) 스케일 조정
  - (3) 이상치 고객 제거
  - (4) 군집화 

<br>

예시

```bash
$ python clustering.py --grid_x=2 --grid_y=3 --iter=20000
```

- (2,3) 크기의 SOM 군집 생성
- SOM 학습 과정을 위한 iteration 20,000회

<br>

### step 3) Task Label 추가 `task_label.py`

- (IN)
  - 군집화된 고객 데이터 : `FA_with_cluster.csv`
- (OUT)
  - (라벨이 추가된) 군집화된 고객 데이터 : `cluster_df_merged.csv`



예시

```bash
$ python task_label.py
```

<br>

### step 4) 응급전화 예측 모델링 `model_{}.py`

- `model_LR.py` : Logistic Regression
- `model_RF.py`  : Random Forest
- `model_MTL.py` : Multi-Task Learning



실험 공통 사항 : 

- task : 이진 분류 ( Binary Classification )
- threshold : 0.2, 0.4, 0.6, 0.8 시도 후, 최적의 경계값 설정

<br>

기타 사항

- `LR`, `RF` 의 경우, 5-fold Cross Validation 수행
- `RF` 의 경우, Random Search를 통해 하이퍼파라미터 최적화

<br>

예시

```bash
$ python model_LR.py
$ python model_RF.py
$ python model_MTL.py --epoch=200
```





