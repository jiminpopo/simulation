# simulation

## 📄 Simulation Data Generator: README

### 🧾 Overview (English)

This project provides a simulation dataset generator for tabular data with both numeric and categorical variables. The generator supports three modes for generating a categorical variable (`cat2`) based on varying dependency assumptions.

### 📂 Directory Structure

```
simulation_project/
├── data/                          # CSV files are saved here
├── generate/  # Generate code for simulation data
├── README.md
```

### 📊 Variable Description

| Variable | Type        | Description                                 |
| -------- | ----------- | ------------------------------------------- |
| num1     | Numeric     | Mean increases as cat1 moves from A → D     |
| num2     | Numeric     | Mean decreases as cat1 moves from A → D     |
| num3     | Numeric     | Arbitrary distribution by cat1              |
| cat1     | Categorical | 4 levels: A, B, C, D                        |
| cat2     | Categorical | 2 levels: a, b (generated by selected mode) |

### 🛠️ Available Modes for `cat2`

* `random`: Generated completely at random
* `rule`: Generated by rule — if `num1 > num2` then 'a', else 'b'
* `logistic`: Generated probabilistically using a logistic model on `num1`, `num2`, `num3`

### 🧪 Usage

To generate data:

```bash
python generate_simulation_boxplot_data.py
```

The output file is saved as:

* `data/s3_random.csv` for `mode='random'`
* `data/s3_rule.csv` for `mode='rule'`
* `data/s3_logistic.csv` for `mode='logistic'`

You can also import the function inside Python:

```python
from generate_simulation_boxplot_data import generate_simulation_boxplot_data

df = generate_simulation_boxplot_data(mode='rule')
```

### 🧹 Deleting the Output Directory

To delete the `data` folder from the terminal:

```bash
rm -r data
```

Add `-f` to force delete without confirmation:

```bash
rm -rf data
```

---

### 🧾 개요 (Korean)

이 프로젝트는 수치형 변수 3개와 범주형 변수 2개를 가지는 시뮬레이션 데이터를 생성하는 Python 스크립트입니다. `cat2` 변수는 선택된 방식에 따라 다르게 생성됩니다.

### 📂 디렉토리 구조

```
simulation_project/
├── data/                          # 생성된 CSV 파일 저장 폴더
├── generate_simulation_boxplot_data.py  # 시뮬레이션 생성 스크립트
├── README.md
```

### 📊 변수 설명

| 변수명  | 유형  | 설명                       |
| ---- | --- | ------------------------ |
| num1 | 수치형 | cat1이 A → D로 갈수록 평균 증가   |
| num2 | 수치형 | cat1이 A → D로 갈수록 평균 감소   |
| num3 | 수치형 | cat1에 따라 임의 설정           |
| cat1 | 범주형 | A, B, C, D의 4단계          |
| cat2 | 범주형 | a, b의 2단계. 생성 방식에 따라 달라짐 |

### 🛠️ `cat2` 생성 방식 (mode)

* `random`: 무작위 생성
* `rule`: num1 > num2이면 'a', 아니면 'b'
* `logistic`: num1, num2, num3 기반 로지스틱 모델을 통해 확률적 생성

### 🧪 사용 방법

데이터 생성 (기본값은 `logistic` 모드):

```bash
python generate_simulation_boxplot_data.py
```

생성된 파일은 다음과 같이 저장됩니다:

* `data/s3_random.csv` (mode='random')
* `data/s3_rule.csv` (mode='rule')
* `data/s3_logistic.csv` (mode='logistic')

Python 내부에서 직접 함수 호출도 가능합니다:

```python
from generate_simulation_boxplot_data import generate_simulation_boxplot_data

df = generate_simulation_boxplot_data(mode='rule')
```

### 🧹 출력 디렉토리 삭제

터미널에서 `data` 폴더를 삭제하려면:

```bash
rm -r data
```

강제로 삭제하려면:

```bash
rm -rf data
```

**주의**: 이 명령은 `data` 폴더 및 그 안의 모든 파일을 복구 불가능하게 삭제합니다. 사용 전에 꼭 필요한 파일이 없는지 확인하세요.

