# Data and Model: Combined Cycle Power Plant

## 1. Problem Formulation
* **Task ($T$):** Predict the net hourly electrical energy output (`PE`, in MW) of a combined cycle power plant running at full load.
* **Experience ($E$):** 9,568 hourly sensor records containing four ambient measurements (`AT`, `V`, `AP`, `RH`) and measured power output (`PE`).
* **Performance Measure ($P$):** Mean Absolute Error (MAE) in MW evaluated on a held-out 20% test set.
* **Learning Type:** Supervised regression.

## 2. Dataset Summary & Preprocessing
The dataset contains **9,568 rows** and **5 columns** with **0 missing values** and **41 exact duplicate rows**. All rows were retained because sensor readings are rounded to two decimal places, meaning identical readings can naturally recur at different hours.

| Variable | Description | Unit | Min | Max | Mean | Median | Std Dev |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **AT** | Ambient Temperature | °C | 1.81 | 37.11 | 19.65 | 20.35 | 7.45 |
| **V** | Exhaust Vacuum | cm Hg | 25.36 | 81.56 | — | — | — |
| **AP** | Ambient Pressure | mbar | 992.89 | 1033.30 | — | — | — |
| **RH** | Relative Humidity | % | 25.56 | 100.16 | — | — | — |
| **PE** | Net Electrical Output | MW | 420.26 | 495.76 | 454.37 | 451.55 | 17.07 |

*Note: All values fall within realistic physical operating bounds, with `RH` slightly exceeding 100% (100.16%) due to minor sensor tolerance.*

## 3. Correlation Analysis
Pearson correlation coefficients ($r$) with target output `PE`:
* **AT:** $-0.948$ (strongest linear relationship, negative)
* **V:** $-0.870$ (strong negative relationship)
* **AP:** $+0.518$ (moderate positive relationship)
* **RH:** $+0.390$ (weak-to-moderate positive relationship)

## 4. Models & Test Set Evaluation
Using an 80/20 train-test split (7,654 training rows, 1,914 test rows):

| Model | Description | Test MAE (MW) |
| :--- | :--- | :---: |
| **Baseline** | Always predicts training mean (`454.38 MW`) | 14.72 |
| **Linear Regression (AT)** | Single-feature line from ambient temperature | 4.33 |
| **Linear Regression (V)** | Single-feature line: $\text{PE} = -1.17 \cdot V + 517.80$ | 6.58 |

### Interpretation & Limitations
* **Ranking:** The `AT` model performs best (`4.33 MW`), followed by the `V` model (`6.58 MW`), and finally the mean baseline (`14.72 MW`), matching the strength of their linear correlations with `PE`.
* **Operational Limit:** Because all training data were recorded at **full load**, the model cannot reliably predict output at **half load**, as partial-capacity operation lies outside the learned domain.
