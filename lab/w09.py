import io
import urllib.request
import zipfile
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00294/CCPP.zip"
response = urllib.request.urlopen(url)
with zipfile.ZipFile(io.BytesIO(response.read())) as z:
    with z.open("CCPP/Folds5x2_pp.xlsx") as f:
        df = pd.read_excel(f)

print(f"Rows: {len(df)}, Columns: {df.shape[1]}")
print(
    f"Missing values: {df.isna().sum().sum()}, Duplicate rows:"
    f" {df.duplicated().sum()}"
)
for col in df.columns:
    print(f"{col}: {df[col].min():.2f} - {df[col].max():.2f}")

print(
    f"AT: mean={df['AT'].mean():.2f}, median={df['AT'].median():.2f},"
    f" std={df['AT'].std():.2f}"
)
print(
    f"PE: mean={df['PE'].mean():.2f}, median={df['PE'].median():.2f},"
    f" std={df['PE'].std():.2f}"
)
corr = df.corr(numeric_only=True)["PE"]
print(
    f"Correlations with PE: AT={corr['AT']:.3f}, V={corr['V']:.3f},"
    f" AP={corr['AP']:.3f}, RH={corr['RH']:.3f}"
)

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
print(f"Train rows: {len(train_df)}, Test rows: {len(test_df)}")

train_mean = train_df["PE"].mean()
baseline_mae = mean_absolute_error(
    test_df["PE"], np.full(len(test_df), train_mean)
)
print(f"Training mean PE: {train_mean:.2f} MW")
print(f"Baseline test MAE: {baseline_mae:.2f} MW")

model_at = LinearRegression().fit(train_df[["AT"]], train_df["PE"])
at_mae = mean_absolute_error(test_df["PE"], model_at.predict(test_df[["AT"]]))
print(f"Line from AT test MAE: {at_mae:.2f} MW")

model_v = LinearRegression().fit(train_df[["V"]], train_df["PE"])
v_mae = mean_absolute_error(test_df["PE"], model_v.predict(test_df[["V"]]))
a = round(model_v.coef_[0], 2)
b = round(model_v.intercept_, 2)
print(f"Line from V: slope={a:.2f}, intercept={b:.2f}, test MAE={v_mae:.2f} MW")

row = df.iloc[1]
pred_pe = round(a * row["V"] + b, 2)
abs_err = round(abs(row["PE"] - pred_pe), 2)
print(
    f"Line 3 check: V={row['V']:.2f}, Predicted PE={pred_pe:.2f} MW, True"
    f" PE={row['PE']:.2f} MW, Abs error={abs_err:.2f} MW"
)
