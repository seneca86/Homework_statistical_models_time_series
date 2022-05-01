# %%
from click import style
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import pmdarima as pm

# %%
from ast import Param
from random import seed
from sqlite3 import paramstyle
from numpy import disp
from pathlib import Path
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import VAR
from pmdarima.arima import StepwiseContext
from pathlib import Path

# %%
plt.style.use("seaborn-darkgrid")
matplotlib.rcParams["axes.labelsize"] = 14
matplotlib.rcParams["xtick.labelsize"] = 12
matplotlib.rcParams["ytick.labelsize"] = 12
matplotlib.rcParams["text.color"] = "k"
matplotlib.rcParams["figure.dpi"] = 200

# %%
directory = "plots"
Path(directory).mkdir(parents=True, exist_ok=True)
# %%
df_raw = pd.read_csv(".lesson/assets/NEW-DATA-1.T15.txt", sep=" ")
df = df_raw.rename(
    {
        "1:Date": "date",
        "2:Time": "time",
        "5:Weather_Temperature": "temp",
        "8:Humedad_Comedor_Sensor": "hum",
        "10:Lighting_Comedor_Sensor": "light",
    },
    axis="columns",
).assign(
    date=lambda x: pd.to_datetime(x.date + "-" + x.time, infer_datetime_format=True)
)
# %%
var = "light"
plt.plot(df.date, df[var], label="actuals")
plt.legend()
plt.savefig(directory + "/actuals")
plt.clf()
# %%
plot_pacf(df[var], lags=20)
plt.savefig(directory + "/pacf_actuals")
pacf_values = pacf(df[var])
print(pacf_values)
# %%
split = 2500
train = df.iloc[0:split, :]
test = df.iloc[split:, :]
# %%
with StepwiseContext(max_dur=15):
    model = pm.auto_arima(
        train[var],
        stepwise=True,
        error_action="ignore",
        seasonal=True,
    )
# %%
print(f"model.summary()")
preds, conf_int = model.predict(n_periods=df.shape[0] - split, return_conf_int=True)

# %%
plt.plot(train.date, train[var], label="actuals", color="black")
plt.plot(test.date, preds, label="autoarima", color="blue")
plt.legend()
plt.savefig(directory + "/autoarima")
plt.clf()
# %%
unemp_raw = pd.read_csv(".lesson/assets/fred_unemployment.csv", sep=",")
unemp = unemp_raw.rename(
    {
        "DATE": "date",
        "UNRATE": "rate",
    },
    axis="columns",
).assign(date=lambda x: pd.to_datetime(x.date, infer_datetime_format=True))
# %%
var = "rate"
plt.plot(unemp.date, unemp[var], label="actuals")
plt.legend()
plt.savefig(directory + "/actuals_unemp")
plt.clf()
# %%
plot_pacf(unemp[var], lags=20)
plt.savefig(directory + "/pacf_actuals_unemp")
pacf_values = pacf(unemp[var])
print(pacf_values)
# %%
split = 850
train = unemp.iloc[0:split, :]
test = unemp.iloc[split:, :]
# %%
with StepwiseContext(max_dur=15):
    model = pm.auto_arima(
        train[var],
        stepwise=True,
        error_action="ignore",
        seasonal=True,
    )
# %%
print(f"model.summary()")
preds, conf_int = model.predict(n_periods=unemp.shape[0] - split, return_conf_int=True)

# %%
plt.plot(train.date, train[var], label="actuals", color="black")
plt.plot(test.date, preds, label="autoarima", color="blue")
plt.legend()
plt.savefig(directory + "/autoarima_unemp")
plt.clf()
# %%
