import streamlit as st
import math
import bisect
import pandas as pd
import numpy as np
import sklearn
from scipy.ndimage import gaussian_filter1d
import pwlf
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime

def fetchData(ticker):
  df = yf.download(ticker, period="2y", ignore_tz=True)
  dfs = yf.download(ticker, period="6mo", interval="1h", ignore_tz=True)
  df.columns = ["Close", "High", "Low", "Open", "Volume"]
  dfs.columns = ["Close", "High", "Low", "Open", "Volume"]
  df.index.tz_localize("Asia/Hong_Kong")
  dfs.index.tz_localize("Asia/Hong_Kong")
  return df, dfs

def heikinashi(df):
  ha = df.copy()
  ha['HA_Close'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
  ha['HA_Open'] = (df['Open'] + df['Close']) / 2
  for i in range(1, len(df)):
    ha.iloc[i, ha.columns.get_loc('HA_Open')] = (ha.iloc[i-1]['HA_Open'] + ha.iloc[i-1]['HA_Close']) / 2
  ha['HA_High'] = ha[['HA_Open', 'HA_Close', 'High']].max(axis=1)
  ha['HA_Low'] = ha[['HA_Open', 'HA_Close', 'Low']].min(axis=1)
  return ha

def alphabeta(df, marketDf):
  abDf = pd.DataFrame({
    "stock_close": df["Close"],
    "market_close": marketDf["Close"]
  }, index=(df.index)).dropna()
  abDf["stock_ret"] = abDf["stock_close"].pct_change()
  abDf["market_ret"] = abDf["market_close"].pct_change()
  abDf = abDf.dropna()
  model = sklearn.linear_model.LinearRegression()
  model.fit(abDf["market_ret"].values.reshape(-1, 1), abDf["stock_ret"].values)
  alpha = model.intercept_
  beta = model.coef_[0]
  return alpha, beta

def srSMA(df):
  tolerance = 0.01 * df["Close"].iloc[-1]
  lookback = 10
  
  results = []
  
  for label in ['10SMA', '20SMA', '50SMA', '100SMA']:
    sma = df[label]
    latest_sma = sma.iloc[-1]
    dist = abs(latest_sma - df["Close"].iloc[-1])
  
    recent_prices = df['Close'].iloc[-lookback:]
    recent_sma = sma.iloc[-lookback:]
  
    support_violations = (recent_prices < recent_sma - tolerance).sum()
    resistance_violations = (recent_prices > recent_sma + tolerance).sum()
  
    results.append({
      'label': label,
      'distance': dist,
      'support_violations': support_violations,
      'resistance_violations': resistance_violations
    })
  
  support_candidates = [r for r in results if r['support_violations'] <= (lookback * 0.2)]
  if len(support_candidates) > 0:
    support_best = min(support_candidates, key=lambda x: (x['support_violations'], x['distance']), default=None)
  else:
    support_best = {'label': None}
  
  resistance_candidates = [r for r in results if r['resistance_violations'] <= (lookback * 0.2)]
  if len(resistance_candidates) > 0:
    resistance_best = min(resistance_candidates, key=lambda x: (x['resistance_violations'], x['distance']), default=None)
  else:
    resistance_best = {'label': None}

  return support_best, resistance_best

def gdCross(df, futureDf):
  smaEx = pd.DataFrame({
      "20SMA": df["20SMA"],
      "50SMA": df["50SMA"],
      "100SMA": df["100SMA"]
  }, index=df.index)
  smaEx = pd.concat([smaEx, futureDf])
  for sma_col in ["20SMA", "50SMA", "100SMA"]:
    recent = df[[sma_col]].dropna().tail(10)
    model = sklearn.linear_model.LinearRegression()
    model.fit(np.arange(len(recent)).reshape(-1, 1), recent[sma_col].values)
    xFuture = np.arange(len(recent), len(recent) + 10).reshape(-1, 1)
    yFuture = model.predict(xFuture)
    nanIdx = smaEx[smaEx[sma_col].isna() & (smaEx.index >= df.index[-1])].index
    smaEx.loc[nanIdx, sma_col] = yFuture
  
  gCrosses = []
  dCrosses = []
  for i in range(1, len(smaEx)):
    for (shortSMA, longSMA) in [("20SMA", "50SMA"), ("20SMA", "100SMA"), ("50SMA", "100SMA")]:
      if (smaEx[shortSMA].iloc[i-1] <= smaEx[longSMA].iloc[i-1]) and (smaEx[shortSMA].iloc[i] > smaEx[longSMA].iloc[i]):
        gCrosses.append(smaEx.index[i])
      elif (smaEx[shortSMA].iloc[i-1] >= smaEx[longSMA].iloc[i-1]) and (smaEx[shortSMA].iloc[i] < smaEx[longSMA].iloc[i]):
        dCrosses.append(smaEx.index[i])

  return gCrosses, dCrosses

def zigzag(df, dfs):
  smoothedHi = gaussian_filter1d(dfs["High"], sigma=1)
  smoothedLo = gaussian_filter1d(dfs["Low"], sigma=1)
  
  peaksXHi = []
  peaksYHi = []
  peaksXLo = []
  peaksYLo = []
  for i in range(1, len(smoothedHi)-1):
    if (smoothedHi[i] >= smoothedHi[i-1]) and (smoothedHi[i+1] < smoothedHi[i]):
      peaksXHi.append(i)
      peaksYHi.append(dfs["High"].iloc[i])
    elif (smoothedLo[i] <= smoothedLo[i-1]) and (smoothedLo[i+1] > smoothedLo[i]):
      peaksXLo.append(i)
      peaksYLo.append(dfs["Low"].iloc[i])
  
  zzPwlfHi = pwlf.PiecewiseLinFit(np.array(peaksXHi), np.array(peaksYHi))
  zzPwlfLo = pwlf.PiecewiseLinFit(np.array(peaksXLo), np.array(peaksYLo))
  resHi = zzPwlfHi.fitfast(8)
  resLo = zzPwlfLo.fitfast(8)

  xHat = []
  for d in df.index.to_pydatetime()[(bisect.bisect_right(df.index.to_pydatetime(), dfs.index.to_pydatetime()[0])-1):]:
    xHat.append(bisect.bisect_left(dfs.index.to_pydatetime(), d))
    
  yHatHi = zzPwlfHi.predict(np.array(xHat))
  yHatLo = zzPwlfLo.predict(np.array(xHat))
  return yHatHi, yHatLo
# ----------------------------------------------
ticker = st.text_input("Ticker", "2600") + ".HK"

# market data
marketDf = yf.download("^HSI", period="2y", ignore_tz=True)
marketDf.columns = ["Close", "High", "Low", "Open", "Volume"]
marketDf.index.tz_localize("Asia/Hong_Kong")

# future dates
futureDates = [marketDf.index[-1] + pd.tseries.offsets.BDay(n=i) for i in range(1, 11)]
futureDf = pd.DataFrame(index=(futureDates))

df, dfs = fetchData(ticker)
df = heikinashi(df)
alpha, beta = alphabeta(df, marketDf)
df["10SMA"] = df["Close"].rolling(window=10).mean()
df["20SMA"] = df["Close"].rolling(window=20).mean()
df["50SMA"] = df["Close"].rolling(window=50).mean()
df["100SMA"] = df["Close"].rolling(window=100).mean()
support_best, resistance_best = srSMA(df)
gCrosses, dCrosses = gdCross(df, futureDf)

df["zzHi"] = [float("nan")] * (len(df))
df["zzLo"] = [float("nan")] * (len(df))
zzHi, zzLo = zigzag(df, dfs)
df["zzHi"].iloc[-len(zzHi):] = zzHi
df["zzLo"].iloc[-len(zzLo):] = zzLo
df["zzDiff"] = df["zzHi"] - df["zzLo"]

# clean data
df = df.reset_index()
df["Date"].iloc = df["Date"].dt.strftime("%Y-%m-%d")

# basic plot
fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                    vertical_spacing=0.05,
                    subplot_titles=["", ""])

df = pd.concat([df, futureDf], ignore_index=True)
fig.add_trace(go.Candlestick(
  x=df['Date'],
  open=df['HA_Open'], high=df['HA_High'],
  low=df['HA_Low'], close=df['HA_Close'],
  increasing_line_color='rgba(0,100,0,0.5)',  # light green
  decreasing_line_color='rgba(100,0,0,0.5)',  # light red
  increasing_fillcolor='rgba(0,100,0,0.5)',
  decreasing_fillcolor='rgba(100,0,0,0.5)',
  opacity=1,
  name='Heikin Ashi', 
  hoverinfo="none"
), row=1, col=1)

fig.add_trace(go.Candlestick(
  x=df['Date'],
  open=df['Open'], high=df['High'],
  low=df['Low'], close=df['Close'],
  increasing_line_color='green',
  decreasing_line_color='red',
  increasing_fillcolor='rgba(0, 0, 0, 0)',
  decreasing_fillcolor='rgba(0, 0, 0, 0)',
  line_width=1,
  opacity=1,
  name='Raw Candlestick',
  hoverinfo="none"
), row=1, col=1)

for sma_label in ["10SMA", "20SMA", "50SMA", "100SMA"]:
  visibility = True if sma_label in [support_best['label'], resistance_best['label']] else 'legendonly'
  fig.add_trace(go.Scatter(
    x=df['Date'],
    y=df[sma_label],
    mode='lines',
    name=sma_label,
    line=dict(width=1.5, color="white"),
    visible=visibility, 
    hoverinfo="none"
  ), row=1, col=1)

fig.add_trace(go.Scatter(
  x=df["Date"], 
  y=df[f"zzHi"], 
  mode="lines", 
  name="Zig Zag High", 
  line=dict(width=1.5, color="yellow"), 
  hoverinfo="none", 
  connectgaps=True, 
  visible="legendonly"
), row=1, col=1)

fig.add_trace(go.Scatter(
  x=df["Date"], 
  y=df[f"zzLo"], 
  mode="lines", 
  name="Zig Zag Low", 
  line=dict(width=1.5, color="yellow"), 
  hoverinfo="none", 
  connectgaps=True, 
  visible="legendonly"
), row=1, col=1)

fig.add_trace(go.Scatter(
  x=df["Date"], 
  y=df["zzDiff"], 
  mode="lines", 
  name="Zig Zag Diff", 
  line=dict(width=1.5, color="white"), 
  hoverinfo="none", 
  connectgaps=True
), row=2, col=1)

# Add layout
fig.update_layout(
  title=f"{ticker} Alpha-Beta Analysis (1Y) | α = {alpha:.5f}, β = {beta:.2f}",
  xaxis2_title="Date",
  yaxis1_title="Price",
  yaxis2_title="Price Diff.", 
  height=600,
  xaxis_rangeslider_visible=False,
  hovermode="x unified",
  spikedistance=-1,
  xaxis=dict(
    type='category',
    categoryorder='array',
    categoryarray=df["Date"].tolist(),
    range=[len(df)-91, len(df)-1],
    autorange=False,
    showspikes=True,
    spikecolor='rgba(255,255,255,0.3)',
    spikedash='solid',
    spikesnap='cursor',
    spikemode='across',
    spikethickness=2
  ),
  yaxis=dict(
    showspikes=True,
    spikecolor='rgba(255,255,255,0.3)',
    spikedash='solid',
    spikesnap='cursor',
    spikemode='across',
    spikethickness=2
  )
)

st.plotly_chart(fig, use_container_width=True)
