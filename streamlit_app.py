import streamlit as st
import math
import pandas as pd
import numpy as np
import sklearn
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime

def fetchData(ticker):
  df = yf.download(ticker, period="2y")
  df.columns = ["Close", "High", "Low", "Open", "Volume"]
  df = df.reset_index()
  df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
  return df

def heikinashi(df):
  ha = df.copy()
  ha['HA_Close'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
  ha['HA_Open'] = (df['Open'] + df['Close']) / 2
  for i in range(1, len(df)):
    ha.iloc[i, ha.columns.get_loc('HA_Open')] = (ha.iloc[i-1]['HA_Open'] + ha.iloc[i-1]['HA_Close']) / 2
  ha['HA_High'] = ha[['HA_Open', 'HA_Close', 'High']].max(axis=1)
  ha['HA_Low'] = ha[['HA_Open', 'HA_Close', 'Low']].min(axis=1)
  return ha

def alphabeta(df):
  abDf = pd.DataFrame({
    "Date": df["Date"],
    "stock_close": df["Close"],
    "market_close": marketDf["Close"]
  }).dropna()
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

def gdCross(df):
  smaEx = pd.DataFrame({
      "Date": df["Date"],
      "20SMA": df["20SMA"],
      "50SMA": df["50SMA"],
      "100SMA": df["100SMA"]
  })
  smaEx = pd.concat([smaEx, futureDf], ignore_index=True)
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
        gCrosses.append(smaEx["Date"].iloc[i])
      elif (smaEx[shortSMA].iloc[i-1] >= smaEx[longSMA].iloc[i-1]) and (smaEx[shortSMA].iloc[i] < smaEx[longSMA].iloc[i]):
        dCrosses.append(smaEx["Date"].iloc[i])

  return gCrosses, dCrosses

def zigzag(arr, func):
  threshold = 0.005

  newArr = [float("nan")] * (len(arr)-1)
  newArr[0] = arr.iloc[0]
  slopeMax = float("inf")
  slopeMin = float("-inf")
  lastAnc = 0
  lastPt = 0
  last2Pt = float("nan")
  for i in range(1, len(arr)):
    if math.isnan(arr.iloc[i]):
      continue

    if math.isnan(last2Pt):
      lastPt = i
      last2Pt = 0
      continue

    if (func(arr.iloc[i] - arr.iloc[lastPt], arr.iloc[lastPt] - arr.iloc[last2Pt])):
      if ((arr.iloc[i]*(1-threshold) - arr.iloc[lastAnc]) / (i - lastAnc)) > slopeMax:
        newArr[lastPt] = arr.iloc[lastPt]
        slopeMax = (arr.iloc[i]*(1+threshold) - arr.iloc[lastPt]) / (i - lastPt)
        slopeMin = (arr.iloc[i]*(1-threshold) - arr.iloc[lastPt]) / (i - lastPt)
        lastAnc = lastPt
      elif ((arr.iloc[i]*(1+threshold) - arr.iloc[lastAnc]) / (i - lastAnc)) < slopeMin:
        newArr[lastPt] = arr.iloc[lastPt]
        slopeMax = (arr.iloc[i]*(1+threshold) - arr.iloc[lastPt]) / (i - lastPt)
        slopeMin = (arr.iloc[i]*(1-threshold) - arr.iloc[lastPt]) / (i - lastPt)
        lastAnc = lastPt
      else:
        slopeMax = min(slopeMax, (arr.iloc[i]*(1+threshold) - arr.iloc[lastAnc]) / (i - lastAnc))
        slopeMin = max(slopeMin, (arr.iloc[i]*(1-threshold) - arr.iloc[lastAnc]) / (i + lastAnc))
    last2Pt = lastPt
    lastPt = i
  newArr.append(arr.iloc[lastAnc] + (slopeMax+slopeMin)*(len(arr)-1 - lastAnc)/2)
  return newArr

ticker = st.text_input("Ticker", "2600") + ".HK"

# market data
marketDf = yf.download("^HSI", period="2y")
marketDf.columns = ["Close", "High", "Low", "Open", "Volume"]
marketDf = marketDf.reset_index()
marketDf["Date"] = marketDf["Date"].dt.strftime("%Y-%m-%d")

# future dates
futureDates = [pd.to_datetime(marketDf["Date"].iloc[-1]) + pd.tseries.offsets.BDay(n=i) for i in range(1, 11)]
futureDatesStr = [d.strftime("%Y-%m-%d") for d in futureDates]
futureDf = pd.DataFrame({"Date": futureDatesStr})

df = fetchData(ticker)
df = heikinashi(df)
alpha, beta = alphabeta(df)
df["10SMA"] = df["Close"].rolling(window=10).mean()
df["20SMA"] = df["Close"].rolling(window=20).mean()
df["50SMA"] = df["Close"].rolling(window=50).mean()
df["100SMA"] = df["Close"].rolling(window=100).mean()
support_best, resistance_best = srSMA(df)
gCrosses, dCrosses = gdCross(df)

df["zz0Hi"] = df["High"]
df["zz0Lo"] = df["Low"]
for i in range(1, 6):
  df[f"zz{i}Hi"] = zigzag(df[f"zz{i-1}Hi"], lambda a, b: a <= b)
  df[f"zz{i}Lo"] = zigzag(df[f"zz{i-1}Lo"], lambda a, b: a >= b)

# basic plot
fig = make_subplots(rows=1, cols=1, shared_xaxes=True,
                    vertical_spacing=0.05,
                    subplot_titles=[""])

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

for i in range(0, 6):
  fig.add_trace(go.Scatter(
    x=df["Date"], 
    y=df[f"zz{i}Hi"], 
    mode="lines", 
    name=f"Zig Zag-{i} High", 
    line=dict(width=1.5, color="yellow"), 
    hoverinfo="none", 
    connectgaps=True, 
    visibility="legendonly"
  ), row=1, col=1)
  
  fig.add_trace(go.Scatter(
    x=df["Date"], 
    y=df[f"zz{i}Lo"], 
    mode="lines", 
    name=f"Zig Zag-{i} Low", 
    line=dict(width=1.5, color="yellow"), 
    hoverinfo="none", 
    connectgaps=True, 
    visibility="legendonly"
  ), row=1, col=1)

# Add layout
fig.update_layout(
  title=f"{ticker} Alpha-Beta Analysis (1Y) | α = {alpha:.5f}, β = {beta:.2f}",
  xaxis_title="Date",
  yaxis_title="Price",
  height=600,
  xaxis_rangeslider_visible=False,
  hovermode="x unified",
  spikedistance=-1,
  xaxis1=dict(
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
  yaxis1=dict(
    showspikes=True,
    spikecolor='rgba(255,255,255,0.3)',
    spikedash='solid',
    spikesnap='cursor',
    spikemode='across',
    spikethickness=2,
  )
)

st.plotly_chart(fig)
