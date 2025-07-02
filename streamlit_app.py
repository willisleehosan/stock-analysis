import streamlit as st
from streamlit_extras.stylable_container import stylable_container
import math
import bisect
import pandas as pd
import numpy as np
import sklearn
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
import pwlf
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from dateutil.relativedelta import relativedelta

def fetchData(ticker):
  df = yf.download(ticker, start=datetime.now() - relativedelta(years=3), end=datetime.now(), ignore_tz=True)
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
  
  smac = []
  smaTag = ["20SMA", "50SMA", "100SMA"]
  for i in range(1, len(smaEx)):
    for (shortSMA, longSMA) in [(0, 1), (0, 2), (1, 2)]:
      if (smaEx[smaTag[shortSMA]].iloc[i-1] <= smaEx[smaTag[longSMA]].iloc[i-1]) and (smaEx[smaTag[shortSMA]].iloc[i] > smaEx[smaTag[longSMA]].iloc[i]):
        smac.append([f"gc{shortSMA}{longSMA}", i-1, i])
      elif (smaEx[smaTag[shortSMA]].iloc[i-1] >= smaEx[smaTag[longSMA]].iloc[i-1]) and (smaEx[smaTag[shortSMA]].iloc[i] < smaEx[smaTag[longSMA]].iloc[i]):
        smac.append([f"dc{shortSMA}{longSMA}", i-1, i])

  return smac

def zigzag(df):  
  troughs = []
  peaks = []
  dates = df.index.to_pydatetime()
  highs = df["High"].values
  lows = df["Low"].values
  closes = df["Close"].values

  seekHi = True
  ext = highs[0]
  thres = lows[0]
  cand = 0
  for i in range(1, len(closes)):
    if seekHi:
      if highs[i] > ext:
        ext = highs[i]
        thres = lows[i]
        cand = i
      elif closes[i] < thres:
        peaks.append([cand, ext])
        seekHi = False
        ext = lows[i]
        thres = highs[i]
        cand = i
    else:
      if lows[i] < ext:
        ext = lows[i]
        thres = highs[i]
        cand = i
      elif closes[i] > thres:
        troughs.append([cand, ext])
        seekHi = True
        ext = highs[i]
        thres = lows[i]
        cand = i

  return peaks, troughs

def fibLev(peaks, troughs):
  fib = []
  zz = peaks + troughs
  zz.sort(key=lambda a: a[0])
  for i in range(len(zz)-1, len(zz)-6, -1):
    start = zz[i-1][1]
    end = zz[i][1]
    fib.append(end)
    fib.append(start*0.236 + end*0.764)
    fib.append(start*0.382 + end*0.618)
    fib.append(start*0.5 + end*0.5)
    fib.append(start*0.618 + end*0.382)
    fib.append(start*0.786 + end*0.214)
    fib.append(start)
    fib.append(start*1.272 - end*0.272)
    fib.append(start*1.618 - end*0.618)
    fib.append(start*2.618 - end*1.618)
    fib.append(start*3.618 - end*2.618)
    fib.append(start*4.236 - end*3.236)

  return fib

def pivLev(df):
  openVals = df["Open"].values
  closeVals = df["Close"].values
  highVals = df["High"].values
  lowVals = df["Low"].values
  indexVals = df.index.to_pydatetime()
  
  week0 = np.searchsorted(indexVals, (pd.Timestamp(indexVals[-1]) - pd.tseries.frequencies.to_offset("W") - pd.DateOffset(weeks=1)).to_pydatetime())
  week1 = np.searchsorted(indexVals, (pd.Timestamp(indexVals[-1]) - pd.tseries.frequencies.to_offset("W")).to_pydatetime())
  month0 = np.searchsorted(indexVals, (pd.offsets.MonthBegin().rollback(pd.Timestamp(indexVals[-1])) - pd.DateOffset(months=1)).to_pydatetime())
  month1 = np.searchsorted(indexVals, (pd.offsets.MonthBegin().rollback(pd.Timestamp(indexVals[-1]))).to_pydatetime())
  st.write(indexVals[week0])
  st.write(indexVals[week1])
  st.write(indexVals[month0])
  st.write(indexVals[month1])

def rsi(arr, l):
  u = [float("nan")]
  d = [float("nan")]
  for i in range(1, len(arr)):
    u.append(max(0, arr[i] - arr[i-1]))
    d.append(max(0, arr[i-1] - arr[i]))
  uSmma = u[:2]
  dSmma = d[:2]
  for i in range(2, len(arr)):
    uSmma.append((uSmma[i-1]*(l-1) + u[i]) / l)
    dSmma.append((dSmma[i-1]*(l-1) + d[i]) / l)
  uSmma[:(3*l)] = [float("nan")] * (3*l)
  dSmma[:(3*l)] = [float("nan")] * (3*l)
  rsi = []
  for i in range(0, len(arr)):
    rsi.append(100 * uSmma[i] / (uSmma[i] + dSmma[i]))
  return rsi

def rsiAn(df, peaks, troughs):
  rsiObs = []
  rsiVals = df["rsi"].values
  indexVals = df.index.to_pydatetime()

  # overbought
  startD = None
  for i in range(len(rsiVals)):
    if startD:
      if rsiVals[i] < 70:
        rsiObs.append(["rsiob", startD, i-1])
        startD = None
    else:
      if rsiVals[i] >= 70:
        startD = i

  # oversold
  startD = None
  for i in range(len(rsiVals)):
    if startD:
      if rsiVals[i] > 30:
        rsiObs.append(["rsios", startD, i-1])
        startD = None
    else:
      if rsiVals[i] <= 30:
        startD = i

  return rsiObs

def divDet(ind, peaks, troughs, bull, bear):
  divObs = []
  
  # bear div
  for i in range(1, len(peaks)):
    if peaks[i][1] > peaks[i-1][1]:
      id0 = peaks[i-1][0]
      id1 = peaks[i][0]
      ind0 = int(np.argmax(ind[max(0, id0-3):min(len(ind), id0+4)]) + max(0, id0-3))
      if ind0 == max(0, id0-3) or ind0 == min(len(ind)-1, id0+3):
        continue
      ind1 = int(np.argmax(ind[max(0, id1-3):min(len(ind), id1+4)]) + max(0, id1-3))
      if ind1 == max(0, id1-3) or ind1 == min(len(ind)-1, id0+3):
        continue
      if ind[ind0] > ind[ind1]:
        divObs.append([bear, id0, id1, ind0, ind1])
        
  # bull div
  for i in range(1, len(troughs)):
    if troughs[i][1] < troughs[i-1][1]:
      id0 = troughs[i-1][0]
      id1 = troughs[i][0]
      ind0 = int(np.argmin(ind[max(0, id0-3):min(len(ind), id0+4)]) + max(0, id0-3))
      if ind0 == max(0, id0-3) or ind0 == min(len(troughs)-1, id1+3):
        continue
      ind1 = int(np.argmin(ind[max(0, id1-3):min(len(ind), id1+4)]) + max(0, id1-3))
      if ind1 == max(0, id0-3) or ind1 == min(len(troughs)-1, id1+3):
        continue
      if ind[ind0] < ind[ind1]:
        divObs.append([bull, id0, id1, ind0, ind1])

  return divObs

def season(df, marketDf, sma):
  ssDf = pd.DataFrame({
    "stock_close": df["Close"], 
    "market_close": marketDf["Close"]
  }, index=(df.index)).dropna()
  ssDf["stock_sma"] = ssDf["stock_close"].rolling(window=sma, center=True).mean()
  ssDf["market_sma"] = ssDf["market_close"].rolling(window=sma, center=True).mean()
  ssDf["stock_ret"] = ssDf["stock_sma"].pct_change()
  ssDf["market_ret"] = ssDf["market_sma"].pct_change()
  ssDf = ssDf.dropna()

  model = sklearn.linear_model.LinearRegression()
  model.fit(ssDf["market_ret"].values.reshape(-1, 1), ssDf["stock_ret"].values)
  ssDf["residue"] = ssDf["stock_ret"] - model.predict(ssDf["market_ret"].values.reshape(-1, 1))
  rsr = ssDf["residue"].rolling(window=sma, center=True).mean()
  rsm = rsr.diff().rolling(window=sma, center=True).mean()
  ssX = []
  ssY = []
  grouped = ssDf.groupby(ssDf.index.to_period("Y"))
  for name, group in grouped:
    ssX.append(np.array([]))
    ssY.append(np.array(group["residue"].values))
    grouped2 = group.groupby(group.index.to_period("M"))
    for name2, group2 in grouped2:
      ssX[len(ssX)-1] = np.append(ssX[len(ssX)-1], np.linspace(int(str(name2).split("-")[1]), int(str(name2).split("-")[1])+1, num=len(group2["residue"].values), endpoint=False))

  xGrid = np.linspace(1, 13, num=365, endpoint=False)
  gridVals = []
  for xVals, yVals in zip(ssX, ssY):
    if len(xVals) < 2:
      continue
    f = interp1d(xVals, yVals, kind="linear", bounds_error=False)
    gridVals.append(f(xGrid))
  meanSs = np.nanmean(gridVals, axis=0)
  return ssX, ssY, xGrid, meanSs, rsr, rsm

obsTit = {
  "gc01": "Golden Cross", 
  "gc02": "Golden Cross", 
  "gc12": "Golden Cross", 
  "dc01": "Death Cross", 
  "dc02": "Death Cross", 
  "dc12": "Death Cross", 
  "rsiob": "RSI Overbought", 
  "rsios": "RSI Oversold", 
  "rsibd": "RSI Bearish Divergence", 
  "rsiwd": "RSI Bullish Divergence"
}

obsDesc = {
  "gc01": "20-D SMA > 50-D SMA", 
  "gc02": "20-D SMA > 100-D SMA", 
  "gc12": "50-D SMA > 100-D SMA", 
  "dc01": "20-D SMA < 50-D SMA", 
  "dc02": "20-D SMA < 100-D SMA", 
  "dc12": "50-D SMA < 100-D SMA", 
  "rsiob": "", 
  "rsios": "", 
  "rsibd": "", 
  "rsiwd": ""
}

obsBull = {
  "gc01": True, 
  "gc02": True, 
  "gc12": True, 
  "dc01": False, 
  "dc02": False, 
  "dc12": False, 
  "rsiob": False, 
  "rsios": True, 
  "rsibd": False, 
  "rsiwd": True
}

obsPlotKey = {
  "gc01": "sma", 
  "gc02": "sma", 
  "gc12": "sma", 
  "dc01": "sma", 
  "dc02": "sma", 
  "dc12": "sma", 
  "rsiob": "rsi", 
  "rsios": "rsi", 
  "rsibd": "rsi", 
  "rsiwd": "rsi"
}

obsPlot = {}

obsPlotName = {
  "sma": "Simple Moving Average Ribbon", 
  "rsi": "Relative Strength Index", 
  "rrg": "Relative Rotation Graph"
}
# ----------------------------------------------
obs = []

ticker = st.text_input("Ticker", "0189") + ".HK"

# market data
marketDf = yf.download("^HSI", start=datetime.now() - relativedelta(years=3), end=datetime.now(), ignore_tz=True)
marketDf.columns = ["Close", "High", "Low", "Open", "Volume"]
marketDf.index.tz_localize("Asia/Hong_Kong")

# future dates
futureDates = [marketDf.index[-1] + pd.tseries.offsets.BDay(n=i) for i in range(1, 11)]
futureDf = pd.DataFrame(index=(futureDates))

df, dfs = fetchData(ticker)
df = heikinashi(df)
resDf = pd.DataFrame(index=df.index)
alpha, beta = alphabeta(df, marketDf)
df["10SMA"] = df["Close"].rolling(window=10).mean()
df["20SMA"] = df["Close"].rolling(window=20).mean()
df["50SMA"] = df["Close"].rolling(window=50).mean()
df["100SMA"] = df["Close"].rolling(window=100).mean()
support_best, resistance_best = srSMA(df)
df["rsi"] = rsi(df["Close"], 14)
peaks, troughs = zigzag(df)
fib = fibLev(peaks, troughs)
pivLev(df)
obs += gdCross(df, futureDf)
obs += rsiAn(df, peaks, troughs)
obs += divDet(df["rsi"].values, peaks, troughs, "rsiwd", "rsibd")
ssX, ssY, meanSsX, meanSsY, df["rsm"], df["rsr"] = season(df, marketDf, 50)

# clean data
df = df.reset_index()
df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

# streamlit layout
c1, c2 = st.columns(2)

# basic plot
fig = make_subplots(rows=2, cols=2, shared_xaxes=True, shared_yaxes=True,
                    horizontal_spacing=0,
                    vertical_spacing=0.05,
                    subplot_titles=["", "", ""], 
                   row_heights=[0.8, 0.2], 
                   column_width=[0.95, 0.05])

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

fig.add_trace(go.Bar(
  x=df["Date"], 
  y=df["Volume"], 
  marker_color=["green" if close >= open_ else "red" for open_, close in zip(df["Open"], df["Close"])], 
  name="Volume"
), row=2, col=1)

for f in fib:
  fig.add_shape(
    type="line", 
    xref="x2 domain", 
    yref="y1", 
    x0=0, 
    y0=f, 
    x1=1, 
    y1=f, 
    line=dict(
      color="rgba(255, 255, 0, 0.3)", 
      width=1
    ), 
    row=1, col=2
  )

# Add layout
fig.update_layout(
  title=f"{ticker} Alpha-Beta Analysis (1Y) | α = {alpha:.5f}, β = {beta:.2f}",
  yaxis1_title="Price",
  yaxis3_title="Volume", 
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
  xaxis2=dict(
    showspikes=False
  ),
  xaxis3=dict(
    type="category", 
    categoryorder="array", 
    categoryarray=df["Date"].tolist(), 
    showspikes=True,
    spikecolor='rgba(255,255,255,0.3)',
    spikedash='solid',
    spikesnap='cursor',
    spikemode='across',
    spikethickness=2
  ), 
  yaxis=dict(
    range=[1.2*df["Low"].iloc[-91:].min() - 0.2*df["High"].iloc[-91:].max(), 1.2*df["High"].iloc[-91:].max() - 0.2*df["Low"].iloc[-91:].min()],
    showspikes=True,
    spikecolor='rgba(255,255,255,0.3)',
    spikedash='solid',
    spikesnap='cursor',
    spikemode='across',
    spikethickness=2
  ), 
  yaxis2=dict(
    range=[1.2*df["Low"].iloc[-91:].min() - 0.2*df["High"].iloc[-91:].max(), 1.2*df["High"].iloc[-91:].max() - 0.2*df["Low"].iloc[-91:].min()],
    showspikes=False
  ), 
  yaxis3=dict(
    range=[0, 1.2*df["Volume"].iloc[-91:].max()],
    showticklabels=False, 
    ticks="", 
    showgrid=False, 
    zeroline=False
  )
)

marketSsFig = make_subplots(rows=1, cols=1, shared_xaxes=True, 
                             vertical_spacing=0.05, 
                             subplot_titles=[""])

for i in range(0, len(ssX)):
  marketSsFig.add_trace(go.Scatter(
    x=ssX[i], 
    y=ssY[i], 
    mode="lines", 
    name=f"Residue {datetime.now().year + i - len(ssX) + 1}", 
    line=dict(width=1, color=f"rgb({i*100//(len(ssX)-1)+100}, {i*100//(len(ssX)-1)+100}, {i*100//(len(ssX)-1)+100})"), 
    hoverinfo="none"
  ), row=1, col=1)

marketSsFig.add_trace(go.Scatter(
  x=meanSsX, 
  y=meanSsY, 
  mode="lines", 
  name="Residue mean", 
  line=dict(width=1.5, color="red"), 
  hoverinfo="none"
), row=1, col=1)

marketSsFig.update_layout(
  title="Market Residue",
  height=600,
  xaxis_rangeslider_visible=False,
  hovermode="x unified",
  spikedistance=-1,
  xaxis=dict(
    tickmode="array", 
    tickvals=[], 
    showticklabels=False,
    showspikes=True,
    spikecolor='rgba(255,255,255,0.3)',
    spikedash='solid',
    spikesnap='cursor',
    spikemode='across',
    spikethickness=2
  ), 
  yaxis=dict(
    showticklabels=False, 
    ticks="", 
    showgrid=False
  ), 
  shapes=[
    dict(
      type="line", 
      xref="paper", 
      yref="y", 
      x0=0, 
      x1=1, 
      y0=0, 
      y1=0, 
      line=dict(
        color="white", 
        width=2
      ), 
      layer="below"
    )
  ]
)

month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

for month in range(1, 13):  
  marketSsFig.add_shape(
    type="line",
    xref="x", 
    yref="paper", 
    x0=month, 
    x1=month,
    y0=0, 
    y1=1,  
    line=dict(
      color="gray", 
      width=0.5, 
      dash="dash"
    ),
    layer="below"
  )

  marketSsFig.add_annotation(
    xref="x", 
    yref="paper", 
    xanchor="center", 
    yanchor="top", 
    x=month + 0.5, 
    y=0, 
    text=month_labels[month-1], 
    font=dict(
      size=10, 
      color="gray"
    ), 
    showarrow=False
  )

obsPlot["sma"] = make_subplots(rows=1, cols=1, shared_xaxes=True, 
                       vertical_spacing=0.05, 
                       subplot_titles=[""])

obsPlot["sma"].add_trace(go.Candlestick(
  x=df['Date'],
  open=df['Open'], high=df['High'],
  low=df['Low'], close=df['Close'],
  increasing_fillcolor='green',
  decreasing_fillcolor='red',
  line_width=1,
  opacity=1,
  name='Raw Candlestick',
  hoverinfo="none"
), row=1, col=1)

obsPlot["sma"].add_trace(go.Scatter(
  x=df['Date'],
  y=df["20SMA"],
  mode='lines',
  name="20-D SMA",
  line=dict(width=1.5, color="red"),
  hoverinfo="none"
), row=1, col=1)

obsPlot["sma"].add_trace(go.Scatter(
  x=df['Date'],
  y=df["50SMA"],
  mode='lines',
  name="50-D SMA",
  line=dict(width=1.5, color="orange"),
  hoverinfo="none"
), row=1, col=1)

obsPlot["sma"].add_trace(go.Scatter(
  x=df['Date'],
  y=df["100SMA"],
  mode='lines',
  name="100-D SMA",
  line=dict(width=1.5, color="yellow"),
  hoverinfo="none"
), row=1, col=1)

obsPlot["sma"].update_layout(
  title="SMA plot",
  yaxis_title="Price",
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
    range=[1.2*df["Low"].iloc[-91:].min() - 0.2*df["High"].iloc[-91:].max(), 1.2*df["High"].iloc[-91:].max() - 0.2*df["Low"].iloc[-91:].min()],
    showspikes=True,
    spikecolor='rgba(255,255,255,0.3)',
    spikedash='solid',
    spikesnap='cursor',
    spikemode='across',
    spikethickness=2
  )
)

obsPlot["rsi"] = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                               vertical_spacing=0.05, 
                               subplot_titles=["", ""], 
                               row_heights=[0.6, 0.4])

obsPlot["rsi"].add_trace(go.Candlestick(
  x=df['Date'],
  open=df['Open'], high=df['High'],
  low=df['Low'], close=df['Close'],
  increasing_fillcolor='green',
  decreasing_fillcolor='red',
  line_width=1,
  opacity=1,
  name='Raw Candlestick',
  hoverinfo="none"
), row=1, col=1)

obsPlot["rsi"].add_trace(go.Scatter(
  x=df["Date"], 
  y=df["rsi"], 
  mode="lines", 
  name="14-D RSI", 
  line=dict(width=1.5, color="white"),
  hoverinfo="none"
), row=2, col=1)

obsPlot["rsi"].update_layout(
  title="RSI plot",
  yaxis_title="Price", 
  yaxis2_title="RSI",
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
  xaxis2=dict(
    type="category", 
    categoryorder="array", 
    categoryarray=df["Date"].tolist(), 
    showspikes=True,
    spikecolor='rgba(255,255,255,0.3)',
    spikedash='solid',
    spikesnap='cursor',
    spikemode='across',
    spikethickness=2
  ), 
  yaxis=dict(
    range=[1.2*df["Low"].iloc[-91:].min() - 0.2*df["High"].iloc[-91:].max(), 1.2*df["High"].iloc[-91:].max() - 0.2*df["Low"].iloc[-91:].min()],
    showspikes=True,
    spikecolor='rgba(255,255,255,0.3)',
    spikedash='solid',
    spikesnap='cursor',
    spikemode='across',
    spikethickness=2
  ), 
  yaxis2=dict(
    range=[0, 100],
    showspikes=True,
    spikecolor='rgba(255,255,255,0.3)',
    spikedash='solid',
    spikesnap='cursor',
    spikemode='across',
    spikethickness=2,
    showticklabels=False, 
    ticks="", 
    showgrid=False, 
    zeroline=False
  ), 
  shapes=[
    dict(
      type="line", 
      xref="paper", 
      yref="y2", 
      x0=0, 
      x1=1, 
      y0=70, 
      y1=70, 
      line=dict(
        color="green", 
        width=1
      ), 
      layer="below"
    ), 
    dict(
      type="line", 
      xref="paper", 
      yref="y2", 
      x0=0, 
      x1=1, 
      y0=50, 
      y1=50, 
      line=dict(
        color="cyan", 
        width=1
      ), 
      layer="below"
    ), 
    dict(
      type="line", 
      xref="paper", 
      yref="y2", 
      x0=0, 
      x1=1, 
      y0=30, 
      y1=30, 
      line=dict(
        color="red", 
        width=1
      ), 
      layer="below"
    )
  ]
)

obsPlot["rrg"] = make_subplots(rows=1, cols=1, shared_xaxes=True, 
                               vertical_spacing=0.05, 
                               subplot_titles=[""])

obsPlot["rrg"].add_trace(go.Scatter(
  x=df["rsr"].iloc[-151:], 
  y=-df["rsm"].iloc[-151:], 
  mode="lines+markers", 
  name="Relative Rotation Trail", 
  text=df["Date"].iloc[-151:], 
  line_shape="spline"
), row=1, col=1)

obsPlot["rrg"].update_layout(
  title="RRG plot",
  xaxis_title="Relative Strength Ratio", 
  yaxis_title="Relative Strength Momentum",
  height=600,
  xaxis=dict(
    showspikes=False
  ),
  yaxis=dict(
    showspikes=False
  )
)

for ob in obs:
  if ob[0] == "rsibd":
    obsPlot["rsi"].add_shape(
      type="line", 
      xref="x", 
      yref="y1", 
      x0=df["Date"][ob[1]],
      y0=df["High"][ob[1]],
      x1=df["Date"][ob[2]], 
      y1=df["High"][ob[2]], 
      line=dict(
        color="yellow", 
        width=1
      ),
      layer="below"
    )
    
    obsPlot["rsi"].add_shape(
      type="line", 
      xref="x", 
      yref="y2", 
      x0=df["Date"][ob[3]],
      y0=df["rsi"][ob[3]],
      x1=df["Date"][ob[4]], 
      y1=df["rsi"][ob[4]], 
      line=dict(
        color="yellow", 
        width=1
      ),
      layer="below"
    )
  elif ob[0] == "rsiwd":
    obsPlot["rsi"].add_shape(
      type="line", 
      xref="x", 
      yref="y1", 
      x0=df["Date"][ob[1]],
      y0=df["Low"][ob[1]],
      x1=df["Date"][ob[2]], 
      y1=df["Low"][ob[2]], 
      line=dict(
        color="yellow", 
        width=1
      ),
      layer="below"
    )
    
    obsPlot["rsi"].add_shape(
      type="line", 
      xref="x", 
      yref="y2", 
      x0=df["Date"][ob[3]],
      y0=df["rsi"][ob[3]],
      x1=df["Date"][ob[4]], 
      y1=df["rsi"][ob[4]], 
      line=dict(
        color="yellow", 
        width=1
      ),
      layer="below"
    )

# --------------------------------------------------

with c1: 
  st.plotly_chart(fig, use_container_width=True)

with c2: 
  st.plotly_chart(marketSsFig, use_container_width=True)

st.markdown("---")

with st.container():
  b1, b2 = st.columns([1, 2.5])
  
  with b1: 
    obs.sort(reverse=True, key=lambda a: a[2])
    st.markdown("### Observations")
    st.markdown("---")
    with st.container(height=300):
      for i, item in enumerate(obs):
        obsKey = item[0]
        startD = df["Date"].iloc[item[1]]
        endD = df["Date"].iloc[item[2]]
        with stylable_container(
          key=f"obs_container_{i}", 
          css_styles=f"""
            button {{
              background-color: {"rgba(0, 255, 0, 0.3)" if obsBull[obsKey] else "rgba(255, 0, 0, 0.3)"}
            }}
          """
        ):
          if st.button(f"{startD} ~ {endD} \n\n**{obsTit[obsKey]}** \n\n{obsDesc[obsKey]}", key=f"obs_button_{i}"):
            if obsKey in obsPlotKey:
              st.session_state["obs_dropdown"] = obsPlotKey[obsKey]

  with b2:
    dropdown = st.selectbox(
      "Select Plot", 
      list(obsPlot.keys()), 
      index=None, 
      key="obs_dropdown", 
      format_func=lambda a: obsPlotName[a]
    )
    
    if dropdown in obsPlot:
      st.plotly_chart(obsPlot[dropdown], use_container_width=True)
    else:
      st.container(border=True, height=600).write("No available plots")
