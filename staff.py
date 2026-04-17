import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go

# --- 頁面配置 ---
st.set_page_config(page_title="台股50強五線譜圖by L.C.", layout="wide")

# --- 完整 50 檔名單 ---
STOCKS = {
    '2330 台積電': '2330.TW', '2317 鴻海': '2317.TW', '2454 聯發科': '2454.TW', 
    '2308 台達電': '2308.TW', '3711 日月光': '3711.TW', '2891 中信金': '2891.TW', 
    '2382 廣達': '2382.TW', '2881 富邦金': '2881.TW', '2303 聯電': '2303.TW', 
    '2882 國泰金': '2882.TW', '2412 中華電': '2412.TW', '2886 兆豐金': '2886.TW', 
    '2884 玉山金': '2884.TW', '1216 統一': '1216.TW', '2892 第一金': '2892.TW', 
    '5880 合庫金': '5880.TW', '2002 中鋼': '2002.TW', '3231 緯創': '3231.TW', 
    '2357 華碩': '2357.TW', '2885 元大金': '2885.TW', '2603 長榮': '2603.TW', 
    '2327 國巨': '2327.TW', '2880 華南金': '2880.TW', '2883 開發金': '2883.TW', 
    '2408 南亞科': '2408.TW', '2379 瑞昱': '2379.TW', '2609 陽明': '2609.TW', 
    '1301 台塑': '1301.TW', '1303 南亞': '1303.TW', '2615 萬海': '2615.TW', 
    '3008 大立光': '3008.TW', '2395 研華': '2395.TW', '3045 台灣大': '3045.TW', 
    '2409 友達': '2409.TW', '3034 聯詠': '3034.TW', '3037 欣興': '3037.TW', 
    '2352 佳世達': '2352.TW', '1101 台泥': '1101.TW', '2912 統一超': '2912.TW', 
    '2313 華通': '2313.TW', '6669 緯穎': '6669.TW', '5876 上海商銀': '5876.TW', 
    '1326 台化': '1326.TW', '4938 和碩': '4938.TW', '9904 寶成': '9904.TW', 
    '2887 台新金': '2887.TW', '6505 台塑化': '6505.TW', '2474 可成': '2474.TW', 
    '1402 遠東新': '1402.TW', '2301 光寶科': '2301.TW'
}

# --- 側邊選單 ---
st.sidebar.title("🛠 控制面板")
selected_label = st.sidebar.selectbox("請選擇股票", list(STOCKS.keys()))
period = st.sidebar.selectbox("回歸長度", ["3y", "5y", "10y"], index=1)
scan_btn = st.sidebar.button("🚀 執行全自動選股掃描")

# --- 計算函數 ---
def analyze_stock(symbol, period):
    # 使用 auto_adjust=True 確保取得調整後收盤價
    df = yf.download(symbol, period=period, auto_adjust=True, progress=False)
    if df.empty: 
        return None
    
    # 修正：確保提取的是單一維度的 Series
    if isinstance(df.columns, pd.MultiIndex):
        p = df['Close'][symbol]
    else:
        p = df['Close']
        
    p = p.dropna()
    y = p.values.reshape(-1, 1)
    x = np.arange(len(y)).reshape(-1, 1)
    
    model = LinearRegression().fit(x, y)
    trend = model.predict(x).flatten()
    std = np.std(y.flatten() - trend)
    
    return p, trend, std

# --- 主畫面顯示 ---
st.title(f"📊 {selected_label} 五線譜報告")
data = analyze_stock(STOCKS[selected_label], period)

if data:
    p, trend, std = data
    # 修正：強制轉換為標量 float
    curr_p = float(p.iloc[-1])
    curr_trend = float(trend[-1])
    z = (curr_p - curr_trend) / std
    
    # 顯示指標卡
    m1, m2 = st.columns(2)
    m1.metric("目前股價", f"{curr_p:.2f}")
    m2.metric("偏離度 (SD)", f"{z:.2f}")

    # Plotly 互動圖表
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=p.index, y=p, name="價格", line=dict(color='#333333')))
    fig.add_trace(go.Scatter(x=p.index, y=trend, name="中線", line=dict(color='green')))
    fig.add_trace(go.Scatter(x=p.index, y=trend+2*std, name="+2SD (極度樂觀)", line=dict(dash='dash', color='red')))
    fig.add_trace(go.Scatter(x=p.index, y=trend-2*std, name="-2SD (極度悲觀)", line=dict(dash='dash', color='purple')))
    
    fig.update_layout(
        height=500, 
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

# --- 掃描功能 ---
if scan_btn:
    st.divider()
    st.subheader("🔎 目前超跌標的 (SD < -1)")
    bar = st.progress(0)
    results = []
    
    all_syms = list(STOCKS.values())
    # 批次下載
    raw_data = yf.download(all_syms, period=period, auto_adjust=True, progress=False)['Close']
    
    for i, (name, sym) in enumerate(STOCKS.items()):
        try:
            # 處理批次下載可能產生的格式問題
            temp_p = raw_data[sym].dropna()
            if len(temp_p) < 100: continue
            
            y_v = temp_p.values.reshape(-1, 1)
            x_v = np.arange(len(y_v)).reshape(-1, 1)
            model_v = LinearRegression().fit(x_v, y_v)
            t_v = model_v.predict(x_v).flatten()
            s_v = np.std(y_v.flatten() - t_v)
            
            # 修正：強制轉 float
            last_p = float(temp_p.iloc[-1])
            last_t = float(t_v[-1])
            z_v = (last_p - last_t) / s_v
            
            if z_v < -1:
                results.append({
                    "股票": name, 
                    "SD": round(float(z_v), 2), 
                    "現價": round(last_p, 1)
                })
        except Exception as e:
            continue
        bar.progress((i+1)/len(STOCKS))
    
    if results:
        st.dataframe(pd.DataFrame(results).sort_values("SD"), use_container_width=True)
    else:
        st.write("目前沒有標的處於低檔。")
