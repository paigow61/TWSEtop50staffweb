import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from datetime import datetime

# --- 1. 頁面配置 ---
st.set_page_config(page_title="台股百大五線譜", layout="wide")

# --- 2. 完整 100 檔名單 (修正重複鍵值並優化清單) ---
STOCKS = {
    '2330 台積電': '2330.TW', '2317 鴻海': '2317.TW', '2454 聯發科': '2454.TW', '2308 台達電': '2308.TW', '2881 富邦金': '2881.TW',
    '2382 廣達': '2382.TW', '2882 國泰金': '2882.TW', '3711 日月光': '3711.TW', '2303 聯電': '2303.TW', '2891 中信金': '2891.TW',
    '2412 中華電': '2412.TW', '2886 兆豐金': '2886.TW', '2884 玉山金': '2884.TW', '1216 統一': '1216.TW', '2892 第一金': '2892.TW',
    '5880 合庫金': '5880.TW', '2002 中鋼': '2002.TW', '3231 緯創': '3231.TW', '2357 華碩': '2357.TW', '2885 元大金': '2885.TW',
    '2603 長榮': '2603.TW', '2327 國巨': '2327.TW', '2880 華南金': '2880.TW', '2883 開發金': '2883.TW', '2408 南亞科': '2408.TW',
    '2379 瑞昱': '2379.TW', '2609 陽明': '2609.TW', '1301 台塑': '1301.TW', '1303 南亞': '1303.TW', '2615 萬海': '2615.TW',
    '3008 大立光': '3008.TW', '2395 研華': '2395.TW', '3045 台灣大': '3045.TW', '2409 友達': '2409.TW', '3034 聯詠': '3034.TW',
    '3037 欣興': '3037.TW', '2352 佳世達': '2352.TW', '1101 台泥': '1101.TW', '2912 統一超': '2912.TW', '2313 華通': '2313.TW',
    '6669 緯穎': '6669.TW', '5876 上海商銀': '5876.TW', '1326 台化': '1326.TW', '4938 和碩': '4938.TW', '9904 寶成': '9904.TW',
    '2887 台新金': '2887.TW', '6505 台塑化': '6505.TW', '2474 可成': '2474.TW', '1402 遠東新': '1402.TW', '2301 光寶科': '2301.TW',
    '3481 群創': '3481.TW', '2345 智邦': '2345.TW', '3661 世芯-KY': '3661.TW', '2377 微星': '2377.TW', '2890 永豐金': '2890.TW', 
    '2801 彰銀': '2801.TW', '5871 中租-KY': '5871.TW', '2618 長榮航': '2618.TW', '2610 華航': '2610.TW', '2360 致茂': '2360.TW', 
    '3017 奇鋐': '3017.TW', '2376 技嘉': '2376.TW', '2049 上銀': '2049.TW', '1504 東元': '1504.TW', '2353 宏碁': '2353.TW', 
    '2324 仁寶': '2324.TW', '2356 英業達': '2356.TW', '3702 大聯大': '3702.TW', '9921 巨大': '9921.TW', '9914 美利達': '9914.TW', 
    '1476 儒鴻': '1476.TW', '1477 聚陽': '1477.TW', '2105 正新': '2105.TW', '2207 和泰車': '2207.TW', '2404 漢唐': '2404.TW', 
    '6239 力成': '6239.TW', '5347 世界': '5347.TW', '3035 智原': '3035.TW', '8046 南電': '8046.TW', '6415 矽力*-KY': '6415.TW', 
    '2383 台光電': '2383.TW', '3532 台勝科': '3532.TW', '6488 環球晶': '6488.TW', '1513 中興電': '1513.TW', '1519 華城': '1519.TW', 
    '1503 士電': '1503.TW', '1722 台肥': '1722.TW', '1717 長興': '1717.TW', '9910 豐泰': '9910.TW', '9945 潤泰新': '9945.TW', 
    '9941 裕融': '9941.TW', '8464 億豐': '8464.TW', '0050 元大台灣50': '0050.TW'
}

# --- 3. 核心數據處理 ---
@st.cache_data(ttl=3600)
def download_stock_data(symbol, period):
    df = yf.download(symbol, period=period, auto_adjust=True, progress=False)
    if df.empty: return pd.Series()
    # 確保只取 Close 欄位並擠壓成單一 Series
    if 'Close' in df.columns:
        p = df['Close']
    else:
        p = df
    return p.squeeze() # 核心修正：確保拿到的是單一維度

def analyze_linear(p_series):
    p_clean = p_series.dropna()
    y = p_clean.values.reshape(-1, 1)
    x = np.arange(len(y)).reshape(-1, 1)
    model = LinearRegression().fit(x, y)
    trend = model.predict(x).flatten()
    std = np.std(y.flatten() - trend)
    return p_clean, trend, std

# --- 4. 側邊欄 ---
st.sidebar.header(f"📅 {datetime.now().strftime('%Y-%m-%d')}")
selected_label = st.sidebar.selectbox("🔎 搜尋標的", list(STOCKS.keys()))
period = st.sidebar.selectbox("回歸長度", ["3y", "5y", "10y"], index=1)
scan_btn = st.sidebar.button("🚀 啟動百大價值掃描")

# --- 5. 主畫面展示 ---
st.title("🏹 台股百大五線譜分析與選股 by L.C.")

p_data = download_stock_data(STOCKS[selected_label], period)

if not p_data.empty:
    p, trend, std = analyze_linear(p_data)
    
    # 安全取值與強制轉型
    last_p = float(p.iloc[-1])
    last_t = float(trend[-1])
    z = (last_p - last_t) / std

    # 狀態判定
    if z < -2: status, color = "極度便宜 (超賣)", "purple"
    elif z < -1: status, color = "便宜 (低檔)", "blue"
    elif z > 2: status, color = "極度昂貴 (超買)", "red"
    elif z > 1: status, color = "昂貴 (高檔)", "orange"
    else: status, color = "合理區域", "green"

    c1, c2, c3 = st.columns(3)
    c1.metric("目前價", f"{last_p:.2f}")
    c2.metric("偏離度 SD", f"{z:.2f}")
    st.markdown(f"### 當前評價：:{color}[**{status}**]")

    # 圖表繪製
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=p.index, y=p, name="價格", line=dict(color='black', width=2)))
    fig.add_trace(go.Scatter(x=p.index, y=trend, name="中線", line=dict(color='green')))
    
    # 繪製四條標準差線
    colors = ['red', 'orange', 'blue', 'purple']
    names = ['+2SD', '+1SD', '-1SD', '-2SD']
    multipliers = [2, 1, -1, -2]
    
    for m, c, n in zip(multipliers, colors, names):
        fig.add_trace(go.Scatter(x=p.index, y=trend + m*std, name=n, 
                                 line=dict(dash='dot', color=c, width=1)))

    fig.update_layout(height=500, margin=dict(l=0, r=0, t=20, b=0), 
                      hovermode="x unified", legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)

# --- 6. 掃描功能 ---
if scan_btn:
    st.divider()
    st.subheader("📡 百大掃描結果 (SD < -1)")
    with st.spinner('分析中...'):
        all_tickers = list(STOCKS.values())
        # 下載所有 Close 資料
        raw_all = yf.download(all_tickers, period=period, auto_adjust=True, progress=False)['Close']
        
        scan_results = []
        for name, ticker in STOCKS.items():
            try:
                # 確保從大表中精確提取該檔股票 Series
                s_p = raw_all[ticker].dropna()
                if len(s_p) < 100: continue
                
                t_v, s_v = analyze_linear(s_p)[1:] # 只取 trend 和 std
                last_val = float(s_p.iloc[-1])
                last_trd = float(t_v[-1])
                z_v = (last_val - last_trd) / s_v
                
                if z_v < -1:
                    scan_results.append({"股票": name, "SD": round(z_v, 2), "現價": round(last_val, 2)})
            except: continue
            
    if scan_results:
        st.table(pd.DataFrame(scan_results).sort_values("SD"))
    else:
        st.info("目前無超跌標的。")
