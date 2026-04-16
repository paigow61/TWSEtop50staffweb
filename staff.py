import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# --- 1. 頁面配置 ---
st.set_page_config(page_title="台股50強五線譜分析", layout="wide")

# --- 2. 完整 50 檔名單 ---
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

# --- 3. 側邊控制 ---
st.sidebar.title("🛠 控制面板")
selected_label = st.sidebar.selectbox("選擇股票", list(STOCKS.keys()))
period = st.sidebar.selectbox("回歸長度", ["3y", "5y", "10y"], index=1)
scan_btn = st.sidebar.button("🚀 執行全自動掃描")

# --- 4. 主畫面分析 ---
st.title(f"📊 {selected_label} 數據簡報")

try:
    # 獲取數據
    df = yf.download(STOCKS[selected_label], period=period, auto_adjust=True, progress=False)
    
    if not df.empty:
        # 核心修正：強制選取 Close 並轉換為單一維度
        if isinstance(df.columns, pd.MultiIndex):
            p = df['Close'][STOCKS[selected_label]].dropna()
        else:
            p = df['Close'].dropna()
            
        y = p.values.reshape(-1, 1)
        x = np.arange(len(y)).reshape(-1, 1)
        
        # 計算回歸
        model = LinearRegression().fit(x, y)
        trend = model.predict(x).flatten()
        std = np.std(y.flatten() - trend)
        
        # 取得最後一個純量數值 (避免陣列報錯)
        last_price = float(p.iloc[-1])
        last_trend = float(trend[-1])
        z = (last_price - last_trend) / std
        
        # 顯示指標
        m1, m2, m3 = st.columns(3)
        m1.metric("目前股價", f"{last_price:.2f}")
        m2.metric("偏離度 (SD)", f"{z:.2f}")
        
        status = "🔥 極度便宜" if z < -2 else "📉 偏低" if z < -1 else "🚀 極度昂貴" if z > 2 else "⚖️ 合理"
        m3.subheader(status)

        # 繪圖 (Matplotlib)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(p.index, y, label="價格", color="#333333", alpha=0.7)
        ax.plot(p.index, trend, label="趨勢線", color="green", linewidth=2)
        ax.plot(p.index, trend + 2*std, '--', color="red", label="+2SD")
        ax.plot(p.index, trend - 2*std, '--', color="purple", label="-2SD")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    else:
        st.error("無法取得該股票數據。")

except Exception as e:
    st.error(f"發生錯誤: {e}")

# --- 5. 掃描功能 ---
if scan_btn:
    st.divider()
    st.subheader("🔎 目前超跌標的 (SD < -1)")
    
    all_syms = list(STOCKS.values())
    raw = yf.download(all_syms, period=period, auto_adjust=True, progress=False)['Close']
    
    results = []
    for name, sym in STOCKS.items():
        try:
            temp_p = raw[sym].dropna()
            y_v = temp_p.values.reshape(-1, 1)
            x_v = np.arange(len(y_v)).reshape(-1, 1)
            m_v = LinearRegression().fit(x_v, y_v)
            t_v = m_v.predict(x_v).flatten()
            s_v = np.std(y_v.flatten() - t_v)
            cur_p = float(temp_p.iloc[-1])
            z_v = (cur_p - t_v[-1]) / s_v
            
            if z_v < -1:
                results.append({"股票": name, "SD": round(z_v, 2), "現價": round(cur_p, 2)})
        except: continue
        
    if results:
        st.dataframe(pd.DataFrame(results).sort_values("SD"), use_container_width=True)
    else:
        st.info("目前無超跌股票。")
