import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from datetime import datetime

# --- 1. 定義台股百大名單 (範例，請確保包含你完整清單) ---
STOCKS = {
    '2330 台積電': '2330.TW', '2317 鴻海': '2317.TW', '2454 聯發科': '2454.TW',
    '2308 台達電': '2308.TW', '2881 富邦金': '2881.TW', '2382 廣達': '2382.TW',
    '2882 國泰金': '2882.TW', '3711 日月光': '3711.TW', '2303 聯電': '2303.TW',
    '2891 中信金': '2891.TW', '2412 中華電': '2412.TW' # 建議依此格式補完
}

# --- 2. 核心分析引擎 ---
def analyze_market_data(tickers_dict):
    symbols = list(tickers_dict.values())
    names = list(tickers_dict.keys())
    
    print(f"📡 正在從 Yahoo Finance 抓取 {len(symbols)} 檔標的數據...")
    # 一次性下載過去三年的日線資料，減少請求次數
    data = yf.download(symbols, period="3y", interval="1d", group_by='ticker', progress=False)
    
    results = []
    
    for name, ticker in tickers_dict.items():
        try:
            # 提取單一標的資料並去除空值
            df = data[ticker].dropna()
            if len(df) < 252: continue # 數據不足一年則跳過
            
            close_prices = df['Close'].values
            current_p = float(close_prices[-1])
            
            # --- 五線譜計算 (線性回歸) ---
            y = close_prices.reshape(-1, 1)
            x = np.arange(len(y)).reshape(-1, 1)
            model = LinearRegression().fit(x, y)
            trend = model.predict(x)
            std = np.std(y - trend)
            last_trend = float(trend[-1])
            
            # --- 判斷位階 ---
            bias = (current_p - last_trend) / std  # 標準差倍數
            
            if bias > 2: status, color = "⚠️ 極度高估", "#FF3B30"
            elif bias > 1: status, color = "📈 偏高", "#FF9500"
            elif bias < -2: status, color = "💎 極度低估", "#34C759"
            elif bias < -1: status, color = "🔍 偏低", "#007AFF"
            else: status, color = "⚖️ 合理區間", "#8E8E93"

            # --- 專家評論邏輯 (結合台股專家意見) ---
            expert_note = ""
            if "極度低估" in status:
                expert_note = "【逆向操作】股價進入價值區，適合長線存股族分批建倉，無須恐慌。"
            elif "偏低" in status:
                expert_note = "【觀察買點】股價低於長期平均，可關注成交量是否放大，確認落底信號。"
            elif "極度高估" in status:
                expert_note = "【風險警示】乖離過大，技術指標已超買。建議逢高部分獲利了結，切勿追高。"
            elif "偏高" in status:
                expert_note = "【謹慎看待】短期多頭強勁但面臨壓力，建議設好移動止損，保護獲利。"
            else:
                expert_note = "【中性整理】股價隨大盤波動，無明顯過熱或委屈，建議按計畫定期定額。"

            results.append({
                "name": name,
                "price": round(current_p, 2),
                "status": status,
                "color": color,
                "note": expert_note
            })
            
        except Exception as e:
            print(f"❌ 分析 {name} 時出錯: {e}")
            
    return results

# --- 3. HTML 郵件模板 ---
def build_html_report(analysis_results):
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M')
    
    rows = ""
    for r in analysis_results:
        rows += f"""
        <tr style="border-bottom: 1px solid #eeeeee;">
            <td style="padding: 12px; font-weight: bold; color: #333;">{r['name']}</td>
            <td style="padding: 12px; color: #444;">${r['price']}</td>
            <td style="padding: 12px; color: {r['color']}; font-weight: bold;">{r['status']}</td>
            <td style="padding: 12px; color: #666; font-size: 13px; line-height: 1.4;">{r['note']}</td>
        </tr>
        """

    html = f"""
    <html>
    <body style="margin: 0; padding: 20px; font-family: 'PingFang TC', 'Microsoft JhengHei', sans-serif; background-color: #f9f9f9;">
        <div style="max-width: 800px; margin: auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.05);">
            <h2 style="color: #1a1a1a; border-left: 5px solid #007AFF; padding-left: 15px;">台股百大五線譜週報</h2>
            <p style="color: #666;">這是您的每週自動 AI 投資掃描。基於過去三年的趨勢回歸分析與專家建議。</p>
            <p style="font-size: 12px; color: #999;">報告生成時間：{now_str}</p>
            <table style="width: 100%; border-collapse: collapse; margin-top: 20px;">
                <thead>
                    <tr style="background-color: #007AFF; color: white; text-align: left;">
                        <th style="padding: 12px; border-top-left-radius: 5px;">標的</th>
                        <th style="padding: 12px;">現價</th>
                        <th style="padding: 12px;">五線譜狀態</th>
                        <th style="padding: 12px; border-top-right-radius: 5px;">專家建議</th>
                    </tr>
                </thead>
                <tbody>
                    {rows}
                </tbody>
            </table>
            <div style="margin-top: 30px; padding: 15px; background: #fffbe6; border: 1px solid #ffe58f; border-radius: 5px;">
                <p style="margin: 0; font-size: 12px; color: #856404;">
                    🔔 <b>免責聲明：</b> 本報告由 AI 自動生成，計算基於歷史數據。投資一定有風險，基金投資有賺有賠，申購前應詳閱公開說明書（或諮詢專業理財顧問）。
                </p>
            </div>
        </div>
    </body>
    </html>
    """
    return html

# --- 4. 發送系統 ---
def send_email(html_body):
    user = os.environ.get('EMAIL_USER')
    password = os.environ.get('EMAIL_PASS')
    
    msg = MIMEMultipart()
    msg['Subject'] = f"📈 每週台股掃描：{len(STOCKS)} 檔標的監測報告"
    msg['From'] = f"AI 投資 Agent <{user}>"
    msg['To'] = user
    
    msg.attach(MIMEText(html_body, 'html'))
    
    try:
        with smtplib.SMTP_SSL('://gmail.com', 465) as server:
            server.login(user, password)
            server.send_message(msg)
        print("🚀 郵件發送成功！")
    except Exception as e:
        print(f"❌ 發送失敗: {e}")

if __name__ == "__main__":
    report_data = analyze_market_data(STOCKS)
    if report_data:
        full_html = build_html_report(report_data)
        send_email(full_html)
