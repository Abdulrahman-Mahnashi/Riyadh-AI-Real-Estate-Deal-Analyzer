import os
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from flask import Flask, jsonify, Response

# -------- إعداد المسارات من البيئة (مع قيم افتراضية) --------
DATA_PATH  = Path(os.environ.get("DATA_PATH",  "data/riyadh_air_quality.csv"))
MODEL_PATH = Path(os.environ.get("MODEL_PATH", "models/riyadh_pm25_xgb.joblib"))
TARGET = os.environ.get("TARGET", "PM2.5")

# -------- تحميل البيانات والنموذج --------
assert DATA_PATH.exists(),  f"DATA_PATH not found: {DATA_PATH}"
assert MODEL_PATH.exists(), f"MODEL_PATH not found: {MODEL_PATH}"

model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)

# تحديد عمود الوقت تلقائيًا
time_col_candidates = [c for c in df.columns if c.lower() in ["timestamp","datetime","date","time"]]
assert time_col_candidates, "No datetime column found. Rename your time column or add it."
TIME_COL = time_col_candidates[0]

# تحويل وتنظيف لسلسلة ساعية مع إزالة التكرارات
df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
df = df.dropna(subset=[TIME_COL]).sort_values(TIME_COL).reset_index(drop=True)
df[TIME_COL] = df[TIME_COL].dt.floor("h")

num_cols   = df.select_dtypes(include=[np.number]).columns.tolist()
other_cols = [c for c in df.columns if c not in num_cols + [TIME_COL]]

agg = {c: "mean" for c in num_cols}
agg.update({c: "last" for c in other_cols})

dfh = (df.groupby(TIME_COL, as_index=True).agg(agg).sort_index())
full_idx = pd.date_range(dfh.index.min(), dfh.index.max(), freq="h")
dfh = dfh.reindex(full_idx)
dfh.index.name = TIME_COL

# تعويض المفقودات
if num_cols:
    dfh[num_cols] = (dfh[num_cols]
                     .interpolate(method="time", limit_direction="both")
                     .fillna(method="ffill")
                     .fillna(method="bfill"))
for c in other_cols:
    if c in dfh and dfh[c].isna().any():
        mode_val = dfh[c].mode().iloc[0] if not dfh[c].mode().empty else "unknown"
        dfh[c] = dfh[c].fillna(mode_val)

assert TARGET in dfh.columns, f"Target '{TARGET}' not found. Available: {list(dfh.columns)}"

# -------- ميزات زمنية --------
def add_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    f = frame.copy()
    f["hour"] = f.index.hour
    f["dayofweek"] = f.index.dayofweek
    f["month"] = f.index.month
    f["is_weekend"] = (f["dayofweek"] >= 5).astype(int)
    return f

def add_lags_rolls(frame: pd.DataFrame, target: str,
                   lags=(1,2,3,6,12,24), rolls=(3,6,12,24)) -> pd.DataFrame:
    f = frame.copy()
    for L in lags:
        f[f"{target}_lag{L}"] = f[target].shift(L)
    for w in rolls:
        f[f"{target}_rollmean{w}"] = f[target].rolling(w, min_periods=max(1, w//2)).mean()
        f[f"{target}_rollstd{w}"]  = f[target].rolling(w, min_periods=max(1, w//2)).std()
    return f

def build_feature_view(history_df: pd.DataFrame):
    Xy = add_time_features(history_df)
    Xy = add_lags_rolls(Xy, TARGET)
    Xy = Xy.dropna().copy()
    features = [c for c in Xy.columns if c != TARGET]
    return Xy, features

# -------- AQI: تحويل PM2.5 → AQI (US EPA - تبسيط تعليمي) --------
_PM25_BREAKS = [
    (0.0,   12.0,   0,   50,  "جيد",                          "aqi-good"),
    (12.1,  35.4,  51,  100,  "متوسط",                        "aqi-moderate"),
    (35.5,  55.4, 101,  150,  "غير صحي للفئات الحساسة",      "aqi-usg"),
    (55.5, 150.4, 151,  200,  "غير صحي",                     "aqi-unhealthy"),
    (150.5,250.4, 201,  300,  "غير صحي جدًا",                "aqi-very"),
    (250.5,500.4, 301,  500,  "خطِر",                         "aqi-hazardous"),
]

def pm25_to_aqi(ug):
    if pd.isna(ug):
        return np.nan, "غير معروف", "aqi-unknown"
    for Cl, Ch, Il, Ih, txt, cls in _PM25_BREAKS:
        if Cl <= ug <= Ch:
            aqi = (Ih - Il) / (Ch - Cl) * (ug - Cl) + Il
            return round(aqi), txt, cls
    return 500, "خطِر", "aqi-hazardous"

def series_pm25_to_aqi(s: pd.Series):
    aqis, txts, clss = [], [], []
    for v in s.values:
        aqi, txt, cls = pm25_to_aqi(v)
        aqis.append(aqi); txts.append(txt); clss.append(cls)
    return pd.Series(aqis, index=s.index), txts, clss

def last_window_arrays(n=72):
    tail = dfh.tail(n).copy()
    aqi_series, _, _ = series_pm25_to_aqi(tail[TARGET])
    return {
        "timestamps": tail.index.astype(str).tolist(),
        "pm25": tail[TARGET].astype(float).round(3).tolist(),
        "aqi": aqi_series.astype(float).tolist(),
    }

# -------- Flask --------
app = Flask(__name__)

@app.route("/api/last")
def api_last():
    return jsonify(last_window_arrays(72))

@app.route("/api/predict")
def api_predict():
    latest_df = dfh.copy()
    Xy_tmp, feats = build_feature_view(latest_df)
    X_latest = Xy_tmp[feats].iloc[[-1]]
    y_hat_ug = float(model.predict(X_latest)[0])
    aqi_val, aqi_txt, aqi_cls = pm25_to_aqi(y_hat_ug)
    return jsonify({
        "next_hour_prediction_ugm3": round(y_hat_ug, 3),
        "next_hour_prediction_aqi": aqi_val,
        "aqi_category_text": aqi_txt,
        "aqi_category_class": aqi_cls,
        "last_timestamp": str(dfh.index.max())
    })

@app.route("/health")
def health():
    return jsonify(ok=True), 200

@app.route("/")
def home():
    html = """
<!doctype html>
<html lang="ar" dir="rtl">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>تنبؤ جودة الهواء في الرياض (PM2.5 & AQI)</title>
<style>
:root { --bg:#0b1117; --card:#111827; --muted:#9ca3af; --accent:#2563eb; --text:#e6edf3; }
body{font-family:system-ui,Segoe UI,Roboto,Arial; margin:24px; background:var(--bg); color:var(--text)}
.wrapper{max-width:1024px; margin:auto}
.card{background:var(--card); padding:20px; border-radius:16px; box-shadow:0 6px 24px rgba(0,0,0,.25); margin-bottom:16px}
h1{margin:0 0 12px; font-size:28px}
.tag{display:inline-block; padding:4px 10px; border-radius:999px; background:#1f2937; color:var(--muted); font-size:12px}
.row{display:flex; gap:16px; align-items:center; justify-content:space-between; flex-wrap:wrap}
.btn{background:var(--accent); color:#fff; padding:10px 14px; border:none; border-radius:10px; cursor:pointer}
.btn:disabled{opacity:.6; cursor:default}
select{background:#0f172a; color:#e2e8f0; border:1px solid #1f2937; border-radius:10px; padding:8px 10px}
canvas{width:100%; height:360px; background:#0b1220; border-radius:12px}
.small{color:#9ca3af; font-size:13px; line-height:1.6}
.kbd{background:#1f2937; padding:2px 6px; border-radius:6px; font-size:12px; color:#cbd5e1}
.footer{margin-top:8px; color:#9ca3af; font-size:12px; text-align:center}
.stat{display:inline-block; padding:6px 10px; border-radius:10px; margin-top:8px; font-weight:600}
.aqi-good{background:#052e11; color:#7aff8a}
.aqi-moderate{background:#2a2a05; color:#fff68a}
.aqi-usg{background:#3a2205; color:#ffce8a}
.aqi-unhealthy{background:#3a0b0b; color:#ff9c9c}
.aqi-very{background:#2a0623; color:#ff9bf0}
.aqi-hazardous{background:#2a0008; color:#ff99b2}
.section-title{margin:6px 0 8px; font-size:18px}
ul {margin-top:4px}
</style>
</head>
<body>
<div class="wrapper">
  <div class="card">
    <div class="row">
      <div>
        <h1>تنبؤ جودة الهواء في الرياض</h1>
        <div class="tag">تجريبي • تنبؤ ساعة-بساعة • وحدات: μg/m³ أو AQI</div>
      </div>
      <div class="row" style="gap:8px">
        <label class="small">العرض:</label>
        <select id="modeSel">
          <option value="ugm3">μg/m³ (PM2.5)</option>
          <option value="aqi">AQI (مؤشر جودة الهواء)</option>
        </select>
        <button id="predictBtn" class="btn">تنبؤ الساعة القادمة</button>
      </div>
    </div>

    <p class="small" style="margin:10px 0 0">
      <strong>الفكرة:</strong> نعالج قياسات <span class="kbd">PM2.5</span> التاريخية ونبني ميزات زمنية (lags والمتوسطات المتحركة)
      لتدريب نموذج يتنبأ بالساعة القادمة. ويمكن عرض النتيجة كوحدة μg/m³ أو كمؤشر <span class="kbd">AQI</span> للتبسيط.
    </p>
    <p class="small" style="margin:6px 0 0">
      <strong>قراءة الرسم:</strong> الخط يُظهر آخر 72 ساعة. اختر نمط العرض من القائمة (μg/m³ أو AQI).
      عند الضغط على زر <em>تنبؤ الساعة القادمة</em> سيظهر الرقم والتصنيف الصحي اللوني.
    </p>

    <div id="predBox"></div>
  </div>

  <div class="card">
    <h2 class="section-title">السلسلة الزمنية (آخر 72 ساعة)</h2>
    <canvas id="chart"></canvas>
    <div class="footer">© Abdulrahman Mahnashi — Built with Flask</div>
  </div>

  <div class="card">
    <h2 class="section-title">ملاحظات مهمة</h2>
    <ul class="small">
      <li>الـ AQI هنا تقريبي لأغراض العرض الفوري. المقياس الرسمي يستخدم تجميعات (مثل 24 ساعة/NowCast).</li>
      <li>التصنيف الصحي إرشادي مبسّط، وليس بديلاً عن المؤشرات الرسمية.</li>
    </ul>
  </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script>
const predBox = document.getElementById('predBox');
const btn = document.getElementById('predictBtn');
const sel = document.getElementById('modeSel');
let cache = { timestamps: [], pm25: [], aqi: [] };
let chart;

function aqiClass(aqi){
  if (aqi <= 50) return {cls:'aqi-good', text:'جيد'};
  if (aqi <= 100) return {cls:'aqi-moderate', text:'متوسط'};
  if (aqi <= 150) return {cls:'aqi-usg', text:'غير صحي للفئات الحساسة'};
  if (aqi <= 200) return {cls:'aqi-unhealthy', text:'غير صحي'};
  if (aqi <= 300) return {cls:'aqi-very', text:'غير صحي جدًا'};
  return {cls:'aqi-hazardous', text:'خطِر'};
}
function yTitle(){ return sel.value === 'aqi' ? 'AQI' : 'PM2.5 (μg/m³)'; }

async function loadLast(){
  const res = await fetch('/api/last');
  cache = await res.json();
  const ctx = document.getElementById('chart').getContext('2d');
  const dataSeries = sel.value === 'aqi' ? cache.aqi : cache.pm25;
  chart = new Chart(ctx, {
    type: 'line',
    data: { labels: cache.timestamps, datasets: [{ data: dataSeries, label: sel.value === 'aqi' ? 'AQI' : 'PM2.5', fill:false, tension:0.2, pointRadius:2 }] },
    options: { responsive:true, plugins: { legend: { labels: { usePointStyle:true } },
                 tooltip: { callbacks: { label: (c) => (sel.value==='aqi' ? 'AQI: ' : 'PM2.5: ') + c.parsed.y + (sel.value==='aqi' ? '' : ' μg/m³') } } },
               scales:{ x:{ display:false }, y:{ title:{ display:true, text:yTitle() } } } }
  });
}
function refreshChart(){
  if(!chart) return;
  const ser = sel.value === 'aqi' ? cache.aqi : cache.pm25;
  chart.data.datasets[0].data = ser;
  chart.data.datasets[0].label = sel.value === 'aqi' ? 'AQI' : 'PM2.5';
  chart.options.scales.y.title.text = yTitle();
  chart.update();
}
async function predict(){
  btn.disabled = true;
  try{
    const res = await fetch('/api/predict');
    const j = await res.json();
    const showAqi = sel.value === 'aqi';
    const value = showAqi ? j.next_hour_prediction_aqi : j.next_hour_prediction_ugm3;
    const cat   = aqiClass(j.next_hour_prediction_aqi);
    predBox.innerHTML = `
      <div class="stat ${cat.cls}">
        ${showAqi ? 'AQI' : 'PM2.5'} (الساعة القادمة):
        <span style="font-weight:800">${value}</span>${showAqi ? '' : ' μg/m³'} — ${cat.text}
      </div>
      <div class="small">آخر قراءة متاحة: ${j.last_timestamp}</div>
    `;
  }catch(e){
    predBox.innerHTML = '<div class="small">تعذر الحصول على التنبؤ.</div>';
  }finally{ btn.disabled = false; }
}
sel.addEventListener('change', refreshChart);
btn.addEventListener('click', predict);
loadLast();
</script>
</body>
</html>
"""
    return Response(html, mimetype="text/html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port)
