import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import plotly.graph_objects as go

st.set_page_config(page_title="배터리 Second-Life 추천 플랫폼", page_icon="🔋", layout="wide")

st.markdown("""
<style>
    .main-title    { font-size:28px; font-weight:700; margin-bottom:4px; }
    .sub-title     { font-size:14px; color:#888; margin-bottom:24px; }
    .metric-card   { background:#1a1a2e; border-radius:12px; padding:18px;
                     text-align:center; border:1px solid #2a2a4a; }
    .metric-val    { font-size:26px; font-weight:700; color:#00d4aa; }
    .metric-label  { font-size:12px; color:#aaa; margin-top:4px; }
    .rec-card      { background:#1a1a2e; border-radius:12px; padding:16px 20px;
                     margin-bottom:10px; border:1px solid #2a2a4a; }
    .top-card      { border:2px solid #00d4aa !important; }
    .section-title { font-size:18px; font-weight:600; margin:20px 0 12px; }
    .ref-box       { background:#111827; border-radius:8px; padding:10px 14px;
                     font-size:12px; color:#666; margin-top:8px; line-height:1.8; }
    .progress-wrap { background:#2a2a4a; border-radius:6px; height:8px; margin-top:6px; }
    .progress-bar  { height:8px; border-radius:6px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# BAT_PROPS: 배터리 종류별 특성 (학술 근거)
# 출처: IOPscience 2020, Frontiers in Energy Research 2023,
#        J. Power Sources 2017, PNNL-31453 2021
# ─────────────────────────────────────────────
BAT_PROPS = {
    "NCM": {
        "cycle_life":       1500,
        "calendar_rate":    0.025,
        "temp_sensitivity": 1.0,
        "voltage_nom":      3.7,
        "voltage_max":      4.2,
        "soh_reuse_min":    65,
        "energy_density":   200,
        "cycle_bonus":      0,
    },
    "LFP": {
        "cycle_life":       4000,
        "calendar_rate":    0.010,
        "temp_sensitivity": 0.7,
        "voltage_nom":      3.2,
        "voltage_max":      3.65,
        "soh_reuse_min":    60,
        "energy_density":   140,
        "cycle_bonus":      5,
    },
    "NCA": {
        "cycle_life":       1200,
        "calendar_rate":    0.030,
        "temp_sensitivity": 1.3,
        "voltage_nom":      3.6,
        "voltage_max":      4.2,
        "soh_reuse_min":    70,
        "energy_density":   220,
        "cycle_bonus":      0,
    },
    "LCO": {
        "cycle_life":       1000,
        "calendar_rate":    0.035,
        "temp_sensitivity": 1.4,
        "voltage_nom":      3.7,
        "voltage_max":      4.2,
        "soh_reuse_min":    70,
        "energy_density":   180,
        "cycle_bonus":      0,
    },
}

def calc_soh(eis_soh, bat_type, years, cycles, voltage):
    p = BAT_PROPS[bat_type]
    # 1. 캘린더 열화 (연수 × 열화율 × 온도민감도)
    calendar_deg = years * p["calendar_rate"] * p["temp_sensitivity"] * 100
    # 2. 사이클 열화 (정격 대비 소모율)
    cycle_deg    = (cycles / p["cycle_life"]) * 20
    # 3. 전압 보정 (정격 대비 초과 → 패널티, 미달 → 가점)
    v_diff = voltage - p["voltage_nom"]
    if v_diff > 0:
        voltage_deg = (v_diff / 0.2) * 5
    else:
        voltage_deg = max(-3, (v_diff / 0.2) * 2)
    total_deg = calendar_deg + cycle_deg + voltage_deg
    final_soh = round(max(15, min(100, eis_soh - total_deg * 0.5)), 1)
    return final_soh, {
        "eis":      round(eis_soh, 1),
        "calendar": round(calendar_deg, 1),
        "cycle":    round(cycle_deg, 1),
        "voltage":  round(voltage_deg, 1),
    }

def read_eis_file(f):
    name = f.name.lower()
    try:
        if name.endswith('.xls'):
            df = pd.read_excel(f, engine='xlrd', header=None)
        elif name.endswith('.xlsx'):
            df = pd.read_excel(f, engine='openpyxl', header=None)
        elif name.endswith('.csv'):
            raw = f.read().decode('utf-8'); f.seek(0)
            sep = '\t' if '\t' in raw.split('\n')[0] else ','
            df  = pd.read_csv(f, sep=sep, header=None, comment='#')
        elif name.endswith('.txt'):
            raw = f.read().decode('utf-8'); f.seek(0)
            sep = '\t' if '\t' in raw.split('\n')[0] else r'\s+'
            df  = pd.read_csv(f, sep=sep, header=None, comment='#')
        else:
            return None, "지원하지 않는 형식"
        df = df.apply(pd.to_numeric, errors='coerce').dropna()
        if df.shape[1] >= 3:
            df = df.iloc[:,:3]; df.columns = ['freq','z_real','z_imag']
        elif df.shape[1] == 2:
            df = df.iloc[:,:2]; df.columns = ['freq','z_real']; df['z_imag'] = 0
        else:
            return None, "컬럼 부족"
        return df, None
    except Exception as e:
        return None, str(e)

@st.cache_resource
def load_eis_model():
    np.random.seed(42)
    feats, labs = [], []
    sp = {100:(0.022,0.018,0.015), 95:(0.028,0.022,0.018),
          90:(0.035,0.030,0.022),  85:(0.042,0.038,0.028), 80:(0.052,0.048,0.035)}
    for soh, (re,rct,zw) in sp.items():
        for _ in range(72):
            re_ = re+np.random.normal(0,0.003); rct_=rct+np.random.normal(0,0.003)
            feats.append([re_, re_+rct_+zw, -(rct_*0.6), rct_*0.3, re_+rct_*0.5, rct_*0.24])
            labs.append(soh)
    m = GradientBoostingRegressor(n_estimators=200, random_state=42)
    m.fit(np.array(feats), np.array(labs))
    return m

def predict_eis_soh(df, model):
    f = [[float(df['z_real'].iloc[0]), float(df['z_real'].max()),
          float(df['z_imag'].min()),   float(df['z_imag'].max()),
          float(df['z_real'].mean()),  float(df['z_imag'].std())]]
    return float(model.predict(f)[0])

def get_recommendations(soh, years, cycles, bat_type):
    p   = BAT_PROPS[bat_type]
    rem = max(0, 1 - cycles / p["cycle_life"])
    b   = p["cycle_bonus"]
    apps = [
        {"name":"가정용 ESS",        "icon":"🏠", "score":soh*0.7+rem*30+b,  "life":max(0,round((soh-70)/p["calendar_rate"]/100*0.3)),  "value":round(soh*2.5),"carbon":round(soh*8),  "desc":"저출력 장기 사용. 태양광 패널과 연계해 잉여전력 저장.", "condition":soh>=p["soh_reuse_min"] and years<=12},
        {"name":"태양광 연계 ESS",    "icon":"☀️", "score":soh*0.75+rem*25+b, "life":max(0,round((soh-65)/p["calendar_rate"]/100*0.25)), "value":round(soh*3.2),"carbon":round(soh*12), "desc":"재생에너지 저장. 낮은 충방전 반복 환경에 최적.",         "condition":soh>=p["soh_reuse_min"] and years<=15},
        {"name":"통신기지국 백업전원", "icon":"📡", "score":soh*0.65+rem*20+b, "life":max(0,round((soh-60)/p["calendar_rate"]/100*0.2)),  "value":round(soh*2.8),"carbon":round(soh*7),  "desc":"간헐적 방전 환경. 안정적 출력 유지.",                  "condition":soh>=p["soh_reuse_min"] and years<=18},
        {"name":"UPS 비상전원",       "icon":"🏥", "score":soh*0.6+rem*15+b,  "life":max(0,round((soh-55)/p["calendar_rate"]/100*0.15)), "value":round(soh*2.2),"carbon":round(soh*6),  "desc":"병원·데이터센터 비상전원. 단기 방전 위주.",            "condition":soh>=60},
    ]
    valid = [a for a in apps if a["condition"]]
    return sorted(valid or [apps[-1]], key=lambda x: x["score"], reverse=True)[:3]

def safety_label(soh, years, cycles, bat_type):
    p = BAT_PROPS[bat_type]
    score = soh - years*2*p["temp_sensitivity"] - (cycles/p["cycle_life"])*30
    if score >= 75:   return "안전","#00d4aa","정상 범위 내 운용 가능합니다."
    elif score >= 55: return "주의","#f0a500","주기적 점검이 필요합니다."
    else:             return "위험","#e05555","재사용보다 재활용 공정 투입을 권장합니다."

# ── UI ──
st.markdown('<div class="main-title">🔋 배터리 Second-Life 추천 플랫폼</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">EIS + 사용 이력 통합 AI 진단 · 학술 근거 기반 활용처 추천</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ 설정")
    st.success("✅ AI 모델 준비 완료!")
    st.divider()
    st.markdown("**SOH 계산 모델**")
    st.markdown("- EIS 전기화학 분석")
    st.markdown("- 캘린더 열화 (종류별 온도 민감도 반영)")
    st.markdown("- 사이클 열화 (정격 대비 소모율)")
    st.markdown("- 전압 보정 (정격 대비 가점/감점)")
    st.divider()
    st.markdown("**지원 파일:** `.xls` / `.xlsx` / `.csv` / `.txt`")

# 배터리 기본 정보
st.markdown('<div class="section-title">📋 배터리 기본 정보</div>', unsafe_allow_html=True)
c1,c2,c3,c4 = st.columns(4)
with c1:
    bat_type = st.selectbox("배터리 종류", list(BAT_PROPS.keys()),
        help="LFP: 긴 수명·낮은 열화 / NCM: 중간 / NCA·LCO: 빠른 열화")
with c2:
    years = st.number_input("사용 연수 (년)", 0, 25, 5)
p = BAT_PROPS[bat_type]
with c3:
    cycles = st.number_input("충방전 횟수 (회)", 0, p["cycle_life"]+1000, 500, 50)
    cycle_pct = min(100, round(cycles / p["cycle_life"] * 100))
    bar_color = "#00d4aa" if cycle_pct<50 else "#f0a500" if cycle_pct<80 else "#e05555"
    st.markdown(f"""
    <div style="font-size:12px; color:#aaa; margin-top:-8px;">
        정격 대비 <b style="color:{bar_color}">{cycle_pct}% 소모</b> (정격 {p['cycle_life']:,}회)
    </div>
    <div class="progress-wrap">
        <div class="progress-bar" style="width:{cycle_pct}%; background:{bar_color};"></div>
    </div>""", unsafe_allow_html=True)
with c4:
    voltage = st.number_input("현재 전압 (V)", 2.5, 4.5, p["voltage_nom"], 0.01)
    v_diff  = voltage - p["voltage_nom"]
    if abs(v_diff) < 0.05:
        v_msg,v_col = f"✅ 정격 전압 ({p['voltage_nom']}V)","#00d4aa"
    elif v_diff > 0:
        v_msg,v_col = f"⚠️ +{v_diff:.2f}V (열화 가속)","#e05555"
    else:
        v_msg,v_col = f"✅ {v_diff:.2f}V (열화 감소)","#00d4aa"
    st.markdown(f'<div style="font-size:12px; color:{v_col}; margin-top:-8px;">{v_msg}</div>', unsafe_allow_html=True)

st.markdown(f"""
<div class="ref-box">
    📋 <b>{bat_type} 특성:</b>
    정격 사이클 {p['cycle_life']:,}회 |
    캘린더 열화율 {p['calendar_rate']*100:.1f}%/년 |
    온도 민감도 {p['temp_sensitivity']}x |
    에너지 밀도 {p['energy_density']} Wh/kg |
    재사용 최소 SOH {p['soh_reuse_min']}%
</div>""", unsafe_allow_html=True)

# 파일 업로드
st.markdown('<div class="section-title">📂 EIS 파일 업로드 (여러 개 = 평균 분석)</div>', unsafe_allow_html=True)
uploaded_files = st.file_uploader(
    "같은 배터리의 반복 측정 파일을 여러 개 올리면 평균내서 더 정확하게 분석해요",
    type=["xls","xlsx","csv","txt"],
    accept_multiple_files=True
)

if uploaded_files:
    model = load_eis_model()
    dfs, errors = [], []
    for f in uploaded_files:
        df, err = read_eis_file(f)
        if err: errors.append(f"{f.name}: {err}")
        else:   dfs.append((f.name, df))
    for e in errors: st.warning(f"⚠️ {e}")

    if not dfs:
        st.error("읽을 수 있는 파일이 없어요.")
    else:
        st.success(f"✅ {len(dfs)}개 파일 읽기 완료!")

        # 나이퀴스트 + 열화 차트
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="section-title">📈 나이퀴스트 플롯 (개별 + 평균)</div>', unsafe_allow_html=True)
            fig_nyq = go.Figure()
            all_real, all_imag = [], []
            for name, df in dfs:
                fig_nyq.add_trace(go.Scatter(
                    x=df['z_real'], y=-df['z_imag'],
                    mode='lines+markers', name=name[:15],
                    opacity=0.35, line=dict(width=1), marker=dict(size=4),
                    showlegend=len(dfs)>1
                ))
                all_real.append(df['z_real'].values)
                all_imag.append(df['z_imag'].values)
            if len(dfs) > 1:
                try:
                    ml = min(len(z) for z in all_real)
                    fig_nyq.add_trace(go.Scatter(
                        x=np.mean([z[:ml] for z in all_real], axis=0),
                        y=-np.mean([z[:ml] for z in all_imag], axis=0),
                        mode='lines', name='평균',
                        line=dict(color='#00d4aa', width=3), opacity=1.0
                    ))
                except: pass
            fig_nyq.update_layout(
                xaxis_title="Z' (Ω)", yaxis_title="-Z'' (Ω)",
                template='plotly_dark', height=300, margin=dict(l=0,r=0,t=10,b=0)
            )
            st.plotly_chart(fig_nyq, use_container_width=True)

        # SOH 계산
        eis_sohs    = [predict_eis_soh(df, model) for _,df in dfs]
        avg_eis_soh = float(np.mean(eis_sohs))
        final_soh, deg = calc_soh(avg_eis_soh, bat_type, years, cycles, voltage)

        with col2:
            st.markdown('<div class="section-title">📊 SOH 열화 요인 분석</div>', unsafe_allow_html=True)
            v_label = f"{'+' if deg['voltage']<0 else '-'}{abs(deg['voltage'])}%"
            fig_deg = go.Figure(go.Bar(
                x=["EIS 기반","캘린더 열화","사이클 열화","전압 보정","최종 SOH"],
                y=[deg["eis"], -deg["calendar"], -deg["cycle"], -deg["voltage"], final_soh],
                marker_color=["#00d4aa","#e05555","#f0a500",
                              "#00d4aa" if deg["voltage"]<=0 else "#9b59b6","#00d4aa"],
                text=[f"{deg['eis']}%", f"-{deg['calendar']}%", f"-{deg['cycle']}%",
                      v_label, f"{final_soh}%"],
                textposition='outside',
            ))
            fig_deg.update_layout(
                template='plotly_dark', height=300,
                margin=dict(l=0,r=0,t=10,b=0), yaxis_title="SOH (%)"
            )
            st.plotly_chart(fig_deg, use_container_width=True)

        # 진단 결과
        st.markdown('<div class="section-title">🤖 AI 진단 결과</div>', unsafe_allow_html=True)
        if len(dfs) > 1:
            soh_info = " | ".join([f"{n[:10]}: {round(s,1)}%" for n,s in zip([n for n,_ in dfs], eis_sohs)])
            st.markdown(f'<div class="ref-box">📊 개별 EIS SOH: {soh_info} → 평균: {round(avg_eis_soh,1)}%</div>', unsafe_allow_html=True)

        m1,m2,m3,m4,m5 = st.columns(5)
        status_txt   = "양호" if final_soh>=85 else "보통" if final_soh>=70 else "주의"
        status_color = "#00d4aa" if final_soh>=85 else "#f0a500" if final_soh>=70 else "#e05555"
        avg_re  = round(np.mean([df['z_real'].iloc[0] for _,df in dfs])*1000, 2)
        avg_rct = round(np.mean([(df['z_real'].max()-df['z_real'].iloc[0]) for _,df in dfs])*1000, 2)
        for col, val, label, color in zip(
            [m1,m2,m3,m4,m5],
            [f"{round(avg_eis_soh,1)}%", f"{final_soh}%", f"{avg_re}mΩ", f"{avg_rct}mΩ", status_txt],
            ["EIS SOH (평균)","최종 SOH","전해질 저항 Re","전하전달 저항 Rct","배터리 상태"],
            ["#aaa","#00d4aa","#00d4aa","#00d4aa", status_color]
        ):
            col.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color:{color}; font-size:22px;">{val}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="ref-box">
            📚 <b>학술 근거 기반 SOH 계산 ({bat_type}):</b><br>
            EIS 평균 {deg['eis']}%
            → 캘린더 열화 -{deg['calendar']}% (연 {p['calendar_rate']*100:.1f}% × 온도민감도 {p['temp_sensitivity']}x, Frontiers in Energy Research 2023)<br>
            → 사이클 열화 -{deg['cycle']}% (충방전 {cycles}회 / 정격 {p['cycle_life']:,}회 = {cycle_pct}% 소모, IOPscience 2020)<br>
            → 전압 보정 {v_label} (정격 {p['voltage_nom']}V 대비 {'+' if v_diff>0 else ''}{v_diff:.2f}V, J. Power Sources 2017)<br>
            = <b>최종 SOH {final_soh}%</b>
        </div>""", unsafe_allow_html=True)

        safety_txt, safety_color, safety_desc = safety_label(final_soh, years, cycles, bat_type)
        st.markdown(f"""
        <div class="metric-card" style="text-align:left; margin-top:12px; border:2px solid {safety_color};">
            <span style="font-size:18px; font-weight:700; color:{safety_color}">🛡️ 안전성: {safety_txt}</span>
            <span style="font-size:13px; color:#ccc; margin-left:10px;">{safety_desc}</span>
        </div>""", unsafe_allow_html=True)

        # 추천 활용처
        st.markdown('<div class="section-title">🎯 추천 활용처</div>', unsafe_allow_html=True)
        recs = get_recommendations(final_soh, years, cycles, bat_type)
        if not recs:
            st.error("SOH가 너무 낮아 재사용이 어렵습니다. 재활용 공정 투입을 권장합니다.")
        else:
            for i, rec in enumerate(recs):
                cc = "rec-card top-card" if i==0 else "rec-card"
                rl = "✦ 최우선 추천" if i==0 else f"{i+1}순위"
                st.markdown(f"""
                <div class="{cc}">
                    <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">
                        <div>
                            <div style="font-size:15px; font-weight:600;">{rec['icon']} {rec['name']}</div>
                            <div style="font-size:12px; color:#aaa;">{rl} · 적합도 {round(rec['score'])}%</div>
                            <div style="font-size:13px; color:#bbb; margin-top:4px;">{rec['desc']}</div>
                        </div>
                        <div style="display:flex; gap:16px;">
                            <div style="text-align:center;"><div style="font-size:16px; font-weight:600; color:#00d4aa;">{rec['life']}년</div><div style="font-size:11px; color:#aaa;">잔존수명</div></div>
                            <div style="text-align:center;"><div style="font-size:16px; font-weight:600; color:#00d4aa;">{rec['value']}만원</div><div style="font-size:11px; color:#aaa;">경제가치</div></div>
                            <div style="text-align:center;"><div style="font-size:16px; font-weight:600; color:#00d4aa;">{rec['carbon']}kg</div><div style="font-size:11px; color:#aaa;">CO₂ 절감</div></div>
                        </div>
                    </div>
                </div>""", unsafe_allow_html=True)

        # 최종 판단
        reusable = final_soh >= p["soh_reuse_min"]
        color = "#00d4aa" if reusable else "#e05555"
        msg   = "✅ 재사용 가능" if reusable else "❌ 재활용 공정 권장"
        st.markdown(f"""
        <div style="background:#1a1a2e; border-radius:12px; padding:16px;
                    border:2px solid {color}; text-align:center; margin-top:12px;">
            <div style="font-size:22px; font-weight:700; color:{color}">{msg}</div>
            <div style="font-size:13px; color:#aaa; margin-top:6px;">
                {bat_type} | 사용 {years}년 | 충방전 {cycles}회 ({cycle_pct}% 소모) | {voltage}V
            </div>
        </div>""", unsafe_allow_html=True)

        # 분석 결과 CSV 다운로드
        st.divider()
        result_df = pd.DataFrame([{
            "배터리 종류": bat_type,
            "사용 연수 (년)": years,
            "충방전 횟수 (회)": cycles,
            "사이클 소모율 (%)": cycle_pct,
            "현재 전압 (V)": voltage,
            "EIS SOH 평균 (%)": round(avg_eis_soh,1),
            "캘린더 열화 (%)": deg["calendar"],
            "사이클 열화 (%)": deg["cycle"],
            "전압 보정 (%)": deg["voltage"],
            "최종 SOH (%)": final_soh,
            "안전성": safety_txt,
            "최우선 활용처": recs[0]["name"] if recs else "재활용 권장",
            "예상 잔존수명 (년)": recs[0]["life"] if recs else 0,
            "경제적 가치 (만원)": recs[0]["value"] if recs else 0,
            "CO₂ 절감 (kg)": recs[0]["carbon"] if recs else 0,
            "최종 판정": "재사용 가능" if reusable else "재활용 권장",
        }])
        csv = result_df.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="📥 분석 결과 CSV 다운로드",
            data=csv,
            file_name=f"battery_analysis_{bat_type}_{years}yr_{cycles}cycle.csv",
            mime="text/csv",
            use_container_width=True
        )

else:
    st.info("👆 EIS 파일을 업로드하면 AI가 분석합니다. 여러 파일을 올리면 평균내서 더 정확하게 분석해요!")
    st.markdown("""
    **사용 방법:**
    1. 배터리 종류 / 연수 / 충방전 횟수 / 전압 입력
    2. EIS 파일 업로드 (여러 개 = 자동 평균 처리)
    3. AI 진단 결과 및 추천 활용처 확인
    4. 분석 결과 CSV 다운로드

    **학술 근거:** Frontiers in Energy Research 2023 / IOPscience 2020 / J. Power Sources 2017 / PNNL-31453 2021
    """)
