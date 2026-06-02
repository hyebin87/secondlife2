import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import os
 
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="배터리 Second-Life 추천 플랫폼",
    page_icon="🔋",
    layout="wide"
)
 
st.markdown("""
<style>
    .main-title  { font-size:28px; font-weight:700; margin-bottom:4px; }
    .sub-title   { font-size:14px; color:#888; margin-bottom:24px; }
    .metric-card { background:#1a1a2e; border-radius:12px; padding:20px;
                   text-align:center; border:1px solid #2a2a4a; }
    .metric-val  { font-size:28px; font-weight:700; color:#00d4aa; }
    .metric-label{ font-size:12px; color:#aaa; margin-top:4px; }
    .rec-card    { background:#1a1a2e; border-radius:12px; padding:16px 20px;
                   margin-bottom:10px; border:1px solid #2a2a4a; }
    .top-card    { border:2px solid #00d4aa !important; }
    .section-title { font-size:18px; font-weight:600; margin:20px 0 12px; }
    .ref-text    { font-size:11px; color:#666; margin-top:4px; }
    .ref-box     { background:#111827; border-radius:8px; padding:10px 14px;
                   font-size:12px; color:#666; margin-top:8px; line-height:1.8; }
</style>
""", unsafe_allow_html=True)
 
# ─────────────────────────────────────────────
# 학술 근거 기반 배터리 특성값
# ─────────────────────────────────────────────
BAT_PROPS = {
    "NCM": dict(soh_reuse=80, soh_recycle=50, cycle_life=2000, nominal_v=3.6),
    "LFP": dict(soh_reuse=80, soh_recycle=50, cycle_life=4000, nominal_v=3.2),
    "NCA": dict(soh_reuse=80, soh_recycle=50, cycle_life=1500, nominal_v=3.6),
    "LCO": dict(soh_reuse=80, soh_recycle=50, cycle_life=800,  nominal_v=3.7),
}
 
# ─────────────────────────────────────────────
# ML 모델 로드 (eis_model.pkl)
# ─────────────────────────────────────────────
@st.cache_resource
def load_ml_model():
    model_path = os.path.join(os.path.dirname(__file__), 'eis_model.pkl')
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None
 
ml_model = load_ml_model()
 
# ─────────────────────────────────────────────
# EIS 특징 추출
# ─────────────────────────────────────────────
def extract_features(df):
    return [
        float(df['z_real'].iloc[0]),
        float(df['z_real'].max()),
        float(df['z_imag'].min()),
        float(df['z_imag'].max()),
        float(df['z_real'].mean()),
        float(df['z_imag'].std()),
    ]
 
def extract_eis_indicators(df):
    df_sorted = df.sort_values('freq', ascending=False).reset_index(drop=True)
    re    = float(df_sorted['z_real'].iloc[0])
    rct   = float(df_sorted['z_real'].max() - re)
    z_low = float(np.sqrt(df_sorted['z_real'].iloc[-1]**2 + df_sorted['z_imag'].iloc[-1]**2))
    return re, rct, z_low
 
# ─────────────────────────────────────────────
# SOH 예측: ML 모델 + 임피던스 비율 병합
# ─────────────────────────────────────────────
def predict_soh_combined(df, bat_type):
    """
    두 가지 방법의 평균으로 SOH 예측:
    1. ML 모델 (DIB 데이터셋 학습, MAE 2.47%)
    2. 임피던스 비율 계산 (Niri et al. 2022, doi:10.1016/j.est.2022.106295)
    """
    # 방법 1: ML 모델 예측
    ml_soh = None
    if ml_model is not None:
        feats  = np.array([extract_features(df)])
        ml_soh = float(ml_model.predict(feats)[0])
        ml_soh = round(np.clip(ml_soh, 0, 100), 1)
 
    # 방법 2: 임피던스 비율 계산
    re, rct, z_low = extract_eis_indicators(df)
    baseline = {
        "NCM": dict(re=0.022, rct=0.018),
        "LFP": dict(re=0.020, rct=0.015),
        "NCA": dict(re=0.020, rct=0.016),
        "LCO": dict(re=0.025, rct=0.022),
    }
    b = baseline[bat_type]
    re_ratio  = re  / b['re']
    rct_ratio = rct / b['rct']
    eis_score = 100 - (re_ratio - 1) * 40 - (rct_ratio - 1) * 60
    eis_score = round(float(np.clip(eis_score, 0, 100)), 1)
 
    # 병합: ML 있으면 평균, 없으면 임피던스 방식만
    if ml_soh is not None:
        combined = round((ml_soh + eis_score) / 2, 1)
        method   = f"ML 예측 {ml_soh}% + 임피던스 비율 {eis_score}% → 평균 {combined}%"
    else:
        combined = eis_score
        method   = f"임피던스 비율 계산 {eis_score}% (ML 모델 미로드)"
 
    return combined, ml_soh, eis_score, method, re, rct, z_low
 
# ─────────────────────────────────────────────
# 활용처 추천
# ─────────────────────────────────────────────
def get_recommendations(health, years, cycles, bat_type, voltage):
    props = BAT_PROPS[bat_type]
    cycle_ratio   = cycles / props['cycle_life']
    cycle_penalty = cycle_ratio * 20
    age_penalty   = years * 2
    v_diff        = abs(voltage - props['nominal_v'])
    v_penalty     = v_diff * 10 if v_diff > 0.3 else 0
    base = health - cycle_penalty - age_penalty - v_penalty
 
    apps = [
        {
            "name": "태양광 연계 ESS", "icon": "☀️",
            "desc": "재생에너지 저장. 낮은 C-rate, 1일 1~2회 충방전 환경에 적합.",
            "ref": "Edge et al. (2023); IEC 62933",
            "score": max(0, base + 5),
            "condition": health >= 70,
        },
        {
            "name": "가정용 ESS", "icon": "🏠",
            "desc": "저출력 장기 사용. 태양광 패널 연계 잉여전력 저장.",
            "ref": "Edge et al. (2023); UL 1974",
            "score": max(0, base),
            "condition": health >= 70,
        },
        {
            "name": "통신기지국 백업전원", "icon": "📡",
            "desc": "간헐적 방전. 부동충전 위주 환경으로 배터리 부담 낮음.",
            "ref": "Martinez-Laserna et al. (2018), Appl. Energy",
            "score": max(0, base - 5),
            "condition": health >= 60,
        },
        {
            "name": "UPS 비상전원", "icon": "🏥",
            "desc": "단기 방전 위주. 충방전 빈도 낮아 열화 부담 적음.",
            "ref": "Edge et al. (2023) mid-range 기준",
            "score": max(0, base - 10),
            "condition": health >= 50,
        },
    ]
 
    if bat_type == "LFP":
        for a in apps:
            a['score'] = min(100, a['score'] + 5)
 
    valid = [a for a in apps if a["condition"]]
    return sorted(valid, key=lambda x: x["score"], reverse=True)[:3]
 
def safety_eval(health, years, cycles, bat_type, voltage):
    props = BAT_PROPS[bat_type]
    cycle_ratio = cycles / props['cycle_life']
    v_diff = abs(voltage - props['nominal_v'])
 
    if voltage < 2.5:
        return "위험", "#e05555", "전압 2.5V 미만 — 안전 기준 미달 (EU 배터리 규정)"
 
    if health >= 80 and cycle_ratio < 0.8 and v_diff <= 0.5:
        return "안전", "#00d4aa", "정상 범위 — 재사용 적합 (IEC 62933 기준 충족)"
    elif health >= 50:
        reason = []
        if health < 80:  reason.append(f"건강도 {round(health)}%")
        if cycle_ratio >= 0.8: reason.append(f"사이클 {round(cycle_ratio*100)}% 소모")
        if v_diff > 0.5: reason.append(f"전압 편차 {round(v_diff,2)}V")
        return "주의", "#f0a500", " · ".join(reason) + " — 점검 필요"
    else:
        return "위험", "#e05555", f"건강도 {round(health)}% — SOH 50% 미만, 해체/재활용 필수"
 
# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
st.markdown('<div class="main-title">🔋 배터리 Second-Life 추천 플랫폼</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">EIS 임피던스 기반 AI 진단 · 학술 근거 기반 활용처 추천</div>', unsafe_allow_html=True)
 
# 모델 로드 상태 표시
with st.sidebar:
    st.header("⚙️ 설정")
    if ml_model is not None:
        st.success("✅ ML 모델 로드 완료 (DIB 데이터셋 기반)")
    else:
        st.warning("⚠️ ML 모델 없음 — 임피던스 비율 방식으로만 분석")
    st.divider()
    st.markdown("**SOH 예측 방식**")
    st.markdown("- ML 모델 (MAE 2.47%)")
    st.markdown("- 임피던스 비율 계산")
    st.markdown("- 두 값의 **평균** 사용")
    st.divider()
    st.markdown("**학술 근거**")
    st.markdown("- Niri et al. (2022)")
    st.markdown("- Edge et al. (2023)")
    st.markdown("- IEC 62933 / UL 1974")
    st.markdown("- Warwick DIB Dataset")
 
# 배터리 기본 정보
st.markdown('<div class="section-title">📋 배터리 기본 정보 입력</div>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
with c1:
    bat_type = st.selectbox("배터리 종류", ["NCM", "LFP", "NCA", "LCO"])
    props = BAT_PROPS[bat_type]
    st.caption(f"설계 사이클: {props['cycle_life']}회 | 정격전압: {props['nominal_v']}V")
with c2:
    years = st.number_input("사용 연수 (년)", min_value=0, max_value=20, value=5)
with c3:
    cycles = st.number_input("충방전 횟수 (회)", min_value=0, max_value=10000, value=500, step=50)
    cycle_ratio = cycles / props['cycle_life']
    if cycle_ratio >= 1.0:   st.caption("⚠️ 설계 사이클 초과")
    elif cycle_ratio >= 0.8: st.caption(f"🟡 사이클 {round(cycle_ratio*100)}% 소모")
    else:                    st.caption(f"🟢 사이클 {round(cycle_ratio*100)}% 소모")
with c4:
    voltage = st.number_input("현재 전압 (V)", min_value=2.0, max_value=4.5,
                               value=props['nominal_v'], step=0.01)
    v_diff = abs(voltage - props['nominal_v'])
    if voltage < 2.5:       st.caption("🔴 2.5V 미만 — 안전 위험")
    elif v_diff > 0.5:      st.caption(f"⚠️ 정격 대비 {v_diff:+.2f}V")
    else:                   st.caption(f"🟢 정격 전압 정상")
 
# SOH 입력 옵션
st.markdown('<div class="section-title">🔢 SOH 정보</div>', unsafe_allow_html=True)
soh_mode = st.radio(
    "SOH 입력 방식",
    ["SOH 모름 — EIS + ML로 자동 예측", "SOH 직접 입력 (용량 측정값 보유 시)"],
    index=0, horizontal=True
)
soh_input = None
if soh_mode == "SOH 직접 입력 (용량 측정값 보유 시)":
    soh_input = st.number_input("SOH (%)", min_value=0, max_value=100, value=80, step=1)
    st.caption("📌 SOH 측정 기준: IEC 62660-1 (용량 측정법)")
 
# EIS 파일 업로드
st.markdown('<div class="section-title">📂 EIS 파일 업로드</div>', unsafe_allow_html=True)
uploaded_files = st.file_uploader(
    "EIS 측정 파일 (.xls, .csv) — 여러 개 동시 업로드 시 자동 평균",
    type=["xls", "csv"],
    accept_multiple_files=True,
)
 
if uploaded_files:
    dfs = []
    for f in uploaded_files:
        try:
            if f.name.endswith('.csv'):
                tmp = pd.read_csv(f, header=None)
            else:
                tmp = pd.read_excel(f, engine='xlrd', header=None)
            tmp = tmp.apply(pd.to_numeric, errors='coerce').dropna()
            tmp.columns = ['freq', 'z_real', 'z_imag'] if tmp.shape[1] >= 3 else tmp.columns
            dfs.append(tmp)
        except Exception as e:
            st.warning(f"⚠️ {f.name} 읽기 실패: {e}")
 
    if not dfs:
        st.error("읽을 수 있는 파일이 없습니다.")
        st.stop()
 
    # 평균 처리
    if len(dfs) == 1:
        df = dfs[0].copy()
        df.columns = ['freq', 'z_real', 'z_imag']
    else:
        df = dfs[0].copy()
        df.columns = ['freq', 'z_real', 'z_imag']
        df['z_real'] = pd.concat([d.iloc[:,1] for d in dfs], axis=1).mean(axis=1)
        df['z_imag'] = pd.concat([d.iloc[:,2] for d in dfs], axis=1).mean(axis=1)
        st.caption(f"📊 {len(dfs)}개 반복 측정 평균값으로 분석합니다.")
 
    # 시각화
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-title">📈 나이퀴스트 플롯</div>', unsafe_allow_html=True)
        fig = go.Figure()
        if len(dfs) > 1:
            for d in dfs:
                d.columns = ['freq','z_real','z_imag']
                fig.add_trace(go.Scatter(
                    x=d['z_real'], y=-d['z_imag'], mode='lines',
                    line=dict(color='rgba(0,212,170,0.2)', width=1), showlegend=False
                ))
        fig.add_trace(go.Scatter(
            x=df['z_real'], y=-df['z_imag'], mode='lines+markers',
            name='평균' if len(dfs) > 1 else '측정값',
            marker=dict(color=np.log10(df['freq']+0.001), colorscale='Plasma', size=7,
                        colorbar=dict(title="log₁₀(Hz)", thickness=12)),
            line=dict(color='rgba(255,255,255,0.8)', width=2),
        ))
        fig.update_layout(
            xaxis_title="Z' (실수부, Ω)", yaxis_title="-Z'' (허수부, Ω)",
            template='plotly_dark', height=320, margin=dict(l=0,r=0,t=10,b=0)
        )
        st.plotly_chart(fig, use_container_width=True)
 
    with col2:
        st.markdown('<div class="section-title">📊 임피던스 크기</div>', unsafe_allow_html=True)
        z_mag = np.sqrt(df['z_real']**2 + df['z_imag']**2) * 1000
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=df['freq'], y=z_mag, mode='lines+markers',
            line=dict(color='#00d4aa', width=2), marker=dict(size=5)
        ))
        fig2.update_layout(
            xaxis_title="주파수 (Hz)", xaxis_type="log",
            yaxis_title="|Z| (mΩ)",
            template='plotly_dark', height=320, margin=dict(l=0,r=0,t=10,b=0)
        )
        st.plotly_chart(fig2, use_container_width=True)
 
    # SOH 예측
    st.divider()
    combined_soh, ml_soh, eis_score, method, re, rct, z_low = predict_soh_combined(df, bat_type)
 
    # 최종 건강도: SOH 직접 입력 > 자동 예측
    health = soh_input if soh_input is not None else combined_soh
    health_label = "입력 SOH" if soh_input is not None else "예측 SOH"
 
    st.markdown('<div class="section-title">🔬 EIS 임피던스 진단</div>', unsafe_allow_html=True)
 
    m1, m2, m3, m4, m5 = st.columns(5)
    re_color     = "#00d4aa" if re  < 0.035 else "#f0a500" if re  < 0.055 else "#e05555"
    rct_color    = "#00d4aa" if rct < 0.030 else "#f0a500" if rct < 0.050 else "#e05555"
    health_color = "#00d4aa" if health >= 80 else "#f0a500" if health >= 50 else "#e05555"
 
    for col, val, label, color, ref in zip(
        [m1, m2, m3, m4, m5],
        [f"{round(re*1000,2)}mΩ", f"{round(rct*1000,2)}mΩ",
         f"{round(z_low*1000,2)}mΩ", f"{round(health,1)}%", f"{voltage}V"],
        ["전해질 저항 (Re)", "전하전달 저항 (Rct)", "저주파 임피던스", health_label, "현재 전압"],
        [re_color, rct_color, "#00d4aa", health_color, "#00d4aa"],
        ["고주파 실수부", "반원 크기", "확산 저항 반영", "ML+임피던스 평균", "측정값"]
    ):
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-val" style="color:{color}">{val}</div>
            <div class="metric-label">{label}</div>
            <div class="ref-text">{ref}</div>
        </div>""", unsafe_allow_html=True)
 
    # SOH 예측 방법 설명
    if soh_input is None:
        st.markdown(f"""
        <div class="ref-box">
            📚 <b>SOH 예측 방법:</b> {method}<br>
            - ML 모델: Warwick DIB Dataset 360개 학습 (MAE 2.47%, R² 0.89)<br>
            - 임피던스 비율: Niri et al. (2022), doi:10.1016/j.est.2022.106295
        </div>""", unsafe_allow_html=True)
 
    # 안전성 평가
    st.markdown('<div class="section-title">🛡️ 안전성 평가</div>', unsafe_allow_html=True)
    safety_txt, safety_color, safety_desc = safety_eval(health, years, cycles, bat_type, voltage)
    st.markdown(f"""
    <div class="metric-card" style="text-align:left; border:2px solid {safety_color};">
        <span style="font-size:20px; font-weight:700; color:{safety_color}">{safety_txt}</span>
        <span style="font-size:14px; color:#ccc; margin-left:12px;">{safety_desc}</span>
    </div>""", unsafe_allow_html=True)
 
    # 추천 활용처
    st.markdown('<div class="section-title">🎯 추천 활용처</div>', unsafe_allow_html=True)
    st.caption("📌 활용처별 SOH 기준 — Edge et al. (2023); Martinez-Laserna et al. (2018); IEC 62933")
 
    recs = get_recommendations(health, years, cycles, bat_type, voltage)
    if not recs:
        st.error("❌ 모든 활용처 기준 미달 — 재활용 공정 투입 권장 (SOH 50% 미만)")
    else:
        for i, rec in enumerate(recs):
            card_class = "rec-card top-card" if i == 0 else "rec-card"
            rank_label = "✦ 최우선 추천" if i == 0 else f"{i+1}순위 추천"
            st.markdown(f"""
            <div class="{card_class}">
                <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px;">
                    <div>
                        <div style="font-size:16px; font-weight:600;">{rec['icon']} {rec['name']}</div>
                        <div style="font-size:12px; color:#aaa;">{rank_label} · 적합도 {round(rec['score'])}점</div>
                        <div style="font-size:13px; color:#bbb; margin-top:6px;">{rec['desc']}</div>
                        <div style="font-size:11px; color:#666; margin-top:4px;">📚 {rec['ref']}</div>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)
 
    # 최종 판단
    st.divider()
    if safety_txt == "위험":
        final_color, final_msg, final_ref = "#e05555", "❌ 재사용 불가 — 재활용 공정 필요", "Edge et al. (2023); IEC 62619"
    elif safety_txt == "주의":
        final_color, final_msg, final_ref = "#f0a500", "⚠️ 조건부 재사용 가능 — 주기적 점검 필요", "Edge et al. (2023)"
    else:
        final_color, final_msg, final_ref = "#00d4aa", "✅ 재사용 가능", "IEC 62933, UL 1974"
 
    st.markdown(f"""
    <div style="background:#1a1a2e; border-radius:12px; padding:20px;
                border:2px solid {final_color}; text-align:center;">
        <div style="font-size:24px; font-weight:700; color:{final_color}">{final_msg}</div>
        <div style="font-size:13px; color:#aaa; margin-top:8px;">
            {bat_type} | 사용 {years}년 | 충방전 {cycles}회 ({round(cycle_ratio*100)}% 소모) | {voltage}V
        </div>
        <div style="font-size:11px; color:#666; margin-top:6px;">📚 근거: {final_ref}</div>
    </div>""", unsafe_allow_html=True)
 
    # 분석 결과 CSV 다운로드
    st.divider()
    result_df = pd.DataFrame([{
        "배터리 종류": bat_type,
        "사용 연수 (년)": years,
        "충방전 횟수 (회)": cycles,
        "현재 전압 (V)": voltage,
        "ML 예측 SOH (%)": ml_soh if ml_soh else "N/A",
        "임피던스 건강도 (%)": eis_score,
        "최종 SOH (%)": round(health, 1),
        "전해질 저항 Re (mΩ)": round(re*1000, 2),
        "전하전달 저항 Rct (mΩ)": round(rct*1000, 2),
        "안전성": safety_txt,
        "최우선 활용처": recs[0]["name"] if recs else "재활용 권장",
        "최종 판정": final_msg,
    }])
    csv = result_df.to_csv(index=False, encoding='utf-8-sig')
    st.download_button(
        label="📥 분석 결과 CSV 다운로드",
        data=csv,
        file_name=f"battery_analysis_{bat_type}.csv",
        mime="text/csv",
        use_container_width=True
    )
 
else:
    st.info("👆 EIS 파일(.xls 또는 .csv)을 업로드하면 분석이 시작됩니다.")
    st.markdown("""
    **사용 방법:**
    1. 배터리 기본 정보 입력 (종류, 연수, 충방전 횟수, 전압)
    2. SOH 입력 방식 선택 (모르면 EIS + ML로 자동 예측)
    3. EIS 파일 업로드 (여러 개 동시 업로드 → 자동 평균)
    4. 임피던스 진단 결과 및 추천 활용처 확인
    5. 분석 결과 CSV 다운로드
    
    **SOH 예측 방식:**
    - ML 모델 (Warwick DIB Dataset 360개 학습, MAE 2.47%)
    - 임피던스 비율 계산 (Niri et al. 2022)
    - 두 값의 평균으로 최종 SOH 산출
    """)
