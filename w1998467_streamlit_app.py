import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="MI Clinical Dashboard", layout="wide")

#Loading data
@st.cache_data
def load_data():
    df = pd.read_csv("MI_finaldf.csv")
    #ensuring datetime columns
    for c in ["admittime", "dischtime"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    #heatmap helper columns
    if "admittime" in df.columns:
        df["admit_hour"] = df["admittime"].dt.hour
        df["admit_dow"] = df["admittime"].dt.day_name()
    return df

df = load_data()

#sidebar - LOS/Readmission highlight toggles

st.sidebar.title("Controls")

if "los_toggle" not in st.session_state:
    st.session_state.los_toggle = False
if "readmit_toggle" not in st.session_state:
    st.session_state.readmit_toggle = False

st.session_state.los_toggle = st.sidebar.toggle("Highlight LOS", value=st.session_state.los_toggle)
st.session_state.readmit_toggle = st.sidebar.toggle("Highlight Readmission", value=st.session_state.readmit_toggle)


#Sidebar 

st.sidebar.divider()
st.sidebar.subheader("Filters")

def make_filter(colname, label):
    if colname not in df.columns:
        return None
    opts = ["All"] + sorted([x for x in df[colname].dropna().unique()])
    return st.sidebar.selectbox(label, opts, index=0)

age_sel = make_filter("age_group", "Age group")
gender_sel = make_filter("gender", "Gender")
admgrp_sel = make_filter("admission_group", "Admission group")
wk_sel = make_filter("admit_weekend", "Weekday/Weekend")

f = df.copy()
if age_sel and age_sel != "All":
    f = f[f["age_group"] == age_sel]
if gender_sel and gender_sel != "All":
    f = f[f["gender"] == gender_sel]
if admgrp_sel and admgrp_sel != "All":
    f = f[f["admission_group"] == admgrp_sel]
if wk_sel and wk_sel != "All":
    f = f[f["admit_weekend"] == wk_sel]


#CSS

def block_class(section):
    """
    section: "los" or "readmit"
    - if section toggled -> highlight
    - if other toggled but this not -> fade
    - if none toggled -> normal
    - if both toggled -> both highlight
    """
    los_on = st.session_state.los_toggle
    read_on = st.session_state.readmit_toggle

    if los_on and read_on:
        return "highlight"

    if section == "los":
        if los_on:
            return "highlight"
        if read_on and not los_on:
            return "fade"
        return "normal"

    if section == "readmit":
        if read_on:
            return "highlight"
        if los_on and not read_on:
            return "fade"
        return "normal"

    return "normal"

st.markdown(
    """
    <style>
    .card {
        padding: 14px;
        border-radius: 16px;
        margin-bottom: 14px;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
    }
    .highlight {
        border: 2px solid rgba(0, 200, 255, 0.9) !important;
        box-shadow: 0 0 14px rgba(0, 200, 255, 0.35);
        opacity: 1;
    }
    .fade {
        opacity: 0.45;
        filter: grayscale(15%);
    }
    .normal {
        opacity: 1;
    }
    </style>
    """,
    unsafe_allow_html=True
)


tab1, tab2, tab3 = st.tabs(["Overview", "Patient Entry & Predictions", "Historical Analytics"])


# TAB 1: OVERVIEW

with tab1:
    st.title("Myocardial Infarction - Dashboard Overview")

    # --- KPI Row (overall summary KPIs) ---
    k1, k2, k3, k4, k5 = st.columns(5)

    total_adm = len(df)
    uniq_pat = df["subject_id"].nunique() if "subject_id" in df.columns else np.nan
    avg_los = df["los_days"].mean() if "los_days" in df.columns else np.nan
    pct_long = df["los_cat"].eq("≥ 7 days").mean() * 100 if "los_cat" in df.columns else np.nan
    pct_readmitted = (df["readmission_risk"].isin(["High", "Medium", "Low"]).mean() * 100) if "readmission_risk" in df.columns else np.nan

    #KPI: Total Admissions
    k1.metric("Total Admissions", f"{total_adm:,}")
    #KPI: Unique Patients
    k2.metric("Unique Patients", f"{int(uniq_pat):,}" if pd.notna(uniq_pat) else "—")
    #KPI: Average LOS
    k3.metric("Average LOS (days)", f"{avg_los:.1f}" if pd.notna(avg_los) else "—")
    #KPI: LOS > 7 days (%)
    k4.metric("LOS ≥ 7 days (%)", f"{pct_long:.1f}%" if pd.notna(pct_long) else "—")
    #KPI: % Patients Readmitted (proxy)
    k5.metric("% Patients Readmitted", f"{pct_readmitted:.1f}%" if pd.notna(pct_readmitted) else "—")

    st.divider()

    c1, c2, c3 = st.columns([1.1, 1.5, 1.4])

    with c1:
        #Chart: Available doctors (dummy list)
        st.subheader("Available Doctors")
        docs = [
            ("Dr. Jaylon Stanton", "Cardiologist", "Available"),
            ("Dr. Carla Schleifer", "Emergency Physician", "Busy"),
            ("Dr. Hanna Geidt", "Internal Medicine", "Available"),
            ("Dr. Nimal Perera", "Cardiology Registrar", "Available"),
        ]
        for name, spec, status in docs:
            st.write(f"**{name}**  \n{spec} • {status}")

    with c2:
        #Chart: Admissions trend (monthly)
        st.subheader("Admissions Trend")
        if "admittime" in df.columns:
            trend = (
                df.dropna(subset=["admittime"])
                  .assign(month=lambda d: d["admittime"].dt.to_period("M").astype(str))
                  .groupby("month").size().reset_index(name="admissions")
            )
            fig = px.line(trend, x="month", y="admissions", markers=True)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("admittime not available.")

    with c3:
        #Chart: LOS category donut
        st.subheader("LOS Category")
        if "los_cat" in df.columns:
            x = df["los_cat"].value_counts().reset_index()
            x.columns = ["los_cat", "count"]
            fig = px.pie(x, names="los_cat", values="count", hole=0.55)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("los_cat not available.")

    #Chart: Readmission risk donut
    st.subheader("Readmission Risk Category")
    if "readmission_risk" in df.columns:
        x = df["readmission_risk"].value_counts().reset_index()
        x.columns = ["readmission_risk", "count"]
        fig = px.pie(x, names="readmission_risk", values="count", hole=0.55)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("readmission_risk not available.")


# TAB 2: PATIENT ENTRY & PREDICTIONS 

with tab2:
    st.title("Patient Entry & Prediction")

    # Section: Input form
    st.subheader("Patient Input Form")

    colA, colB = st.columns(2)

    with colA:
        patient_name = st.text_input("Patient Name (optional)")
        gender = st.selectbox("Gender", ["M", "F"])
        age_group = st.selectbox("Age Group", ["18–45", "46–60", "61–74", "75+"])

        admission_group = st.selectbox("Admission Group", ["Emergency / Unplanned", "Non-Emergency / Planned"])
        admit_weekend = st.selectbox("Weekend Admission?", ["Weekday", "Weekend"])
        prior_mi = st.selectbox("Prior MI?", ["N", "Y"])

    with colB:
        other_diag = st.selectbox("Comorbidity present?", ["N", "Y"])

        comorbidity_type = None
        if other_diag == "Y":
            comorbidity_type = st.selectbox(
                "Comorbidity Type (for visual purposes)",
                ["Hypertension", "Diabetes", "Smoking", "Obesity", "Hyperlipidemia", "CKD"]
            )
        notes = st.text_area("Comments / Notes (optional)", height=120)
        save_to_db = st.toggle("Save to database", value=True)

    submitted = st.button("Generate Prediction")

    if submitted:
        #Placeholder predictions (we replace with real model later)
        predicted_los = "≥ 7 days" if age_group in ["61–74", "75+"] and admission_group.startswith("Emergency") else "< 7 days"
        predicted_readmit = "High" if prior_mi == "Y" else "Medium"

        st.divider()

        #Prediction cards
        p1, p2 = st.columns(2)

        #KPI: Predicted LOS Category (LOS section styling)
        st.markdown(f'<div class="card {block_class("los")}">', unsafe_allow_html=True)
        p1.metric("Predicted LOS Category", predicted_los)
        st.markdown("</div>", unsafe_allow_html=True)

        #KPI: Predicted Readmission Risk (Readmission section styling)
        st.markdown(f'<div class="card {block_class("readmit")}">', unsafe_allow_html=True)
        p2.metric("Predicted Readmission Risk", predicted_readmit)
        st.markdown("</div>", unsafe_allow_html=True)

        #Historical averages for similar patients (quick, using filtered dataset)
        sim = df.copy()
        if "age_group" in sim.columns:
            sim = sim[sim["age_group"] == age_group]
        if "admission_group" in sim.columns:
            sim = sim[sim["admission_group"] == admission_group]
        if "other_diag" in sim.columns:
            sim = sim[sim["other_diag"] == other_diag]

        avg_los_sim = sim["los_days"].mean() if "los_days" in sim.columns and len(sim) else np.nan

        #KPI: Average LOS for similar patients
        st.markdown(f'<div class="card {block_class("los")}">', unsafe_allow_html=True)
        st.metric("Average LOS for similar historical patients (days)", f"{avg_los_sim:.1f}" if pd.notna(avg_los_sim) else "—")
        st.markdown("</div>", unsafe_allow_html=True)

        st.caption("Predictions are placeholders for now. Models + MongoDB will be connected after final model selection.")

#TAB 3: HISTORICAL ANALYTICS
with tab3:
    st.title("Historical Analytics (MI_finaldf)")
    st.caption("Filters on the left apply here.")

    #LOS Section (KPIs + charts)
    st.subheader("Length of Stay (LOS) Analytics")

    st.markdown(f'<div class="card {block_class("los")}">', unsafe_allow_html=True)

    los_k1, los_k2, los_k3 = st.columns(3)

    #KPI: Total filtered admissions
    los_k1.metric("Filtered Admissions", f"{len(f):,}")

    #KPI: Avg LOS days
    avg_los_f = f["los_days"].mean() if "los_days" in f.columns and len(f) else np.nan
    los_k2.metric("Avg LOS (days)", f"{avg_los_f:.1f}" if pd.notna(avg_los_f) else "—")

    #KPI: % long LOS
    pct_long_f = f["los_cat"].eq("≥ 7 days").mean() * 100 if "los_cat" in f.columns and len(f) else np.nan
    los_k3.metric("LOS ≥ 7 days (%)", f"{pct_long_f:.1f}%" if pd.notna(pct_long_f) else "—")

    #Chart: LOS donut
    if "los_cat" in f.columns and len(f):
        x = f["los_cat"].value_counts().reset_index()
        x.columns = ["los_cat", "count"]
        fig = px.pie(x, names="los_cat", values="count", hole=0.55)
        st.plotly_chart(fig, use_container_width=True)

    #Chart: LOS by age group (bar)
    if "age_group" in f.columns and "los_cat" in f.columns and len(f):
        x = f.groupby(["age_group", "los_cat"]).size().reset_index(name="count")
        fig = px.bar(x, x="age_group", y="count", color="los_cat", barmode="group")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    #Readmission Section (KPIs + charts)
    st.subheader("Readmission Risk Analytics")

    st.markdown(f'<div class="card {block_class("readmit")}">', unsafe_allow_html=True)

    r1, r2, r3 = st.columns(3)

    #KPI: % high readmission
    pct_high = f["readmission_risk"].eq("High").mean() * 100 if "readmission_risk" in f.columns and len(f) else np.nan
    r1.metric("% High Readmission Risk", f"{pct_high:.1f}%" if pd.notna(pct_high) else "—")

    #KPI: % medium
    pct_med = f["readmission_risk"].eq("Medium").mean() * 100 if "readmission_risk" in f.columns and len(f) else np.nan
    r2.metric("% Medium Readmission Risk", f"{pct_med:.1f}%" if pd.notna(pct_med) else "—")

    #KPI: % low
    pct_low = f["readmission_risk"].eq("Low").mean() * 100 if "readmission_risk" in f.columns and len(f) else np.nan
    r3.metric("% Low Readmission Risk", f"{pct_low:.1f}%" if pd.notna(pct_low) else "—")

    #Chart: Readmission risk donut
    if "readmission_risk" in f.columns and len(f):
        x = f["readmission_risk"].value_counts().reset_index()
        x.columns = ["readmission_risk", "count"]
        fig = px.pie(x, names="readmission_risk", values="count", hole=0.55)
        st.plotly_chart(fig, use_container_width=True)

    #Chart: Readmission by age group
    if "age_group" in f.columns and "readmission_risk" in f.columns and len(f):
        x = f.groupby(["age_group", "readmission_risk"]).size().reset_index(name="count")
        fig = px.bar(x, x="age_group", y="count", color="readmission_risk", barmode="group")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    #heatmap section
    st.subheader("Admissions Heatmap (Weekday × Hour)")

    # Chart: Heatmap of admissions by weekday and hour
    if "admit_dow" in f.columns and "admit_hour" in f.columns and len(f):
        order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        heat = (
            f.dropna(subset=["admit_dow", "admit_hour"])
             .groupby(["admit_dow", "admit_hour"])
             .size()
             .reset_index(name="count")
        )
        heat["admit_dow"] = pd.Categorical(heat["admit_dow"], categories=order, ordered=True)
        heat = heat.sort_values(["admit_dow", "admit_hour"])

        fig = px.density_heatmap(
            heat, x="admit_hour", y="admit_dow", z="count",
            nbinsx=24, nbinsy=7
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Heatmap requires admittime to calculate weekday and hour.")