import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from scipy.sparse import hstack
from datetime import datetime
from pymongo import MongoClient

st.set_page_config(
    page_title="MI Analytics : IIT National Hospital",
    page_icon="🩺", layout="wide",
    initial_sidebar_state="expanded"
)

#CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Sora:wght@600;700&display=swap');
html,body,[class*="css"]{ font-family:'Inter',sans-serif; }

.stApp { background:#f0f4f9; color:#1e2a3a; }
#MainMenu,footer,header{ visibility:hidden; }
.block-container{ padding:0 1.5rem 2rem 1.5rem !important; max-width:100% !important; }

[data-testid="stSidebar"]{
    background:#1a2035 !important;
    border-right:1px solid rgba(255,255,255,0.06) !important;
}
[data-testid="stSidebar"] * { color:#cbd5e1 !important; }
[data-testid="stSidebar"] .stRadio label { color:#cbd5e1 !important; font-size:0.82rem !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown strong { color:#94a3b8 !important; font-size:0.75rem !important; }
[data-testid="stSidebar"] .stSelectbox>div>div {
    background:#273050 !important;
    border:1px solid rgba(255,255,255,0.1) !important;
    border-radius:8px !important;
    color:#e2e8f0 !important;
}
[data-testid="stSidebar"] .stSelectbox>div>div>div { color:#e2e8f0 !important; }
[data-testid="stSidebar"] .stRadio>div { gap:0.4rem !important; }
[data-testid="collapsedControl"]{
    background:#2563eb !important; border-radius:0 8px 8px 0 !important;
    color:#ffffff !important; top:50% !important;
    box-shadow:3px 0 12px rgba(37,99,235,0.4) !important;
}
[data-testid="collapsedControl"] svg { color:#ffffff !important; fill:#ffffff !important; }
/* Reset button styling */
[data-testid="stSidebar"] .stButton>button {
    background:rgba(37,99,235,0.15) !important;
    border:1px solid rgba(37,99,235,0.4) !important;
    color:#93c5fd !important;
    border-radius:8px !important;
    font-size:0.78rem !important;
    font-weight:600 !important;
    width:100% !important;
    animation:none !important;
}
[data-testid="stSidebar"] .stButton>button:hover {
    background:rgba(37,99,235,0.3) !important;
    border-color:rgba(147,197,253,0.6) !important;
    color:#ffffff !important;
    box-shadow:none !important;
}

/* CHART CARD WRAPPER */
.chart-wrap {
    background: rgba(236,239,245,0.75);
    border-radius: 16px;
    padding: 0.5rem 0.5rem 0.1rem 0.5rem;
    border: 1px solid rgba(15,45,94,0.08);
    box-shadow: 0 2px 10px rgba(15,45,94,0.06);
    margin-bottom: 0.4rem;
    overflow: hidden;
    position: relative;
}
.chart-wrap > div {
    border-radius: 16px !important;
    overflow: hidden !important;
}
/* Target streamlit plotly wrapper */
.chart-wrap [data-testid="stPlotlyChart"] {
    border-radius: 16px !important;
    overflow: hidden !important;
}
.chart-wrap [data-testid="stPlotlyChart"] > div {
    border-radius: 16px !important;
    overflow: hidden !important;
}
.chart-wrap iframe {
    border-radius: 16px !important;
}

/* HEADER */
.dash-header{
    background:linear-gradient(100deg,#0a1628 0%,#0f2550 55%,#153a7a 100%);
    padding:1.2rem 2.2rem; margin:-1rem -1.5rem 0 -1.5rem;
    display:flex; align-items:center; justify-content:space-between;
    box-shadow:0 4px 24px rgba(0,0,0,0.4); position:relative; overflow:hidden;
}
.dash-header::before{
    content:''; position:absolute; top:-50px; right:-50px;
    width:200px; height:200px; background:rgba(255,255,255,0.03); border-radius:50%;
}
.header-logo{ display:flex; align-items:center; gap:1rem; position:relative; z-index:1; }
.header-logo-icon{
    width:44px; height:44px; background:rgba(255,255,255,0.1);
    border:1px solid rgba(255,255,255,0.18); border-radius:12px;
    display:flex; align-items:center; justify-content:center; font-size:1.3rem;
}
.header-title{ font-family:'Sora',sans-serif; font-size:1.2rem; font-weight:700; color:#fff; margin:0; }
.header-sub{ font-size:0.67rem; color:rgba(255,255,255,0.45); margin:0.15rem 0 0 0; text-transform:uppercase; letter-spacing:0.12em; }
.header-right{ text-align:right; position:relative; z-index:1; }
.header-date{ font-size:1rem; font-weight:600; color:#fff; }
.header-hospital{ font-size:0.7rem; color:rgba(255,255,255,0.45); margin-top:0.15rem; }

/* KPI CARDS */
.kpi-card{
    background:#ffffff; border-radius:16px; padding:1.3rem 1.4rem 1.2rem 1.4rem;
    box-shadow:0 2px 16px rgba(15,45,94,0.09); border:1px solid rgba(15,45,94,0.07);
    position:relative; overflow:hidden; transition:transform 0.2s,box-shadow 0.2s; min-height:110px;
}
.kpi-card:hover{ transform:translateY(-3px); box-shadow:0 8px 28px rgba(15,45,94,0.14); }
.kpi-card::before{
    content:''; position:absolute; top:0; left:0; right:0; height:3px;
    background:linear-gradient(90deg,#2563eb,#38bdf8); border-radius:16px 16px 0 0;
}
.kpi-label{ font-size:0.75rem; color:#1e2a3a; text-transform:uppercase; letter-spacing:0.1em; font-weight:800; margin-bottom:0.55rem; }
.kpi-value{ font-family:'Sora',sans-serif; font-size:1.9rem; font-weight:700; color:#0a1f44; line-height:1; }
.kpi-sub{ font-size:0.72rem; color:#4a5a74; margin-top:0.4rem; font-weight:600; }
.kpi-icon{ position:absolute; top:1rem; right:1.1rem; font-size:2.2rem; opacity:0.28; }

/* SECTION TITLE */
.section-title{
    font-size:0.85rem; font-weight:800; color:#0a1f44;
    text-transform:uppercase; letter-spacing:0.14em;
    margin:1.8rem 0 0.9rem 0; display:flex; align-items:center; gap:0.7rem;
}
.section-title::before{ content:''; width:18px; height:3px; background:linear-gradient(90deg,#2563eb,#38bdf8); border-radius:2px; flex-shrink:0; }
.section-title::after{ content:''; flex:1; height:1px; background:rgba(15,45,94,0.1); }

/* WARD CARDS */
.ward-card{
    background:#ffffff; border-radius:12px; padding:0.85rem 1rem; margin-bottom:0.6rem;
    box-shadow:0 1px 6px rgba(15,45,94,0.07); border:1px solid rgba(15,45,94,0.06);
}
.ward-name{ font-size:0.82rem; font-weight:700; color:#1e2a3a; }
.ward-meta{ font-size:0.67rem; color:#7a8ba8; margin-bottom:0.3rem; }

/* DOCTOR CARDS */
.doctor-card{
    background:#ffffff; border-radius:14px; padding:1rem 1.1rem;
    box-shadow:0 2px 12px rgba(15,45,94,0.08); border:1px solid rgba(15,45,94,0.07);
    transition:transform 0.2s,box-shadow 0.2s; margin-bottom:0.3rem;
}
.doctor-card:hover{ transform:translateY(-3px); box-shadow:0 6px 20px rgba(15,45,94,0.14); }
.doctor-name{ font-weight:700; font-size:0.84rem; color:#0a1f44; margin-bottom:0.1rem; }
.doctor-specialty{ font-size:0.69rem; color:#7a8ba8; margin-bottom:0.35rem; }
.doctor-contact{ font-size:0.69rem; color:#2563eb; margin-bottom:0.5rem; }
.badge-on{ display:inline-flex; align-items:center; gap:0.3rem; background:rgba(5,150,105,0.08); color:#059669; border:1px solid rgba(5,150,105,0.25); border-radius:20px; padding:0.15rem 0.75rem; font-size:0.63rem; font-weight:700; }
.badge-off{ display:inline-flex; align-items:center; gap:0.3rem; background:rgba(220,38,38,0.06); color:#dc2626; border:1px solid rgba(220,38,38,0.2); border-radius:20px; padding:0.15rem 0.75rem; font-size:0.63rem; font-weight:700; }

@keyframes shimmer {
    0%   { border-color: rgba(37,99,235,0.3); box-shadow: 0 0 0 rgba(37,99,235,0); }
    50%  { border-color: rgba(56,189,248,0.8); box-shadow: 0 0 8px rgba(56,189,248,0.35); }
    100% { border-color: rgba(37,99,235,0.3); box-shadow: 0 0 0 rgba(37,99,235,0); }
}

/* PRED CARDS */
.pred-card{ background:#ffffff; border-radius:16px; padding:1.5rem 1.6rem; text-align:center; box-shadow:0 2px 16px rgba(15,45,94,0.08); border:1px solid rgba(15,45,94,0.07); }
.pred-label{ font-size:0.63rem; color:#7a8ba8; text-transform:uppercase; letter-spacing:0.12em; font-weight:700; margin-bottom:0.6rem; }
.pred-good{ font-family:'Sora',sans-serif; font-size:1.8rem; font-weight:700; color:#059669; }
.pred-warn{ font-family:'Sora',sans-serif; font-size:1.8rem; font-weight:700; color:#d97706; }
.pred-bad { font-family:'Sora',sans-serif; font-size:1.8rem; font-weight:700; color:#dc2626; }
.pred-sub { font-size:0.71rem; color:#a0aec0; margin-top:0.4rem; line-height:1.5; }

.live-card{ background:#ffffff; border-radius:14px; padding:1.2rem 1.4rem; border:1px solid rgba(37,99,235,0.2); box-shadow:0 2px 16px rgba(15,45,94,0.08); margin-bottom:0.8rem; }
.live-label{ font-size:0.62rem; color:#2563eb; text-transform:uppercase; letter-spacing:0.12em; font-weight:700; margin-bottom:0.5rem; }

.next-pid{ display:inline-flex; align-items:center; gap:0.6rem; background:rgba(37,99,235,0.06); border:1px solid rgba(37,99,235,0.2); border-radius:10px; padding:0.45rem 1.1rem; font-size:0.8rem; color:#1d4ed8; margin-bottom:1.3rem; }
.next-pid span{ font-weight:700; font-size:1.05rem; }

.stTabs [data-baseweb="tab-list"]{ background:rgba(15,45,94,0.03); border-bottom:1px solid rgba(15,45,94,0.1); gap:0; padding:0 0.5rem; }
.stTabs [data-baseweb="tab"]{ background:transparent; color:#7a8ba8; border-radius:0; font-size:0.73rem; font-weight:600; letter-spacing:0.08em; text-transform:uppercase; padding:0.85rem 1.4rem; border-bottom:2px solid transparent; margin-bottom:-1px; }
.stTabs [aria-selected="true"]{ color:#1d4ed8 !important; border-bottom:2px solid #2563eb !important; background:transparent !important; }
.stTabs [data-baseweb="tab"]:hover{ color:#1e2a3a !important; }

.stTextInput>div>div>input,.stNumberInput>div>div>input{ background:#ffffff !important; border:1px solid rgba(15,45,94,0.15) !important; border-radius:10px !important; color:#1e2a3a !important; }
.stSelectbox>div>div{ background:#ffffff !important; border:1px solid rgba(15,45,94,0.15) !important; border-radius:10px !important; color:#1e2a3a !important; }
label,.stSelectbox label,.stNumberInput label,.stTextInput label,.stRadio label{ color:#1e2a3a !important; font-size:0.77rem !important; font-weight:600 !important; }

.stButton>button{
    background:rgba(255,255,255,0.8) !important; border:1.5px solid rgba(37,99,235,0.35) !important;
    color:#1d4ed8 !important; border-radius:10px !important; font-size:0.75rem !important;
    font-weight:600 !important; padding:0.35rem 0.8rem !important; transition:all 0.2s !important;
    animation: shimmer 2.5s ease-in-out infinite !important;
}
.stButton>button:hover{
    border-color:rgba(56,189,248,0.9) !important; color:#0f2d5e !important;
    background:rgba(219,234,254,0.6) !important; box-shadow:0 0 10px rgba(56,189,248,0.4) !important;
    animation:none !important;
}

[data-testid="stMetric"]{ background:#ffffff; border-radius:12px; padding:0.8rem 1rem; border:1px solid rgba(15,45,94,0.07); box-shadow:0 1px 6px rgba(15,45,94,0.06); }
[data-testid="stMetricLabel"]{ color:#7a8ba8 !important; font-size:0.72rem !important; }
[data-testid="stMetricValue"]{ color:#0a1f44 !important; }

.stSuccess{ background:rgba(5,150,105,0.08) !important; border:1px solid rgba(5,150,105,0.25) !important; color:#059669 !important; border-radius:10px !important; }
.stWarning{ background:rgba(217,119,6,0.08) !important; border:1px solid rgba(217,119,6,0.25) !important; border-radius:10px !important; }
.stError{ background:rgba(220,38,38,0.07) !important; border:1px solid rgba(220,38,38,0.2) !important; border-radius:10px !important; }
.stInfo{ background:rgba(37,99,235,0.07) !important; border:1px solid rgba(37,99,235,0.2) !important; color:#1d4ed8 !important; border-radius:10px !important; }

[data-testid="stDataFrameResizable"]{ background:#ffffff !important; border:1px solid rgba(15,45,94,0.08) !important; border-radius:12px !important; }
::-webkit-scrollbar{ width:5px; height:5px; }
::-webkit-scrollbar-track{ background:#f0f4f9; }
::-webkit-scrollbar-thumb{ background:rgba(15,45,94,0.15); border-radius:10px; }
</style>
""", unsafe_allow_html=True)

#loading data and models
@st.cache_data
def load_data():
    df = pd.read_csv("MI_finaldf.csv")
    df["admittime"] = pd.to_datetime(df["admittime"])
    df["year"] = df["admittime"].dt.year
   
    df["curr_service"] = df["curr_service"].fillna("UNKNOWN")
    return df

@st.cache_resource
def load_models():
    return (
        joblib.load("Final_LOS_Model.pkl"),       joblib.load("LOS_Encoder.pkl"),
        joblib.load("Final_Mortality_Model.pkl"),  joblib.load("Mortality_Admission_Encoder.pkl"),
        joblib.load("Final_Mortality_Post_Model.pkl"), joblib.load("Mortality_Post_Encoder.pkl"),
        joblib.load("Final_Readmission_Model.pkl"),   joblib.load("readmission_encoder.pkl"),
        joblib.load("readmission_threshold.pkl"), joblib.load("LOS_Regression_Model.pkl"),
    )

df = load_data()
(los_model,los_enc,mort_model,mort_enc,
 mort2_model,mort2_enc,readmit_model,readmit_enc, readmit_thresh,los_reg_model) = load_models()

#mongo db connection function
@st.cache_resource
def init_mongo():
    client = MongoClient(st.secrets["mongo"]["uri"])
    return client["w1998467_MI_Analytics"]["Patients"]

mongo_patients = init_mongo()

MI_SERVICES = ["CMED","CSURG","MED","NMED","VSURG","SURG","OMED","NSURG","TRAUM","TSURG","UNKNOWN"]
mi_svcs_in_data = [s for s in MI_SERVICES if s in df["curr_service"].unique()]

DISCHARGE_LOCATIONS = [
    "HOME", "HOME HEALTH CARE", "SKILLED NURSING FACILITY", "REHAB",
    "CHRONIC/LONG TERM ACUTE CARE", "HOSPICE", "ACUTE HOSPITAL",
    "AGAINST ADVICE", "ASSISTED LIVING", "HEALTHCARE FACILITY",
    "OTHER", "OTHER FACILITY", "PSYCH FACILITY", "DIED"
]

#Plot colours
CHART_BG = "rgba(236,239,245,0.75)"
GRID     = "rgba(15,45,94,0.06)"
FC       = "#4a5a74"
TITLE_C  = "#0f2d5e" 
BLUE     = "#1e40af"
RED      = "#be123c"
GREEN    = "#34d399"
AMBER    = "#fbbf24"
PURP     = "#a78bfa"
TEAL     = "#22d3ee"
NAVY     = "#1d4ed8"
TEAL2    = "#0891b2"
PAL      = [BLUE,RED,GREEN,AMBER,PURP,TEAL,NAVY,"#e879f9","#4ade80","#38bdf8"]

LEG = dict(
    bgcolor="rgba(15,45,94,0.88)",
    bordercolor="rgba(255,255,255,0.2)", borderwidth=1,
    font=dict(size=12, color="#ffffff"),
)

def bl(fig, title="", h=None, margin=None):
    m = margin or dict(l=12, r=20, t=52, b=12)
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>",
                   font=dict(size=14, color=TITLE_C, family="Inter"), x=0),
        plot_bgcolor=CHART_BG, paper_bgcolor=CHART_BG,
        font=dict(color=FC, family="Inter", size=12),
        margin=m,
        legend=LEG,
        xaxis=dict(gridcolor="rgba(0,0,0,0)", linecolor="rgba(15,45,94,0.1)",
                   tickfont=dict(color=FC, size=11), showgrid=False),
        yaxis=dict(gridcolor=GRID, linecolor="rgba(15,45,94,0.1)",
                   tickfont=dict(color=FC, size=11), showgrid=True),
    )
    if h:
        fig.update_layout(height=h)
    return fig

def chart(fig, outcome="None"):
    dimmed = is_dimmed(outcome)
    if dimmed:
        fig.update_layout(
            plot_bgcolor="rgba(236,239,245,0.75)",
            paper_bgcolor="rgba(236,239,245,0.75)",
            title_font_color="rgba(15,45,94,0.18)",
            font_color="rgba(15,45,94,0.18)",
            xaxis=dict(tickfont_color="rgba(15,45,94,0.18)",
                       title_font_color="rgba(15,45,94,0.18)",
                       gridcolor="rgba(15,45,94,0.02)",
                       linecolor="rgba(15,45,94,0.05)"),
            yaxis=dict(tickfont_color="rgba(15,45,94,0.18)",
                       title_font_color="rgba(15,45,94,0.18)",
                       gridcolor="rgba(15,45,94,0.02)",
                       linecolor="rgba(15,45,94,0.05)"),
        )
        for trace in fig.data:
            if hasattr(trace, 'opacity'):
                trace.opacity = 0.12
            if hasattr(trace, 'marker') and trace.marker:
                try: trace.marker.opacity = 0.12
                except: pass
            if hasattr(trace, 'line') and trace.line:
                try: trace.line.color = "rgba(15,45,94,0.12)"
                except: pass
    else:
        fig.update_layout(
            plot_bgcolor="rgba(236,239,245,0.75)",
            paper_bgcolor="rgba(236,239,245,0.75)",
        )
    st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

def make_gauge(value, title):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        number={"suffix":"%","font":{"size":30,"color":TITLE_C}},
        gauge={
            "axis":{"range":[0,100],"tickcolor":FC,"tickfont":{"color":FC}},
            "bar":{"color":BLUE}, "bgcolor":"#f0f4f9","bordercolor":"rgba(15,45,94,0.1)",
            "steps":[
                {"range":[0,10],"color":"rgba(5,150,105,0.1)"},
                {"range":[10,25],"color":"rgba(217,119,6,0.08)"},
                {"range":[25,100],"color":"rgba(220,38,38,0.07)"},
            ],
            "threshold":{"line":{"color":TITLE_C,"width":2},"thickness":0.75,"value":value}
        },
        title={"text":title,"font":{"color":FC,"size":12}}))
    fig.update_layout(paper_bgcolor=CHART_BG, font={"color":FC,"family":"Inter"},
                      margin=dict(l=30,r=30,t=60,b=10), height=260)
    return fig

#Prediction variables
CAT_LOS   = ["gender","admission_type","curr_service","admit_weekend","prior_mi"]
NUM_LOS   = ["age","num_diagnoses_at_admission"]
CAT_MORT  = ["gender","admission_type","curr_service","admit_weekend","prior_mi"]
NUM_MORT  = ["age","num_diagnoses_at_admission"]
CAT_MORT2 = ["gender","admission_type","curr_service","admit_weekend","prior_mi"]
NUM_MORT2 = ["age","num_diagnoses_at_admission","procedure_count","drg_severity","drg_mortality"]

CAT_READ  = ["gender","admission_type","curr_service","discharge_location","admit_weekend","prior_mi"]
NUM_READ  = ["age","num_diagnoses_at_admission","los_days","procedure_count","drg_severity","drg_mortality","cardiac_proc_flag"]

def to_prior_mi_yn(val):
    """Convert any prior_mi representation to the Y/N the encoders expect."""
    return "Y" if str(val).strip() in ["1", "1.0", "Y", "y", "Yes", "yes"] else "N"

def enc_pred(model, encoder, cat_cols, num_cols, inp):
    """General prediction helper — uses sparse hstack (for RF/CatBoost)."""
    row = pd.DataFrame([inp])
    row["prior_mi"] = to_prior_mi_yn(row["prior_mi"].iloc[0])
    for c in cat_cols: row[c] = row[c].astype(str)
    X_cat = encoder.transform(row[cat_cols])
    X_num = row[num_cols].astype(float).values
    return float(model.predict_proba(hstack([X_cat,X_num]))[0][1])

def enc_pred_readmit(model, encoder, inp):
    """Readmission prediction — XGBoost requires dense array."""
    row = pd.DataFrame([inp])
    row["prior_mi"] = to_prior_mi_yn(row["prior_mi"].iloc[0]) 
    for c in CAT_READ: row[c] = row[c].astype(str)
    X_cat = encoder.transform(row[CAT_READ])
    X_num = row[NUM_READ].astype(float).values
    X_dense = hstack([X_cat, X_num]).toarray()
    return float(model.predict_proba(X_dense)[0][1])

def predict_los_days(inp):
    """Predict continuous LOS in days using Linear Regression model."""
    row = pd.DataFrame([inp])
    row["prior_mi"] = to_prior_mi_yn(row["prior_mi"].iloc[0])
    for c in CAT_LOS: row[c] = row[c].astype(str)
    X_cat = los_enc.transform(row[CAT_LOS])
    X_num = row[NUM_LOS].astype(float).values
    pred  = los_reg_model.predict(hstack([X_cat, X_num]))[0]
    return round(max(0.1, pred), 1)  #to avoid negative prediction

def avg_read_sim(inp):
    s = df[(df["readmit_30d"]==1) & (df["age"].between(inp["age"]-10,inp["age"]+10))]
    return round(s["days_to_readmit"].mean() if len(s)>=5
                 else df[df["readmit_30d"]==1]["days_to_readmit"].mean(), 1)

#including available/unavailable doctors
if "patients" not in st.session_state: st.session_state.patients = {}
if "doc_avail" not in st.session_state:
    st.session_state.doc_avail = {
        "Dr. Pandula Athaudaarchchi":True,"Dr. Dev Kesava":True,"Dr. M. H. M. Zacky":False,
        "Dr. Amila Walawwatta":True,"Dr. Rajitha Y. De Silva":False,
        "Dr. Asunga Dunuwille":True,"Dr. Sandamali Premarathna":True,
        "Dr. M.B.F Rahuman":False,"Dr. Shehan Perera":True,
        "Acute MI team":True,
    }

DOCTORS = {
    "Dr. Pandula Athaudaarchchi":   {"spec":"Cardiologist","tel":"+94 11 911 1000"},
    "Dr. Dev Kesava":               {"spec":"Consultant Cardiothoracic Surgeon","tel":"+94 11 911 1001"},
    "Dr. M. H. M. Zacky":           {"spec":"Interventional Cardiologist","tel":"+94 11 911 1002"},
    "Dr. Amila Walawwatta":         {"spec":"Consultant Cardiologist","tel":"+94 11 911 1003"},
    "Dr. Rajitha Y. De Silva":      {"spec":"Cardiothoracic Surgeon","tel":"+94 11 911 1004"},
    "Dr. Asunga Dunuwille":         {"spec":"Cardiac Electrophysiologist","tel":"+94 11 911 1005"},
    "Dr. Sandamali Premarathna":    {"spec":"Cardiologist","tel":"+94 11 911 1006"},
    "Dr. M.B.F Rahuman":            {"spec":"Cardiologist","tel":"+94 11 911 1007"},
    "Dr. Shehan Perera":            {"spec":"Pediatric Cardiologist","tel":"+94 11 911 1008"},
    "Acute MI team":                {"spec":"Emergency Medicine/Internal Medicine","tel":"+94 11 911 1009"},
}

if st.session_state.get("_reset", False):
    for k in ["hl_radio","svc_sel","adm_sel","gen_sel","age_sel"]:
        if k in st.session_state:
            del st.session_state[k]
    del st.session_state["_reset"]

#Sidebar
with st.sidebar:
    st.markdown("""
    <div style="padding:1.2rem 0.5rem 0.8rem 0.5rem;border-bottom:1px solid rgba(255,255,255,0.08);margin-bottom:1rem">
      <div style="font-size:2rem;margin-bottom:0.4rem">🏥</div>
      <div style="font-size:1.1rem;font-weight:700;color:#f1f5f9;font-family:Sora,sans-serif;letter-spacing:0.01em">IIT National Hospital</div>
      <div style="font-size:0.72rem;color:#64748b;margin-top:0.1rem">Colombo</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p style="font-size:1rem;font-weight:700;color:#f1f5f9;margin-bottom:0.8rem">Filters</p>',
                unsafe_allow_html=True)
    st.markdown('<hr style="border-color:rgba(255,255,255,0.08);margin:0 0 1rem 0">', unsafe_allow_html=True)

    st.markdown('<p style="font-size:0.72rem;font-weight:600;color:#64748b;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.4rem">Outcome Highlight</p>',
                unsafe_allow_html=True)
    st.session_state.setdefault("hl_radio", "All")
    highlight = st.radio("Highlight",["All","Length of Stay","Mortality","Readmission"],
                          label_visibility="collapsed", key="hl_radio")

    st.markdown('<hr style="border-color:rgba(255,255,255,0.08);margin:0.8rem 0">', unsafe_allow_html=True)

    st.markdown('<p style="font-size:0.72rem;font-weight:600;color:#64748b;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.4rem">Clinical Service</p>',
                unsafe_allow_html=True)
    svc_opts = ["All"]+mi_svcs_in_data
    st.session_state.setdefault("svc_sel", "All")
    sel_svc  = st.selectbox("Clinical Service", svc_opts,
                             label_visibility="collapsed", key="svc_sel")

    st.markdown('<p style="font-size:0.72rem;font-weight:600;color:#64748b;text-transform:uppercase;letter-spacing:0.1em;margin:0.7rem 0 0.4rem 0">Admission Type</p>',
                unsafe_allow_html=True)
    adm_opts = ["All"]+sorted(df["admission_type"].dropna().unique().tolist())
    st.session_state.setdefault("adm_sel", "All")
    sel_adm  = st.selectbox("Admission Type", adm_opts,
                             label_visibility="collapsed", key="adm_sel")

    st.markdown('<p style="font-size:0.72rem;font-weight:600;color:#64748b;text-transform:uppercase;letter-spacing:0.1em;margin:0.7rem 0 0.4rem 0">Gender</p>',
                unsafe_allow_html=True)
    st.session_state.setdefault("gen_sel", "All")
    sel_gender = st.selectbox("Gender",["All","Male (M)","Female (F)"],
                               label_visibility="collapsed", key="gen_sel")

    st.markdown('<p style="font-size:0.72rem;font-weight:600;color:#64748b;text-transform:uppercase;letter-spacing:0.1em;margin:0.7rem 0 0.4rem 0">Age Group</p>',
                unsafe_allow_html=True)
    st.session_state.setdefault("age_sel", "All")
    sel_age_grp = st.selectbox("Age Group",
                                ["All","<40","40–50","50–60","60–70","70–80","80+"],
                                label_visibility="collapsed", key="age_sel")

    st.markdown('<hr style="border-color:rgba(255,255,255,0.08);margin:1rem 0 0.7rem 0">', unsafe_allow_html=True)

    count_placeholder = st.empty()

    if st.button("↺  Reset Filters", key="reset_btn", use_container_width=True):
        st.session_state["_reset"] = True
        st.rerun()

    st.markdown('<p style="font-size:0.65rem;color:#334155;margin-top:0.8rem">Filters apply to Hospital Overview tab.</p>',
                unsafe_allow_html=True)

#Filters
dff = df.copy()
_age_bins = {"<40":(0,39),"40–50":(40,49),"50–60":(50,59),
             "60–70":(60,69),"70–80":(70,79),"80+":(80,999)}
if sel_age_grp != "All":
    _lo, _hi = _age_bins[sel_age_grp]
    dff = dff[(dff["age"]>=_lo)&(dff["age"]<=_hi)]
if sel_gender!="All":
    dff = dff[dff["gender"]==("M" if "M" in sel_gender else "F")]
if sel_adm!="All":
    dff = dff[dff["admission_type"]==sel_adm]
if sel_svc!="All":
    dff = dff[dff["curr_service"]==sel_svc]
count_placeholder.markdown(
    f'<p style="font-size:0.82rem;font-weight:700;color:#93c5fd;margin:0">'
    f'Showing: <span style="font-size:1rem;color:#f1f5f9">{len(dff):,}</span> patients</p>',
    unsafe_allow_html=True
)

def is_dimmed(outcome):
    h = st.session_state.get("hl_radio", "All")
    if h == "All": return False
    if outcome == "None": return False
    return outcome != h

def dim_start(outcome): return ""
def dim_end(): return ""

#Header
st.markdown(f"""
<div class="dash-header">
  <div class="header-logo">
    <div class="header-logo-icon">🩺</div>
    <div>
      <div class="header-title">Myocardial Infarction - Patient Admission Analytics</div>
      <div class="header-sub">Clinical Decision Support System &nbsp;:&nbsp; IIT National Hospital</div>
    </div>
  </div>
  <div class="header-right">
    <div class="header-date">{datetime.now().strftime("%d %B %Y")}</div>
    <div class="header-hospital">{datetime.now().strftime("%H:%M")} &nbsp;:&nbsp; IIT National Hospital, Colombo</div>
  </div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs([
    "🏥  HOSPITAL OVERVIEW",
    "📋  PATIENT ADMISSION & PREDICTION",
    "🔂  POST-ADMISSION UPDATE",
])

#TAB 1
with tab1:
    if len(dff)==0:
        st.warning("No data matches the current filters.")
        st.stop()

    k1,k2,k3,k4,k5 = st.columns(5)
    for col,icon,label,val,sub,outcome in [
        (k1,"💊","Total MI Admissions",  f"{len(dff):,}", "13,152 unique patients", "None"),
        (k2,"📆","Average Length of Stay", f"{round(dff['los_days'].mean(),1)} days", "All admissions", "Length of Stay"),
        (k3,"⌛","Length of Stay ≥ 7 Days",    f"{round((dff['los_cat']=='≥ 7 days').mean()*100,1)}%", "Extended stays", "Length of Stay"),
        (k4,"💔","In-Hospital Mortality", f"{round(dff['hospital_expire_flag'].mean()*100,1)}%", "Expired in-hospital", "Mortality"),
        (k5,"🔃","30-Day Readmission",    f"{round(dff['readmit_30d'].mean()*100,1)}%", "Readmitted within 30 day", "Readmission"),
    ]:
        with col:
            _h = st.session_state.get("hl_radio","All")
            _dim = "" if _h=="All" or outcome=="None" or outcome==_h else "opacity:0.12;filter:grayscale(0.4);"
            st.markdown(f"""<div class="kpi-card" style="transition:opacity 0.4s;{_dim}">
              <div class="kpi-icon">{icon}</div>
              <div class="kpi-label">{label}</div>
              <div class="kpi-value">{val}</div>
              <div class="kpi-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Patient Demographics & Trends</div>', unsafe_allow_html=True)
    c1,c2,c3 = st.columns([1.4,2,1.8])

#Gender distribution 
    with c1:
        gc = dff["gender"].value_counts()
        m_c = gc.get("M",0); f_c = gc.get("F",0)
        fig = go.Figure(go.Pie(
            labels=["Male","Female"], values=[m_c,f_c],
            hole=0.65, marker_colors=[BLUE,RED],
            textinfo="percent", textfont=dict(size=13,color="#fff"),
            insidetextorientation="auto",
        ))
        fig.update_layout(
            plot_bgcolor=CHART_BG, paper_bgcolor=CHART_BG,
            font=dict(family="Inter",color=FC),
            margin=dict(l=10,r=10,t=52,b=60),
            title=dict(text="<b>Gender Distribution</b>",
                       font=dict(size=14,color=TITLE_C,family="Inter"),x=0),
            legend=dict(
                orientation="h", y=-0.16, x=0.5, xanchor="center",
                font=dict(size=12,color="#ffffff"),
                bgcolor="rgba(15,45,94,0.88)",
                bordercolor="rgba(255,255,255,0.2)", borderwidth=1,
            ),
        )
        chart(fig)

#MI admission over time
    with c2:
        yearly = dff.groupby("year").size().reset_index(name="count").sort_values("year")
        fig = go.Figure(go.Scatter(
            x=yearly["year"], y=yearly["count"],
            mode="lines",
            line=dict(color=TEAL2, width=2.5),
            fill="tozeroy", fillcolor="rgba(8,145,178,0.08)",
        ))
        bl(fig,"Myocardial Infarction Admissions Over Time")
        fig.update_xaxes(title_text="Year", nticks=8, tickangle=0, showgrid=False)
        fig.update_yaxes(title_text="Admissions")
        chart(fig)

#admission type breakdown
    with c3:
        ac = dff["admission_type"].value_counts().sort_values(ascending=True)
        adm_pal = ["#0f4c81","#1e6091","#2e86c1","#117a8b","#0d9488",
                   "#1d4ed8","#1e3a5f","#4f46e5","#0c4a6e","#164e63"]
        fig = go.Figure(go.Bar(
            x=ac.values, y=ac.index, orientation="h",
            marker=dict(color=adm_pal[:len(ac)]),
            text=ac.values, textposition="outside",
            textfont=dict(size=11,color=TITLE_C),
            cliponaxis=False,
        ))
        bl(fig,"Admission Type Breakdown", margin=dict(l=12,r=90,t=52,b=12))
        fig.update_xaxes(title_text="Patient count", tickfont=dict(color=FC,size=11),
                         showgrid=True, gridcolor="rgba(15,45,94,0.06)")
        fig.update_layout(showlegend=False,
                          yaxis=dict(tickfont=dict(color=TITLE_C,size=9), showgrid=False,
                                     categoryorder="total ascending"))
        chart(fig)

    st.markdown('<div class="section-title">Clinical Outcomes Analysis</div>', unsafe_allow_html=True)
    c4,c5,c6 = st.columns(3)

#Age distribution
    with c4:
        fig = go.Figure(go.Histogram(x=dff["age"],nbinsx=25,marker_color=PURP,opacity=0.85))
        bl(fig,"Age Distribution")
        fig.update_xaxes(title_text="Age", showgrid=False)
        fig.update_yaxes(title_text="Patient count")
        chart(fig)

#LOS category by service
    with c5:
        top_svcs = dff["curr_service"].value_counts().head(6).index
        svc_sub  = dff[dff["curr_service"].isin(top_svcs)]
        if len(svc_sub)>0:
            los_svc = pd.crosstab(svc_sub["curr_service"],svc_sub["los_cat"],normalize="index")*100
            los_clrs = {"< 7 days":"#0d9488","≥ 7 days":"#92400e"}
            fig = go.Figure()
            for col in los_svc.columns:
                fig.add_trace(go.Bar(y=los_svc.index,x=los_svc[col],name=col,
                                     orientation="h",marker_color=los_clrs.get(col,BLUE)))
            fig.update_layout(barmode="stack")
            bl(fig,"Length of Stay Category by Service (%)", margin=dict(l=12,r=12,t=52,b=65))
            fig.update_xaxes(title_text="%", showgrid=True, gridcolor="rgba(15,45,94,0.06)")
            fig.update_yaxes(showgrid=False, tickfont=dict(color=TITLE_C,size=10))
            fig.update_layout(
                legend=dict(orientation="h",y=-0.22,x=0.2,
                            font=dict(size=12,color="#ffffff"),
                            bgcolor="rgba(15,45,94,0.88)",
                            bordercolor="rgba(255,255,255,0.2)",borderwidth=1),
            )
            chart(fig, "Length of Stay")

#Moetality rate by age and gender
    with c6:
        sub_mort = dff.copy()
        sub_mort["age_bin"] = (sub_mort["age"]//5)*5
        ma = sub_mort.groupby(["age_bin","gender"])["hospital_expire_flag"].mean().unstack()*100
        fig = go.Figure()
        for g,color,lbl in [("M",BLUE,"Male"),("F",RED,"Female")]:
            if g in ma.columns:
                fig.add_trace(go.Scatter(
                    x=ma.index, y=ma[g],
                    mode="lines+markers", name=lbl,
                    line=dict(color=color,width=2.5),
                    marker=dict(size=6,color=color,line=dict(color="#ffffff",width=1.5)),
                ))
        bl(fig,"Mortality Rate by Age & Gender")
        fig.update_xaxes(title_text="Age", showgrid=False)
        fig.update_yaxes(title_text="Mortality Rate (%)")
        fig.update_layout(legend=LEG)
        chart(fig, "Mortality")

    c9,c10 = st.columns(2)

#In hospital mortality rate by admission
    with c9:
        mort_adm = dff.groupby("admission_type")["hospital_expire_flag"].mean()*100
        mort_adm = mort_adm.sort_values(ascending=True)
        mort_pal = ["#0369a1","#0891b2","#0d9488","#1d4ed8","#4f46e5",
                    "#7c3aed","#9333ea","#0f4c81","#1e6091","#2e86c1"]
        fig = go.Figure(go.Bar(
            x=mort_adm.values, y=mort_adm.index, orientation="h",
            marker_color=mort_pal[:len(mort_adm)],
            text=[f"{v:.1f}%" for v in mort_adm.values],
            textposition="outside", textfont=dict(size=11,color=TITLE_C),
            cliponaxis=False,
        ))
        bl(fig,"In-Hospital Mortality Rate by Admission Type",
           margin=dict(l=12,r=70,t=52,b=12))
        fig.update_xaxes(title_text="Mortality Rate (%)", showgrid=True,
                         gridcolor="rgba(15,45,94,0.06)")
        fig.update_yaxes(showgrid=False, tickfont=dict(color=TITLE_C,size=9))
        fig.update_layout(showlegend=False)
        chart(fig, "Mortality")

#clincial outcome by service
    with c10:
        top6 = dff["curr_service"].value_counts().head(6).index
        svc_heat = dff[dff["curr_service"].isin(top6)]
        heat_data = svc_heat.groupby("curr_service").agg(
            LOS_long=("los_cat", lambda x: (x=="≥ 7 days").mean()*100),
            Mortality=("hospital_expire_flag","mean"),
            Readmission=("readmit_30d","mean"),
        )
        heat_data["Mortality"]   = heat_data["Mortality"]*100
        heat_data["Readmission"] = heat_data["Readmission"]*100
        z_vals = heat_data[["LOS_long","Mortality","Readmission"]].values
        fig = go.Figure(go.Heatmap(
            z=z_vals.T,
            x=heat_data.index.tolist(),
            y=["LOS ≥7d (%)","Mortality (%)","Readmission (%)"],
            colorscale="Blues",
            text=[[f"{v:.1f}%" for v in row] for row in z_vals.T],
            texttemplate="%{text}",
            textfont=dict(size=12,color=TITLE_C),
            showscale=False,
        ))
        bl(fig,"Clinical Outcomes by Service (%)")
        fig.update_xaxes(tickfont=dict(color=TITLE_C,size=10), showgrid=False)
        fig.update_yaxes(tickfont=dict(color=TITLE_C,size=11), showgrid=False)
        chart(fig)

    st.markdown('<div class="section-title">Hospital Operations</div>', unsafe_allow_html=True)
    op1,op2 = st.columns(2)

#30-day readmission over time
    with op1:
        ry = dff.groupby("year")["readmit_30d"].mean()*100
        fig = go.Figure(go.Scatter(
            x=ry.index, y=ry.values,
            mode="lines",
            line=dict(color=TEAL2,width=2.5),
            fill="tozeroy", fillcolor="rgba(8,145,178,0.08)",
        ))
        bl(fig,"30-Day Readmission Rate Over Time (%)")
        fig.update_xaxes(title_text="Year", nticks=8, showgrid=False)
        fig.update_yaxes(title_text="Readmission Rate (%)")
        chart(fig, "Readmission")

#30-day readmission by age group
    with op2:
        age_read = dff.copy()
        age_read["age_group"] = pd.cut(age_read["age"],
                                        bins=[0,40,50,60,70,80,120],
                                        labels=["<40","40–50","50–60","60–70","70–80","80+"])
        read_age = age_read.groupby("age_group",observed=True)["readmit_30d"].mean()*100
        read_age = read_age.reset_index()
        read_age.columns = ["Age Group","Readmission Rate (%)"]
        age_pal = ["#075985","#1e40af","#5b21b6","#065f46","#92400e","#164e63"]
        fig = go.Figure(go.Bar(
            x=read_age["Age Group"].astype(str),
            y=read_age["Readmission Rate (%)"],
            marker_color=age_pal[:len(read_age)],
            text=[f"{v:.1f}%" for v in read_age["Readmission Rate (%)"]],
            textposition="outside",
            textfont=dict(size=12,color=TITLE_C),
        ))
        bl(fig,"30-Day Readmission Rate by Age Group (%)")
        fig.update_xaxes(title_text="Age Group", showgrid=False)
        fig.update_yaxes(title_text="Readmission Rate (%)",
                         range=[0, read_age["Readmission Rate (%)"].max()*1.3])
        fig.update_layout(showlegend=False)
        chart(fig, "Readmission")

    st.markdown("<div style='height:0.2rem'></div>", unsafe_allow_html=True)
    c7,c8 = st.columns([1.1,1.9])

#Ward occupancy chart 
    with c7:
        st.markdown(
            '<div class="chart-wrap" style="padding:0.8rem 1rem 0.5rem 1rem">'
            '<p style="font-size:14px;font-weight:700;color:#0f2d5e;font-family:Inter,sans-serif;'
            'margin:0 0 0.7rem 0"><b>Ward Occupancy (Est.)</b></p>',
            unsafe_allow_html=True)
        for name,pct,beds,color in [
            ("Cardiac ICU",78,40,"#ef4444"),("Coronory care unit",61,60,"#d97706"),
            ("Emergency",85,30,"#dc2626"),("General Medicine",44,80,"#059669"),
            ("Step-Down Unit",55,50,"#2563eb"),
        ]:
            used = int(beds*pct/100)
            st.markdown(f"""<div class="ward-card">
              <div style="display:flex;justify-content:space-between;align-items:flex-start">
                <div><div class="ward-name">{name}</div><div class="ward-meta">{used}/{beds} beds occupied</div></div>
                <div style="font-size:1.05rem;font-weight:700;color:{color}">{pct}%</div>
              </div>
              <div style="background:rgba(15,45,94,0.08);border-radius:4px;height:6px;margin-top:0.35rem">
                <div style="background:{color};width:{pct}%;height:6px;border-radius:4px"></div>
              </div>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

#Hospital cost chart
    with c8:
        cost_df = pd.DataFrame({
            "LOS":  ["< 7 days"]*4 + ["≥ 7 days"]*4,
            "Type": ["Ward Cost","ICU Cost","Avg Procedure","Total Average"]*2,
            "LKR":  [34000, 51000, 37400, 122400,
                    157000, 235500, 583000, 975500],         
        })
        fig = go.Figure()
        for ct,color in [
            ("Ward Cost",    "#0369a1"),
            ("ICU Cost",     "#0891b2"),
            ("Avg Procedure","#0d9488"),
            ("Total Average","#1e40af"),
        ]:
            sc = cost_df[cost_df["Type"]==ct]
            fig.add_trace(go.Bar(
                x=sc["LOS"], y=sc["LKR"], name=ct,
                marker_color=color,
                text=[f"Rs.{v:,.0f}" for v in sc["LKR"]],
                textposition="outside",
                textfont=dict(size=9,color=TITLE_C),
            ))
        fig.update_layout(barmode="group")
        bl(fig,"Estimated Hospital Cost by LOS Category - LKR (Est.)")
        fig.update_yaxes(title_text="LKR")
        fig.update_layout(
            legend=dict(
                bgcolor="rgba(15,45,94,0.88)",
                bordercolor="rgba(255,255,255,0.2)", borderwidth=1,
                font=dict(size=11,color="#ffffff"),
                orientation="h", x=0.5, y=-0.18,
                xanchor="center", yanchor="top",
            ),
            margin=dict(l=12,r=20,t=52,b=70),
        )
        chart(fig)

#medical staff doctors
    st.markdown('<div class="section-title">Medical Staff</div>', unsafe_allow_html=True)
    dcols = st.columns(5)
    for i,(name,info) in enumerate(DOCTORS.items()):
        with dcols[i%5]:
            avail = st.session_state.doc_avail[name]
            badge = ('<span class="badge-on">● Available</span>'
                     if avail else '<span class="badge-off">● Unavailable</span>')
            st.markdown(f"""<div class="doctor-card">
              <div class="doctor-name">{name}</div>
              <div class="doctor-specialty">{info['spec']}</div>
              <div class="doctor-contact">📞 {info['tel']}</div>
              {badge}
            </div>""", unsafe_allow_html=True)
            lbl = "Set Available" if not avail else "Set Unavailable"
            if st.button(lbl,key=f"doc_{i}",use_container_width=True):
                st.session_state.doc_avail[name] = not avail
                st.rerun()


#TAB 2
with tab2:
    if st.session_state.get("_reset_form", False):
        st.session_state["form_v"] = st.session_state.get("form_v", 0) + 1
        del st.session_state["_reset_form"]

    fv = st.session_state.get("form_v", 0)
    try:
        existing = list(mongo_patients.find({}, {"patient_id": 1, "_id": 0}))
        if existing:
            max_id = max([r["patient_id"] for r in existing if "patient_id" in r])
            next_pid = max_id + 1
        else:
            next_pid = 1001
    except:
        next_pid = max(st.session_state.patients.keys())+1 if st.session_state.patients else 1001

    next_hadm = next_pid + 20000

    id1, id2, _ = st.columns([0.9, 0.9, 3.2])
    with id1:
        st.markdown(f'<div class="next-pid">🪪 &nbsp; Suggested Patient ID: &nbsp;<span>P-{next_pid}</span></div>',
                    unsafe_allow_html=True)
    with id2:
        st.markdown(f'<div class="next-pid">🏷️ &nbsp; Suggested HADM ID: &nbsp;<span>{next_hadm}</span></div>',
                    unsafe_allow_html=True)

    form_col, prev_col = st.columns([1.6, 1])

    with form_col:
        st.markdown('<div class="section-title" style="margin-top:0.3rem">Patient Input Form</div>',
                    unsafe_allow_html=True)
        fc1, fc2 = st.columns(2)
        with fc1:
            patient_id   = st.number_input("Patient ID",  min_value=1, value=next_pid, step=1, key=f"pid_{fv}")
            hadm_id      = st.number_input("HADM ID",     min_value=1, value=next_hadm, step=1, key=f"hadm_{fv}")
            patient_name = st.text_input("Patient Name",  placeholder="e.g. Sehan Herath", key=f"pname_{fv}")
            age_inp      = st.number_input("Age",         min_value=18, max_value=110, value=65, key=f"age_inp_{fv}")
            gender_inp   = st.selectbox("Gender",         ["M","F"], key=f"gender_inp_{fv}")
        with fc2:
            adm_inp  = st.selectbox("Admission Type",     sorted(df["admission_type"].dropna().unique()), key=f"adm_inp_{fv}")
            svc_inp  = st.selectbox("Clinical Service",   mi_svcs_in_data, key=f"svc_inp_{fv}")
            wknd_inp = st.selectbox("Admission Day",      ["Weekday","Weekend"], key=f"wknd_inp_{fv}")
            pmi_inp  = st.selectbox("Prior MI History",   [0,1],
                                     format_func=lambda x:"Yes" if x==1 else "No", key=f"pmi_inp_{fv}")
            ndiag_inp = st.number_input("No. of Diagnoses at Admission",
                                         min_value=0, max_value=50, value=5, key=f"ndiag_inp_{fv}")
        comments_inp = st.text_area("Additional Notes / Comments",
                                     placeholder="e.g. Other details/diagnoses of patient",
                                     height=90, key=f"comments_inp_{fv}")
        save_clicked = st.button("💾  Save Patient Record", key=f"save_btn_{fv}", use_container_width=True)

    with prev_col:
        st.markdown('<div class="section-title" style="margin-top:0.3rem">Live Predictions</div>',
                    unsafe_allow_html=True)
        live_inp = dict(age=age_inp, gender=gender_inp, admission_type=adm_inp,
                        curr_service=svc_inp, admit_weekend=wknd_inp,
                        prior_mi=str(pmi_inp), num_diagnoses_at_admission=ndiag_inp)
        try:
            lp  = enc_pred(los_model,los_enc,CAT_LOS,NUM_LOS,live_inp)
            mp  = enc_pred(mort_model,mort_enc,CAT_MORT,NUM_MORT,live_inp)
            lpd = "≥ 7 days" if lp>=0.5 else "< 7 days"
            mpv = round(mp*100,1)
            lal = predict_los_days(live_inp)
            lc  = "pred-warn" if lpd=="≥7 days" else "pred-good" 
            mc  = "pred-good" if mpv<10 else ("pred-warn" if mpv<25 else "pred-bad")

            cost_per_day  = 10000
            cost_est      = round(lal * cost_per_day / 1000) * 1000
            cost_low      = max(0, round(cost_est * 0.9 / 1000) * 1000)
            cost_high     = round(cost_est * 1.1 / 1000) * 1000
            def fmt_lkr(v): return f"Rs. {v:,.0f}"
            
            #live prediction - LOS
            st.markdown(f"""
            <div class="live-card">
              <div class="live-label">🗓️ Live Length of Stay Estimate</div>
              <div class="{lc}" style="font-family:'Sora',sans-serif;font-size:1.8rem;font-weight:700;margin:0.3rem 0">{lpd}</div>
              <div style="font-size:0.8rem;color:#64748b;margin-top:0.4rem">
                Predicted LOS:&nbsp;
                <span style="font-size:1rem;font-weight:700;color:#0f2d5e">{lal} days</span>
              </div>
              <hr style="border:none;border-top:1px solid rgba(15,45,94,0.08);margin:0.6rem 0">
              <div style="font-size:0.62rem;color:#2563eb;font-weight:700;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.3rem">💵 Estimated Ward Cost</div>
              <div style="font-family:'Sora',sans-serif;font-size:1.3rem;font-weight:700;color:#0f2d5e">{fmt_lkr(cost_low)} – {fmt_lkr(cost_high)}</div>
              <div style="font-size:0.68rem;color:#94a3b8;margin-top:0.2rem">Based on {lal} days × LKR 10,000/day - ward room only</div>
            </div>""", unsafe_allow_html=True)

            st.markdown(f"""
            <div class="live-card">
              <div class="live-label">💔 Live Mortality Risk Estimate</div>
              <div class="{mc}" style="font-family:'Sora',sans-serif;font-size:2.2rem;font-weight:700;margin:0.3rem 0">{mpv}%</div>
            </div>""", unsafe_allow_html=True)

            fig_live = make_gauge(mpv,"Mortality Risk")
            fig_live.update_layout(height=200,margin=dict(l=20,r=20,t=45,b=5))
            st.plotly_chart(fig_live,use_container_width=True)

            _pmi = int(pmi_inp)
            sim = df[
                (df["age"].between(age_inp-10,age_inp+10)) &
                (df["gender"]==gender_inp) &
                (df["admission_type"]==adm_inp) &
                (df["curr_service"]==svc_inp) &
                (df["admit_weekend"]==wknd_inp) &
                (df["prior_mi"].apply(lambda x: 1 if str(x).strip() in ["1","1.0","Yes","yes","Y","y"] else 0)==_pmi) &
                (df["num_diagnoses_at_admission"].between(
                    max(0,ndiag_inp-5), ndiag_inp+5))
            ]
            if len(sim) < 10:
                sim = df[
                    (df["age"].between(age_inp-10,age_inp+10)) &
                    (df["gender"]==gender_inp) &
                    (df["admission_type"]==adm_inp) &
                    (df["curr_service"]==svc_inp)
                ]
            if len(sim) < 10:
                sim = df[(df["age"].between(age_inp-10,age_inp+10)) & (df["gender"]==gender_inp)]
            ld  = sim["los_cat"].value_counts()
            n_sim = len(sim)
            
        #LOS pattern - similar patient pie chart    
            fig_s = go.Figure(go.Pie(
                labels=ld.index.tolist(), values=ld.values.tolist(),
                hole=0.58,
                marker_colors=["#0d9488","#92400e"],
                textinfo="percent+label",
                textfont=dict(size=12,color="#0f2d5e",family="Inter"),
                insidetextorientation="auto",
                pull=[0.03,0.03],
            ))
            fig_s.update_layout(
                plot_bgcolor=CHART_BG, paper_bgcolor=CHART_BG,
                font=dict(color=FC,family="Inter"),
                margin=dict(l=10,r=10,t=65,b=55),
                height=300,
                title=dict(
                    text=f"<b>LOS Pattern - Similar Patients</b><br>"
                         f"<span style='font-size:11px;color:{FC}'>"
                         f"{n_sim:,} patients - same age ±10, gender, admission & service</span>",
                    font=dict(size=12,color=TITLE_C,family="Inter"), x=0,
                ),
                legend=dict(
                    orientation="h", y=-0.14, x=0.5, xanchor="center",
                    font=dict(size=12,color="#ffffff"),
                    bgcolor="rgba(15,45,94,0.88)",
                    bordercolor="rgba(255,255,255,0.2)", borderwidth=1,
                ),
                annotations=[dict(
                    text=f"<b>{n_sim:,}</b><br><span style='font-size:10px'>patients</span>",
                    x=0.5, y=0.5, font=dict(size=13,color=TITLE_C),
                    showarrow=False,
                )],
            )
            st.plotly_chart(fig_s, use_container_width=True)
        except Exception:
            st.info("Fill in the form to see live predictions.")

    if save_clicked:
        if not patient_name.strip():
            st.error("Please enter a patient name.")
        elif patient_id in st.session_state.patients:
            st.warning(f"Patient ID {patient_id} already exists.")
        else:
            si = dict(age=age_inp, gender=gender_inp, admission_type=adm_inp,
                      curr_service=svc_inp, admit_weekend=wknd_inp,
                      prior_mi=str(pmi_inp), num_diagnoses_at_admission=ndiag_inp)
            lp2 = enc_pred(los_model,los_enc,CAT_LOS,NUM_LOS,si)
            mp2 = enc_pred(mort_model,mort_enc,CAT_MORT,NUM_MORT,si)
            st.session_state.patients[patient_id] = dict(
                patient_id=patient_id, hadm_id=hadm_id, name=patient_name,
                comments=comments_inp,
                saved_at=datetime.now().isoformat(), **si,
                los_prediction="≥ 7 days" if lp2>=0.5 else "< 7 days",
                los_prob=round(lp2,4), predicted_los_days=predict_los_days(si),
                mortality_risk_admission=round(mp2*100,2),
                procedure_count=None, drg_severity=None, drg_mortality=None,
                cardiac_proc_flag=None, discharge_location=None, los_days=None,
                updated_mortality_risk=None,
                readmission_prediction=None, readmission_prob=None,
                avg_days_readmission=None, updated_at=None,
            )
            mongo_patients.insert_one({**st.session_state.patients[patient_id]})
            st.success(f"✅  Patient {patient_name} (ID: P-{patient_id} | HADM: {hadm_id}) saved.")
            st.session_state["_reset_form"] = True
            st.rerun()

    if st.session_state.patients:
        st.markdown('<div class="section-title">Recent Admissions</div>', unsafe_allow_html=True)
        rows = [{"Patient ID": f"P-{r['patient_id']}",
                 "HADM ID":    r.get("hadm_id","—"),
                 "Name":       r["name"],
                 "Age":        r["age"],
                 "Gender":     r["gender"],
                 "LOS Pred":   r["los_prediction"],
                 "Admission Mortality %":  f"{r['mortality_risk_admission']}%",
                 "Saved":      datetime.fromisoformat(r["saved_at"]).strftime("%d %b %Y %H:%M")}
                for r in list(st.session_state.patients.values())[-10:]]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.markdown('<div class="section-title">All Saved Patient Records (MongoDB)</div>', unsafe_allow_html=True)
        try:
            all_records = list(mongo_patients.find({}, {"_id": 0}))
            if all_records:
                records_df = pd.DataFrame(all_records)[[
                    "patient_id", "hadm_id", "name", "age", "gender",
                    "los_prediction", "mortality_risk_admission",
                    "readmission_prediction", "saved_at"
                ]]
                records_df.columns = [
                    "Patient ID", "HADM ID", "Name", "Age", "Gender",
                    "LOS Pred", "Admission Mortality %", "Readmission Risk", "Saved"
                ]
                records_df["Saved"] = pd.to_datetime(records_df["Saved"]).dt.strftime("%d %b %Y %H:%M")
                records_df["Admission Mortality %"] = records_df["Admission Mortality %"].apply(lambda x: f"{x}%" if pd.notna(x) else "—")
                st.dataframe(records_df, use_container_width=True, hide_index=True)
            else:
                st.info("No records saved yet.")
        except Exception as e:
            st.warning("Could not load records from database")


#TAB 3
with tab3:
    st.markdown('<div class="section-title">Search Patient Record</div>', unsafe_allow_html=True)
    st.caption("Enter the Patient ID saved in the Admission tab to load their record.")
    search_id = st.number_input("Enter Patient ID", min_value=1, value=1001, step=1, key="search_pid")
    sb1, sb2 = st.columns([1.6, 1])
    with sb1:
        search_clicked = st.button("🔍  Load Patient Record", key="search_btn", use_container_width=True)
    with sb2:
        clear_clicked  = st.button("✖️ Clear", key="clear_btn", use_container_width=True)

    if clear_clicked:
        if "loaded_pid" in st.session_state:
            del st.session_state["loaded_pid"]
        st.rerun()

#search logic in tab 3    
    if search_clicked:
        if search_id in st.session_state.patients:
            st.session_state["loaded_pid"] = search_id
        else: #trying mongo db if not in the current session state
            mongo_rec = mongo_patients.find_one({"patient_id": search_id})
            if mongo_rec:
                mongo_rec.pop("_id", None)
                st.session_state.patients[search_id] = mongo_rec
                st.session_state["loaded_pid"] = search_id
            else:
                st.error(f"No patient found with ID {search_id}.")

    loaded_pid = st.session_state.get("loaded_pid", None)

    if loaded_pid and loaded_pid in st.session_state.patients:
        rec = st.session_state.patients[loaded_pid]

        st.markdown('<div class="section-title">Admission Summary</div>', unsafe_allow_html=True)
        sc1,sc2,sc3,sc4,sc5,sc6 = st.columns(6)
        sc1.metric("Patient Name",   rec["name"])
        sc2.metric("Patient ID",     f"P-{rec['patient_id']}")
        sc3.metric("HADM ID",        rec.get("hadm_id","—"))
        sc4.metric("Age / Gender",   f"{rec['age']} / {rec['gender']}")
        sc5.metric("LOS Prediction", rec["los_prediction"])
        sc6.metric("Admission Mortality", f"{rec['mortality_risk_admission']}%")
        st.markdown("""<style>
        [data-testid="stMetricValue"]{font-size:1.15rem !important;}
        </style>""", unsafe_allow_html=True)

        st.markdown('<div class="section-title">Admission Details</div>', unsafe_allow_html=True)
        ad1,ad2,ad3,ad4,ad5 = st.columns(5)
        for _col, _lbl, _val in [
            (ad1,"Admission Type",   rec["admission_type"]),
            (ad2,"Clinical Service", rec["curr_service"]),
            (ad3,"Admission Day",    rec["admit_weekend"]),
            (ad4,"Prior MI",         "Yes" if str(rec["prior_mi"]) in ["1","1.0","Yes","yes","Y"] else "No"),
            (ad5,"No. of Diagnoses", str(rec["num_diagnoses_at_admission"])),
        ]:
            val_size = "0.88rem" if _lbl == "Admission Type" else "1.1rem"
            _col.markdown(f"""<div style="background:#ffffff;border-radius:12px;padding:0.8rem 1rem;
                border:1px solid rgba(15,45,94,0.07);box-shadow:0 1px 6px rgba(15,45,94,0.06)">
                <div style="font-size:0.68rem;color:#7a8ba8;font-weight:600;
                text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.3rem">{_lbl}</div>
                <div style="font-size:{val_size};font-weight:700;color:#0f2d5e;
                word-break:break-word">{_val}</div>
            </div>""", unsafe_allow_html=True)

        #Post admission variables
        st.markdown('<div class="section-title">Post-Admission Variables</div>', unsafe_allow_html=True)
        st.caption("These variables are collected after admission. All are used for updated mortality and readmission predictions.")

        with st.form("post_form"):
            pc1,pc2,pc3,pc4 = st.columns(4)
            with pc1:
                proc  = st.number_input("Procedure Count",
                    help="Total number of procedures performed during admission",
                    min_value=0, max_value=30, value=1)
            with pc2:
                drgs  = st.number_input("DRG Severity (1–4)",
                    help="APR-DRG severity score: 1=Minor, 2=Moderate, 3=Major, 4=Extreme",
                    min_value=0, max_value=4, value=2)
            with pc3:
                drgm  = st.number_input("DRG Mortality (1–4)",
                    help="APR-DRG mortality risk: 1=Minor, 2=Moderate, 3=Major, 4=Extreme",
                    min_value=0, max_value=4, value=2)
            with pc4:
                cardiac_procs = st.multiselect(
                    "Cardiac Procedure(s) Performed",
                    ["None",
                     "Coronary Angioplasty (PCI)",
                     "Stent Placement",
                     "Bypass Surgery (CABG)",
                     "Other Coronary Procedure"],
                    help="Select all that apply. If any cardiac procedure was performed, flag = 1"
                )
                cflag = 0 if (not cardiac_procs or cardiac_procs == ["None"]) else 1
                st.caption(f"Cardiac procedure flag: {'1 - Yes' if cflag==1 else '0 - None'}")

            #Discharge Location & Actual LOS
            st.markdown("<div style='height:0.3rem'></div>", unsafe_allow_html=True)
            dl1, dl2 = st.columns(2)
            with dl1:
                discharge_loc = st.selectbox(
                    "Discharge Location",
                    DISCHARGE_LOCATIONS,
                    index=0,
                    help="Where the patient is being discharged to. Used in the readmission risk model."
                )
            with dl2:
                #computing days since the patient was saved/admitted
                _admitted_str = rec.get("saved_at", None)
                _days_elapsed = None
                if _admitted_str:
                    try:
                        _admitted_dt = datetime.fromisoformat(_admitted_str)
                        _days_elapsed = (datetime.now() - _admitted_dt).days
                    except Exception:
                        _days_elapsed = None

                #building label with inline suggestion
                _suggestion_text = ""
                if _days_elapsed is not None:
                    _suggestion_text = f"  ·  💡 Suggested: **{_days_elapsed} day{'s' if _days_elapsed != 1 else ''}** (since admission)"

                los_days_inp = st.number_input(
                    "Actual Length of Stay (days)",
                    min_value=0.0, max_value=365.0,
                    value=float(rec.get("predicted_los_days") or
                                (7.0 if rec["los_prediction"]=="≥ 7 days" else 3.5)),
                    step=0.5,
                    help="Actual LOS in days at time of discharge. Pre-filled from LOS model estimate - override if known."
                )
                if _suggestion_text:
                    st.caption(_suggestion_text)
                else:
                    st.caption(f"Model predicted: {rec['los_prediction']} · Predicted LOS: {rec.get('predicted_los_days','—')} days")
            
            upd = st.form_submit_button("🔄  Update & Predict", use_container_width=True)

        if upd:
            
            post = dict(
                age=rec["age"],
                gender=rec["gender"],
                admission_type=rec["admission_type"],
                curr_service=rec["curr_service"],
                admit_weekend=rec["admit_weekend"],
                prior_mi=to_prior_mi_yn(rec["prior_mi"]), 
                num_diagnoses_at_admission=rec["num_diagnoses_at_admission"],
                procedure_count=proc,
                drg_severity=drgs,
                drg_mortality=drgm,
            )
            
            ri = dict(
                age=rec["age"],
                gender=rec["gender"],
                admission_type=rec["admission_type"],
                curr_service=rec["curr_service"],
                discharge_location=discharge_loc,
                admit_weekend=rec["admit_weekend"],
                prior_mi=to_prior_mi_yn(rec["prior_mi"]),
                num_diagnoses_at_admission=rec["num_diagnoses_at_admission"],
                los_days=float(los_days_inp),
                procedure_count=proc,
                drg_severity=drgs,
                drg_mortality=drgm,
                cardiac_proc_flag=cflag,
            )

            m2p = enc_pred(mort2_model, mort2_enc, CAT_MORT2, NUM_MORT2, post)
            rap = enc_pred_readmit(readmit_model, readmit_enc, ri)
            rpd = "High Risk" if rap >= readmit_thresh else "Low Risk"
            m2v = round(m2p*100,1)
            rv  = round(rap*100,1)

            rec.update(dict(
                procedure_count=proc, drg_severity=drgs, drg_mortality=drgm,
                cardiac_proc_flag=cflag,
                discharge_location=discharge_loc,
                los_days=float(los_days_inp),
                updated_mortality_risk=round(m2p*100,2),
                readmission_prediction=rpd,
                readmission_prob=round(rap*100,2),
                avg_days_readmission=avg_read_sim(post),
                updated_at=datetime.now().isoformat(),
            ))

            #updating Tab 3 after update and predict
            st.session_state.patients[loaded_pid] = rec   
            mongo_patients.update_one(
                {"patient_id": loaded_pid},
                {"$set": rec},
                upsert=True
            )
            st.success("✅  Post-admission record updated successfully.")

    #Updated predictions - post admission mortality
            st.markdown('<div class="section-title">Updated Predictions</div>', unsafe_allow_html=True)
            u1,u2,u3 = st.columns(3)
            m2c = "pred-good" if m2v<10 else ("pred-warn" if m2v<25 else "pred-bad")
            rc2 = "pred-bad"  if rpd=="High Risk" else "pred-good"
            with u1:
                st.markdown(f'''<div class="pred-card">
                  <div class="pred-label">Admission Mortality Risk</div>
                  <div class="pred-warn">{rec["mortality_risk_admission"]}%</div>
                  <div class="pred-sub">At time of admission</div>
                </div>''', unsafe_allow_html=True)
            with u2:
                st.markdown(f'''<div class="pred-card">
                  <div class="pred-label">Updated Mortality Risk</div>
                  <div class="{m2c}">{m2v}%</div>
                  <div class="pred-sub">Post-procedure</div>
                </div>''', unsafe_allow_html=True)
            with u3:
                st.markdown(f'''<div class="pred-card">
                  <div class="pred-label">30-Day Readmission Risk</div>
                  <div class="{rc2}">{rv}% - {rpd}</div>
                  <div class="pred-sub">Avg similar: <b style="color:#1e2a3a">{rec["avg_days_readmission"]} days to readmit</b></div>
                </div>''', unsafe_allow_html=True)

            st.markdown('<div class="section-title">Mortality Risk - Admission vs Post-Procedure</div>',
                        unsafe_allow_html=True)
            gg1,gg2 = st.columns(2)
            with gg1:
                st.plotly_chart(make_gauge(float(rec["mortality_risk_admission"]),"At Admission"),
                                use_container_width=True)
            with gg2:
                st.plotly_chart(make_gauge(m2v,"Post-Procedure"),use_container_width=True)


