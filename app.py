import streamlit as st, pandas as pd, numpy as np, joblib, json, os
st.set_page_config(page_title="Prediksi Churn Pelanggan Telco", page_icon="📡", layout="wide")
BASE = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load():
    return joblib.load(os.path.join(BASE, "models", "churn_model.joblib"))

@st.cache_data
def load_results():
    return json.load(open(os.path.join(BASE, "results.json")))

art, R = load(), load_results()
model, th = art["model"], art["threshold"]

def build(d):
    df = pd.DataFrame([d])
    svc = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
    df["NumServices"] = sum((df[c] == "Yes").astype(int) for c in svc)
    df["AvgCharge"] = df.TotalCharges / df.tenure.clip(lower=1)
    df["ChargeDiff"] = df.MonthlyCharges - df.AvgCharge
    df["TenureGroup"] = pd.cut(df.tenure, [-1, 12, 24, 48, 72], labels=["0-12", "13-24", "25-48", "49-72"]).astype(str)
    df["AutoPay"] = df.PaymentMethod.str.contains("automatic").astype(int)
    return df

def score(df):
    return model.predict_proba(df)[:, 1]

st.title("📡 Prediksi Churn Pelanggan Telekomunikasi")
st.caption(f"Model: **{art['name']}** · threshold keputusan = {th:.2f} · Dataset: Telco Customer Churn (IBM/Kaggle)")
tab1, tab2, tab3 = st.tabs(["🔮 Prediksi Individu", "📂 Prediksi Batch (CSV)", "📊 Performa Model"])

with tab1:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Profil")
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Ya" if x else "Tidak")
        partner = st.selectbox("Punya Pasangan", ["Yes", "No"])
        dep = st.selectbox("Punya Tanggungan", ["Yes", "No"])
        tenure = st.slider("Lama berlangganan (bulan)", 0, 72, 12)
    with c2:
        st.subheader("Layanan")
        phone = st.selectbox("Layanan Telepon", ["Yes", "No"])
        ml = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        inet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        opts = ["Yes", "No"] if inet != "No" else ["No internet service"]
        osec = st.selectbox("Online Security", opts); obk = st.selectbox("Online Backup", opts)
        dp = st.selectbox("Device Protection", opts); ts = st.selectbox("Tech Support", opts)
        stv = st.selectbox("Streaming TV", opts); smv = st.selectbox("Streaming Movies", opts)
    with c3:
        st.subheader("Kontrak & Tagihan")
        contract = st.selectbox("Kontrak", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        pay = st.selectbox("Metode Bayar", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        monthly = st.number_input("Tagihan Bulanan (USD)", 18.0, 120.0, 70.0, 0.5)
        total = st.number_input("Total Tagihan (USD)", 0.0, 9000.0, float(round(monthly * max(tenure, 1), 2)), 10.0)
    if st.button("Prediksi", type="primary"):
        row = dict(gender=gender, SeniorCitizen=senior, Partner=partner, Dependents=dep, tenure=tenure, PhoneService=phone,
                   MultipleLines=ml, InternetService=inet, OnlineSecurity=osec, OnlineBackup=obk, DeviceProtection=dp,
                   TechSupport=ts, StreamingTV=stv, StreamingMovies=smv, Contract=contract, PaperlessBilling=paperless,
                   PaymentMethod=pay, MonthlyCharges=monthly, TotalCharges=total)
        p = float(score(build(row))[0])
        a, b = st.columns([1, 2])
        a.metric("Probabilitas Churn", f"{p:.1%}")
        a.progress(min(p, 1.0))
        if p >= th:
            b.error("⚠️ **Berisiko tinggi churn.** Prioritaskan retensi: tawarkan kontrak jangka panjang, diskon, atau paket layanan tambahan.")
        else:
            b.success("✅ **Risiko churn rendah.** Pertahankan layanan dan pantau secara berkala.")

with tab2:
    st.write("Unggah CSV dengan kolom yang sama seperti dataset Telco (tanpa kolom `Churn`).")
    f = st.file_uploader("File CSV", type="csv")
    if f:
        d = pd.read_csv(f)
        d["TotalCharges"] = pd.to_numeric(d["TotalCharges"], errors="coerce").fillna(0)
        X = d.drop(columns=[c for c in ["customerID", "Churn"] if c in d.columns])
        feats = build_df = None
        svc = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
        X["NumServices"] = sum((X[c] == "Yes").astype(int) for c in svc)
        X["AvgCharge"] = X.TotalCharges / X.tenure.clip(lower=1)
        X["ChargeDiff"] = X.MonthlyCharges - X.AvgCharge
        X["TenureGroup"] = pd.cut(X.tenure, [-1, 12, 24, 48, 72], labels=["0-12", "13-24", "25-48", "49-72"]).astype(str)
        X["AutoPay"] = X.PaymentMethod.str.contains("automatic").astype(int)
        d["Prob_Churn"] = score(X); d["Prediksi"] = np.where(d.Prob_Churn >= th, "Churn", "Tidak Churn")
        st.dataframe(d.sort_values("Prob_Churn", ascending=False), use_container_width=True)
        st.download_button("Unduh hasil", d.to_csv(index=False).encode(), "hasil_prediksi.csv", "text/csv")

with tab3:
    m = pd.DataFrame(R["models"]).T[["cv_f1", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]].round(3)
    st.subheader("Perbandingan Model (Data Uji, threshold 0.5)"); st.dataframe(m, use_container_width=True)
    st.subheader(f"Model Final: {R['winner']} (threshold {R['threshold']:.2f})")
    st.json({k: round(v, 3) for k, v in R["final_test"].items()})
    cols = st.columns(3)
    for c, img in zip(cols, ["roc.png", "cm.png", "imp.png"]):
        p = os.path.join(BASE, "figures", img)
        if os.path.exists(p): c.image(p)
