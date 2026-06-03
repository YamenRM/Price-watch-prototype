import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="Gaza Price Watch - ML Anomaly Engine", page_icon="📈", layout="wide")

# Title
st.title("نظام التحقق")
st.markdown("---")

# 1. HELPER: Generate a realistic, volatile dataset history
@st.cache_data(ttl=60) # Caches data for 1 minute so it doesn't refresh on every click
def generate_market_history(base_price, volatility=0.15, outliers_count=3):
    # Generate 30 normal reports using a Gaussian distribution
    normal_reports = np.random.normal(loc=base_price, scale=base_price * volatility, size=30)
    
    # Inject a few intentional high/low anomalies (trolls/scalpers)
    high_outliers = np.random.uniform(base_price * 1.8, base_price * 2.2, size=outliers_count)
    low_outliers = np.random.uniform(base_price * 0.1, base_price * 0.3, size=outliers_count // 2)
    
    all_reports = np.concatenate([normal_reports, high_outliers, low_outliers])
    np.random.shuffle(all_reports) # Mix them up chronologically
    return np.clip(all_reports, 0.1, None) # Ensure no negative prices

# 2. Select Product Configuration
products = {
    "طحين - 25 كغ": 60.0,
    "زيت طهي - 1 لتر": 17.0,
    "أرز - 1 كغ": 10.0,
}

col_setup, col_visuals = st.columns([1, 2])

with col_setup:
    st.header("معطيات السوق")
    selected_prod = st.selectbox("اختار المنتج", list(products.keys()))
    user_status = st.selectbox("حالة المستخدم", ["Guest Profile", "Registered User", "Vetted Trusted User"])
    
    # Generate history based on selection
    base = products[selected_prod]
    history = generate_market_history(base)
    
    # Calculate Real-time Metrics from our "Self-Made Dataset"
    market_mean = np.mean(history)
    market_std = np.std(history)
    
    st.metric("المتوسط ($\mu$)", f"{market_mean:.2f} $")
    st.metric("الانحراف المعياري ($\sigma$)", f"{market_std:.2f} $")
    
    st.markdown("---")
    reported_price = st.number_input("ادخل سعر جديد للتجربة", min_value=0.0, value=float(market_mean), step=0.5)

with col_visuals:
    st.header("تمثيل البيانات")
    
    # Plot the histogram of our self-made dataset
    df = pd.DataFrame(history, columns=["السعر"])
    st.bar_chart(df["السعر"].value_counts().sort_index())
    st.caption("تمثيل لاخر 55 إدخالاً.")

# 3. ML / Statistical Verification Logic
if st.button("مقارنة السعر المبلغ عنه مع بيانات السوق", type="primary"):
    st.markdown("### المقارنة")
    
    # Calculate Z-score
    z_score = (reported_price - market_mean) / market_std
    st.write(f"• Calculated Z-Score: `{z_score:.2f}`")
    
    # Threshold rules based on standard deviations
    if abs(z_score) <= 0.5:
        st.success("✅ .مقبول** السعر ضمن النطاق الطبيعي للتقلبات السوقية**")
    elif abs(z_score) <= 1 and user_status == "Vetted Trusted User":
        st.warning("⚠️. مستخدم موثوق** السعر خارج النطاق الطبيعي، لكن المستخدم موثوق. سيتم اعتماد السعر مع ملاحظة أنه قد يكون غير دقيق**")
    else:
        st.error("⏳ في انتظار التحقق** اذا تم التصويت من قبل 3 مستخدمين اخرين سيتم اعتماد السعر**")