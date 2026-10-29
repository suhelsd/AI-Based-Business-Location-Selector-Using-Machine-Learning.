import streamlit as st, pandas as pd, numpy as np
st.set_page_config(page_title="AI Business Location Selector", layout="wide")
st.title("AI-Based Business Location Selector – Café, Bhopal")
st.warning("Demo uses SYNTHETIC data. Scores are decision support, not a guarantee of profit.")

@st.cache_data
def load(): return pd.read_csv("outputs/ranked_locations.csv")
df = load()

with st.sidebar:
    st.header("Inputs")
    budget = st.slider("Monthly rent budget (₹)", 15000, 90000, 40000, 1000)
    areas = st.multiselect("Areas", sorted(df.area.unique()), default=sorted(df.area.unique()))
    st.subheader("Weights (auto-normalised)")
    names = {"D":"Demand (predicted sales)","A":"Accessibility","F":"Footfall","P":"Parking/Transport","C":"Competition opportunity","R":"Rent affordability"}
    defaults = dict(D=30, A=20, F=15, P=15, C=10, R=10)
    w = {k: st.slider(v, 0, 100, defaults[k]) for k, v in names.items()}
    tot = sum(w.values()) or 1

d = df[df.area.isin(areas)].copy()
d["s_R"] = (100*(1 - d.monthly_rent/budget)).clip(0, 100)
d["score"] = sum(d[f"s_{k}"]*w[k]/tot for k in names)
d = d.sort_values("score", ascending=False)
top = d.head(5)

c = st.columns(4)
c[0].metric("Business", "Café"); c[1].metric("Budget", f"₹{budget:,}/mo")
c[2].metric("Candidates", len(d)); c[3].metric("Top result", top.iloc[0].location_name if len(top) else "-")

if len(top):
    l, r = st.columns(2)
    with l:
        st.subheader("Top-5 ranking")
        st.dataframe(top[["location_name","monthly_rent","predicted_sales","daily_footfall","competitor_count","score"]].round(1), hide_index=True)
    with r:
        st.subheader("Score components (0–100)")
        st.bar_chart(top.set_index("location_name")[[f"s_{k}" for k in names]].rename(columns=lambda x: names[x[2:]]))
    st.subheader("Explanation")
    for _, row in top.iterrows():
        comps = {names[k]: row[f"s_{k}"] for k in names}
        best, worst = max(comps, key=comps.get), min(comps, key=comps.get)
        st.markdown(f"**{row.location_name}** – score {row.score:.1f}. Strongest: *{best}* ({comps[best]:.0f}); weakest trade-off: *{worst}* ({comps[worst]:.0f}).")
    st.subheader("Clusters of similar locations")
    st.scatter_chart(d, x="daily_footfall", y="monthly_rent", color="cluster", size=20)
    st.subheader("Comparison table & export")
    st.dataframe(d.head(50).round(1), hide_index=True)
    st.download_button("Download ranking (CSV)", d.to_csv(index=False), "ranking.csv", "text/csv")
else:
    st.info("Select at least one area.")
st.caption("Limitations: synthetic data; predicted sales come from a model trained on synthetic labels; no real coordinates/map.")
