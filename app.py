import pandas as pd, plotly.express as px, streamlit as st
from pathlib import Path

st.set_page_config(page_title="Interactive BI Dashboard", page_icon="📊", layout="wide")
D = Path(__file__).parent / "data"

@st.cache_data
def load():
    web = pd.read_csv(D / "web_traffic.csv", parse_dates=["Date"])
    web["Registered"] = web.Registered.eq("Yes"); web["Bounced"] = web.Bounced.eq("Yes")
    cus = pd.read_csv(D / "customer_churn.csv").drop_duplicates()          # 5 duplicate rows removed
    cus["Region"] = cus.Region.fillna("Unknown")                            # 13 missing regions
    cus["TotalChargesINR"] = pd.to_numeric(cus.TotalChargesINR, errors="coerce")  # 'unknown' -> NaN
    cus["Churned"] = cus.Churn.eq("Yes")
    mkt = pd.read_csv(D / "regional_market.csv")
    ab = pd.read_csv(D / "checkout_ab_test.csv", parse_dates=["Timestamp"])
    ab["Date"] = ab.Timestamp.dt.normalize(); ab["Order"] = ab.Converted.eq("Yes")
    return web, cus, mkt, ab

web, cus, mkt, ab = load()

st.title("📊 Interactive Business Intelligence Dashboard")
st.caption("Synthetic learning data (Workora data pack) - not real business results.")

# ---------- Filters (slicers)
sb = st.sidebar; sb.header("Filters")
dmin, dmax = web.Date.min().date(), web.Date.max().date()
dr = sb.date_input("Date range (web + checkout)", (dmin, dmax), min_value=dmin, max_value=dmax)
d1, d2 = (dr if len(dr) == 2 else (dmin, dmax))
device = sb.multiselect("Device (web + checkout)", sorted(ab.Device.unique()), default=sorted(ab.Device.unique()))
region = sb.multiselect("Region (customers)", sorted(cus.Region.unique()), default=sorted(cus.Region.unique()))
plan = sb.multiselect("Category: subscription plan (customers)", sorted(cus.SubscriptionType.unique()), default=sorted(cus.SubscriptionType.unique()))
state = sb.multiselect("State (market map)", sorted(mkt.State.unique()), default=sorted(mkt.State.unique()))

dt = lambda s: (s >= pd.Timestamp(d1)) & (s <= pd.Timestamp(d2))
W = web[dt(web.Date) & web.DeviceCategory.isin(device)]
A = ab[dt(ab.Date) & ab.Device.isin(device)]
C = cus[cus.Region.isin(region) & cus.SubscriptionType.isin(plan)]
M = mkt[mkt.State.isin(state)]

# ---------- KPI cards
rev, orders = A.OrderValueINR.sum(), int(A.Order.sum())
k = st.columns(4)
k[0].metric("Total Revenue", f"₹{rev:,.0f}", help="Sum of converted checkout orders (data covers Jul 1-22, 2026)")
k[1].metric("Active Users", f"{int((~C.Churned).sum()):,}", help="Customers with Churn = No")
k[2].metric("Churn %", f"{C.Churned.mean():.1%}" if len(C) else "-")
k[3].metric("Avg Ticket Size", f"₹{rev / orders:,.0f}" if orders else "-", help="Revenue / converted orders")

t1, t2, t3, t4 = st.tabs(["Sales & Checkout", "Retention", "Web Traffic", "Regional Market"])

with t1:
    c1, c2 = st.columns(2)
    daily = A.groupby("Date", as_index=False).OrderValueINR.sum()
    c1.plotly_chart(px.line(daily, x="Date", y="OrderValueINR", markers=True, title="Daily revenue (INR)"), width='stretch', key='pc1')
    v = A.groupby("Variant").agg(Visitors=("VisitorID", "count"), Orders=("Order", "sum"), Revenue=("OrderValueINR", "sum")).reset_index()
    v["Conversion %"] = (v.Orders / v.Visitors * 100).round(2)
    c2.plotly_chart(px.bar(v, x="Variant", y="Conversion %", color="Variant", text="Conversion %", title="Checkout conversion by variant"), width='stretch', key='pc2')
    st.dataframe(v, hide_index=True, width='stretch')

with t2:
    c1, c2 = st.columns(2)
    by = lambda col: C.groupby(col).Churned.agg(["mean", "size"]).reset_index().rename(columns={"mean": "Churn %", "size": "Customers"})
    ct = by("ContractType"); ct["Churn %"] *= 100
    c1.plotly_chart(px.bar(ct, x="ContractType", y="Churn %", text=ct["Churn %"].round(1), title="Churn % by contract type"), width='stretch', key='pc3')
    pl = by("SubscriptionType"); pl["Churn %"] *= 100
    c2.plotly_chart(px.bar(pl, x="SubscriptionType", y="Churn %", text=pl["Churn %"].round(1), title="Churn % by plan"), width='stretch', key='pc4')
    st.caption("Extreme MonthlyCharges values (~6 customers) are kept for churn counts but not used in any metric.")

with t3:
    c1, c2 = st.columns(2)
    wk = W.groupby(W.Date.dt.to_period("W").dt.start_time).agg(Sessions=("SessionID", "count"), Registered=("Registered", "sum")).reset_index()
    c1.plotly_chart(px.line(wk.melt("Date"), x="Date", y="value", color="variable", title="Weekly sessions vs registrations"), width='stretch', key='pc5')
    src = W.groupby("Source").agg(Sessions=("SessionID", "count"), Reg=("Registered", "mean"), Bounce=("Bounced", "mean")).reset_index()
    src["Registration %"] = (src.Reg * 100).round(1)
    c2.plotly_chart(px.bar(src, x="Source", y="Registration %", text="Registration %", title="Registration rate by source"), width='stretch', key='pc6')
    w = st.columns(3); w[0].metric("Sessions", f"{len(W):,}"); w[1].metric("Bounce rate", f"{W.Bounced.mean():.1%}" if len(W) else "-"); w[2].metric("Avg session", f"{W.SessionDurationSec.mean():.0f} sec" if len(W) else "-")

with t4:
    fig = px.scatter_map(M, lat="Latitude", lon="Longitude", color="LeadConversionRate", size="MonthlySearchDemand", hover_name="City",
                         hover_data=["PopulationSegment", "EstimatedAcquisitionCostINR"], zoom=3.8, height=520, title="Market areas: demand (size) and lead conversion (colour)") \
        if hasattr(px, "scatter_map") else px.scatter_mapbox(M, lat="Latitude", lon="Longitude", color="LeadConversionRate", size="MonthlySearchDemand", hover_name="City", zoom=3.8, height=520, mapbox_style="carto-positron")
    fig.update_layout(map_style="carto-positron") if hasattr(px, "scatter_map") else None
    st.plotly_chart(fig, width='stretch', key='pc7')
    st.dataframe(M.groupby("City").agg(Areas=("RegionID", "count"), AvgConversion=("LeadConversionRate", "mean"), AvgCAC=("EstimatedAcquisitionCostINR", "mean")).round(3).reset_index(), hide_index=True, width='stretch')

with st.expander("Data notes and assumptions"):
    st.markdown("""
- **Cleaning:** 5 duplicate customer rows removed; 13 missing regions labelled *Unknown*; 11 unparseable TotalCharges set to blank.
- **Revenue / Avg ticket:** converted orders in `checkout_ab_test.csv` (Jul 1-22, 2026 only).
- **Active users / churn:** `customer_churn.csv` (Churn = No / Yes).
- **Filters:** the four files share no region or category field, so each filter applies only to the data it fits (labelled in the sidebar).
""")
