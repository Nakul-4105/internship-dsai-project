import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from prophet import Prophet
from sklearn.ensemble import IsolationForest
import smtplib
from email.message import EmailMessage


# Load dataset
df = pd.read_csv("india_residential_smart_meter_2025_Jan_Jun.csv")
df["Date"] = pd.to_datetime(df["Date"])

# Static data - households per state (estimated, based on population/4.5)
households_per_state = {
    "Uttar Pradesh": 250_000_000 // 4.5,
    "Maharashtra": 125_000_000 // 4.5,
    "Bihar": 130_000_000 // 4.5,
    "West Bengal": 100_000_000 // 4.5,
    "Madhya Pradesh": 85_000_000 // 4.5,
    "Tamil Nadu": 80_000_000 // 4.5,
    "Rajasthan": 80_000_000 // 4.5,
    "Karnataka": 70_000_000 // 4.5,
    "Gujarat": 70_000_000 // 4.5,
    "Andhra Pradesh": 55_000_000 // 4.5
}

st.title("🏠 Residential Smart Meter Analyzer (India)")

# Select year and month
year = st.number_input("Enter year for analysis", min_value=2025, max_value=2025, value=2025)
month = st.number_input("Enter month for analysis (1-12)", min_value=1, max_value=6, value=1)

# Filter data for selected month and year
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df_filtered = df[(df["Year"] == year) & (df["Month"] == month)]

if df_filtered.empty:
    st.warning(f"No data available for {month}/{year}")
    st.stop()

# State selection for comparison
states_selected = st.multiselect(
    "Select States to Compare",
    options=df["State"].unique(),
    default=[df_filtered["State"].iloc[0]]
)

df_compare = df_filtered[df_filtered["State"].isin(states_selected)]

# Plot daily residential consumption per state as bar plot
st.header(f"Daily Residential Consumption in {month}/{year} (kWh)")
fig = px.bar(
    df_compare,
    x="Date",
    y="Residential_Consumption_kWh",
    color="State",
    barmode="group",
    title="State-wise Daily Residential Electricity Consumption"
)
st.plotly_chart(fig)

# Calculate cost per household and show summary per state
st.header("Estimated Monthly Cost per Residential Household")

# User input: tariff rate
tariff = st.number_input("Enter your electricity tariff rate (₹/kWh)", value=5.88)

summary_data = []
for state in states_selected:
    state_data = df_filtered[df_filtered["State"] == state]
    total_consumption = state_data["Residential_Consumption_kWh"].sum()  # kWh total for month
    households = households_per_state[state]
    avg_consumption_per_household = total_consumption / households
    cost_per_household = avg_consumption_per_household * tariff
    summary_data.append({
        "State": state,
        "Total_Consumption_kWh": total_consumption,
        "Households": households,
        "Avg_Consumption_Per_Household_kWh": avg_consumption_per_household,
        "Estimated_Cost_Per_Household_₹": cost_per_household
    })

summary_df = pd.DataFrame(summary_data)
st.dataframe(summary_df.style.format({
    "Total_Consumption_kWh": "{:,.0f}",
    "Households": ":,",
    "Avg_Consumption_Per_Household_kWh": "{:.2f}",
    "Estimated_Cost_Per_Household_₹": "₹{:,.2f}"
}))

# Optional: Forecast consumption for one selected state
if len(states_selected) == 1:
    st.header("Load Forecasting for Jul-Dec 2025")
    ts = df[df["State"] == states_selected[0]][["Date", "Residential_Consumption_kWh"]].rename(columns={"Date": "ds", "Residential_Consumption_kWh": "y"})
    model = Prophet()
    model.fit(ts)
    future = model.make_future_dataframe(periods=184)  # 6 months
    forecast = model.predict(future)
    forecast_filtered = forecast[forecast["ds"].dt.month > 6]
    fig2 = px.line(forecast_filtered, x="ds", y="yhat", title=f"Forecasted Consumption (Jul-Dec 2025) for {states_selected[0]}")
    st.plotly_chart(fig2)

    # Email section
    st.header("📧 Send Forecast Report via Email")
    email_id = st.text_input("Enter your email to receive report")
    threshold_kwh = st.number_input("Set alert threshold (kWh)", value=400.0)

    if st.button("Send Report"):
        report_df = forecast_filtered[["ds", "yhat"]]
        report_df.columns = ["Date", "Forecast_kWh"]
        report_csv = report_df.to_csv(index=False)
        avg_forecast = report_df["Forecast_kWh"].mean()

        msg = EmailMessage()
        msg["Subject"] = f"Electricity Forecast Report - {states_selected[0]} (Jul-Dec 2025)"
        msg["From"] = "youremail@example.com"  # replace
        msg["To"] = email_id
        body = f"Hello,\n\nPlease find attached the electricity forecast for {states_selected[0]} for Jul–Dec 2025.\n\nAverage Daily Forecast: {avg_forecast:.2f} kWh.\n"
        if avg_forecast > threshold_kwh:
            body += f"⚠️ Alert: Forecasted load exceeds your threshold of {threshold_kwh} kWh!\n"
        body += "\nRegards,\nSmart Meter Analyzer"

        msg.set_content(body)
        msg.add_attachment(report_csv, subtype='csv', filename="forecast_report.csv")

        try:
            with smtplib.SMTP("smtp.gmail.com", 587) as smtp:
                smtp.starttls()
                smtp.login("youremail@example.com", "yourpassword")  # replace with real
                smtp.send_message(msg)
                st.success("📩 Report sent successfully!")
        except Exception as e:
            st.error(f"Failed to send email: {e}")

# Anomaly detection for one selected state
if len(states_selected) == 1:
    st.header("Anomaly Detection")
    df_state = df_compare[df_compare["State"] == states_selected[0]]
    model = IsolationForest(contamination=0.03, random_state=42)
    df_state["Anomaly"] = model.fit_predict(df_state[["Residential_Consumption_kWh"]])
    fig3 = px.scatter(df_state, x="Date", y="Residential_Consumption_kWh", color=df_state["Anomaly"].astype(str),
                      title=f"Anomalies in Consumption for {states_selected[0]}")
    st.plotly_chart(fig3)
