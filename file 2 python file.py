import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import timedelta
from io import BytesIO

# Set up page
st.set_page_config(page_title="Milestone Achievement Forecast", layout="centered")

st.title("📈 Milestone Achievement Forecast")
st.markdown("This app simulates milestone achievement and forecasts future completions using linear regression.")

# 1. Generate Synthetic Data
np.random.seed(42)
milestone_count = st.slider("Number of Completed Milestones", min_value=5, max_value=20, value=10)
start_date = pd.to_datetime("2024-01-01")
achievement_days = np.cumsum(np.random.randint(10, 20, size=milestone_count))
achievement_dates = [start_date + timedelta(days=int(day)) for day in achievement_days]

df = pd.DataFrame({
    "Milestone": [f"Milestone {i}" for i in range(1, milestone_count + 1)],
    "Achievement Date": achievement_dates
})
df["Days Since Start"] = (df["Achievement Date"] - start_date).dt.days

# Display synthetic data
st.subheader("🔍 Actual Milestone Data")
st.dataframe(df)

# 2. Forecast Future Milestones
future_count = st.slider("Number of Future Milestones to Forecast", min_value=1, max_value=10, value=5)
X = np.array(range(1, milestone_count + 1)).reshape(-1, 1)
y = df["Days Since Start"].values

model = LinearRegression()
model.fit(X, y)

future_indices = np.array(range(milestone_count + 1, milestone_count + future_count + 1)).reshape(-1, 1)
future_days = model.predict(future_indices)
future_dates = [start_date + timedelta(days=int(day)) for day in future_days]

df_forecast = pd.DataFrame({
    "Milestone": [f"Milestone {i}" for i in range(milestone_count + 1, milestone_count + future_count + 1)],
    "Predicted Achievement Date": future_dates,
    "Forecast Days Since Start": future_days.astype(int)
})

# Display forecast
st.subheader("🔮 Forecasted Milestone Data")
st.dataframe(df_forecast)

# 3. Combine and Download
combined_df = pd.concat([
    df[["Milestone", "Achievement Date", "Days Since Start"]],
    df_forecast.rename(columns={
        "Predicted Achievement Date": "Achievement Date",
        "Forecast Days Since Start": "Days Since Start"
    })
], ignore_index=True)

# Download button
def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Milestones')
    return output.getvalue()

st.download_button("📥 Download Forecast Data as Excel", data=to_excel(combined_df),
                   file_name="Milestone_Achievement_Forecast.xlsx")

# 4. Plotting
st.subheader("📊 Forecast Plot")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(X.flatten(), y, 'o-', label="Actual", color="blue")
ax.plot(future_indices.flatten(), future_days, 'x--', label="Forecast", color="orange")
ax.set_xlabel("Milestone Number")
ax.set_ylabel("Days Since Start")
ax.set_title("Milestone Achievement Forecast")
ax.grid(True)
ax.legend()
st.pyplot(fig)
