import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import timedelta
import random

# 1. Generate Synthetic Data
np.random.seed(42)
milestones = [f"Milestone {i}" for i in range(1, 11)]
start_date = pd.to_datetime("2024-01-01")
achievement_days = np.cumsum(np.random.randint(10, 20, size=10))
achievement_dates = [start_date + timedelta(days=int(day)) for day in achievement_days]

# 2. Create DataFrame
df = pd.DataFrame({
    "Milestone": milestones,
    "Achievement Date": achievement_dates
})
df["Days Since Start"] = (df["Achievement Date"] - start_date).dt.days

# 3. Forecast Future Milestones (Milestone 11–15)
future_milestones = [f"Milestone {i}" for i in range(11, 16)]
future_indices = np.array(range(11, 16)).reshape(-1, 1)

# 4. Linear Regression Model
X = np.array(range(1, 11)).reshape(-1, 1)  # Milestone numbers
y = df["Days Since Start"].values

model = LinearRegression()
model.fit(X, y)

future_days = model.predict(future_indices)
future_dates = [start_date + timedelta(days=int(day)) for day in future_days]

# 5. Combine Data
df_forecast = pd.DataFrame({
    "Milestone": future_milestones,
    "Predicted Achievement Date": future_dates,
    "Forecast Days Since Start": future_days.astype(int)
})

# 6. Export to Excel
combined_df = pd.concat([
    df[["Milestone", "Achievement Date", "Days Since Start"]],
    df_forecast.rename(columns={
        "Predicted Achievement Date": "Achievement Date",
        "Forecast Days Since Start": "Days Since Start"
    })
], ignore_index=True)

combined_df.to_excel("Milestone_Achievement_Forecast.xlsx", index=False)

# 7. Plot Progress
plt.figure(figsize=(10, 6))
plt.plot(X, y, 'o-', label="Actual")
plt.plot(future_indices, future_days, 'x--', label="Forecast")
plt.xlabel("Milestone Number")
plt.ylabel("Days Since Start")
plt.title("Milestone Achievement Forecast")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("Milestone_Forecast_Chart.png")
plt.show()
