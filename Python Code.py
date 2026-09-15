import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
generation = pd.read_csv(r"C:\Users\osman\Videos\AML_A_!\data\Plant_1_Generation_Data.csv")
weather = pd.read_csv(r"C:\Users\osman\Videos\AML_A_!\data\Plant_1_Weather_Sensor_Data.csv")
print("Generation shape:", generation.shape)
print("Weather shape:", weather.shape)

generation["DATE_TIME"] = pd.to_datetime(
    generation["DATE_TIME"],
    dayfirst=True
)

weather["DATE_TIME"] = pd.to_datetime(
    weather["DATE_TIME"]
)
print(generation["DATE_TIME"].head())
print(weather["DATE_TIME"].head())

groupby("DATE_TIME")
plant_generation = (
    generation
    .groupby("DATE_TIME")["AC_POWER"]
    .sum()
    .reset_index()
)
print(plant_generation.head())
print(plant_generation.shape)

plant_weather = weather[
    [
        "DATE_TIME",
        "AMBIENT_TEMPERATURE",
        "MODULE_TEMPERATURE",
        "IRRADIATION"
    ]
].copy()

plant_data = pd.merge(
    plant_generation,
    plant_weather,
    on="DATE_TIME",
    how="inner"
)
print(plant_data.head())
print(plant_data.shape)

print(
    "Duplicate timestamps:",
    plant_data["DATE_TIME"].duplicated().sum()
)
plant_data = plant_data.sort_values("DATE_TIME")
plant_data = plant_data.reset_index(drop=True)

time_difference = plant_data["DATE_TIME"].diff()

print(time_difference.value_counts().head(10))
plant_data = plant_data.set_index("DATE_TIME")
hourly_data = plant_data.resample("1h").mean()
print(hourly_data.isnull().sum())

hourly_data = hourly_data.dropna()
print("Hourly dataset shape:", hourly_data.shape)

plt.figure(figsize=(12, 5))

plt.plot(
    hourly_data.index,
    hourly_data["AC_POWER"]
)

plt.xlabel("Time")
plt.ylabel("AC Power (kW)")
plt.title("Plant 1 Hourly AC Power")

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
print(hourly_data.head())
print(hourly_data.columns)
X = hourly_data[
    [
        "IRRADIATION",
        "AMBIENT_TEMPERATURE",
        "MODULE_TEMPERATURE"
    ]
].values

y = hourly_data["AC_POWER"].values
print("X shape:", X.shape)
print("y shape:", y.shape)
split = int(0.8 * len(X))

X_train_raw = X[:split]
X_test_raw = X[split:]

y_train = y[:split]
y_test = y[split:]

print("Training samples:", len(X_train_raw))
print("Testing samples:", len(X_test_raw))
plt.figure(figsize=(10, 6))
plt.scatter(X_train_raw[:, 0], y_train, alpha=0.5, label="Training")
plt.scatter(X_test_raw[:, 0], y_test, alpha=0.5, label="Testing")
plt.xlabel("Irradiation (kW/m²)")
plt.ylabel("AC Power (kW)")
plt.title("AC Power vs Irradiation")
plt.legend()
plt.grid(True)
plt.show()
plt.figure(figsize=(10, 6))
plt.scatter(X_train_raw[:, 1], y_train, alpha=0.5, label="Training")
plt.scatter(X_test_raw[:, 1], y_test, alpha=0.5, label="Testing")
plt.xlabel("Ambient Temperature (°C)")
plt.ylabel("AC Power (kW)")
plt.title("AC Power vs Ambient Temperature")
plt.legend()
plt.grid(True)
plt.show()
plt.figure(figsize=(10, 6))
plt.scatter(X_train_raw[:, 1], y_train, alpha=0.5, label="Training")
plt.scatter(X_test_raw[:, 1], y_test, alpha=0.5, label="Testing")
plt.xlabel("Ambient Temperature (°C)")
plt.ylabel("AC Power (kW)")
plt.title("AC Power vs Ambient Temperature")
plt.legend()
plt.grid(True)
plt.show()
plt.figure(figsize=(10, 6))
plt.scatter(X_train_raw[:, 2], y_train, alpha=0.5, label="Training")
plt.scatter(X_test_raw[:, 2], y_test, alpha=0.5, label="Testing")
plt.xlabel("Module Temperature (°C)")
plt.ylabel("AC Power (kW)")
plt.title("AC Power vs Module Temperature")
plt.legend()
plt.grid(True)
plt.show()
#Normal Equation
X_train = np.c_[
    np.ones(len(X_train_raw)),
    X_train_raw
]

X_test = np.c_[
    np.ones(len(X_test_raw)),
    X_test_raw
]
print(X_train.shape)
print(X_train[:3])
theta_normal = (
    np.linalg.pinv(X_train.T @ X_train)
    @ X_train.T
    @ y_train
)

print("Theta:")
print(theta_normal)
#prediction
y_pred_normal = X_test @ theta_normal
print("Actual:")
print(y_test[:10])

print("\nPredicted:")
print(y_pred_normal[:10])

#Cost Fucntion
def compute_cost(X, y, theta):
    m = len(y)

    predictions = X @ theta

    errors = predictions - y

    cost = (1 / (2 * m)) * np.sum(errors ** 2)

    return cost
 train_cost = compute_cost(
    X_train,
    y_train,
    theta_normal
)

print("Training Cost:", train_cost)
#RMSE
rmse_normal = np.sqrt(
    np.mean(
        (y_test - y_pred_normal) ** 2
    )
)

print("Normal Equation RMSE:", rmse_normal)
# R^2
ss_res = np.sum(
    (y_test - y_pred_normal) ** 2
)

ss_tot = np.sum(
    (y_test - np.mean(y_test)) ** 2
)

r2_normal = 1 - (ss_res / ss_tot)

print("Normal Equation R²:", r2_normal)
plt.figure(figsize=(12, 5))

plt.plot(
    y_test,
    label="Actual AC Power"
)

plt.plot(
    y_pred_normal,
    label="Predicted AC Power"
)

plt.xlabel("Test Sample")
plt.ylabel("AC Power (kW)")
plt.title("Normal Equation: Actual vs Predicted")

plt.legend()
plt.tight_layout()
plt.show()
#Batch Gradient Descent
mean = np.mean(X_train_raw, axis=0)
std = np.std(X_train_raw, axis=0)

X_train_scaled = (
    X_train_raw - mean
) / std

X_test_scaled = (
    X_test_raw - mean
) / std
print("Training mean:")
print(mean)

print("\nTraining standard deviation:")
print(std)
#Adding intercept 
X_train_gd = np.c_[
    np.ones(len(X_train_scaled)),
    X_train_scaled
]

X_test_gd = np.c_[
    np.ones(len(X_test_scaled)),
    X_test_scaled
]
def batch_gradient_descent(X, y, alpha, iterations):

    m, n = X.shape

    theta = np.zeros(n)

    cost_history = []

    for i in range(iterations):

        predictions = X @ theta

        errors = predictions - y

        gradient = (1 / m) * (X.T @ errors)

        theta = theta - alpha * gradient

        cost = compute_cost(X, y, theta)

        cost_history.append(cost)

    return theta, cost_history
theta_bgd, cost_history = batch_gradient_descent(
    X_train_gd,
    y_train,
    alpha=0.01,
    iterations=5000
)

print(theta_bgd)
plt.figure(figsize=(8, 5))

plt.plot(cost_history)

plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Batch Gradient Descent Cost")

plt.tight_layout()
plt.show()
y_pred_bgd = X_test_gd @ theta_bgd

rmse_bgd = np.sqrt(
    np.mean(
        (y_test - y_pred_bgd) ** 2
    )
)

print("Batch GD RMSE:", rmse_bgd)
ss_res = np.sum(
    (y_test - y_pred_bgd) ** 2
)

ss_tot = np.sum(
    (y_test - np.mean(y_test)) ** 2
)

r2_bgd = 1 - (ss_res / ss_tot)

print("Batch GD R²:", r2_bgd)
#Stochastic Gradient Descent
def stochastic_gradient_descent(
    X,
    y,
    alpha,
    epochs
):

    m, n = X.shape

    theta = np.zeros(n)

    cost_history = []

    for epoch in range(epochs):

        for i in range(m):

            xi = X[i]
            yi = y[i]

            prediction = xi @ theta

            error = prediction - yi

            gradient = xi * error

            theta = theta - alpha * gradient

        cost = compute_cost(
            X,
            y,
            theta
        )

        cost_history.append(cost)

    return theta, cost_history
theta_sgd, sgd_cost = stochastic_gradient_descent(
    X_train_gd,
    y_train,
    alpha=0.001,
    epochs=50
)

print(theta_sgd)
plt.figure(figsize=(8, 5))

plt.plot(
    range(1, len(sgd_cost) + 1),
    sgd_cost,
    marker='o'
)

plt.xlabel("Epoch")
plt.ylabel("Cost J(θ)")
plt.title("Stochastic Gradient Descent")
plt.grid(True)

plt.show()
y_pred_sgd = X_test_gd @ theta_sgd

rmse_sgd = np.sqrt(
    np.mean(
        (y_test - y_pred_sgd) ** 2
    )
)

print("SGD RMSE:", rmse_sgd)

ss_res = np.sum(
    (y_test - y_pred_sgd) ** 2
)

ss_tot = np.sum(
    (y_test - np.mean(y_test)) ** 2
)

r2_sgd = 1 - (ss_res / ss_tot)

print("SGD R²:", r2_sgd)
results = pd.DataFrame({
    "Method": [
        "Normal Equation",
        "Batch Gradient Descent",
        "Stochastic Gradient Descent"
    ],

    "RMSE": [
        rmse_normal,
        rmse_bgd,
        rmse_sgd
    ],

    "R2": [
        r2_normal,
        r2_bgd,
        r2_sgd
    ]
})

print(results)
url = "https://archive-api.open-meteo.com/v1/archive"

params = {
    "latitude": 14.82,
    "longitude": 78.28,
    "start_date": "2020-05-15",
    "end_date": "2020-06-17",
    "hourly": "temperature_2m,shortwave_radiation",
    "timezone": "Asia/Kolkata"
}

response = requests.get(
    url,
    params=params
)

print("Status code:", response.status_code)
data = response.json()
print(type(data))
public_weather = pd.DataFrame({
    "DATE_TIME": data["hourly"]["time"],

    "PUBLIC_TEMPERATURE":
        data["hourly"]["temperature_2m"],

    "PUBLIC_IRRADIATION":
        data["hourly"]["shortwave_radiation"]
})
print(public_weather.head())
public_weather["DATE_TIME"] = pd.to_datetime(
    public_weather["DATE_TIME"]
)
print(public_weather.dtypes)
print("Rows:", len(public_weather))

print("\nStart:")
print(public_weather["DATE_TIME"].min())

print("\nEnd:")
print(public_weather["DATE_TIME"].max())
public_weather["PUBLIC_IRRADIATION"] = (
    public_weather["PUBLIC_IRRADIATION"] / 1000
)
hourly_data_public = hourly_data.reset_index()
print(hourly_data_public.head())

comparison_data = pd.merge(
    hourly_data_public,
    public_weather,
    on="DATE_TIME",
    how="inner"
)
print(comparison_data.head())
print("\nShape:", comparison_data.shape)
print(comparison_data.columns.tolist())
X_public = comparison_data[
    [
        "PUBLIC_IRRADIATION",
        "PUBLIC_TEMPERATURE"
    ]
].values

y_public = comparison_data["AC_POWER"].values

split_public = int(
    0.8 * len(X_public)
)

X_public_train_raw = X_public[:split_public]
X_public_test_raw = X_public[split_public:]

y_public_train = y_public[:split_public]
y_public_test = y_public[split_public:]
X_public_train = np.c_[
    np.ones(len(X_public_train_raw)),
    X_public_train_raw
]

X_public_test = np.c_[
    np.ones(len(X_public_test_raw)),
    X_public_test_raw
]
theta_public = (
    np.linalg.pinv(
        X_public_train.T @ X_public_train
    )
    @ X_public_train.T
    @ y_public_train
)

print("Public Weather Theta:")
print(theta_public)
y_public_pred = (
    X_public_test @ theta_public
)
rmse_public = np.sqrt(
    np.mean(
        (y_public_test - y_public_pred) ** 2
    )
)

print(
    "Public Weather RMSE:",
    rmse_public
)
ss_res_public = np.sum(
    (y_public_test - y_public_pred) ** 2
)

ss_tot_public = np.sum(
    (y_public_test - np.mean(y_public_test)) ** 2
)

r2_public = (
    1 -
    (ss_res_public / ss_tot_public)
)

print(
    "Public Weather R²:",
    r2_public
)
#How much prediction accuracy is lost when on-site irradiation and temperature sensors are
#replaced by free public weather data?
r2_loss = r2_normal - r2_public

relative_r2_loss = (
    (r2_normal - r2_public)
    / r2_normal
) * 100

print("R² loss:", r2_loss)
print(
    "Relative R² loss (%):",
    relative_r2_loss
)
rmse_increase = (
    (rmse_public - rmse_normal)
    / rmse_normal
) * 100

print(
    "RMSE increase (%):",
    rmse_increase
)
final_results = pd.DataFrame({
    "Model": [
        "On-site Sensor Data",
        "Public Weather Data"
    ],

    "RMSE": [
        rmse_normal,
        rmse_public
    ],

    "R2": [
        r2_normal,
        r2_public
    ]
})

print(final_results)
plt.figure(figsize=(12, 5))

plt.plot(
    y_public_test,
    label="Actual AC Power"
)

plt.plot(
    y_public_pred,
    label="Public Weather Prediction"
)

plt.xlabel("Test Sample")
plt.ylabel("AC Power (kW)")
plt.title(
    "Public Weather Model: Actual vs Predicted"
)

plt.legend()
plt.tight_layout()
plt.show()
plt.figure(figsize=(12, 5))

plt.plot(
    y_test,
    label="Actual"
)

plt.plot(
    y_pred_normal,
    label="On-site Sensor Prediction"
)

plt.plot(
    y_public_test,
    label="Public Model Actual"
)

plt.plot(
    y_public_pred,
    label="Public Weather Prediction"
)

plt.xlabel("Test Sample")
plt.ylabel("AC Power (kW)")
plt.title("On-site vs Public Weather Prediction")

plt.legend()
plt.tight_layout()
plt.show()