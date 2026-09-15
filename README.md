## Project Description

This project predicts the AC power output of a solar power plant using weather and irradiation data.

The dataset contains Plant 1 generation data and weather sensor data. The project performs data preprocessing, hourly aggregation, exploratory data analysis, and linear regression using three different methods:

- Normal Equation
- Batch Gradient Descent
- Stochastic Gradient Descent

Public weather data is also obtained from Open-Meteo and compared with the on-site weather sensor data.

## Dataset

The project uses the following Plant 1 datasets:

- `Plant_1_Generation_Data.csv`
- `Plant_1_Weather_Sensor_Data.csv`

The generation data contains AC power, DC power, daily yield, and total yield.

The weather data contains:

- Ambient temperature
- Module temperature
- Irradiation

## Libraries Used

```text
NumPy
Pandas
Matplotlib
Requests
JSON
```

## How to Run

### 1. Install Python

Make sure Python is installed on your computer.

### 2. Install Required Libraries

Open the terminal and run:

```bash
pip install numpy pandas matplotlib requests
```

### 3. Open the Jupyter Notebook

Open the notebook:

```text
AML (1)(1).ipynb
```

using Jupyter Notebook or VS Code.

### 4. Add the Dataset

Place these files in the same folder as the notebook:

```text
Plant_1_Generation_Data.csv
Plant_1_Weather_Sensor_Data.csv
```

### 5. Run the Notebook

Run the cells from top to bottom.

The notebook performs the following steps:

1. Imports the required libraries.
2. Loads the generation and weather datasets.
3. Converts the `DATE_TIME` columns into datetime format.
4. Aggregates Plant 1 AC power using the sum of inverter values.
5. Merges generation and weather data using `DATE_TIME`.
6. Resamples the data into hourly values.
7. Creates input features and target AC power.
8. Splits the data into training and testing data.
9. Scales the training features.
10. Adds the intercept term.
11. Implements the cost function.
12. Implements Batch Gradient Descent.
13. Implements Stochastic Gradient Descent.
14. Calculates RMSE and R².
15. Fetches public weather data from Open-Meteo.
16. Compares public weather data with on-site weather data.
17. Generates graphs for analysis.

## Regression Models

### Normal Equation

The Normal Equation is used to calculate the regression parameters directly:

```text
θ = (XᵀX)⁻¹Xᵀy
```

### Batch Gradient Descent

Batch Gradient Descent updates the parameters using the complete training dataset in every iteration.

### Stochastic Gradient Descent

Stochastic Gradient Descent updates the parameters one training sample at a time and calculates the cost after each epoch.

## Evaluation Metrics

The models are evaluated using:

### RMSE

Root Mean Squared Error measures the difference between actual and predicted AC power.

### R² Score

R² measures how well the model explains the variation in AC power.

## Graphs and Results

The notebook generates graphs including:

- Average AC power per hour
- Gradient Descent cost convergence
- Learning-rate comparison
- Actual vs predicted AC power
- Residuals vs hour
- On-site weather vs public weather comparison

The generated graphs can be found directly in the notebook outputs.

## Open-Meteo Weather Data

Public historical weather data is retrieved using the Open-Meteo archive API.

The public weather data is used to investigate whether publicly available weather information can be used instead of on-site sensor measurements for solar power prediction.

## front End Interface
<img width="386" height="308" alt="image" src="https://github.com/user-attachments/assets/0cf990ef-5548-4c58-8d95-193043fe5a1f" />


## Project Conclusion

The project compares different regression methods and investigates the effect of replacing on-site weather measurements with public weather data.

The results are used to determine the prediction accuracy obtained from each feature set and regression method.
