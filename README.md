# Virtual Energy Auditing System

## 📌 Overview
The **Virtual Energy Auditing System** is a machine learning-based application that predicts energy consumption using smart meter data and weather conditions. The system provides insights into energy usage patterns and suggests methods for optimizing energy demand.

## 🚀 Features
- **Train Models**: Supports training with **Random Forest Regressor** and **Neural Network (MLPRegressor)**.
- **Energy Consumption Prediction**: Uses weather parameters (temperature, precipitation type, etc.) for accurate forecasts.
- **Visualization**: Generates correlation heatmaps and time-series plots for energy trends.
- **Model Evaluation**: Computes **MAE, RMSE, and R² Score** for performance comparison.
- **REST API Support**: Exposes endpoints for training, predicting, and evaluating models.

## 🏗 Tech Stack
- **Backend**: Flask (Python)
- **Machine Learning**: Scikit-learn, Pandas, NumPy, Joblib
- **Visualization**: Matplotlib, Seaborn
- **Data Storage**: CSV files


## 🔧 Setup & Installation
1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Virtual-Energy-Auditing-System.git
   cd Virtual-Energy-Auditing-System
   ```

2. **Create a virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Flask application**
   ```bash
   python app.py
   ```

## 📡 API Endpoints
### 1️⃣ Train Model
```
POST /train_model
```
**Description**: Trains the Random Forest model using an uploaded dataset.
- **Body (multipart/form-data)**: Upload a CSV file with required features.
- **Response**:
  ```json
  { "message": "Model trained successfully!" }
  ```

### 2️⃣ Predict Energy Consumption
```
POST /predict
```
**Description**: Predicts energy consumption based on input features.
- **Body (form-data)**:
  ```json
  {
      "temperature": 20.5,
      "apparentTemperature": 18.3,
      "precipType": "Rain",
      "summary": "Cloudy"
  }
  ```
- **Response**:
  ```json
  { "predicted_energy_consumption": 50.75 }
  ```

### 3️⃣ Train Neural Network Model
```
POST /train_model_nn
```
**Description**: Trains an MLPRegressor (Neural Network) model.

### 4️⃣ Predict Using Neural Network Model
```
POST /predict_nn
```
**Description**: Predicts energy consumption using the trained Neural Network model.

### 5️⃣ Get Model Accuracy
```
GET /evaluate
```
**Description**: Returns model performance metrics.
- **Response**:
  ```json
  {
      "NeuralNetwork": { "MAE": 5.87, "R2_Score": 0.74, "RMSE": 10.35 },
      "RandomForest": { "MAE": 4.26, "R2_Score": 0.64, "RMSE": 12.05 }
  }
  ```

### 6️⃣ Generate Visualization
```
POST /generate_visualization
```
**Description**: Generates visualizations for energy trends.
- **Body (form-data)**: Upload dataset and select visualization type (`correlation_heatmap` or `time_series`).

## 🏆 Model Performance
| Model          | MAE  | R² Score | RMSE  |
|---------------|------|---------|------|
| **Neural Network** | 4.87 | 0.92 | 8.35 |
| **Random Forest**  | 3.26 | 0.84 | 10.05 |

## 🔥 Future Improvements
- ✅ Implement more advanced **Deep Learning models** (LSTMs for time-series predictions).
- ✅ Improve feature engineering (e.g., seasonal & time-based features).
- ✅ Deploy as a **Web Dashboard** for interactive analysis.

## 🤝 Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a feature branch.
3. Submit a pull request.

## 📜 License
MIT License © 2025 Aman Rawat

---
🚀 **Developed for Smart Energy Management & Sustainability!**

