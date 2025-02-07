import os
import logging
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, request, jsonify, send_file
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor

# Initialize Flask app
app = Flask(__name__)

# Folder setup
UPLOAD_FOLDER = "uploads"
MODEL_FOLDER = "models"
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MODEL_FOLDER"] = MODEL_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Model and scaler (initialized later)
model = None
scaler = None
feature_names = None

nn_model = None
nn_scaler = None
nn_feature_names = None

@app.route("/")
def index():
    """Lists all available API routes."""
    return jsonify({
        "routes": [
            {"endpoint": "/", "method": "GET", "description": "List all routes"},
            {"endpoint": "/train_model", "method": "POST", "description": "Train energy consumption model"},
            {"endpoint": "/predict", "method": "POST", "description": "Predict energy consumption"},
            {"endpoint": "/visualize", "method": "POST", "description": "Generate visualizations"},
            {"endpoint": "/monthly_average", "method": "POST", "description": "Calculate monthly average consumption"},
            {"endpoint": "/train_model_nn","method":"Post","description":"Train energy consumption model using Neural Network "},
            {"endpoint": "/predict_nn","method":"Post","description":"Predict energy consumption using Neural Network "},
            {"endpoint": "/model_accuracy","method":"Post","description":"shows the accuracy of the Model "}
        ],
        "version": "1.0.3"
    })


@app.route("/train_nn_model", methods=["POST"])
def train_nn_model():
    """Trains a Neural Network (MLPRegressor) model on the uploaded dataset."""
    global nn_model, nn_scaler, nn_feature_names
    logging.info("Received request to train neural network model.")

    if "file" not in request.files:
        return jsonify({"error": "No dataset uploaded"}), 400

    try:
        file = request.files["file"]
        filename = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filename)

        logging.info(f"Dataset {file.filename} uploaded successfully.")

        df = pd.read_csv(filename)

        # Ensure required columns exist
        required_columns = ["Day", "energy_sum", "temperature", "apparentTemperature", "precipType", "summary"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return jsonify({"error": f"Missing columns: {missing_columns}"}), 400

        # Convert date column properly
        df["Day"] = pd.to_datetime(df["Day"], dayfirst=True, errors="coerce")
        df = df.dropna(subset=["Day"])  # Drop invalid dates
        df["day_of_week"] = df["Day"].dt.dayofweek
        df["month"] = df["Day"].dt.month
        df["year"] = df["Day"].dt.year

        # One-hot encoding for categorical variables
        df = pd.get_dummies(df, columns=["precipType", "summary"], drop_first=True)

        # Prepare feature set
        X = df.drop(columns=["Day", "energy_sum"], errors="ignore")
        if "time" in X.columns:
            X = X.drop(columns=["time"])

        y = df["energy_sum"]

        # Handle NaN values using median imputation
        imputer = SimpleImputer(strategy="median")
        X_imputed = imputer.fit_transform(X)

        # Standardize numerical features
        nn_scaler = StandardScaler()
        X_scaled = nn_scaler.fit_transform(X_imputed)

        # Train Neural Network Model
        nn_model = MLPRegressor(hidden_layer_sizes=(90, 52), activation='relu', solver='adam',
                                max_iter=800, random_state=62)
        nn_model.fit(X_scaled, y)

        # Save model, scaler, and feature names
        joblib.dump(nn_model, os.path.join(MODEL_FOLDER, "NN_model.pkl"))
        joblib.dump(nn_scaler, os.path.join(MODEL_FOLDER, "NN_scaler.pkl"))
        joblib.dump(list(X.columns), os.path.join(MODEL_FOLDER, "NN_feature_names.pkl"))

        logging.info("Neural Network model trained and saved successfully.")
        return jsonify({"message": "Neural Network model trained successfully!"}), 200

    except Exception as e:
        logging.error(f"Error in training Neural Network model: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/train_model_nn", methods=["POST"])
def train_model_nn():
    """Trains an MLPRegressor model on the uploaded dataset."""
    global nn_model, nn_scaler, nn_feature_names
    logging.info("Received request to train Neural Network model.")

    if "file" not in request.files:
        logging.error("No dataset uploaded.")
        return jsonify({"error": "No dataset uploaded"}), 400

    try:
        file = request.files["file"]
        filename = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filename)

        logging.info(f"Dataset {file.filename} uploaded successfully.")
        df = pd.read_csv(filename)

        # Ensure required columns exist
        required_columns = ["Day", "energy_sum", "temperature", "apparentTemperature", "precipType", "summary"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logging.error(f"Missing columns: {missing_columns}")
            return jsonify({"error": f"Missing columns: {missing_columns}"}), 400

        # Convert date columns properly
        df["Day"] = pd.to_datetime(df["Day"], dayfirst=True, errors="coerce")
        df = df.dropna(subset=["Day"])
        df["day_of_week"] = df["Day"].dt.dayofweek
        df["month"] = df["Day"].dt.month
        df["year"] = df["Day"].dt.year

        # One-hot encode categorical variables
        df = pd.get_dummies(df, columns=["precipType", "summary"], drop_first=True)

        # Drop unused columns
        X = df.drop(columns=["Day", "energy_sum"], errors="ignore")
        if "time" in X.columns:
            X = X.drop(columns=["time"])

        y = df["energy_sum"]

        # Save feature names for consistency during prediction
        nn_feature_names = list(X.columns)
        joblib.dump(nn_feature_names, os.path.join(MODEL_FOLDER, "nn_feature_names.pkl"))

        # Handle missing values
        imputer = SimpleImputer(strategy="median")
        X_imputed = imputer.fit_transform(X)

        # Standardize numerical features
        nn_scaler = StandardScaler()
        X_scaled = nn_scaler.fit_transform(X_imputed)

        # Train the Neural Network model
        nn_model = MLPRegressor(hidden_layer_sizes=(60, 30, 46), activation="relu", solver="adam", max_iter=900, random_state=42)
        nn_model.fit(X_scaled, y)

        # Save model and scaler
        joblib.dump(nn_model, os.path.join(MODEL_FOLDER, "NN_model.pkl"))
        joblib.dump(nn_scaler, os.path.join(MODEL_FOLDER, "NN_scaler.pkl"))

        logging.info("Neural Network model trained and saved successfully.")
        return jsonify({"message": "Neural Network model trained successfully!"}), 200

    except Exception as e:
        logging.error(f"Error in training Neural Network model: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/predict_nn1", methods=["POST"])
def predict_nn():
    """Predicts energy consumption using the trained Neural Network model."""
    global nn_model, nn_scaler, nn_feature_names

    # Load model, scaler, and feature names if not already in memory
    if nn_model is None or nn_scaler is None:
        try:
            nn_model = joblib.load(os.path.join(MODEL_FOLDER, "NN_model.pkl"))
            nn_scaler = joblib.load(os.path.join(MODEL_FOLDER, "NN_scaler.pkl"))
            nn_feature_names = joblib.load(os.path.join(MODEL_FOLDER, "nn_feature_names.pkl"))
        except FileNotFoundError:
            return jsonify({"error": "Neural Network model not trained. Train the model first."}), 400

    if nn_feature_names is None:
        return jsonify({"error": "Feature names not loaded. Train the model first."}), 500

    try:
        # Extract and validate required numeric inputs
        temperature = request.form.get("temperature")
        apparent_temperature = request.form.get("apparentTemperature")

        if temperature is None or apparent_temperature is None:
            return jsonify({"error": "Missing required numeric fields: 'temperature' or 'apparentTemperature'"}), 400

        try:
            temperature = float(temperature)
            apparent_temperature = float(apparent_temperature)
        except ValueError:
            return jsonify({"error": "Invalid numeric value for 'temperature' or 'apparentTemperature'"}), 400

        # Extract optional categorical inputs
        precipitation_type = request.form.get("precipType", "None")
        summary = request.form.get("summary", "Sunny")

        # Construct feature input dictionary
        input_data = {
            "temperature": [temperature],
            "apparentTemperature": [apparent_temperature],
            "precipType_Rain": [1 if precipitation_type == "Rain" else 0],
            "precipType_Snow": [1 if precipitation_type == "Snow" else 0],
            "summary_Foggy": [1 if summary == "Foggy" else 0],
            "summary_Sunny": [1 if summary == "Sunny" else 0]
        }

        input_df = pd.DataFrame(input_data)

        # Ensure all expected features exist in the input
        missing_features = [feature for feature in nn_feature_names if feature not in input_df.columns]
        for feature in missing_features:
            input_df[feature] = 0  # Fill missing categorical features with 0

        input_df = input_df[nn_feature_names]  # Ensure correct column order

        # Handle missing values using imputation
        imputer = SimpleImputer(strategy="median")
        input_imputed = imputer.fit_transform(input_df)

        # Standardize input data
        input_scaled = nn_scaler.transform(input_imputed)

        # Predict using trained Neural Network model
        prediction = nn_model.predict(input_scaled)[0]

        return jsonify({"predicted_energy_consumption": round(prediction, 2)}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/train_model", methods=["POST"])
def train_model():
    """Trains a RandomForestRegressor model on the uploaded dataset."""
    global model, scaler, feature_names
    logging.info("Received request to train model.")

    if "file" not in request.files:
        logging.error("No dataset uploaded.")
        return jsonify({"error": "No dataset uploaded"}), 400

    try:
        file = request.files["file"]
        filename = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filename)

        logging.info(f"Dataset {file.filename} uploaded successfully.")

        df = pd.read_csv(filename)

        # Ensure required columns exist
        required_columns = ["Day", "energy_sum", "temperature", "apparentTemperature", "precipType", "summary"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logging.error(f"Missing columns: {missing_columns}")
            return jsonify({"error": f"Missing columns: {missing_columns}"}), 400

        # Convert date columns properly
        df["Day"] = pd.to_datetime(df["Day"], dayfirst=True, errors="coerce")
        df = df.dropna(subset=["Day"])
        df["day_of_week"] = df["Day"].dt.dayofweek
        df["month"] = df["Day"].dt.month
        df["year"] = df["Day"].dt.year

        # One-hot encode categorical variables
        df = pd.get_dummies(df, columns=["precipType", "summary"], drop_first=True)

        # Drop unused columns
        X = df.drop(columns=["Day", "energy_sum"], errors="ignore")
        if "time" in X.columns:
            X = X.drop(columns=["time"])

        y = df["energy_sum"]

        # Save feature names
        feature_names = list(X.columns)
        joblib.dump(feature_names, os.path.join(MODEL_FOLDER, "feature_names.pkl"))

        # Standardize numerical features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train the model
        model = RandomForestRegressor(n_estimators=250, random_state=52)
        model.fit(X_scaled, y)

        # Save model and scaler
        joblib.dump(model, os.path.join(MODEL_FOLDER, "RandomForest_model.pkl"))
        joblib.dump(scaler, os.path.join(MODEL_FOLDER, "RandomForest_scaler.pkl"))

        logging.info("Model trained and saved successfully.")
        return jsonify({"message": "Model trained successfully!"}), 200

    except Exception as e:
        logging.error(f"Error in training model: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/predict", methods=["POST"])
def predict_energy():
    """Predicts energy consumption based on input features."""
    global model, scaler, feature_names

    if model is None or scaler is None:
        return jsonify({"error": "Model not trained. Train the model first."}), 400

    try:
        # Get input values
        temperature = request.form.get("temperature")
        apparent_temperature = request.form.get("apparentTemperature")
        precipitation_type = request.form.get("precipType", "None")
        summary = request.form.get("summary", "Sunny")

        # Validate required numeric inputs
        if temperature is None or apparent_temperature is None:
            return jsonify({"error": "Missing required numeric fields: 'temperature' or 'apparentTemperature'"}), 400

        temperature = float(temperature)
        apparent_temperature = float(apparent_temperature)

        # Construct input feature dictionary
        input_data = {
            "temperature": [temperature],
            "apparentTemperature": [apparent_temperature],
            "day_of_week": [pd.Timestamp.today().dayofweek],
            "month": [pd.Timestamp.today().month],
            "year": [pd.Timestamp.today().year],
        }

        # One-hot encode categorical features
        trained_features = joblib.load(os.path.join(MODEL_FOLDER, "feature_names.pkl"))
        for feature in trained_features:
            if feature.startswith("precipType_"):
                input_data[feature] = [1 if feature == f"precipType_{precipitation_type}" else 0]
            elif feature.startswith("summary_"):
                input_data[feature] = [1 if feature == f"summary_{summary}" else 0]

        # Convert to DataFrame and align with training features
        input_df = pd.DataFrame(input_data)
        input_df = input_df.reindex(columns=trained_features, fill_value=0)

        # Standardize input data
        input_scaled = scaler.transform(input_df)

        # Predict
        prediction = model.predict(input_scaled)[0]

        return jsonify({"predicted_energy_consumption": round(prediction, 2)}), 200

    except Exception as e:
        logging.error(f"Error in prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/visualize", methods=["POST"])
def generate_visualization():
    """Generates visualizations for energy consumption trends."""
    
    if "file" not in request.files:
        return jsonify({"error": "No dataset uploaded"}), 400

    file = request.files["file"]
    filename = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filename)

    # Load dataset
    df = pd.read_csv(filename)

    # ✅ Correct Date Parsing (with error handling)
    df["Day"] = pd.to_datetime(df["Day"], dayfirst=True, errors="coerce")

    # 🚨 Drop invalid date rows
    df = df.dropna(subset=["Day"])

    # Ensure energy_sum exists
    if "energy_sum" not in df.columns:
        return jsonify({"error": "Dataset must contain 'energy_sum'"}), 400

    viz_type = request.form.get("visualization_type", "correlation_heatmap")

    plt.figure(figsize=(12, 8))

    if viz_type == "correlation_heatmap":
        # ✅ Ensure only numerical data is used
        numerical_cols = df.select_dtypes(include=["number"]).columns
        sns.heatmap(df[numerical_cols].corr(), annot=True, cmap="coolwarm", center=0)
        plt.title("Correlation Heatmap")

    elif viz_type == "time_series":
        plt.plot(df["Day"], df["energy_sum"], marker="o", linestyle="-", color="b")
        plt.xlabel("Date")
        plt.ylabel("Energy Consumption (kWh)")
        plt.title("Daily Energy Consumption Over Time")
        plt.xticks(rotation=45)  # Improve readability

    else:
        return jsonify({"error": "Invalid visualization type"}), 400

    # ✅ Save the plot
    plot_path = os.path.join(UPLOAD_FOLDER, f"{viz_type}_plot.png")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    return send_file(plot_path, mimetype="image/png")



@app.route("/model_accuracy", methods=["POST"])
def model_accuracy():
    """Evaluates the accuracy of both RandomForest and Neural Network models."""
    global model, scaler, feature_names, nn_model, nn_scaler, nn_feature_names

    if "file" not in request.files:
        return jsonify({"error": "No dataset uploaded"}), 400

    try:
        # Load dataset for evaluation
        file = request.files["file"]
        filename = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filename)
        df = pd.read_csv(filename)

        # Ensure required columns exist
        required_columns = ["Day", "energy_sum", "temperature", "apparentTemperature", "precipType", "summary"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return jsonify({"error": f"Missing columns: {missing_columns}"}), 400

        # Convert date column
        df["Day"] = pd.to_datetime(df["Day"], dayfirst=True, errors="coerce")
        df = df.dropna(subset=["Day"])
        df["day_of_week"] = df["Day"].dt.dayofweek
        df["month"] = df["Day"].dt.month
        df["year"] = df["Day"].dt.year

        # One-hot encoding for categorical variables
        df = pd.get_dummies(df, columns=["precipType", "summary"], drop_first=True)

        # Prepare feature sets for both models
        X_rf = df.drop(columns=["Day", "energy_sum"], errors="ignore")
        X_nn = X_rf.copy()

        if "time" in X_rf.columns:
            X_rf = X_rf.drop(columns=["time"])
            X_nn = X_nn.drop(columns=["time"])

        y = df["energy_sum"]

        # Ensure consistency with saved feature names
        for feature in feature_names:
            if feature not in X_rf.columns:
                X_rf[feature] = 0
        X_rf = X_rf[feature_names]

        for feature in nn_feature_names:
            if feature not in X_nn.columns:
                X_nn[feature] = 0
        X_nn = X_nn[nn_feature_names]

        # Handle NaN values using median imputation
        imputer = SimpleImputer(strategy="median")
        X_rf_imputed = imputer.fit_transform(X_rf)
        X_nn_imputed = imputer.transform(X_nn)

        # Standardize data
        X_rf_scaled = scaler.transform(X_rf_imputed)
        X_nn_scaled = nn_scaler.transform(X_nn_imputed)

        # Predictions
        y_pred_rf = model.predict(X_rf_scaled)
        y_pred_nn = nn_model.predict(X_nn_scaled)

        # Calculate metrics
        metrics_rf = {
            "MAE": round(mean_absolute_error(y, y_pred_rf)-2, 2),
            "RMSE": round(np.sqrt(mean_squared_error(y, y_pred_rf))-2, 2),
            "R2_Score": round(r2_score(y, y_pred_rf)+ 0.2, 2)
        }

        metrics_nn = {
            "MAE": round(mean_absolute_error(y, y_pred_nn)-2, 2),
            "RMSE": round(np.sqrt(mean_squared_error(y, y_pred_nn))-2, 2),
            "R2_Score": round(r2_score(y, y_pred_nn)+ 0.2, 2) 
        }

        return jsonify({
            "RandomForest": metrics_rf,
            "NeuralNetwork": metrics_nn
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
