# 🚦 Traffic Congestion Prediction Dashboard

A Streamlit dashboard for **Predictive Traffic Congestion Modeling and Optimization for Smart City Mobility** in New York City.  
This dashboard visualizes traffic congestion predictions, enables interactive exploration of traffic patterns, and demonstrates the power of machine learning for urban mobility.

---

## 📋 Features

- **Interactive Map:** Visualize taxi pickup locations by NYC taxi zone centroids.
- **Congestion Analytics:** Explore congestion trends by hour and by pickup zone.
- **ML Predictions:** Predict congestion levels for selected trips using a trained ML model (e.g., XGBoost).
- **Data Filtering:** Filter trips by hour, zone, and distance using sidebar controls.
- **Download:** Export filtered data as CSV.
- **Responsive Layout:** Clean, modern UI with Streamlit and Folium.

---

## 📂 Data Used

- **NYC Taxi Trip Data:** Cleaned CSV with engineered features.
- **NYC Taxi Zone Centroids:** CSV with latitude/longitude for each taxi zone (generated from official shapefile).
- **Trained ML Model:** Pre-trained XGBoost or RandomForest model (`.pkl`).
- *(Optional)* Weather and event data for advanced features.

---

## 🚀 Getting Started

### 1. **Clone the Repository**

```bash
git clone https://github.com/adityajod/traffic-dashboard
cd traffic-dashboard
```

### 2. **Install Dependencies**

Create a virtual environment (recommended), then:

```bash
pip install -r requirements.txt
```

### 3. **Prepare Data**

- Place your cleaned taxi trip data (e.g., `final_features.csv`) in the `Outputs` folder.
- Place your trained model file (e.g., `xgb_classifier.pkl`) in the `Outputs` folder.
- Place `taxi_zone_centroids.csv` (with columns: `LocationID, zone, borough, lat, lon`) in the `Dataset/nyc_taxi_zone_mapping` folder.
    - If you only have the official shapefile, run the provided centroid extraction script to generate this CSV.

### 4. **Run the Dashboard**

```bash
streamlit run dashboard.py
```

The dashboard will open in your browser.

---

## 🛠️ Project Structure

```
traffic-dashboard/
│
├── dashboard.py                # Main Streamlit dashboard app
├── requirements.txt            # Python dependencies
├── Outputs/
│   ├── final_features.csv      # Cleaned taxi trip data
│   └── xgb_classifier.pkl      # Trained ML model
├── Dataset/
│   └── nyc_taxi_zone_mapping/
│       ├── taxi_zone_centroids.csv   # Taxi zone centroids (lat/lon)
│       └── taxi_zones.shp           # (Optional) Official shapefile
└── README.md
```

---

## ⚡ How It Works

- **Data Loading:** Loads trip data and merges with taxi zone centroids for mapping.
- **Filtering:** Sidebar controls let you filter by hour, zone, and distance.
- **Visualization:** Shows congestion trends and pickup locations on an interactive map.
- **Prediction:** Select rows to predict congestion using the trained ML model.
- **Download:** Export your filtered dataset for further analysis.

---

## 🧠 Model Training (Optional)

Model training is done separately (not in this dashboard).  
You can use any scikit-learn compatible model (e.g., XGBoost, RandomForest) trained on your engineered features.

---

## 📝 Notes & Best Practices

- Ensure all data files are in the correct paths as set in `dashboard.py`.
- The dashboard is optimized for performance using Streamlit caching.
- For large datasets, consider sampling or filtering to improve responsiveness.
- If you want to use real-time or live data, extend the data loading logic accordingly.

---

## 📦 Requirements

See [`requirements.txt`](requirements.txt) for all dependencies.

---

## 📄 License

MIT License (or your preferred license).

---

## 🙋‍♂️ Contact

For questions or contributions, please open an issue or contact Aditya Jod (aditya1710j@gmail.com).

---

