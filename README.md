# Traffic Congestion Prediction Dashboard

A Streamlit dashboard for **Predictive Traffic Congestion Modeling and Optimization for Smart City Mobility** in New York City.  
This dashboard visualizes traffic congestion predictions, enables interactive exploration of traffic patterns, and demonstrates the power of machine learning for urban mobility.

---

## Features

- **Interactive Map:** Visualize taxi pickup locations by NYC taxi zone centroids.
- **Congestion Analytics:** Explore congestion trends by hour and by pickup zone.
- **ML Predictions:** Predict congestion levels for selected trips using a trained ML model (e.g., XGBoost).
- **Data Filtering:** Filter trips by hour, zone, and distance using sidebar controls.
- **Download:** Export filtered data as CSV.
- **Responsive Layout:** Clean, modern UI with Streamlit and Folium.

---

## Data Used

- **NYC Taxi Trip Data:** Cleaned CSV with engineered features.
- **NYC Taxi Zone Centroids:** CSV with latitude/longitude for each taxi zone (generated from official shapefile).
- **Trained ML Model:** Pre-trained XGBoost or RandomForest model (`.pkl`).
- *(Optional)* Weather and event data for advanced features.

---

## Getting Started

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

- Place a sampled dataset (e.g., `sample_data.csv`) in the project root or appropriate path as referenced in `dashboard.py`.
    - This should contain engineered features used for prediction.
    - If using full data (`final_features.csv`), ensure it's under hosting size limits (e.g., <100 MB for Streamlit Cloud).
- Place your trained model file (e.g., `xgb_classifier.pkl`) in the `Outputs` folder.
- Place `taxi_zone_centroids.csv` (with columns: `LocationID, zone, borough, lat, lon`) in the `Dataset/nyc_taxi_zone_mapping` folder.
    - If you only have the official shapefile, run the provided centroid extraction script to generate this CSV.

### 4. **Run the Dashboard**

```bash
streamlit run dashboard.py
```

The dashboard will open in your browser.

---

## Project Structure
```
traffic-dashboard/
│
├── dashboard.py             # Main Streamlit dashboard app
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
└── data/                    # Data and model files
    ├── sample_data.csv          # Sampled taxi trip data (~25,000 rows)
    ├── taxi_zone_centroids.csv  # NYC taxi zone centroid data (LocationID, zone, borough, lat, lon)
    └── xgb_classifier.pkl       # Trained ML model (XGBoost)
```

```

---

## Deployment (Streamlit Cloud)

To deploy the dashboard on [Streamlit Cloud](https://streamlit.io/cloud):

1. **Push your repo** to GitHub, making sure it includes:
   - `dashboard.py`
   - `sample_data.csv`
   - `taxi_zone_centroids.csv`
   - `xgb_classifier.pkl`
   - `requirements.txt`

2. **Go to** [https://share.streamlit.io](https://share.streamlit.io) and log in with your GitHub account.

3. **Select your repository** and choose `dashboard.py` as the entry point.

4. Click **Deploy**. Streamlit will automatically install dependencies and launch the app.

> ⚠️ Make sure all files are under 100 MB. Avoid using large datasets like `final_features.csv` — use `sample_data.csv` (25,000 rows) instead for better performance and compatibility with free hosting limits.

---

## How It Works

- **Data Loading:** Loads trip data and merges with taxi zone centroids for mapping.
- **Filtering:** Sidebar controls let you filter by hour, zone, and distance.
- **Visualization:** Shows congestion trends and pickup locations on an interactive map.
- **Prediction:** Select rows to predict congestion using the trained ML model.
- **Download:** Export your filtered dataset for further analysis.

---

## Model Training (Optional)

Model training is done separately (not in this dashboard).  
You can use any scikit-learn compatible model (e.g., XGBoost, RandomForest) trained on your engineered features.

---

## Notes & Best Practices

- Ensure all data files are in the correct paths as set in `dashboard.py`.
- The dashboard is optimized for performance using Streamlit caching.
- For large datasets, consider sampling or filtering to improve responsiveness.
- If you want to use real-time or live data, extend the data loading logic accordingly.

---

## Requirements

See [`requirements.txt`](requirements.txt) for all dependencies.

---

## License

This project is licensed under the [MIT License](LICENSE).

---
