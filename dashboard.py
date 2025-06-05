import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import folium
from streamlit_folium import st_folium

# Set up Streamlit page configuration
st.set_page_config(page_title="Traffic Congestion Prediction Dashboard", layout="wide")

# --- Paths to model and data files ---
model_path = r"D:\Data Science Advance Projects\Predictive Traffic Congestion Modeling and Optimization for Smart City Mobility\traffic-dashboard\xgb_classifier.pkl"
data_path = r"D:\Data Science Advance Projects\Predictive Traffic Congestion Modeling and Optimization for Smart City Mobility\traffic-dashboard\final_features.csv"
TAXI_ZONE_CENTROIDS_PATH = r"D:\Data Science Advance Projects\Predictive Traffic Congestion Modeling and Optimization for Smart City Mobility\traffic-dashboard\taxi_zone_centroids.csv"

# --- Configuration for Map ---
PICKUP_LAT_COL_NAME_IN_DF = 'pickup_centroid_lat'
PICKUP_LON_COL_NAME_IN_DF = 'pickup_centroid_lon'

# --- Function Definitions ---
@st.cache_resource
def load_model(path):
    # Load a trained ML model from disk
    try:
        return joblib.load(path)
    except FileNotFoundError:
        st.error(f"Model file not found at {path}.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_data
def load_data(path, name="Data", nrows=10000):
    # Load a CSV file as a DataFrame, with error handling
    try:
        df_loaded = pd.read_csv(path, low_memory=False, nrows=nrows)
        return df_loaded
    except FileNotFoundError:
        st.error(f"{name} file not found at {path}.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading {name} from {path}: {e}")
        return pd.DataFrame()

# --- 1. Load Main Trip Data ---
df_trips = load_data(data_path, name="Trip Features Data")
if df_trips.empty:
    st.error("Main trip data could not be loaded. Dashboard cannot proceed.")
    st.stop()
else:
    st.success("Successfully loaded Trip Features Data.")

# --- 2. Initial Processing of df_trips (BEFORE MERGING CENTROIDS) ---
# Convert all datetime columns to string, handle missing values
datetime_col_names = ['tpep_pickup_datetime', 'tpep_dropoff_datetime']
for col_in_df in df_trips.columns:
    if ('date' in col_in_df.lower() or 'time' in col_in_df.lower()) and col_in_df not in datetime_col_names:
        datetime_col_names.append(col_in_df)
for col_name in list(set(datetime_col_names)):
    if col_name in df_trips.columns:
        df_trips[col_name] = pd.to_datetime(df_trips[col_name], errors='coerce').astype(str).replace('NaT', 'N/A')

# Prepare model features: ensure correct types and handle missing values
model_features = [
    'pickup_hour', 'pickup_day_of_week', 'is_weekend', 'is_rain', 'is_snow',
    'PU_zone', 'DO_zone', 'rush_hour', 'is_holiday', 'trip_distance',
    'avg_speed_mph', 'PRCP', 'SNOW'
]
for col in model_features:
    if col in df_trips.columns:
        if col not in ['PU_zone', 'DO_zone']:
            df_trips[col] = pd.to_numeric(df_trips[col], errors='coerce')
        elif col in ['PU_zone', 'DO_zone']:
            df_trips[col] = df_trips[col].astype(str).fillna('Unknown').astype('category')

# Fill missing values in numeric columns with 0
numeric_cols_in_trips = df_trips.select_dtypes(include=np.number).columns
df_trips.loc[:, numeric_cols_in_trips] = df_trips[numeric_cols_in_trips].fillna(0)

# --- 3. Load and Merge Taxi Zone Centroids Data ---
zone_centroids_df = load_data(TAXI_ZONE_CENTROIDS_PATH, name="Taxi Zone Centroids", nrows=None)
df = df_trips.copy()

if not zone_centroids_df.empty:
    ORIGINAL_CENTROID_ID_COL = 'LocationID'
    ORIGINAL_CENTROID_LAT_COL = 'lat'
    ORIGINAL_CENTROID_LON_COL = 'lon'
    
    # Check if required columns exist in centroids data
    if all(col in zone_centroids_df.columns for col in [ORIGINAL_CENTROID_ID_COL, ORIGINAL_CENTROID_LAT_COL, ORIGINAL_CENTROID_LON_COL]):
        centroids_to_merge = zone_centroids_df[[ORIGINAL_CENTROID_ID_COL, ORIGINAL_CENTROID_LAT_COL, ORIGINAL_CENTROID_LON_COL]].copy()
        try:
            # Ensure merge keys are of the same type
            if 'PULocationID' in df.columns:
                df['PULocationID'] = pd.to_numeric(df['PULocationID'], errors='coerce')
                centroids_to_merge[ORIGINAL_CENTROID_ID_COL] = pd.to_numeric(centroids_to_merge[ORIGINAL_CENTROID_ID_COL], errors='coerce')
                df.dropna(subset=['PULocationID'], inplace=True)
                centroids_to_merge.dropna(subset=[ORIGINAL_CENTROID_ID_COL], inplace=True)
                df['PULocationID'] = df['PULocationID'].astype(int)
                centroids_to_merge[ORIGINAL_CENTROID_ID_COL] = centroids_to_merge[ORIGINAL_CENTROID_ID_COL].astype(int)
        except Exception as e:
            st.warning(f"Could not ensure type consistency for merge keys: {e}")

        # Rename columns for merging and merge centroids into main DataFrame
        centroids_to_merge.rename(columns={
            ORIGINAL_CENTROID_ID_COL: 'PULocationID',
            ORIGINAL_CENTROID_LAT_COL: PICKUP_LAT_COL_NAME_IN_DF,
            ORIGINAL_CENTROID_LON_COL: PICKUP_LON_COL_NAME_IN_DF
        }, inplace=True)
        df = df.merge(centroids_to_merge, on='PULocationID', how='left')
        st.success("Successfully merged taxi zone centroids.")
    else:
        st.error(f"One or more original columns NOT found in Taxi Zone Centroids CSV. Please update the script.")
else:
    st.error(f"Taxi Zone Centroids data ('{TAXI_ZONE_CENTROIDS_PATH}') could not be loaded. Map will be empty.")

# --- 4. Load Model and Final Setup ---
model = load_model(model_path)
if model is None: st.warning("Model could not be loaded. Predictions unavailable.")
if df.empty:
    st.error("Main DataFrame `df` is empty before dashboard rendering. Halting.")
    st.stop()

def prepare_df_for_display(input_df):
    # Prepare DataFrame for display: convert types, handle missing values
    if not isinstance(input_df, pd.DataFrame): return input_df
    display_df = input_df.copy()
    display_df.columns = [str(col_name) for col_name in display_df.columns]
    for col in display_df.columns:
        if pd.api.types.is_object_dtype(display_df[col].dtype) or pd.api.types.is_string_dtype(display_df[col].dtype):
            display_df[col] = display_df[col].astype(str).fillna('N/A').replace(['NaT', 'nan', 'None', 'nat', '<NA>'], 'N/A')
        elif pd.api.types.is_datetime64_any_dtype(display_df[col].dtype):
            display_df[col] = display_df[col].dt.strftime('%Y-%m-%d %H:%M:%S').fillna('N/A')
    return display_df

# --- Page Title and Layout ---
st.title("🚦 Traffic Congestion Prediction Dashboard")
st.sidebar.header("Filter Data")

# --- Sidebar Filters ---
filtered_df = df.copy()
# Apply sidebar filters for pickup hour, pickup zone, and trip distance
if not df.empty:
    if 'pickup_hour' in df.columns:
        try: unique_pickup_hours = sorted(df['pickup_hour'].dropna().astype(int).unique())
        except ValueError: unique_pickup_hours = sorted(df['pickup_hour'].dropna().astype(str).unique())
        if unique_pickup_hours:
            pickup_hours = st.sidebar.multiselect("Pickup Hour", unique_pickup_hours, default=unique_pickup_hours)
            if pickup_hours : filtered_df = filtered_df[filtered_df['pickup_hour'].isin(pickup_hours)]
    if 'PU_zone' in df.columns:
        unique_pu_zones = sorted(df['PU_zone'].astype(str).dropna().unique())
        if unique_pu_zones:
            pu_zones = st.sidebar.multiselect("Pickup Zone", unique_pu_zones, default=unique_pu_zones)
            if pu_zones: filtered_df = filtered_df[filtered_df['PU_zone'].isin(pu_zones)]
    if 'trip_distance' in df.columns:
        min_dist, max_dist = 0.0, 1.0
        numeric_trip_distances = pd.to_numeric(df['trip_distance'], errors='coerce').dropna()
        if not numeric_trip_distances.empty:
            min_val, max_val = numeric_trip_distances.min(), numeric_trip_distances.max()
            min_dist, max_dist = float(min_val), float(max_val)
            if max_dist <= min_dist: max_dist = min_dist + 1.0
        trip_distance_range = st.sidebar.slider("Trip Distance", min_dist, max_dist, (min_dist, max_dist))
        if trip_distance_range[0] is not None:
            filtered_df = filtered_df[filtered_df['trip_distance'].between(trip_distance_range[0], trip_distance_range[1])]

# --- Main Page Layout ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Congestion Analysis")
    # Show congestion over time and by pickup zone
    if not filtered_df.empty:
        st.write("**Congestion Over Time (Actual)**")
        if 'pickup_hour' in filtered_df.columns and 'is_congested_speed' in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df['is_congested_speed']):
            hourly_congestion = filtered_df.groupby('pickup_hour', observed=True)['is_congested_speed'].mean().sort_index()
            st.line_chart(hourly_congestion, use_container_width=True)
        st.write("**Top 10 Congested Pickup Zones (Actual)**")
        if 'PU_zone' in filtered_df.columns and 'is_congested_speed' in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df['is_congested_speed']):
            zone_congestion = filtered_df.groupby('PU_zone', observed=True)['is_congested_speed'].mean().sort_values(ascending=False).head(10)
            st.bar_chart(zone_congestion, use_container_width=True)

with col2:
    st.subheader("Pickup Location Map")
    # Display pickup locations on a Folium map
    if not filtered_df.empty and PICKUP_LAT_COL_NAME_IN_DF in filtered_df.columns and PICKUP_LON_COL_NAME_IN_DF in filtered_df.columns:
        map_data_for_plot = filtered_df[[PICKUP_LAT_COL_NAME_IN_DF, PICKUP_LON_COL_NAME_IN_DF]].copy()
        map_data_for_plot.dropna(subset=[PICKUP_LAT_COL_NAME_IN_DF, PICKUP_LON_COL_NAME_IN_DF], inplace=True)
        map_data_for_plot.rename(columns={PICKUP_LAT_COL_NAME_IN_DF: 'lat', PICKUP_LON_COL_NAME_IN_DF: 'lon'}, inplace=True)
        if not map_data_for_plot.empty:
            map_data_for_plot['lat'] = pd.to_numeric(map_data_for_plot['lat'], errors='coerce')
            map_data_for_plot['lon'] = pd.to_numeric(map_data_for_plot['lon'], errors='coerce')
            map_data_for_plot.dropna(subset=['lat', 'lon'], inplace=True)

            if not map_data_for_plot.empty:
                st.info(f"Displaying {min(len(map_data_for_plot), 5000)} pickup locations on the map.")
                nyc_map = folium.Map(location=[40.7128, -73.935242], zoom_start=11)
                for idx, row in map_data_for_plot.head(5000).iterrows():
                    folium.CircleMarker(location=[row['lat'], row['lon']], radius=2, color='blue', fill=True, fill_color='blue', fill_opacity=0.6).add_to(nyc_map)
                st_folium(nyc_map, width=725, height=450)

                with st.expander("Show Map Data Debugging Info"):
                    st.write("**Descriptive statistics for map coordinates (`lat`, `lon`):**")
                    st.dataframe(map_data_for_plot[['lat', 'lon']].describe())

# --- Prediction Section (FIXED and RESTORED) ---
st.header("Predict Congestion on Selected Trips")
if not filtered_df.empty:
    # Allow user to select rows for prediction
    selectable_indices = filtered_df.index.tolist()
    default_selection_count = min(5, len(selectable_indices))
    default_selected_indices = [idx for idx in selectable_indices[:default_selection_count] if idx in filtered_df.index]
    
    selected_indices = st.multiselect("Select one or more rows for prediction:", options=selectable_indices, default=default_selected_indices)

    if selected_indices:
        valid_selected_indices = [idx for idx in selected_indices if idx in filtered_df.index]
        if valid_selected_indices:
            to_predict_base_df = filtered_df.loc[valid_selected_indices].copy()
            prediction_input_df = to_predict_base_df.copy()
            # Ensure all model features are present
            for feature in model_features:
                if feature not in prediction_input_df.columns:
                    prediction_input_df[feature] = 'Unknown' if feature in ['PU_zone', 'DO_zone'] else 0
            
            # Encode categorical columns as codes
            for cat_col in ['PU_zone', 'DO_zone']:
                if cat_col in prediction_input_df.columns:
                    if not isinstance(prediction_input_df[cat_col].dtype, pd.CategoricalDtype):
                        prediction_input_df[cat_col] = prediction_input_df[cat_col].astype('category')
                    prediction_input_df[cat_col] = prediction_input_df[cat_col].cat.codes
            
            prediction_input_df = prediction_input_df[model_features].fillna(0)

            try:
                if model is not None:
                    # Get predictions from the model
                    predictions = model.predict(prediction_input_df)
                    
                    # Prepare results for display
                    result_df = to_predict_base_df.copy()
                    
                    # Handle Regressor vs Classifier model type
                    if 'Regressor' in type(model).__name__:
                        result_df['Predicted_Value'] = predictions # Regressors predict a value
                        cols_to_show = ['PULocationID', 'DOLocationID', 'trip_distance', 'pickup_hour', 'Predicted_Value']
                    else: # Assume it's a classifier
                        result_df['Predicted_Congestion_Class'] = predictions # Classifiers predict a class
                        cols_to_show = ['PULocationID', 'DOLocationID', 'trip_distance', 'pickup_hour', 'Predicted_Congestion_Class']
                        # Only get probabilities if the model is a classifier
                        if hasattr(model, 'predict_proba'):
                            probabilities = model.predict_proba(prediction_input_df)[:, 1]
                            result_df['Probability_Congested'] = probabilities
                            cols_to_show.append('Probability_Congested')

                    st.write("**Prediction Results:**")
                    final_display_cols_results = [c for c in cols_to_show if c in result_df.columns]
                    st.dataframe(prepare_df_for_display(result_df[final_display_cols_results]))
                else: st.error("Model is not loaded.")
            except Exception as e:
                st.error(f"Prediction error: {e}")
                st.error("This could be due to a mismatch between the data features and what the model was trained on.")
else:
    st.info("No data available to make predictions.")

# --- Download and Footer ---
st.sidebar.markdown("---")
if not filtered_df.empty:
    # Allow user to download filtered data as CSV
    display_cols = [col for col in df.columns if col not in ['store_and_fwd_flag']]
    download_df = prepare_df_for_display(filtered_df[[c for c in display_cols if c in filtered_df.columns]])
    csv_data = download_df.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button("Download Filtered Data as CSV", csv_data, "filtered_traffic_data.csv", "text/csv")
else: st.sidebar.info("No filtered data to download.")

# --- Footer ---
st.sidebar.markdown("---")
st.markdown("<div style='text-align: center; color: gray; font-size: 20px; margin-top: 30px;'>© 2025 Traffic Congestion Prediction Dashboard </div>", unsafe_allow_html=True)

# --- End of Dashboard Code ---
