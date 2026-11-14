# [import of library]
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from pathlib import Path


# [start of preprocessing functions]

# (Helper: Missing Value Imputation for Week 1 Part B1 Benchmark)
def apply_imputation(df, method="zero"):
    """
    Handles missing or artificially masked values for Week 1 benchmarking.
    Options: 'zero' (Zero-Fill), 'ffill' (Forward-Fill).
    """
    df_copy = df.copy()
    if method == "zero":
        return df_copy.fillna(0)
    elif method == "ffill":
        return df_copy.ffill().bfill()
    return df_copy


# (Log Scaling)
def apply_log_scaling(df, columns):
    """Applies log1p scaling to handle heavily skewed financial/downtime features."""
    df_copy = df.copy()
    for col in columns:
        if col in df_copy.columns:
            # Clip at 0 to ensure no negative values cause NaN in log1p
            df_copy[col] = np.log1p(np.maximum(0, df_copy[col]))
    return df_copy


# (StandardScaler)
def apply_standard_scaler(df, columns):
    """Standardizes features by removing the mean and scaling to unit variance."""
    df_copy = df.copy()
    valid_cols = [c for c in columns if c in df_copy.columns]
    if valid_cols:
        scaler = StandardScaler()
        df_copy[valid_cols] = scaler.fit_transform(df_copy[valid_cols])
    return df_copy


# (One-Hot Encoding)
def apply_one_hot_encoding(df, columns):
    """Converts categorical variables into dummy/indicator variables (0 and 1)."""
    df_copy = df.copy()
    valid_cols = [c for c in columns if c in df_copy.columns]
    if valid_cols:
        df_copy = pd.get_dummies(df_copy, columns=valid_cols, drop_first=False, dtype=int)
    return df_copy


# (Sliding Windows + RobustScaler)
def apply_sliding_windows_and_robust(df, group_cols, sort_col, sensor_columns, window_size=3):
    """
    Sorts by usage progression within vehicle cohorts, creates t-1 to t-(window_size-1)
    lag features to capture degradation trajectories, and scales via RobustScaler.
    """
    df_copy = df.copy()
    valid_sensors = [c for c in sensor_columns if c in df_copy.columns]

    # Sort chronologically by Usage_Hours within vehicle cohorts to form degradation trajectories
    df_copy = df_copy.sort_values(by=group_cols + [sort_col]).reset_index(drop=True)

    all_sensor_cols = list(valid_sensors)
    grouped = df_copy.groupby(group_cols)

    for lag in range(1, window_size):
        for col in valid_sensors:
            lag_col_name = f"{col}_lag_{lag}"
            # Create lag and backfill the initial boundary rows of each cohort
            df_copy[lag_col_name] = grouped[col].shift(lag).bfill()
            all_sensor_cols.append(lag_col_name)

    # Restore original Vehicle_ID ordering
    if "Vehicle_ID" in df_copy.columns:
        df_copy = df_copy.sort_values(by="Vehicle_ID").reset_index(drop=True)

    # Apply RobustScaler across both current and lagged sensor columns
    if all_sensor_cols:
        scaler = RobustScaler()
        df_copy[all_sensor_cols] = scaler.fit_transform(df_copy[all_sensor_cols])

    return df_copy


# (IQR Clipping + StandardScaler)
def apply_iqr_clipping_and_standard(df, columns):
    """Clips extreme outliers using the 1.5x Interquartile Range rule, then standardizes."""
    df_copy = df.copy()
    valid_cols = [c for c in columns if c in df_copy.columns]

    for col in valid_cols:
        Q1 = df_copy[col].quantile(0.25)
        Q3 = df_copy[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_copy[col] = np.clip(df_copy[col], lower_bound, upper_bound)

    if valid_cols:
        scaler = StandardScaler()
        df_copy[valid_cols] = scaler.fit_transform(df_copy[valid_cols])

    return df_copy


# (RobustScaler)
def apply_robust_scaler(df, columns):
    """Scales features using median and IQR statistics that are robust to outliers."""
    df_copy = df.copy()
    valid_cols = [c for c in columns if c in df_copy.columns]
    if valid_cols:
        scaler = RobustScaler()
        df_copy[valid_cols] = scaler.fit_transform(df_copy[valid_cols])
    return df_copy


# (MinMaxScaler)
def apply_min_max_scaler(df, columns):
    """Scales features to a strictly bounded [0, 1] range."""
    df_copy = df.copy()
    valid_cols = [c for c in columns if c in df_copy.columns]
    if valid_cols:
        scaler = MinMaxScaler()
        df_copy[valid_cols] = scaler.fit_transform(df_copy[valid_cols])
    return df_copy


# (Ordinal Encoding)
def apply_ordinal_encoding(df, mappings):
    """Maps sequential string categories to integers based on a provided dictionary."""
    df_copy = df.copy()
    for col, mapping in mappings.items():
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].map(mapping)
    return df_copy


# [end of preprocessing functions]


# [start of the main function]
def main():
    # Dynamically resolve the ML_Assignment root directory and Data_Set folder
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "Data_Set"

    # (import the dataset)
    file_path = data_dir / "logistics_predictive_maintenanceV2.csv"
    print(f"1. Loading dataset: {file_path}...")
    df = pd.read_csv(file_path)
    print(f"   Initial shape: {df.shape}")

    # (Drop column)
    cols_to_drop = [
        "Last_Maintenance_Date",
        "Maintenance_Severity_ID",
        "Data_Split",
        "FL_Client_ID",
        "Partition_Type",
        "Local_Epochs_Per_Round",
        "Communication_Rounds",
        "Pre_Event_Record",
    ]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    print(f"2. Dropped metadata/duplicate columns. Shape: {df.shape}")

    # Feature Engineering: Convert Year_of_Manufacture -> Vehicle_Age
    if "Year_of_Manufacture" in df.columns:
        current_year = 2026
        df["Vehicle_Age"] = current_year - df["Year_of_Manufacture"]
        df = df.drop(columns=["Year_of_Manufacture"])
        print("3. Engineered 'Vehicle_Age' from 'Year_of_Manufacture'.")

    # Optional: Run baseline imputation (change 'zero' to 'ffill' for Week 1 benchmarking)
    # df = apply_imputation(df, method="zero")

    # Define feature groups according to the preprocessing strategy
    ordinal_mappings = {
        "Brake_Condition": {"Poor": 0, "Fair": 1, "Good": 2},
        "Maintenance_Severity": {"Normal": 0, "Minor Maintenance": 1, "Major Maintenance": 2},
        "Edge_Device_Class": {
            "ARM_Cortex_A53": 0,
            "ARM_Cortex_A72": 1,
            "Automotive_ECU": 2,
            "NVIDIA_Jetson_Nano": 3
        }
    }

    one_hot_cols = [
        "Make_and_Model", "Vehicle_Type", "Route_Info", "Weather_Conditions",
        "Road_Conditions", "Communication_Interface", "Telematics_Gateway",
        "Maintenance_Type"
    ]

    sensor_sliding_cols = [
        "Engine_Temperature", "Tire_Pressure", "Fuel_Consumption",
        "Battery_Status", "Vibration_Levels", "Oil_Quality"
    ]

    robust_cols = ["Usage_Hours", "Load_Capacity", "Actual_Load"]

    iqr_standard_cols = ["CAN_Message_Rate_Hz", "Sensor_Packet_Loss_Rate"]

    standard_cols = [
        "Days_Since_Last_Maintenance", "PCR", "UIR", "TPI",
        "MBF", "ADS", "OHI", "CMES", "UER"
    ]

    minmax_cols = [
        "Vehicle_Age", "Failure_History", "Anomalies_Detected",
        "Diagnostic_Trouble_Code_Count"
    ]

    log_reward_cols = [
        "Maintenance_Cost", "Historical_Maintenance_Cost",
        "Downtime_Maintenance", "Impact_on_Efficiency"
    ]

    # Go through the pipeline
    print("4. Running preprocessing transformations...")

    # Step A: Sliding windows MUST run before One-Hot Encoding so we can group by Make_and_Model & Vehicle_Type
    df = apply_sliding_windows_and_robust(
        df,
        group_cols=["Make_and_Model", "Vehicle_Type"],
        sort_col="Usage_Hours",
        sensor_columns=sensor_sliding_cols,
        window_size=3
    )
    print("   -> Sliding Windows (t-1, t-2) + RobustScaler applied to core sensors.")

    # Step B: Encodings
    df = apply_ordinal_encoding(df, ordinal_mappings)
    print("   -> Ordinal Encoding applied.")

    df = apply_one_hot_encoding(df, one_hot_cols)
    print("   -> One-Hot Encoding applied.")

    # Step C: Scalers & Clipping
    df = apply_robust_scaler(df, robust_cols)
    print("   -> RobustScaler applied to usage/load metrics.")

    df = apply_iqr_clipping_and_standard(df, iqr_standard_cols)
    print("   -> IQR Clipping + StandardScaler applied to CAN/Telemetry rates.")

    df = apply_standard_scaler(df, standard_cols)
    print("   -> StandardScaler applied to advanced engineering indices.")

    df = apply_min_max_scaler(df, minmax_cols)
    print("   -> MinMaxScaler applied to age and discrete fault counts.")

    df = apply_log_scaling(df, log_reward_cols)
    print("   -> Log1p Scaling applied to cost/downtime reward features.")

    # Drop Vehicle_ID after sorting/grouping so it doesn't pollute neural network weights
    #if "Vehicle_ID" in df.columns:
    #    df = df.drop(columns=["Vehicle_ID"])
    #    print("5. Dropped 'Vehicle_ID' post-alignment.")

     # Update at the bottom of main() to save inside Data_Set/processed_dataset.csv
    output_path = data_dir / "processed_dataset.csv"
    print(f"6. Saving processed dataset to {output_path} (Final Shape: {df.shape})...")
    df.to_csv(output_path, index=False)
    print("Preprocessing complete!")


if __name__ == "__main__":
    main()
# [end of the main function]