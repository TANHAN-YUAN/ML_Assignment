# [import of library]
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler


# [start of preprocessing functions]

# (Physics & Real-World Domain Filter)
def apply_physics_and_domain_filter(df, drop_ev_mode="strict"):
    """
    Enforces real-world ICE automotive physics and fleet domain constraints.
    drop_ev_mode='strict': Drops any row where Vehicle_Type == 'Electric Truck' OR Make_and_Model == 'Tesla Semi' (retains 174,876 rows).
    """
    df_copy = df.copy()

    # 1. Strictly drop Electric Vehicles
    if drop_ev_mode == "strict":
        ev_mask = (df_copy["Vehicle_Type"] == "Electric Truck") | (df_copy["Make_and_Model"] == "Tesla Semi")
        df_copy = df_copy[~ev_mask].reset_index(drop=True)

    # 2. Compute Vehicle_Age (needed for physical Usage_Hours ceiling)
    current_year = 2026
    if "Year_of_Manufacture" in df_copy.columns:
        df_copy["Vehicle_Age"] = np.maximum(1, current_year - df_copy["Year_of_Manufacture"])
        df_copy = df_copy.drop(columns=["Year_of_Manufacture"])

    # 3. Preserve original payload ratio (PCR) before correcting Load_Capacity
    raw_load_ratio = np.clip(df_copy["Actual_Load"] / df_copy["Load_Capacity"], 0.10, 1.05)

    # 4. Align ICE Vehicle_Type and Load_Capacity (kg) with real-world Make_and_Model specs
    light_models = ["Ford F-150", "Chevy Silverado"]
    medium_models = ["Isuzu NPR"]
    heavy_ice_models = ["Volvo FH", "Scania R-Series", "Mercedes Actros", "MAN TGX"]

    df_copy.loc[df_copy["Make_and_Model"].isin(light_models), "Vehicle_Type"] = "Light Truck"
    df_copy.loc[
        df_copy["Make_and_Model"].isin(medium_models) &
        (~df_copy["Vehicle_Type"].isin(["Medium Truck", "Refrigerated Truck"])),
        "Vehicle_Type"
    ] = "Medium Truck"
    df_copy.loc[
        df_copy["Make_and_Model"].isin(heavy_ice_models) &
        (~df_copy["Vehicle_Type"].isin(["Heavy Truck", "Refrigerated Truck"])),
        "Vehicle_Type"
    ] = "Heavy Truck"

    # Clamp Load_Capacity to realistic chassis payload ratings (kg)
    df_copy.loc[df_copy["Make_and_Model"].isin(light_models), "Load_Capacity"] = (
        df_copy.loc[df_copy["Make_and_Model"].isin(light_models), "Load_Capacity"].clip(1500, 2500)
    )
    df_copy.loc[df_copy["Make_and_Model"].isin(medium_models), "Load_Capacity"] = (
        df_copy.loc[df_copy["Make_and_Model"].isin(medium_models), "Load_Capacity"].clip(5000, 10000)
    )
    df_copy.loc[df_copy["Make_and_Model"].isin(heavy_ice_models), "Load_Capacity"] = (
        df_copy.loc[df_copy["Make_and_Model"].isin(heavy_ice_models), "Load_Capacity"].clip(15000, 25000)
    )

    # Recompute Actual_Load from realistic Load_Capacity and bounded load ratio
    df_copy["Actual_Load"] = df_copy["Load_Capacity"] * raw_load_ratio

    # 5. Enforce physical Usage_Hours ceiling (Max 4,500 operating hours/year of age)
    max_possible_hours = df_copy["Vehicle_Age"] * 4500.0
    df_copy["Usage_Hours"] = np.minimum(df_copy["Usage_Hours"], max_possible_hours)

    # 6. Physical ICE sensor & downtime boundary checks
    df_copy["Engine_Temperature"] = df_copy["Engine_Temperature"].clip(70.0, 125.0)
    df_copy["Tire_Pressure"] = df_copy["Tire_Pressure"].clip(22.0, 44.0)
    df_copy["Battery_Status"] = df_copy["Battery_Status"].clip(20.0, 100.0)
    df_copy["Oil_Quality"] = df_copy["Oil_Quality"].clip(20.0, 100.0)

    minor_mask = df_copy["Maintenance_Severity"] == "Minor Maintenance"
    major_mask = df_copy["Maintenance_Severity"] == "Major Maintenance"
    df_copy.loc[minor_mask, "Downtime_Maintenance"] = np.maximum(1.0, df_copy.loc[minor_mask, "Downtime_Maintenance"])
    df_copy.loc[major_mask, "Downtime_Maintenance"] = np.maximum(4.0, df_copy.loc[major_mask, "Downtime_Maintenance"])

    # 7. Synchronize derived engineering features (PCR, UIR, UER)
    df_copy["PCR"] = (df_copy["Actual_Load"] / df_copy["Load_Capacity"]).round(4)
    df_copy["UIR"] = (df_copy["Usage_Hours"] / (df_copy["Vehicle_Age"] * 2000.0)).round(4)
    df_copy["UER"] = (df_copy["Usage_Hours"] / (df_copy["Maintenance_Cost"] + 1.0)).round(4)

    return df_copy


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

    df_copy = df_copy.sort_values(by=group_cols + [sort_col]).reset_index(drop=True)

    all_sensor_cols = list(valid_sensors)
    grouped = df_copy.groupby(group_cols)

    for lag in range(1, window_size):
        for col in valid_sensors:
            lag_col_name = f"{col}_lag_{lag}"
            df_copy[lag_col_name] = grouped[col].shift(lag).bfill()
            all_sensor_cols.append(lag_col_name)

    if "Vehicle_ID" in df_copy.columns:
        df_copy = df_copy.sort_values(by="Vehicle_ID").reset_index(drop=True)

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
    # Dynamically resolve paths relative to SRC/data_prepprocessing.py
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "Data_Set"

    # (import the dataset)
    file_path = data_dir / "logistics_predictive_maintenanceV2.csv"
    print(f"1. Loading dataset: {file_path}...")
    df = pd.read_csv(file_path)
    print(f"   Initial shape: {df.shape}")

    # (Drop column - includes metadata, hardware tags, and ignored CAN/Packet_Loss columns)
    cols_to_drop = [
        "Last_Maintenance_Date", "Maintenance_Severity_ID", "Data_Split",
        "FL_Client_ID", "Partition_Type", "Local_Epochs_Per_Round",
        "Communication_Rounds", "Pre_Event_Record", "Historical_Maintenance_Cost",
        "Communication_Interface", "Telematics_Gateway", "Edge_Device_Class",
        "CAN_Message_Rate_Hz", "Sensor_Packet_Loss_Rate"
    ]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    print(f"2. Dropped metadata/hardware/CAN columns. Shape: {df.shape}")

    # Apply Physics & Real-World Domain Filter with strict EV removal
    df = apply_physics_and_domain_filter(df, drop_ev_mode="strict")
    print(f"3. Applied Physics & Domain Filter (strictly removed EVs, fixed ICE specs). Shape: {df.shape}")

    # Optional: Run baseline imputation (change 'zero' to 'ffill' for Week 1 benchmarking)
    df = apply_imputation(df, method="zero")

    # Define feature groups according to the preprocessing & EDA outlier strategy
    ordinal_mappings = {
        "Brake_Condition": {"Poor": 0, "Fair": 1, "Good": 2},
        "Maintenance_Severity": {"Normal": 0, "Minor Maintenance": 1, "Major Maintenance": 2}
    }

    one_hot_cols = [
        "Make_and_Model", "Vehicle_Type", "Route_Info", "Weather_Conditions",
        "Road_Conditions", "Maintenance_Type"
    ]

    sensor_sliding_cols = [
        "Engine_Temperature", "Tire_Pressure", "Fuel_Consumption",
        "Battery_Status", "Vibration_Levels", "Oil_Quality"
    ]

    # Included MBF, ADS, CMES in RobustScaler due to >9% heavy right-tail outliers found in EDA
    robust_cols = ["Usage_Hours", "Load_Capacity", "Actual_Load", "MBF", "ADS", "CMES"]

    standard_cols = [
        "Days_Since_Last_Maintenance", "PCR", "UIR", "TPI", "OHI", "UER"
    ]

    minmax_cols = [
        "Vehicle_Age", "Failure_History", "Anomalies_Detected",
        "Diagnostic_Trouble_Code_Count"
    ]

    log_reward_cols = [
        "Maintenance_Cost", "Downtime_Maintenance", "Impact_on_Efficiency"
    ]

    # Go through the pipeline
    print("4. Running preprocessing transformations...")

    df = apply_sliding_windows_and_robust(
        df,
        group_cols=["Make_and_Model", "Vehicle_Type"],
        sort_col="Usage_Hours",
        sensor_columns=sensor_sliding_cols,
        window_size=3
    )
    print("   -> Sliding Windows (t-1, t-2) + RobustScaler applied to core sensors.")

    df = apply_ordinal_encoding(df, ordinal_mappings)
    print("   -> Ordinal Encoding applied.")

    df = apply_one_hot_encoding(df, one_hot_cols)
    print("   -> One-Hot Encoding applied.")

    df = apply_robust_scaler(df, robust_cols)
    print("   -> RobustScaler applied to usage, load, and heavy-tailed engineering indices (MBF, ADS, CMES).")

    df = apply_standard_scaler(df, standard_cols)
    print("   -> StandardScaler applied to standard engineering indices.")

    df = apply_min_max_scaler(df, minmax_cols)
    print("   -> MinMaxScaler applied to age and discrete fault counts.")

    df = apply_log_scaling(df, log_reward_cols)
    print("   -> Log1p Scaling applied to cost/downtime reward features.")

    if "Vehicle_ID" in df.columns:
        df = df.drop(columns=["Vehicle_ID"])
        print("5. Dropped 'Vehicle_ID' post-alignment.")

    # (save the processed data into a dataset named "processed_dataset.csv" inside Data_Set/)
    output_path = data_dir / "processed_dataset.csv"
    print(f"6. Saving processed dataset to {output_path} (Final Shape: {df.shape})...")
    df.to_csv(output_path, index=False)
    print("Preprocessing complete!")


if __name__ == "__main__":
    main()
# [end of the main function]