import pandas as pd
import numpy as np


def filter_and_process_logistics_v2(
    input_csv: str = "logistics_predictive_maintenanceV2.csv",
    output_csv: str = "logistics_predictive_maintenanceV2(filtered).csv",
    strict_row_filter_only: bool = False,
    drop_leaked_features: bool = True,
    drop_post_event_outcomes: bool = False,
) -> pd.DataFrame:
    """
    Loads, filters, and sanitizes logistics_predictive_maintenanceV2.csv,
    then saves the cleaned dataset to 'logistics_predictive_maintenanceV2(filtered).csv'.

    Parameters:
    -----------
    input_csv : str
        Path to the raw 250,000-row dataset.
    output_csv : str
        Output filename for the filtered CSV.
    strict_row_filter_only : bool (default=False)
        If False (recommended), retains all 174,876 valid ICE truck rows and repairs
        their Vehicle_Type and Load_Capacity to match their real Make_and_Model.
        If True, strictly drops rows where Make_and_Model, Vehicle_Type, and Load_Capacity
        were randomly mismatched.
    drop_leaked_features : bool (default=True)
        Drops closed-form algebraic indices (UIR, TPI, MBF, ADS, OHI, CMES, UER)
        and the pre-assigned target score (Predictive_Score) to prevent data leakage.
    drop_post_event_outcomes : bool (default=False)
        Set to True if training a strictly pre-service classifier where post-event
        Maintenance_Cost, Downtime_Maintenance, and Impact_on_Efficiency are unavailable at inference.
    """
    print(f"[1/5] Loading dataset from '{input_csv}'...")
    df = pd.read_csv(input_csv)
    initial_rows, initial_cols = df.shape
    print(f"      Loaded {initial_rows:,} rows and {initial_cols} columns.")

    # -------------------------------------------------------------------------
    # STEP 1: Remove Electric Truck & Tesla Semi Contradictions
    # (All EV rows in this dataset contain internal combustion Oil_Quality,
    # Fuel_Consumption, Engine_Temperature, and Oil Change / Engine Overhaul logs)
    # -------------------------------------------------------------------------
    ev_mask = (df["Make_and_Model"] == "Tesla Semi") | (
        df["Vehicle_Type"] == "Electric Truck"
    )
    df = df[~ev_mask].copy()
    print(
        f"[2/5] Removed {ev_mask.sum():,} EV/ICE contradiction rows -> {len(df):,} rows remaining."
    )

    # -------------------------------------------------------------------------
    # STEP 2: Fix Make_and_Model vs. Vehicle_Type & Load_Capacity Mismatches
    # -------------------------------------------------------------------------
    # Preserve the original payload utilization ratio (PCR = Actual_Load / Load_Capacity)
    original_pcr = (df["Actual_Load"] / df["Load_Capacity"]).clip(
        lower=0.10, upper=1.10
    )

    heavy_models = ["Volvo FH", "Scania R-Series", "MAN TGX", "Mercedes Actros"]
    medium_models = ["Isuzu NPR"]
    light_models = ["Ford F-150", "Chevy Silverado"]

    if strict_row_filter_only:
        valid_type_mask = (
            (
                df["Make_and_Model"].isin(light_models)
                & (df["Vehicle_Type"] == "Light Truck")
                & (df["Load_Capacity"].isin([1500, 2500]))
            )
            | (
                df["Make_and_Model"].isin(medium_models)
                & (df["Vehicle_Type"].isin(["Medium Truck", "Refrigerated Truck"]))
                & (df["Load_Capacity"].isin([5000, 10000]))
            )
            | (
                df["Make_and_Model"].isin(heavy_models)
                & (df["Vehicle_Type"].isin(["Heavy Truck", "Refrigerated Truck"]))
                & (df["Load_Capacity"].isin([15000, 20000, 25000]))
            )
        )
        df = df[valid_type_mask].copy()
        print(
            f"[3/5] Strict row filter applied -> {len(df):,} physically consistent rows remaining."
        )
    else:
        # Align Vehicle_Type to Make_and_Model while preserving Refrigerated Truck on Cold-Chain routes
        conditions_type = [
            df["Make_and_Model"].isin(heavy_models)
            & (df["Route_Info"] == "Cold-Chain Route"),
            df["Make_and_Model"].isin(heavy_models),
            df["Make_and_Model"].isin(medium_models)
            & (df["Route_Info"] == "Cold-Chain Route"),
            df["Make_and_Model"].isin(medium_models),
            df["Make_and_Model"].isin(light_models),
        ]
        choices_type = [
            "Refrigerated Truck",
            "Heavy Truck",
            "Refrigerated Truck",
            "Medium Truck",
            "Light Truck",
        ]
        df["Vehicle_Type"] = np.select(
            conditions_type, choices_type, default=df["Vehicle_Type"]
        )

        # Assign realistic Load_Capacity (kg) per model and rescale Actual_Load proportionally
        model_capacity_map = {
            "Ford F-150": 1500,
            "Chevy Silverado": 2500,
            "Isuzu NPR": 10000,
            "MAN TGX": 20000,
            "Mercedes Actros": 25000,
            "Scania R-Series": 25000,
            "Volvo FH": 25000,
        }
        df["Load_Capacity"] = df["Make_and_Model"].map(model_capacity_map).astype(int)
        df["Actual_Load"] = (df["Load_Capacity"] * original_pcr).round(4)
        df["PCR"] = (df["Actual_Load"] / df["Load_Capacity"]).round(4)

        # Reassign light pickup trucks off Cold-Chain Routes to Urban Delivery
        light_cold_mask = df["Make_and_Model"].isin(light_models) & (
            df["Route_Info"] == "Cold-Chain Route"
        )
        df.loc[light_cold_mask, "Route_Info"] = "Urban Delivery"
        print(
            f"[3/5] Aligned Vehicle_Type, Load_Capacity, and Actual_Load across {len(df):,} ICE rows."
        )

    # -------------------------------------------------------------------------
    # STEP 3: Enforce Physical Sensor & Operational Bounds
    # -------------------------------------------------------------------------
    valid_sensor_mask = (
        (df["Usage_Hours"] >= 0.0)
        & (df["Engine_Temperature"].between(60.0, 135.0))
        & (df["Tire_Pressure"].between(15.0, 55.0))
        & (df["Oil_Quality"].between(0.0, 100.0))
        & (df["Battery_Status"].between(0.0, 100.0))
        & (df["Vibration_Levels"] >= 0.0)
        & (df["Days_Since_Last_Maintenance"] >= 0)
    )
    df = df[valid_sensor_mask].copy()

    # -------------------------------------------------------------------------
    # STEP 4: Drop Duplicate, Zero-Variance Constant, and Leaked Columns
    # -------------------------------------------------------------------------
    cols_to_drop = [
        "Communication_InterFace",
        "Historical_Maintenance_Cost",  # 100% identical to Maintenance_Cost
        "Pre_Event_Record",             # Constant = 1 across all rows
        "Partition_Type",               # Constant = 'Non-IID Dirichlet alpha=0.5'
        "Local_Epochs_Per_Round",       # Constant = 2 across all rows
        "Communication_Rounds",         # Constant = 200 across all rows
        "FL_Client_ID",                 # Synthetic federated client partition ID
    ]

    if drop_leaked_features:
        cols_to_drop.extend(
            [
                "Predictive_Score",  # Directly encodes target severity class
                "UIR",               # Algebraic formula of Usage_Hours & Year
                "TPI",               # Engine_Temperature / Tire_Pressure
                "MBF",               # Maintenance_Cost * (1 + 0.25 * Failure_History)
                "ADS",               # 0.5*Failure + 1.0*Anomalies + 0.2*DTC
                "OHI",               # 1 + 0.0001*Usage + 0.1*Vib - 0.01*Oil
                "CMES",              # Downtime + Fuel_Consumption * Impact
                "UER",               # Usage_Hours / (Maintenance_Cost + 1)
            ]
        )

    if drop_post_event_outcomes:
        cols_to_drop.extend(
            [
                "Maintenance_Cost",
                "Downtime_Maintenance",
                "Impact_on_Efficiency",
            ]
        )

    existing_drop_cols = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=existing_drop_cols)
    print(
        f"[4/5] Dropped {len(existing_drop_cols)} redundant/constant/leaked columns -> {df.shape[1]} columns remaining."
    )

    # Re-index Vehicle_ID cleanly from 1 to N
    df = df.reset_index(drop=True)
    df["Vehicle_ID"] = np.arange(1, len(df) + 1, dtype=int)

    # -------------------------------------------------------------------------
    # STEP 5: Save Processed Dataset
    # -------------------------------------------------------------------------
    df.to_csv(output_csv, index=False)
    print(
        f"[5/5] Successfully saved '{output_csv}' ({len(df):,} rows x {df.shape[1]} columns)."
    )
    return df


if __name__ == "__main__":
    filtered_df = filter_and_process_logistics_v2(
        input_csv="logistics_predictive_maintenanceV2.csv",
        output_csv="logistics_predictive_maintenanceV2(filtered).csv",
        strict_row_filter_only=False,
        drop_leaked_features=True,
        drop_post_event_outcomes=False,
    )