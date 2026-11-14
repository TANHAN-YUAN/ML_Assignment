from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def run_eda():
    # Resolve paths relative to SRC/EDA/eda.py
    eda_dir = Path(__file__).resolve().parent
    base_dir = eda_dir.parent.parent
    data_path = base_dir / "Data_Set" / "logistics_predictive_maintenanceV2.csv"

    print(f"1. Loading original dataset from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"   Dataset shape: {df.shape}")

    # Active numerical features (excluding dropped metadata & ignored CAN/Packet_Loss columns)
    active_num_cols = [
        "Year_of_Manufacture", "Usage_Hours", "Load_Capacity", "Actual_Load",
        "Engine_Temperature", "Tire_Pressure", "Fuel_Consumption", "Battery_Status",
        "Vibration_Levels", "Oil_Quality", "Failure_History", "Anomalies_Detected",
        "Diagnostic_Trouble_Code_Count", "Maintenance_Cost", "Downtime_Maintenance",
        "Impact_on_Efficiency", "Days_Since_Last_Maintenance",
        "PCR", "UIR", "TPI", "MBF", "ADS", "OHI", "CMES", "UER",
        "Predictive_Score", "Maintenance_Required"
    ]
    valid_cols = [c for c in active_num_cols if c in df.columns]

    plt.style.use("seaborn-v0_8-whitegrid")

    # ==========================================
    # PART 1: CORRELATION MATRIX & TARGET RANKING
    # ==========================================
    print("2. Computing correlation matrix and generating correlation_analysis.png...")
    corr_matrix = df[valid_cols].corr()
    corr_csv_path = eda_dir / "eda_correlation_matrix.csv"
    corr_matrix.to_csv(corr_csv_path)

    target_corr = corr_matrix[["Predictive_Score", "Maintenance_Required"]].sort_values(
        by="Predictive_Score", ascending=False
    )

    fig = plt.figure(figsize=(22, 10), dpi=150)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1], wspace=0.25)

    # Panel A: Lower-triangle Heatmap
    ax1 = fig.add_subplot(gs[0])
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap="coolwarm",
        vmin=-1, vmax=1,
        center=0,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 6.5},
        linewidths=0.4,
        cbar_kws={"label": "Pearson Correlation (r)", "shrink": 0.8},
        ax=ax1
    )
    ax1.set_title("A. Feature Correlation Matrix (Original Dataset, N = 250,000)", fontsize=13, fontweight="bold", pad=12)
    ax1.tick_params(axis="x", rotation=45, labelsize=8)
    ax1.tick_params(axis="y", labelsize=8)

    # Panel B: Target Correlation Bar Chart
    ax2 = fig.add_subplot(gs[1])
    target_sub = target_corr.drop(index=["Predictive_Score", "Maintenance_Required"]).sort_values(
        by="Predictive_Score", ascending=True
    )
    y_pos = np.arange(len(target_sub))
    height = 0.38

    bars1 = ax2.barh(y_pos + height / 2, target_sub["Predictive_Score"], height, label="Predictive_Score", color="#1f77b4", alpha=0.9)
    bars2 = ax2.barh(y_pos - height / 2, target_sub["Maintenance_Required"], height, label="Maintenance_Required", color="#ff7f0e", alpha=0.9)

    ax2.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(target_sub.index, fontsize=9)
    ax2.set_xlabel("Pearson Correlation Coefficient (r)", fontsize=10, fontweight="bold")
    ax2.set_title("B. Feature Correlations with Target Variables", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xlim(-1.05, 1.08)
    ax2.legend(loc="lower right", frameon=True, fontsize=10)

    for bar in bars1:
        w = bar.get_width()
        offset = 0.02 if w >= 0 else -0.02
        ha = "left" if w >= 0 else "right"
        ax2.text(w + offset, bar.get_y() + bar.get_height() / 2, f"{w:.2f}", va="center", ha=ha, fontsize=7.5, color="#1f77b4", fontweight="bold")

    corr_plot_path = eda_dir / "correlation_analysis.png"
    plt.savefig(corr_plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   Saved: {corr_plot_path}")
    print(f"   Saved: {corr_csv_path}")

    # ==========================================
    # PART 2: OUTLIER AUDIT & BOXPLOT VISUALIZATION
    # ==========================================
    print("3. Auditing 1.5x IQR outliers and generating outlier_analysis.png...")
    feature_cols = [c for c in valid_cols if c not in ["Predictive_Score", "Maintenance_Required"]]
    outlier_summary = []

    for col in feature_cols:
        s = df[col]
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = int(((s < lower) | (s > upper)).sum())
        pct = (outliers / len(s)) * 100
        outlier_summary.append({
            "Feature": col,
            "Mean": s.mean(),
            "Std": s.std(),
            "Min": s.min(),
            "Q1_25%": q1,
            "Median_50%": s.median(),
            "Q3_75%": q3,
            "Max": s.max(),
            "IQR_Lower_Bound": lower,
            "IQR_Upper_Bound": upper,
            "Outlier_Count": outliers,
            "Outlier_Pct": pct
        })

    outlier_df = pd.DataFrame(outlier_summary).sort_values(by="Outlier_Pct", ascending=False)
    outlier_csv_path = eda_dir / "eda_outlier_summary.csv"
    outlier_df.to_csv(outlier_csv_path, index=False)

    ordered_features = outlier_df["Feature"].tolist()
    df_z = (df[ordered_features] - df[ordered_features].mean()) / df[ordered_features].std()

    # Subsample 25,000 points for clean boxplot rendering
    np.random.seed(42)
    sample_idx = np.random.choice(len(df_z), size=min(25000, len(df_z)), replace=False)
    df_z_sample = df_z.iloc[sample_idx]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10), dpi=150, gridspec_kw={"width_ratios": [1.3, 1], "wspace": 0.25})

    sns.boxplot(
        data=df_z_sample,
        orient="h",
        ax=ax1,
        palette="Spectral",
        fliersize=1.8,
        linewidth=1.0,
        flierprops={"marker": "o", "alpha": 0.35, "markeredgecolor": "none", "markerfacecolor": "#d62728"}
    )
    ax1.set_title("A. Standardized Feature Distributions & Outliers (Z-Score Boxplots)", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("Standardized Value (Z-Score: Standard Deviations from Mean)", fontsize=10, fontweight="bold")
    ax1.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax1.set_xlim(-4.5, 9.5)
    ax1.tick_params(axis="y", labelsize=9)

    y_pos = np.arange(len(outlier_df))
    colors = ["#d62728" if pct >= 8.0 else "#ff7f0e" if pct >= 3.0 else "#2ca02c" for pct in outlier_df["Outlier_Pct"]]
    bars = ax2.barh(y_pos, outlier_df["Outlier_Pct"], color=colors, alpha=0.88, height=0.68)
    ax2.invert_yaxis()
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(outlier_df["Feature"], fontsize=9)
    ax2.set_xlabel("Outlier Percentage (% of 250,000 Rows Outside 1.5x IQR)", fontsize=10, fontweight="bold")
    ax2.set_title("B. Severity of Outliers per Feature (1.5x IQR Rule)", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xlim(0, 15.0)

    for bar, row in zip(bars, outlier_df.itertuples()):
        w = bar.get_width()
        ax2.text(
            w + 0.18, bar.get_y() + bar.get_height() / 2,
            f"{row.Outlier_Pct:.2f}% ({row.Outlier_Count:,})",
            va="center", ha="left", fontsize=8, fontweight="bold"
        )

    outlier_plot_path = eda_dir / "outlier_analysis.png"
    plt.savefig(outlier_plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   Saved: {outlier_plot_path}")
    print(f"   Saved: {outlier_csv_path}")
    print("4. EDA Complete!")


if __name__ == "__main__":
    run_eda()