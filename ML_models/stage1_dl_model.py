from pathlib import Path
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
)


# ==========================================
# 1. REPRODUCIBILITY & DEVICE SETUP
# ==========================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# 2. PYTORCH STAGE 1 ARCHITECTURES
# ==========================================
class DenoisingAutoencoder(nn.Module):
    """
    Reconstructs clean vehicle sensor telemetry from masked/noisy inputs
    and extracts a compact 24-dim denoised latent representation (z).
    """

    def __init__(self, input_dim: int, latent_dim: int = 24):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(64, 48),
            nn.LayerNorm(48),
            nn.GELU(),
            nn.Linear(48, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 48),
            nn.LayerNorm(48),
            nn.GELU(),
            nn.Linear(48, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z


class Stage1HealthMLP(nn.Module):
    """
    Dual-Head MLP Predictor taking fused [reconstructed_sensors, latent_z]
    to output:
      - Head 1: Maintenance_Required probability (via BCEWithLogitsLoss)
      - Head 2: Continuous Predictive_Score degradation risk in [0, 1]
    """

    def __init__(self, input_dim: int, latent_dim: int = 24):
        super().__init__()
        fused_dim = input_dim + latent_dim
        self.backbone = nn.Sequential(
            nn.Linear(fused_dim, 96),
            nn.LayerNorm(96),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(96, 48),
            nn.LayerNorm(48),
            nn.GELU(),
            nn.Dropout(0.10),
        )
        self.cls_head = nn.Linear(48, 1)
        self.reg_head = nn.Linear(48, 1)

    def forward(self, x_recon: torch.Tensor, z: torch.Tensor):
        fused = torch.cat([x_recon, z], dim=-1)
        hidden = self.backbone(fused)
        logit_cls = self.cls_head(hidden).squeeze(-1)
        pred_score = torch.sigmoid(self.reg_head(hidden)).squeeze(-1)
        return logit_cls, pred_score


class BaselineMLP(nn.Module):
    """
    Standard single-stage MLP baseline used in Part B1 to evaluate
    Zero-Fill and Forward-Fill imputation without Autoencoder denoising.
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 96),
            nn.LayerNorm(96),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(96, 48),
            nn.LayerNorm(48),
            nn.GELU(),
            nn.Dropout(0.10),
        )
        self.cls_head = nn.Linear(48, 1)
        self.reg_head = nn.Linear(48, 1)

    def forward(self, x: torch.Tensor):
        h = self.net(x)
        return self.cls_head(h).squeeze(-1), torch.sigmoid(self.reg_head(h)).squeeze(-1)


# ==========================================
# 3. REUSABLE INFERENCE ENGINE (FOR RL & STREAMLIT)
# ==========================================
class Stage1InferenceEngine:
    """
    Lightweight wrapper loaded by fleet_maintenance_env.py (Week 1/3)
    and the Streamlit prototype (Week 4) for single-step or batch predictions.
    """

    def __init__(self, checkpoint_path: Path, device=None):
        self.device = device or get_device()
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.feature_cols = ckpt["feature_cols"]
        self.input_dim = ckpt["input_dim"]
        self.latent_dim = ckpt["latent_dim"]

        self.dae = DenoisingAutoencoder(self.input_dim, self.latent_dim).to(self.device)
        self.mlp = Stage1HealthMLP(self.input_dim, self.latent_dim).to(self.device)
        self.dae.load_state_dict(ckpt["dae_state_dict"])
        self.mlp.load_state_dict(ckpt["mlp_state_dict"])
        self.dae.eval()
        self.mlp.eval()

    @torch.no_grad()
    def predict(self, X_array: np.ndarray):
        """
        Accepts shape (N, input_dim) or (input_dim,) NumPy float32 array.
        Returns dict with failure_prob, health_score, and latent_z.
        """
        single_input = X_array.ndim == 1
        if single_input:
            X_array = X_array[None, :]

        x_tensor = torch.tensor(X_array, dtype=torch.float32, device=self.device)
        x_recon, z = self.dae(x_tensor)
        logit_cls, pred_score = self.mlp(x_recon, z)
        fail_prob = torch.sigmoid(logit_cls).cpu().numpy()
        health_score = pred_score.cpu().numpy()
        latent_z = z.cpu().numpy()

        if single_input:
            return {
                "failure_prob": float(fail_prob[0]),
                "health_score": float(health_score[0]),
                "latent_z": latent_z[0],
            }
        return {
            "failure_prob": fail_prob,
            "health_score": health_score,
            "latent_z": latent_z,
        }


# ==========================================
# 4. TRAINING & BENCHMARKING HELPERS
# ==========================================
def corrupt_continuous_features(x_batch: torch.Tensor, cont_indices: list, mask_prob: float = 0.15, noise_std: float = 0.05):
    """Injects random sensor dropout (0-masking) and Gaussian noise on continuous features."""
    x_corrupted = x_batch.clone()
    if not cont_indices:
        return x_corrupted

    sub = x_corrupted[:, cont_indices]
    keep_mask = (torch.rand_like(sub) > mask_prob).float()
    gaussian_noise = torch.randn_like(sub) * noise_std
    x_corrupted[:, cont_indices] = (sub + gaussian_noise) * keep_mask
    return x_corrupted


def create_masked_test_data(X_clean: np.ndarray, cont_indices: list, mask_prob: float = 0.15, seed: int = 42):
    """
    Creates a reproducible 15% sensor-dropout test set with NaN values to benchmark
    Zero-Fill vs. Forward-Fill vs. Denoising Autoencoder imputation (Part B1).
    """
    rng = np.random.default_rng(seed)
    X_nan = X_clean.copy()
    mask_matrix = np.zeros_like(X_clean, dtype=bool)

    dropout_draw = rng.random((X_clean.shape[0], len(cont_indices))) < mask_prob
    mask_matrix[:, cont_indices] = dropout_draw
    X_nan[mask_matrix] = np.nan

    # Method 1: Zero-Fill
    X_zero = np.nan_to_num(X_nan, nan=0.0).astype(np.float32)

    # Method 2: Forward-Fill (with Back-Fill edge handling)
    df_temp = pd.DataFrame(X_nan)
    X_ffill = df_temp.ffill().bfill().fillna(0.0).values.astype(np.float32)

    return X_zero, X_ffill, mask_matrix


def train_denoising_autoencoder(
    dae: DenoisingAutoencoder,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cont_indices: list,
    device: torch.device,
    epochs: int = 20,
    lr: float = 1e-3,
):
    print("\n--- Training Stage 1A: Denoising Autoencoder ---")
    optimizer = torch.optim.AdamW(dae.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_weights = None

    for epoch in range(1, epochs + 1):
        dae.train()
        train_loss = 0.0
        for (x_clean,) in train_loader:
            x_clean = x_clean.to(device)
            x_noisy = corrupt_continuous_features(x_clean, cont_indices, mask_prob=0.15, noise_std=0.05)

            optimizer.zero_grad()
            x_recon, _ = dae(x_noisy)
            loss = criterion(x_recon, x_clean)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x_clean.size(0)

        scheduler.step()
        train_loss /= len(train_loader.dataset)

        # Validation reconstruction loss on 15% masked inputs
        dae.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (x_clean,) in val_loader:
                x_clean = x_clean.to(device)
                x_noisy = corrupt_continuous_features(x_clean, cont_indices, mask_prob=0.15, noise_std=0.0)
                x_recon, _ = dae(x_noisy)
                val_loss += criterion(x_recon, x_clean).item() * x_clean.size(0)
        val_loss /= len(val_loader.dataset)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in dae.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            print(f"   Epoch {epoch:02d}/{epochs:02d} | Train Recon MSE: {train_loss:.5f} | Val Masked MSE: {val_loss:.5f}")

    dae.load_state_dict(best_weights)
    return dae


def train_stage1_mlp(
    dae: DenoisingAutoencoder,
    mlp: Stage1HealthMLP,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cont_indices: list,
    pos_weight: float,
    device: torch.device,
    epochs: int = 25,
    lr: float = 1e-3,
):
    print("\n--- Training Stage 1B: Dual-Head Health Predictor (DAE + MLP Fusion) ---")
    dae.eval()  # Freeze DAE encoder so latent space stays stable
    for param in dae.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(mlp.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    bce_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    reg_criterion = nn.SmoothL1Loss()

    best_val_auc = 0.0
    best_weights = None

    for epoch in range(1, epochs + 1):
        mlp.train()
        total_loss = 0.0

        for x_clean, y_cls, y_reg in train_loader:
            x_clean = x_clean.to(device)
            y_cls = y_cls.to(device)
            y_reg = y_reg.to(device)

            # Train MLP on both clean and 15% sensor-masked states for robustness
            x_in = corrupt_continuous_features(x_clean, cont_indices, mask_prob=0.10, noise_std=0.02)
            with torch.no_grad():
                x_recon, z = dae(x_in)

            optimizer.zero_grad()
            logit_cls, pred_score = mlp(x_recon, z)
            loss_cls = bce_criterion(logit_cls, y_cls)
            loss_reg = reg_criterion(pred_score, y_reg)
            loss = loss_cls + 0.5 * loss_reg
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x_clean.size(0)

        scheduler.step()
        total_loss /= len(train_loader.dataset)

        # Validation Metrics
        mlp.eval()
        val_probs, val_scores, val_y_cls, val_y_reg = [], [], [], []
        with torch.no_grad():
            for x_clean, y_cls, y_reg in val_loader:
                x_clean = x_clean.to(device)
                x_recon, z = dae(x_clean)
                logit_cls, pred_score = mlp(x_recon, z)
                val_probs.append(torch.sigmoid(logit_cls).cpu().numpy())
                val_scores.append(pred_score.cpu().numpy())
                val_y_cls.append(y_cls.numpy())
                val_y_reg.append(y_reg.numpy())

        val_probs = np.concatenate(val_probs)
        val_scores = np.concatenate(val_scores)
        val_y_cls = np.concatenate(val_y_cls)
        val_y_reg = np.concatenate(val_y_reg)

        val_preds = (val_probs >= 0.5).astype(int)
        val_f1 = f1_score(val_y_cls, val_preds)
        val_auc = roc_auc_score(val_y_cls, val_probs)
        val_mae = mean_absolute_error(val_y_reg, val_scores)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_weights = {k: v.cpu().clone() for k, v in mlp.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"   Epoch {epoch:02d}/{epochs:02d} | Loss: {total_loss:.4f} | "
                f"Val F1: {val_f1:.4f} | Val ROC-AUC: {val_auc:.4f} | Val Score MAE: {val_mae:.4f}"
            )

    mlp.load_state_dict(best_weights)
    return mlp


def train_baseline_mlp(
    model: BaselineMLP,
    train_loader: DataLoader,
    pos_weight: float,
    device: torch.device,
    epochs: int = 15,
    lr: float = 1e-3,
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    bce_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    reg_criterion = nn.SmoothL1Loss()

    model.train()
    for _ in range(epochs):
        for x_batch, y_cls, y_reg in train_loader:
            x_batch, y_cls, y_reg = x_batch.to(device), y_cls.to(device), y_reg.to(device)
            optimizer.zero_grad()
            logit_cls, pred_score = model(x_batch)
            loss = bce_criterion(logit_cls, y_cls) + 0.5 * reg_criterion(pred_score, y_reg)
            loss.backward()
            optimizer.step()
    return model


# ==========================================
# 5. MAIN EXECUTION & PART B1 BENCHMARK
# ==========================================
def main():
    set_seed(42)
    device = get_device()
    print(f"Running Stage 1 DL Pipeline on device: {device}")

    # Resolve paths
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir if (script_dir / "Data_Set").exists() else script_dir.parent
    data_dir = base_dir / "Data_Set"
    model_dir = base_dir / "Saved_Models"
    model_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = data_dir / "processed_dataset.csv"
    print(f"1. Loading preprocessed dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"   Loaded shape: {df.shape}")

    # Separate Stage 1 Input X (pre-event features) from RL Metadata and Targets y
    maint_type_cols = [c for c in df.columns if c.startswith("Maintenance_Type_")]
    rl_and_target_cols = maint_type_cols + [
        "Maintenance_Cost",
        "Downtime_Maintenance",
        "Impact_on_Efficiency",
        "Maintenance_Severity",
        "Predictive_Score",
        "Maintenance_Required",
    ]
    feature_cols = [c for c in df.columns if c not in rl_and_target_cols]
    print(f"   Stage 1 Input Features (X): {len(feature_cols)} columns")

    # Identify continuous sensor/telemetry indices (non-binary columns) for Part B1 masking
    cont_indices = [
        i for i, col in enumerate(feature_cols)
        if not col.startswith(("Make_and_Model_", "Vehicle_Type_", "Route_Info_", "Weather_Conditions_", "Road_Conditions_"))
    ]

    X_all = df[feature_cols].values.astype(np.float32)
    y_cls_all = df["Maintenance_Required"].values.astype(np.float32)
    y_reg_all = df["Predictive_Score"].values.astype(np.float32)

    # Stratified Train (80%) / Val (10%) / Test (10%) Split
    X_train, X_temp, y_cls_train, y_cls_temp, y_reg_train, y_reg_temp = train_test_split(
        X_all, y_cls_all, y_reg_all, test_size=0.20, stratify=y_cls_all, random_state=42
    )
    X_val, X_test, y_cls_val, y_cls_test, y_reg_val, y_reg_test = train_test_split(
        X_temp, y_cls_temp, y_reg_temp, test_size=0.50, stratify=y_cls_temp, random_state=42
    )
    print(f"2. Split sizes -> Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

    # Compute positive class weight for balanced failure detection
    pos_count = float(y_cls_train.sum())
    neg_count = float(len(y_cls_train) - pos_count)
    pos_weight = neg_count / max(pos_count, 1.0)

    # DataLoaders
    batch_size = 512
    dae_train_loader = DataLoader(TensorDataset(torch.tensor(X_train)), batch_size=batch_size, shuffle=True)
    dae_val_loader = DataLoader(TensorDataset(torch.tensor(X_val)), batch_size=batch_size, shuffle=False)

    mlp_train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_cls_train), torch.tensor(y_reg_train)),
        batch_size=batch_size,
        shuffle=True,
    )
    mlp_val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val), torch.tensor(y_cls_val), torch.tensor(y_reg_val)),
        batch_size=batch_size,
        shuffle=False,
    )

    # Train Stage 1A (Denoising Autoencoder) & Stage 1B (Dual-Head MLP)
    input_dim = len(feature_cols)
    latent_dim = 24
    dae = DenoisingAutoencoder(input_dim=input_dim, latent_dim=latent_dim).to(device)
    mlp = Stage1HealthMLP(input_dim=input_dim, latent_dim=latent_dim).to(device)

    dae = train_denoising_autoencoder(dae, dae_train_loader, dae_val_loader, cont_indices, device, epochs=20)
    mlp = train_stage1_mlp(dae, mlp, mlp_train_loader, mlp_val_loader, cont_indices, pos_weight, device, epochs=25)

    # Save Stage 1 Checkpoint
    ckpt_path = model_dir / "stage1_fusion_model.pt"
    torch.save(
        {
            "input_dim": input_dim,
            "latent_dim": latent_dim,
            "feature_cols": feature_cols,
            "dae_state_dict": dae.state_dict(),
            "mlp_state_dict": mlp.state_dict(),
        },
        ckpt_path,
    )
    print(f"\n3. Saved Stage 1 Fusion Model checkpoint to: {ckpt_path}")

    # ==========================================
    # PART B1 BENCHMARK: ZERO-FILL VS FFILL VS DAE
    # ==========================================
    print("\n4. Running Week 1 Part B1 Imputation Benchmark on 15% Masked Test Telemetry...")
    X_test_zero, X_test_ffill, mask_matrix = create_masked_test_data(X_test, cont_indices, mask_prob=0.15, seed=42)

    # Train Baseline MLPs for Zero-Fill and Forward-Fill comparison
    print("   Training Baseline MLP for Zero-Fill & Forward-Fill comparison...")
    baseline_zero_mlp = train_baseline_mlp(BaselineMLP(input_dim).to(device), mlp_train_loader, pos_weight, device, epochs=15)
    baseline_ffill_mlp = train_baseline_mlp(BaselineMLP(input_dim).to(device), mlp_train_loader, pos_weight, device, epochs=15)

    dae.eval()
    mlp.eval()
    baseline_zero_mlp.eval()
    baseline_ffill_mlp.eval()

    with torch.no_grad():
        # 1. Zero-Fill Evaluation
        t_zero = torch.tensor(X_test_zero, device=device)
        z_logits, z_scores = baseline_zero_mlp(t_zero)
        z_probs = torch.sigmoid(z_logits).cpu().numpy()
        z_scores = z_scores.cpu().numpy()
        zero_rmse = np.sqrt(mean_squared_error(X_test[mask_matrix], X_test_zero[mask_matrix]))

        # 2. Forward-Fill Evaluation
        t_ffill = torch.tensor(X_test_ffill, device=device)
        f_logits, f_scores = baseline_ffill_mlp(t_ffill)
        f_probs = torch.sigmoid(f_logits).cpu().numpy()
        f_scores = f_scores.cpu().numpy()
        ffill_rmse = np.sqrt(mean_squared_error(X_test[mask_matrix], X_test_ffill[mask_matrix]))

        # 3. Denoising Autoencoder + MLP Fusion Evaluation (fed 0-masked sensors)
        x_dae_recon, z_latent = dae(t_zero)
        d_logits, d_scores = mlp(x_dae_recon, z_latent)
        d_probs = torch.sigmoid(d_logits).cpu().numpy()
        d_scores = d_scores.cpu().numpy()
        x_dae_recon_np = x_dae_recon.cpu().numpy()
        dae_rmse = np.sqrt(mean_squared_error(X_test[mask_matrix], x_dae_recon_np[mask_matrix]))

        # 4. Clean Uncorrupted Stage 1 Reference
        t_clean = torch.tensor(X_test, device=device)
        c_recon, c_latent = dae(t_clean)
        c_logits, c_scores = mlp(c_recon, c_latent)
        c_probs = torch.sigmoid(c_logits).cpu().numpy()
        c_scores = c_scores.cpu().numpy()

    benchmark_rows = [
        {
            "Imputation_Method": "Zero-Fill + MLP (15% Masked)",
            "Masked_Sensor_RMSE": round(float(zero_rmse), 4),
            "Test_Accuracy": round(float(accuracy_score(y_cls_test, (z_probs >= 0.5).astype(int))), 4),
            "Test_F1_Score": round(float(f1_score(y_cls_test, (z_probs >= 0.5).astype(int))), 4),
            "Test_ROC_AUC": round(float(roc_auc_score(y_cls_test, z_probs)), 4),
            "Predictive_Score_MAE": round(float(mean_absolute_error(y_reg_test, z_scores)), 4),
        },
        {
            "Imputation_Method": "Forward-Fill + MLP (15% Masked)",
            "Masked_Sensor_RMSE": round(float(ffill_rmse), 4),
            "Test_Accuracy": round(float(accuracy_score(y_cls_test, (f_probs >= 0.5).astype(int))), 4),
            "Test_F1_Score": round(float(f1_score(y_cls_test, (f_probs >= 0.5).astype(int))), 4),
            "Test_ROC_AUC": round(float(roc_auc_score(y_cls_test, f_probs)), 4),
            "Predictive_Score_MAE": round(float(mean_absolute_error(y_reg_test, f_scores)), 4),
        },
        {
            "Imputation_Method": "Denoising Autoencoder + MLP (15% Masked)",
            "Masked_Sensor_RMSE": round(float(dae_rmse), 4),
            "Test_Accuracy": round(float(accuracy_score(y_cls_test, (d_probs >= 0.5).astype(int))), 4),
            "Test_F1_Score": round(float(f1_score(y_cls_test, (d_probs >= 0.5).astype(int))), 4),
            "Test_ROC_AUC": round(float(roc_auc_score(y_cls_test, d_probs)), 4),
            "Predictive_Score_MAE": round(float(mean_absolute_error(y_reg_test, d_scores)), 4),
        },
        {
            "Imputation_Method": "Stage 1 DAE + MLP (Clean Uncorrupted)",
            "Masked_Sensor_RMSE": 0.0000,
            "Test_Accuracy": round(float(accuracy_score(y_cls_test, (c_probs >= 0.5).astype(int))), 4),
            "Test_F1_Score": round(float(f1_score(y_cls_test, (c_probs >= 0.5).astype(int))), 4),
            "Test_ROC_AUC": round(float(roc_auc_score(y_cls_test, c_probs)), 4),
            "Predictive_Score_MAE": round(float(mean_absolute_error(y_reg_test, c_scores)), 4),
        },
    ]

    bench_df = pd.DataFrame(benchmark_rows)
    bench_csv_path = model_dir / "part_b1_imputation_benchmark.csv"
    bench_df.to_csv(bench_csv_path, index=False)
    print("\n=== WEEK 1 PART B1 IMPUTATION BENCHMARK RESULTS ===")
    print(bench_df.to_string(index=False))
    print(f"Saved benchmark table to: {bench_csv_path}")

    # Plot Benchmark Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    masked_only = bench_df.iloc[:3]
    methods_short = ["Zero-Fill", "Forward-Fill", "DAE + MLP"]

    ax1.bar(methods_short, masked_only["Masked_Sensor_RMSE"], color=["#d62728", "#ff7f0e", "#2ca02c"], width=0.5)
    ax1.set_title("Sensor Reconstruction Error (Lower is Better)", fontweight="bold")
    ax1.set_ylabel("Masked Sensor RMSE")
    for i, v in enumerate(masked_only["Masked_Sensor_RMSE"]):
        ax1.text(i, v + 0.01, f"{v:.4f}", ha="center", fontweight="bold")

    x_idx = np.arange(len(methods_short))
    ax2.bar(x_idx - 0.18, masked_only["Test_F1_Score"], width=0.35, label="F1-Score", color="#1f77b4")
    ax2.bar(x_idx + 0.18, masked_only["Test_ROC_AUC"], width=0.35, label="ROC-AUC", color="#17becf")
    ax2.set_xticks(x_idx)
    ax2.set_xticklabels(methods_short)
    ax2.set_ylim(0.5, 1.05)
    ax2.set_title("Downstream Health Prediction (Higher is Better)", fontweight="bold")
    ax2.legend()

    plt.tight_layout()
    plot_path = model_dir / "part_b1_benchmark_plot.png"
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved benchmark chart to: {plot_path}")

    # ==========================================
    # 6. EXPORT STAGE 1 PREDICTIONS FOR RL ENV
    # ==========================================
    print("\n5. Generating Stage 1 state estimates across all 174,876 rows for RL Environment...")
    engine = Stage1InferenceEngine(ckpt_path, device=device)
    preds = engine.predict(X_all)

    df["Stage1_Failure_Prob"] = np.round(preds["failure_prob"], 5)
    df["Stage1_Health_Score"] = np.round(preds["health_score"], 5)

    rl_dataset_path = data_dir / "dataset_with_stage1_preds.csv"
    df.to_csv(rl_dataset_path, index=False)
    print(f"   Saved RL-ready dataset to: {rl_dataset_path} (Shape: {df.shape})")
    print("Stage 1 DL Engine complete! Ready to build fleet_maintenance_env.py.")


if __name__ == "__main__":
    main()