"""
Page 2: Training Dashboard.

Monitor training progress for both the ViT classifier and the
RL treatment recommendation agents (PPO and DQN).

Shows:
- Training status and best metrics
- Training curves (loss, accuracy, learning rate) from history JSON
- Model checkpoint listing
- Training commands for reference
"""

import json
import streamlit as st
from pathlib import Path

import plotly.graph_objects as go

from app.utils.ui_components import (
    apply_custom_css,
    render_header,
    render_disclaimer,
    COLORS,
)
from src.utils.dqn_log_parser import parse_dqn_eval_log, get_dqn_eval_path

st.set_page_config(page_title="Training Dashboard", layout="wide")
apply_custom_css()

render_header("Training Dashboard", "Monitor training progress for the ViT classifier and RL agents.")

# --- Paths ---
CHECKPOINT_DIR = Path("models/checkpoints")
HISTORY_PATH = CHECKPOINT_DIR / "training_history.json"
LOG_DIR = Path("logs")


# =============================================================================
# HELPERS
# =============================================================================

def load_training_history() -> dict | None:
    """Load training history from JSON file saved by the Trainer."""
    if HISTORY_PATH.exists():
        with open(HISTORY_PATH) as f:
            return json.load(f)
    return None


def count_checkpoints(directory: Path, pattern: str) -> int:
    """Count checkpoint files matching a glob pattern."""
    if directory.exists():
        return len(list(directory.glob(pattern)))
    return 0


# =============================================================================
# TRAINING STATUS
# =============================================================================

st.header("Training Status")

history = load_training_history()

col1, col2, col3, col4 = st.columns(4)

with col1:
    vit_checkpoints = count_checkpoints(CHECKPOINT_DIR, "*.pth")
    st.metric("ViT Checkpoints", vit_checkpoints)

with col2:
    if history and history.get("val_acc"):
        best_acc = max(history["val_acc"])
        st.metric("Best Val Accuracy", f"{best_acc:.2%}")
    else:
        st.metric("Best Val Accuracy", "N/A")

with col3:
    if history and history.get("val_loss"):
        best_loss = min(history["val_loss"])
        st.metric("Best Val Loss", f"{best_loss:.4f}")
    else:
        st.metric("Best Val Loss", "N/A")

with col4:
    if history and history.get("train_loss"):
        epochs_done = len(history["train_loss"])
        st.metric("Epochs Completed", epochs_done)
    else:
        st.metric("Epochs Completed", "0")


# =============================================================================
# TRAINING CURVES
# =============================================================================

st.header("Training Curves")

if history and history.get("train_loss"):
    epochs = list(range(1, len(history["train_loss"]) + 1))

    # --- Loss curves ---
    fig_loss = go.Figure()
    fig_loss.add_trace(go.Scatter(
        x=epochs, y=history["train_loss"],
        name="Train Loss", mode="lines+markers",
        line=dict(color=COLORS["primary"], width=2),
        marker=dict(size=5),
    ))
    fig_loss.add_trace(go.Scatter(
        x=epochs, y=history["val_loss"],
        name="Val Loss", mode="lines+markers",
        line=dict(color=COLORS["danger"], width=2),
        marker=dict(size=5),
    ))
    fig_loss.update_layout(
        title="Loss per Epoch",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400,
    )
    st.plotly_chart(fig_loss, use_container_width=True)

    # --- Accuracy curves ---
    fig_acc = go.Figure()
    fig_acc.add_trace(go.Scatter(
        x=epochs, y=history["train_acc"],
        name="Train Accuracy", mode="lines+markers",
        line=dict(color=COLORS["primary"], width=2),
        marker=dict(size=5),
    ))
    fig_acc.add_trace(go.Scatter(
        x=epochs, y=history["val_acc"],
        name="Val Accuracy", mode="lines+markers",
        line=dict(color=COLORS["success"], width=2),
        marker=dict(size=5),
    ))
    fig_acc.update_layout(
        title="Accuracy per Epoch",
        xaxis_title="Epoch",
        yaxis_title="Accuracy",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(range=[0, 1]),
        height=400,
    )
    st.plotly_chart(fig_acc, use_container_width=True)

    # --- Learning rate schedule ---
    if history.get("learning_rate"):
        fig_lr = go.Figure()
        fig_lr.add_trace(go.Scatter(
            x=epochs, y=history["learning_rate"],
            name="Learning Rate", mode="lines+markers",
            line=dict(color=COLORS["info"], width=2),
            marker=dict(size=5),
        ))
        fig_lr.update_layout(
            title="Learning Rate Schedule",
            xaxis_title="Epoch",
            yaxis_title="Learning Rate",
            template="plotly_white",
            height=350,
        )
        st.plotly_chart(fig_lr, use_container_width=True)

    # --- Epoch timing ---
    if history.get("epoch_time"):
        fig_time = go.Figure()
        fig_time.add_trace(go.Bar(
            x=epochs, y=history["epoch_time"],
            name="Epoch Time",
            marker_color=COLORS["primary_light"],
        ))
        fig_time.update_layout(
            title="Time per Epoch",
            xaxis_title="Epoch",
            yaxis_title="Seconds",
            template="plotly_white",
            height=350,
        )
        st.plotly_chart(fig_time, use_container_width=True)

else:
    st.info("No training history found. Train the ViT model first to see curves here.")
    st.markdown("Run `python train.py` to start training.")


# =============================================================================
# PER-CLASS METRICS (precision / recall / F1)
# =============================================================================

st.header("Per-Class Test Metrics")

# These would normally be saved by evaluate_model() into a JSON next to the
# best checkpoint. If present, display them; otherwise prompt the user.
test_metrics_path = CHECKPOINT_DIR / "test_metrics.json"

if test_metrics_path.exists():
    with open(test_metrics_path) as f:
        test_metrics = json.load(f)

    rows = test_metrics.get("per_class", [])
    if rows:
        st.dataframe(
            rows,
            use_container_width=True,
            hide_index=True,
            column_config={
                "class": st.column_config.TextColumn("Class"),
                "precision": st.column_config.ProgressColumn(
                    "Precision", min_value=0, max_value=1, format="%.3f"
                ),
                "recall": st.column_config.ProgressColumn(
                    "Recall", min_value=0, max_value=1, format="%.3f"
                ),
                "f1": st.column_config.ProgressColumn(
                    "F1-Score", min_value=0, max_value=1, format="%.3f"
                ),
                "support": st.column_config.NumberColumn("Support"),
            },
        )

        # Overall numbers
        ocol1, ocol2, ocol3, ocol4 = st.columns(4)
        ocol1.metric("Accuracy", f"{test_metrics.get('accuracy', 0):.2%}")
        ocol2.metric("Precision (weighted)", f"{test_metrics.get('precision', 0):.3f}")
        ocol3.metric("Recall (weighted)", f"{test_metrics.get('recall', 0):.3f}")
        ocol4.metric("F1 (weighted)", f"{test_metrics.get('f1', 0):.3f}")
    else:
        st.info("`test_metrics.json` is empty. Re-run `python train.py` to populate it.")
else:
    st.info(
        "No per-class metrics file found at `models/checkpoints/test_metrics.json`. "
        "These are written automatically by `train.py` after test evaluation."
    )


# =============================================================================
# DQN TREATMENT AGENT DASHBOARD
# =============================================================================

st.header("DQN Treatment Agent")

dqn_log = parse_dqn_eval_log(get_dqn_eval_path(LOG_DIR))

if dqn_log:
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Eval Points", len(dqn_log["timesteps"]))
    col_b.metric(
        "Best Mean Reward",
        f"{max(dqn_log['mean_rewards']):+.3f}",
    )
    col_c.metric(
        "Latest Mean Reward",
        f"{dqn_log['mean_rewards'][-1]:+.3f}",
    )

    # Reward curve with std band
    fig_dqn = go.Figure()
    fig_dqn.add_trace(go.Scatter(
        x=dqn_log["timesteps"],
        y=dqn_log["mean_rewards"],
        mode="lines+markers",
        name="Mean episode reward",
        line=dict(color=COLORS["primary"], width=2),
    ))
    upper = [m + s for m, s in zip(dqn_log["mean_rewards"], dqn_log["std_rewards"])]
    lower = [m - s for m, s in zip(dqn_log["mean_rewards"], dqn_log["std_rewards"])]
    fig_dqn.add_trace(go.Scatter(
        x=dqn_log["timesteps"] + dqn_log["timesteps"][::-1],
        y=upper + lower[::-1],
        fill="toself",
        fillcolor="rgba(0, 119, 182, 0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        name="+/- 1 std",
        showlegend=True,
    ))

    fig_dqn.update_layout(
        title="DQN Mean Episode Reward over Training",
        xaxis_title="Timestep",
        yaxis_title="Mean reward",
        template="plotly_white",
        height=400,
    )
    st.plotly_chart(fig_dqn, use_container_width=True)

    st.caption(f"Source: `{dqn_log['source_path']}`")
else:
    st.info(
        "No DQN evaluation log found. Train the agent with "
        "`python -m src.agent.train_dqn` -- EvalCallback will write "
        "`evaluations.npz` to `logs/eval_dqn/`."
    )

# DQN recommendation accuracy summary
dqn_eval_path = CHECKPOINT_DIR / "dqn_treatment_eval.json"
if dqn_eval_path.exists():
    with open(dqn_eval_path) as f:
        dqn_eval = json.load(f)
    st.subheader("DQN Recommendation Accuracy")
    r1, r2, r3 = st.columns(3)
    r1.metric("Accuracy", f"{dqn_eval.get('accuracy', 0):.1%}")
    r2.metric("Avg Reward", f"{dqn_eval.get('avg_reward', 0):+.3f}")
    r3.metric("Episodes", dqn_eval.get("total_episodes", 0))


# =============================================================================
# MODEL CHECKPOINTS
# =============================================================================

st.header("Model Checkpoints")

tab_vit, tab_dqn, tab_ppo = st.tabs(["ViT Classifier", "DQN Treatment Agent", "PPO Agent"])

with tab_vit:
    vit_dir = CHECKPOINT_DIR
    if vit_dir.exists():
        pth_files = sorted(vit_dir.glob("*.pth"))
        if pth_files:
            for pth in pth_files:
                size_mb = pth.stat().st_size / (1024 * 1024)
                col_name, col_size = st.columns([3, 1])
                with col_name:
                    st.code(str(pth.name))
                with col_size:
                    st.text(f"{size_mb:.1f} MB")
        else:
            st.info("No ViT checkpoints found. Train the model first.")
    else:
        st.info("Checkpoint directory not found.")

with tab_dqn:
    dqn_dir = CHECKPOINT_DIR / "dqn_treatment"
    if dqn_dir.exists():
        zip_files = sorted(dqn_dir.glob("**/*.zip"))
        if zip_files:
            for zf in zip_files:
                size_mb = zf.stat().st_size / (1024 * 1024)
                col_name, col_size = st.columns([3, 1])
                with col_name:
                    st.code(str(zf.relative_to(dqn_dir)))
                with col_size:
                    st.text(f"{size_mb:.1f} MB")
        else:
            st.info("No DQN checkpoints found. Train the DQN agent first.")
    else:
        st.info("DQN checkpoint directory not found.")

with tab_ppo:
    ppo_dir = CHECKPOINT_DIR
    if ppo_dir.exists():
        zip_files = sorted(ppo_dir.glob("ppo_*.zip"))
        zip_files += sorted(ppo_dir.glob("best/*.zip"))
        if zip_files:
            for zf in zip_files:
                size_mb = zf.stat().st_size / (1024 * 1024)
                col_name, col_size = st.columns([3, 1])
                with col_name:
                    st.code(str(zf.name))
                with col_size:
                    st.text(f"{size_mb:.1f} MB")
        else:
            st.info("No PPO checkpoints found. Train the PPO agent first.")
    else:
        st.info("PPO checkpoint directory not found.")


# =============================================================================
# TRAINING COMMANDS
# =============================================================================

st.header("How to Train")

st.markdown("""
Use these commands from the project root to train the models:
""")

st.code("# 1. Prepare the dataset (split into train/val/test)\npython prepare_data.py --yes", language="bash")
st.code("# 2. Train the ViT classifier\npython train.py", language="bash")
st.code("# 3. Train the DQN treatment recommendation agent\npython -m src.agent.train_dqn", language="bash")
st.code("# 4. (Optional) Train the PPO classification agent\npython -m src.agent.train_agent", language="bash")

st.markdown("""
After training, launch the app with:
""")
st.code("streamlit run app/main.py", language="bash")


render_disclaimer()
