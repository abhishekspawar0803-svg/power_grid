import streamlit as st
import torch
import numpy as np
import joblib
import json
import plotly.graph_objects as go
import pandas as pd
from models import Autoencoder, LSTMClassifier
from plotly.subplots import make_subplots

st.set_page_config(page_title="Grid Anomaly Detector", layout="wide")


@st.cache_resource
def load_assets():
    with open("model_config.json", "r") as f:
        config = json.load(f)

    scaler = joblib.load("feature_scaler.pkl")

    autoencoder = Autoencoder(5, 64, 1)
    autoencoder.load_state_dict(
        torch.load("lstm_autoencoder_final.pth", map_location=torch.device("cpu"))
    )
    autoencoder.eval()

    classifier = LSTMClassifier(5, 64, 4)
    classifier.load_state_dict(
        torch.load("lstm_classifier.pth", map_location=torch.device("cpu"))
    )
    classifier.eval()

    return config, scaler, autoencoder, classifier


@st.cache_data
def load_dataset():
    df = pd.read_csv("power_system_multiclass_anomaly_data.csv")
    return df


config, scaler, autoencoder, classifier = load_assets()
df = load_dataset()

LABEL_MAP = {
    0: "Normal",
    1: "Voltage Anomaly",
    2: "Frequency Anomaly",
    3: "Power Factor Anomaly",
    4: "Combined Anomaly",
}

FEATURE_NAMES = [
    "Voltage (V)",
    "Current (A)",
    "Power (kW)",
    "Frequency (Hz)",
    "Power_Factor"
]

FEATURE_COLOR_MAP = {
    "Voltage (V)": "#ff6b6b",
    "Current (A)": "#4dabf7",
    "Power (kW)": "#51cf66",
    "Frequency (Hz)": "#ffd43b",
    "Power_Factor": "#845ef7",
}

CLASS_COLOR_MAP = {
    "Voltage Anomaly": "#ff6b6b",
    "Frequency Anomaly": "#ffd43b",
    "Power Factor Anomaly": "#845ef7",
    "Combined Anomaly": "#51cf66",
}

WINDOW_SIZE = 12
SEGMENT_LENGTH = 12


def build_sliding_windows(x_scaled, sequence_length=12):
    windows = []
    for i in range(len(x_scaled) - sequence_length + 1):
        windows.append(x_scaled[i:i + sequence_length])
    return np.array(windows, dtype=np.float32)


def compute_window_error(input_windows, reconstructed_windows):
    residual = torch.abs(reconstructed_windows - input_windows) ** 2
    window_error = torch.max(residual, dim=1)[0].mean(dim=1)
    return window_error


def get_blind_real_anomaly_start(df, segment_length=12):
    valid_starts = []
    max_start = max(0, len(df) - segment_length)

    for start_idx in range(max_start + 1):
        segment_labels = df.iloc[start_idx:start_idx + segment_length]["Anomaly_Label"].to_numpy()
        if len(segment_labels) == segment_length and np.any(segment_labels > 0):
            valid_starts.append(start_idx)

    if len(valid_starts) == 0:
        return 0

    return int(np.random.choice(valid_starts))


def build_xy_windows(x_scaled, y_labels, sequence_length=12):
    xy_combined = np.hstack((x_scaled, y_labels.reshape(-1, 1)))
    windows = []
    for i in range(len(xy_combined) - sequence_length + 1):
        windows.append(xy_combined[i:i + sequence_length])
    all_windows = np.array(windows, dtype=np.float32)

    if len(all_windows) == 0:
        return (
            np.empty((0, sequence_length, x_scaled.shape[1]), dtype=np.float32),
            np.empty((0, sequence_length), dtype=np.float32)
        )

    x_windows = all_windows[:, :, :-1]
    y_windows = all_windows[:, :, -1]
    return x_windows, y_windows


def run_diagnostic(raw_signal):
    threshold = float(config["anomaly_threshold"])
    scaled_signal = scaler.transform(raw_signal)
    x_windows = build_sliding_windows(scaled_signal, WINDOW_SIZE)

    if len(x_windows) == 0:
        return {
            "scaled_signal": scaled_signal,
            "x_windows": x_windows,
            "window_errors": np.array([]),
            "threshold": threshold,
            "anomalous_window_idx": np.array([], dtype=int),
            "global_error": 0.0,
            "diagnosis": "Normal",
            "pred_idx": None,
            "class_probs": None,
            "worst_window_idx": None
        }

    x_tensor = torch.tensor(x_windows, dtype=torch.float32)
    infer_dataset = torch.utils.data.TensorDataset(x_tensor)
    infer_loader = torch.utils.data.DataLoader(infer_dataset, batch_size=32, shuffle=False)

    all_reconstructed = []
    all_window_errors = []

    with torch.no_grad():
        for (batch_x,) in infer_loader:
            reconstructed = autoencoder(batch_x)
            batch_errors = compute_window_error(batch_x, reconstructed)

            all_reconstructed.append(reconstructed.cpu())
            all_window_errors.append(batch_errors.cpu())

    reconstructed_all = torch.cat(all_reconstructed, dim=0)
    window_errors = torch.cat(all_window_errors, dim=0).numpy()

    anomalous_window_idx = np.where(window_errors >= threshold)[0]
    global_error = float(np.max(window_errors))

    diagnosis = "Normal"
    pred_idx = None
    class_probs = None
    worst_window_idx = None

    if len(anomalous_window_idx) > 0:
        worst_window_idx = int(np.argmax(window_errors))
        classifier_input = x_tensor[worst_window_idx].unsqueeze(0)

        cls_dataset = torch.utils.data.TensorDataset(classifier_input)
        cls_loader = torch.utils.data.DataLoader(cls_dataset, batch_size=1, shuffle=False)

        with torch.no_grad():
            for (batch_x,) in cls_loader:
                class_logits = classifier(batch_x)
                probabilities = torch.softmax(class_logits, dim=1)
                _, predicted_class = torch.max(class_logits, dim=1)

                pred_idx = str(predicted_class.item())
                diagnosis = config["label_map"].get(pred_idx, "Unknown Fault")
                class_probs = probabilities.squeeze(0).cpu().numpy()

    return {
        "scaled_signal": scaled_signal,
        "x_windows": x_windows,
        "window_errors": window_errors,
        "threshold": threshold,
        "anomalous_window_idx": anomalous_window_idx,
        "global_error": global_error,
        "diagnosis": diagnosis,
        "pred_idx": pred_idx,
        "class_probs": class_probs,
        "worst_window_idx": worst_window_idx
    }

st.title("Power Grid Anomaly Detection")
st.markdown("Real-time telemetry analysis using Deep Learning LSTM Autoencoder + LSTM Classifier.")
st.divider()

st.sidebar.header("Dataset Controls")

max_start = max(0, len(df) - SEGMENT_LENGTH)

if "start_idx" not in st.session_state:
    st.session_state.start_idx = 0

if "blind_injected" not in st.session_state:
    st.session_state.blind_injected = False

if "blind_meta" not in st.session_state:
    st.session_state.blind_meta = None

manual_start = st.sidebar.slider("Start Row", 0, max_start, st.session_state.start_idx, 1)
st.session_state.start_idx = manual_start

col_btn1, col_btn2 = st.sidebar.columns(2)

if col_btn1.button("New Window", use_container_width=True):
    st.session_state.start_idx = np.random.randint(0, max_start + 1)
    st.session_state.blind_injected = False
    st.session_state.blind_meta = None

if col_btn2.button("Blind Injection", use_container_width=True):
    st.session_state.start_idx = get_blind_real_anomaly_start(df, SEGMENT_LENGTH)
    st.session_state.blind_injected = True
    st.session_state.blind_meta = {
        "mode": "real_anomaly_segment"
    }

window_df = df.iloc[st.session_state.start_idx:st.session_state.start_idx + SEGMENT_LENGTH].copy()

if len(window_df) == 0:
    st.warning("No data available for the selected range.")
    st.stop()

raw_signal = window_df[FEATURE_NAMES].to_numpy(dtype=np.float32)
true_labels = window_df["Anomaly_Label"].to_numpy()
x_axis = np.arange(len(window_df))

st.subheader("Live Sensor Telemetry")

fig = make_subplots(
    rows=5,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.03,
    subplot_titles=FEATURE_NAMES
)

for i, feature in enumerate(FEATURE_NAMES, start=1):
    fig.add_trace(
        go.Scatter(
            x=x_axis,
            y=raw_signal[:, i - 1],
            mode="lines+markers",
            name=feature,
            line=dict(color=FEATURE_COLOR_MAP[feature], width=2),
            showlegend=False
        ),
        row=i,
        col=1
    )
    fig.update_yaxes(title_text=feature, row=i, col=1)

fig.update_xaxes(title_text="Row Index", row=5, col=1)

fig.update_layout(
    height=950,
    margin=dict(l=0, r=0, t=30, b=0),
    template="plotly_dark",
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)

col_a, col_b, col_c = st.columns(3)
with col_a:
    st.write(f"Selected Segment Length: {len(raw_signal)} rows")
with col_b:
    st.write(f"Feature Count: {raw_signal.shape[1]}")
with col_c:
    st.write(f"Diagnostic Windows: {max(0, len(raw_signal) - WINDOW_SIZE + 1)}")

if st.session_state.blind_meta is not None:
    st.info("Blind real-anomaly segment loaded from the internal dataset. Ground truth remains hidden until reveal.")

st.divider()
st.subheader("Deep Learning Diagnostics")

if st.button("Run Full Diagnostic Analysis", type="primary", use_container_width=True):
    with st.spinner("Applying scaler, building 12-step windows, and passing telemetry through LSTM Autoencoder..."):
        results = run_diagnostic(raw_signal)

    st.markdown("### Phase 1: Autoencoder Results")

    error_col1, error_col2, error_col3 = st.columns(3)
    error_col1.metric("Max Window Error", f"{results['global_error']:.5f}")
    error_col2.metric("Threshold", f"{results['threshold']:.5f}")
    error_col3.metric("Flagged Windows", int(len(results["anomalous_window_idx"])))

    # Feature deviation view instead of single-point error chart
    dev_fig = go.Figure()

    # Voltage deviation (% from mean of this segment)
    v = raw_signal[:, 0]
    v_mean = np.mean(v) if len(v) > 0 else 0.0
    v_dev_pct = (v - v_mean) / v_mean * 100.0 if v_mean != 0 else np.zeros_like(v)
    dev_fig.add_trace(
        go.Scatter(
            x=x_axis,
            y=v_dev_pct,
            mode="lines+markers",
            name="Voltage Deviation (%)",
            line=dict(color=FEATURE_COLOR_MAP["Voltage (V)"], width=2)
        )
    )

    # Frequency deviation from 50 Hz
    f = raw_signal[:, 3]
    f_dev = f - 50.0
    dev_fig.add_trace(
        go.Scatter(
            x=x_axis,
            y=f_dev,
            mode="lines+markers",
            name="Frequency Deviation (Hz)",
            line=dict(color=FEATURE_COLOR_MAP["Frequency (Hz)"], width=2)
        )
    )

    # Power factor deviation from 1.0
    pf = raw_signal[:, 4]
    pf_dev = pf - 1.0
    dev_fig.add_trace(
        go.Scatter(
            x=x_axis,
            y=pf_dev,
            mode="lines+markers",
            name="Power Factor Deviation",
            line=dict(color=FEATURE_COLOR_MAP["Power_Factor"], width=2)
        )
    )

    dev_fig.update_layout(
        title="Feature Deviation View (Segment)",
        xaxis_title="Row Index",
        yaxis_title="Deviation",
        height=380,
        margin=dict(l=0, r=0, t=50, b=0),
        template="plotly_dark"
    )
    st.plotly_chart(dev_fig, use_container_width=True)

    if len(results["anomalous_window_idx"]) == 0:
        st.success(
            f"GRID STABLE — Max Window Error: {results['global_error']:.5f} is below threshold {results['threshold']:.5f}"
        )
        st.info("All 12-step windows were reconstructed cleanly. No anomaly window was detected, so the LSTM classifier is resting.")
    else:
        st.error(
            f"ANOMALY DETECTED — {len(results['anomalous_window_idx'])} window(s) exceeded threshold {results['threshold']:.5f}"
        )

        spans = []
        for idx in results["anomalous_window_idx"]:
            spans.append(f"[{idx}–{idx + WINDOW_SIZE - 1}]")
        st.write("Flagged Signal Regions:", ", ".join(spans))

        st.markdown("### Phase 2: LSTM Classifier Diagnosis")

        with st.spinner("Routing the most abnormal 12-step window to LSTM Classifier..."):
            diagnosis = results["diagnosis"]
            pred_idx = results["pred_idx"]
            class_probs = results["class_probs"]

        st.warning(f"DIAGNOSIS: {diagnosis}")
        st.write(f"Raw Model Output Class Index: {pred_idx}")

        if class_probs is not None:
            class_names = [
                config["label_map"].get("0", "Voltage Anomaly"),
                config["label_map"].get("1", "Frequency Anomaly"),
                config["label_map"].get("2", "Power Factor Anomaly"),
                config["label_map"].get("3", "Combined Anomaly"),
            ]

            prob_fig = go.Figure(
                data=[
                    go.Bar(
                        x=class_names,
                        y=class_probs,
                        marker_color=[CLASS_COLOR_MAP[name] for name in class_names]
                    )
                ]
            )
            prob_fig.update_layout(
                title="Classifier Confidence",
                yaxis_title="Probability",
                height=380,
                margin=dict(l=0, r=0, t=50, b=0),
                template="plotly_dark"
            )
            st.plotly_chart(prob_fig, use_container_width=True)

    st.markdown("### Signal Snapshot")
    snapshot_df = pd.DataFrame(raw_signal, columns=FEATURE_NAMES)
    snapshot_df["Anomaly_Label"] = true_labels
    st.dataframe(
        snapshot_df.head(20),
        use_container_width=True
    )

    scaled_signal = scaler.transform(raw_signal)
    x_gt_windows, y_gt_windows = build_xy_windows(scaled_signal, true_labels, WINDOW_SIZE)

    pure_normal_idx = np.where((y_gt_windows == 0).all(axis=1))[0]
    mixed_or_anom_idx = np.where((y_gt_windows != 0).any(axis=1))[0]

    if len(y_gt_windows) == 0 or len(pure_normal_idx) > 0:
        segment_ground_label = 0
        ground_truth_text = "Normal"
        gt_classifier_idx = None
    else:
        y_anomaly = y_gt_windows[mixed_or_anom_idx]
        y_anomaly = np.max(y_anomaly, axis=1)
        y_anomaly = y_anomaly - 1
        gt_classifier_idx = str(int(y_anomaly[0]))
        segment_ground_label = int(np.max(true_labels))
        ground_truth_text = config["label_map"].get(gt_classifier_idx, "Unknown Fault")

    st.markdown("### Reveal Ground Label")
    with st.expander("Show ground truth label for this selected segment"):
        st.write(f"Ground Truth Class Index: {segment_ground_label}")
        st.write(f"Ground Truth Label: {ground_truth_text}")
        st.write(f"LSTM Predicted Label: {results['diagnosis']}")

        if results["pred_idx"] is None:
            st.info("The detector marked this segment as normal, so the classifier was not triggered.")
        else:
            predicted_label = config["label_map"].get(results["pred_idx"], "Unknown Fault")
            gt_classifier_label = "Normal" if gt_classifier_idx is None else config["label_map"].get(gt_classifier_idx, "Unknown Fault")

            if predicted_label == gt_classifier_label:
                st.success("Classifier output matches the revealed ground label.")
            else:
                st.error("Classifier output does not match the revealed ground label.")