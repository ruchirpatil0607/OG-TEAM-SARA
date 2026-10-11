"""
analytics.py — Plotly Interactive Visualization and Analytics Engine for PulseVision.
Generates polished dark-theme interactive charts for raw color signals, filtered optical
pulse waveforms, BPM trends with gap preservation, power spectra, SQI, and comparisons.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import plotly.graph_objects as go

from src.config import (
    BPM_MAX,
    BPM_MIN,
    COLOR_BG_DARK,
    COLOR_CARD_BG,
    COLOR_CARD_BORDER,
    COLOR_EMERALD_GREEN,
    COLOR_PRIMARY_TEAL,
    COLOR_SECONDARY_CYAN,
    COLOR_TEXT_MAIN,
    COLOR_TEXT_MUTED,
    HR_HIGH_HZ,
    HR_LOW_HZ,
)


def _base_dark_layout(title: str, xaxis_title: str, yaxis_title: str) -> go.Layout:
    """Returns consistent dark navy Plotly layout."""
    return go.Layout(
        title=dict(text=title, font=dict(color=COLOR_TEXT_MAIN, size=15)),
        paper_bgcolor=COLOR_CARD_BG,
        plot_bgcolor=COLOR_CARD_BG,
        font=dict(color=COLOR_TEXT_MUTED, family="Inter, -apple-system, sans-serif"),
        xaxis=dict(
            title=xaxis_title,
            gridcolor=COLOR_CARD_BORDER,
            zerolinecolor=COLOR_CARD_BORDER,
            showgrid=True,
            color=COLOR_TEXT_MUTED
        ),
        yaxis=dict(
            title=yaxis_title,
            gridcolor=COLOR_CARD_BORDER,
            zerolinecolor=COLOR_CARD_BORDER,
            showgrid=True,
            color=COLOR_TEXT_MUTED
        ),
        margin=dict(l=50, r=30, t=50, b=40),
        legend=dict(
            bgcolor="rgba(28, 37, 65, 0.7)",
            bordercolor=COLOR_CARD_BORDER,
            font=dict(color=COLOR_TEXT_MAIN)
        ),
        hovermode="x unified"
    )


def create_raw_color_chart(
    time_series: List[float],
    r_vals: List[float],
    g_vals: List[float],
    b_vals: List[float],
    channel_selection: str = "All"
) -> go.Figure:
    """Graph 1: Raw facial color signals over time."""
    fig = go.Figure()
    if not time_series or len(time_series) == 0:
        fig.add_annotation(text="No raw color data collected yet", showarrow=False,
                           font=dict(color=COLOR_TEXT_MUTED, size=14))
        fig.update_layout(_base_dark_layout("Raw Facial Skin Color Signals", "Elapsed Time (s)", "Raw Intensity"))
        return fig

    t = np.array(time_series)
    if channel_selection in ["All", "Red"]:
        fig.add_trace(go.Scatter(x=t, y=r_vals, mode="lines", name="Red Channel",
                                 line=dict(color="#EF4444", width=1.5)))
    if channel_selection in ["All", "Green"]:
        fig.add_trace(go.Scatter(x=t, y=g_vals, mode="lines", name="Green Channel",
                                 line=dict(color="#10B981", width=1.5)))
    if channel_selection in ["All", "Blue"]:
        fig.add_trace(go.Scatter(x=t, y=b_vals, mode="lines", name="Blue Channel",
                                 line=dict(color="#3B82F6", width=1.5)))

    fig.update_layout(_base_dark_layout("Raw Facial Skin Color Signals (RGB)", "Elapsed Time (s)", "Channel Intensity (a.u.)"))
    return fig


def create_filtered_pulse_chart(
    pulse_waveform: np.ndarray,
    fs: float = 30.0,
    title_suffix: str = ""
) -> go.Figure:
    """Graph 2: Filtered optical blood volume pulse waveform (rPPG)."""
    fig = go.Figure()
    if pulse_waveform is None or len(pulse_waveform) == 0:
        fig.add_annotation(text="Accumulating pulse signal...", showarrow=False,
                           font=dict(color=COLOR_TEXT_MUTED, size=14))
        fig.update_layout(_base_dark_layout(f"Filtered Pulse Waveform {title_suffix}", "Window Time (s)", "Normalized Amplitude"))
        return fig

    n = len(pulse_waveform)
    t = np.linspace(0, n / max(1.0, fs), n)

    fig.add_trace(go.Scatter(
        x=t,
        y=pulse_waveform,
        mode="lines",
        name="Optical Pulse (rPPG)",
        line=dict(color=COLOR_PRIMARY_TEAL, width=2.2)
    ))

    layout = _base_dark_layout(
        f"Filtered Optical Pulse Waveform {title_suffix}",
        "Window Time (s)",
        "Relative Amplitude (a.u.)"
    )
    # Subtitle note clarifying optical pulse vs ECG
    layout.annotations = [
        dict(
            text="Note: Optical photoplethysmogram (rPPG) from webcam video, not an electrocardiogram (ECG).",
            showarrow=False,
            xref="paper", yref="paper",
            x=0.0, y=1.12,
            font=dict(size=11, color=COLOR_TEXT_MUTED)
        )
    ]
    fig.update_layout(layout)
    return fig


def create_heart_rate_trend_chart(
    bpm_records: List[Any],
    session_avg_bpm: Optional[float] = None
) -> go.Figure:
    """Graph 3: Heart rate trend over elapsed time with gap preservation."""
    fig = go.Figure()
    if not bpm_records:
        fig.add_annotation(text="No valid heart rate estimates yet", showarrow=False,
                           font=dict(color=COLOR_TEXT_MUTED, size=14))
        fig.update_layout(_base_dark_layout("Heart Rate Trend", "Elapsed Time (s)", "Beats Per Minute (BPM)"))
        return fig

    # Separate points and preserve gaps when delta_t > 2.5s
    times = []
    bpms = []
    for i, rec in enumerate(bpm_records):
        # Support either tuple (t, bpm, sqi) or dict {"t": t, "bpm": bpm}
        t = rec[0] if isinstance(rec, (list, tuple)) else rec["t"]
        b = rec[1] if isinstance(rec, (list, tuple)) else rec["bpm"]

        if i > 0:
            prev_t = bpm_records[i-1][0] if isinstance(bpm_records[i-1], (list, tuple)) else bpm_records[i-1]["t"]
            if t - prev_t > 2.5:
                # Insert NaN to break line across unmeasured gap
                times.append(None)
                bpms.append(None)

        times.append(t)
        bpms.append(b)

    fig.add_trace(go.Scatter(
        x=times,
        y=bpms,
        mode="lines+markers",
        name="Estimated BPM",
        line=dict(color=COLOR_SECONDARY_CYAN, width=2.5),
        marker=dict(size=6, color=COLOR_PRIMARY_TEAL)
    ))

    # Add reference line for session average
    if session_avg_bpm is not None and len([b for b in bpms if b is not None]) > 0:
        valid_times = [t for t in times if t is not None]
        if valid_times:
            fig.add_trace(go.Scatter(
                x=[valid_times[0], valid_times[-1]],
                y=[session_avg_bpm, session_avg_bpm],
                mode="lines",
                name=f"Session Avg ({session_avg_bpm:.1f} BPM)",
                line=dict(color=COLOR_EMERALD_GREEN, width=1.8, dash="dash")
            ))

    layout = _base_dark_layout("Heart Rate Trend (BPM)", "Elapsed Time (s)", "Heart Rate (BPM)")
    layout.yaxis.range = [max(35, min(bpms if bpms else [50]) - 10), min(220, max(bpms if bpms else [100]) + 15)]
    fig.update_layout(layout)
    return fig


def create_frequency_spectrum_chart(
    frequencies: np.ndarray,
    power_spectrum: np.ndarray,
    peak_bpm: Optional[float] = None
) -> go.Figure:
    """Graph 4: Power Spectral Density with highlighted peak in cardiac band."""
    fig = go.Figure()
    if frequencies is None or len(frequencies) == 0:
        fig.add_annotation(text="Spectral analysis unavailable", showarrow=False,
                           font=dict(color=COLOR_TEXT_MUTED, size=14))
        fig.update_layout(_base_dark_layout("Power Spectral Density (PSD)", "Frequency (Hz)", "Power Density"))
        return fig

    # Convert Hz to BPM for secondary axis or hover
    bpms_axis = frequencies * 60.0

    fig.add_trace(go.Scatter(
        x=frequencies,
        y=power_spectrum,
        mode="lines",
        name="Spectral Power",
        line=dict(color="#A78BFA", width=2.0),
        hovertemplate="Freq: %{x:.2f} Hz (%{customdata:.1f} BPM)<br>Power: %{y:.2e}<extra></extra>",
        customdata=bpms_axis
    ))

    # Highlight cardiac passband [0.7, 3.5] Hz
    fig.add_vrect(
        x0=HR_LOW_HZ, x1=HR_HIGH_HZ,
        fillcolor="rgba(0, 242, 254, 0.08)",
        layer="below", line_width=0,
        annotation_text="Human Cardiac Band (42-210 BPM)",
        annotation_position="top left",
        annotation_font=dict(size=10, color=COLOR_TEXT_MUTED)
    )

    # Highlight peak if provided
    if peak_bpm is not None and peak_bpm > 0:
        peak_hz = peak_bpm / 60.0
        fig.add_vline(
            x=peak_hz,
            line=dict(color=COLOR_EMERALD_GREEN, width=2, dash="dash"),
            annotation_text=f"Peak: {peak_bpm:.1f} BPM ({peak_hz:.2f} Hz)",
            annotation_position="top right",
            annotation_font=dict(size=11, color=COLOR_EMERALD_GREEN)
        )

    layout = _base_dark_layout("Power Spectral Density (Frequency Spectrum)", "Frequency (Hz)", "Power (a.u.)")
    layout.xaxis.range = [0.4, 4.0]
    fig.update_layout(layout)
    return fig


def create_sqi_trend_chart(sqi_records: List[Any]) -> go.Figure:
    """Graph 5: Signal Quality Index trend with reliability thresholds."""
    fig = go.Figure()
    if not sqi_records:
        fig.add_annotation(text="No quality records available", showarrow=False,
                           font=dict(color=COLOR_TEXT_MUTED, size=14))
        fig.update_layout(_base_dark_layout("Signal Quality Index (SQI) Trend", "Elapsed Time (s)", "Quality (%)"))
        return fig

    times = [rec[0] for rec in sqi_records]
    sqis = [rec[1] for rec in sqi_records]

    fig.add_trace(go.Scatter(
        x=times,
        y=sqis,
        mode="lines",
        name="Quality Index",
        line=dict(color=COLOR_PRIMARY_TEAL, width=2.0)
    ))

    # Reliability threshold lines
    fig.add_hline(y=60.0, line=dict(color=COLOR_EMERALD_GREEN, width=1, dash="dot"),
                  annotation_text="Reliable (60%)", annotation_position="bottom right")
    fig.add_hline(y=40.0, line=dict(color="#F59E0B", width=1, dash="dot"),
                  annotation_text="Marginal (40%)", annotation_position="bottom right")

    layout = _base_dark_layout("Signal Quality Index (SQI) Over Time", "Elapsed Time (s)", "SQI Score (%)")
    layout.yaxis.range = [0, 105]
    fig.update_layout(layout)
    return fig


def create_session_comparison_chart(sessions_data: List[Dict[str, Any]]) -> go.Figure:
    """Graph 6: Multi-session BPM comparison chart."""
    fig = go.Figure()
    colors = ["#00F2FE", "#10B981", "#A78BFA", "#F59E0B", "#EF4444"]

    for i, sess in enumerate(sessions_data):
        c = colors[i % len(colors)]
        bpm_series = sess.get("bpm_series", [])
        if bpm_series:
            times = [item["t"] for item in bpm_series]
            bpms = [item["bpm"] for item in bpm_series]
            label = f"{sess.get('user_label', 'Session')} ({sess.get('session_id')})"
            fig.add_trace(go.Scatter(
                x=times, y=bpms, mode="lines+markers",
                name=label,
                line=dict(color=c, width=2.0),
                marker=dict(size=4)
            ))

    layout = _base_dark_layout("Historical Session Comparison", "Elapsed Time (s)", "Heart Rate (BPM)")
    fig.update_layout(layout)
    return fig
