"""
ASPIRATION BEHAVIOR ANALYSIS - Complete Streamlit App v3
Analyse complète avec explication pédagogique du comportement d'Absaugen
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Aspiration Analysis - Complete Study",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔬 Aspiration Behavior Analysis - Complete Study")
st.markdown("Complete detailed analysis of Aspiration  behavior - 240 test runs")

# ============================================================
# LOAD AND ANALYZE DATA
# ============================================================

@st.cache_data
def load_and_analyze_data():
    """Load data and perform complete analysis"""
    
    # Load JSONL
    data_path = Path(r"Q:\1CHOM\Old tubings without pressure sensor in line\_parsed_output\washer_aspiration_traces.jsonl")
    
    if not data_path.exists():
        return None
    
    traces_list = []
    with open(data_path, 'r') as f:
        for line in f:
            try:
                record = json.loads(line)
                if isinstance(record.get('trace'), list) and len(record['trace']) > 100:
                    traces_list.append(record)
            except:
                continue
    
    df = pd.DataFrame(traces_list)
    df['event_timestamp'] = pd.to_datetime(df['event_timestamp'], errors='coerce')
    df['trace'] = df['trace'].apply(lambda x: np.array(x, dtype=float) if isinstance(x, list) else None)
    df = df[df['trace'].notna()].copy()
    
    return df

# Custom Hampel filter
def hampel_filter_custom(x, window_length=9, n_sigma=4.0):
    """Custom Hampel filter"""
    x = np.asarray(x, dtype=float)
    filtered = x.copy()
    half_window = window_length // 2
    
    for i in range(len(x)):
        start = max(0, i - half_window)
        end = min(len(x), i + half_window + 1)
        window = x[start:end]
        
        median_val = np.median(window)
        mad = np.median(np.abs(window - median_val))
        threshold = n_sigma * mad
        
        if threshold > 0 and abs(x[i] - median_val) > threshold:
            filtered[i] = median_val
    
    return filtered

# Drop detection
def detect_drops(df):
    """Detect drops in all traces"""
    
    DROP_DETECTION_START = 1200
    DROP_DETECTION_END = 1700
    HAMPEL_WINDOW = 9
    HAMPEL_K = 4.0
    BASELINE_PERCENTILE = 75
    
    drop_results = []
    
    for idx, trace_arr in enumerate(df['trace']):
        if trace_arr is None or len(trace_arr) < DROP_DETECTION_END:
            drop_results.append({
                'trace_idx': idx,
                'drop_detected': False,
                'drop_idx': None,
                'drop_depth': None,
                'min_value': None,
                'pre_drop_mean': None
            })
            continue
        
        window = trace_arr[DROP_DETECTION_START:DROP_DETECTION_END]
        filtered = hampel_filter_custom(window, window_length=HAMPEL_WINDOW, n_sigma=HAMPEL_K)
        filtered_smooth = pd.Series(filtered).rolling(window=9, center=True, min_periods=1).median().values
        
        baseline = np.percentile(filtered_smooth[:200], BASELINE_PERCENTILE) if len(filtered_smooth) > 200 else np.median(filtered_smooth)
        drop_threshold = baseline * 0.4
        below_threshold = filtered_smooth < drop_threshold
        
        drop_segments = []
        segment_start = None
        count = 0
        
        for i, below in enumerate(below_threshold):
            if below:
                if segment_start is None:
                    segment_start = i
                count += 1
            else:
                if count >= 5:
                    drop_segments.append((segment_start, i-1, count))
                segment_start = None
                count = 0
        
        if count >= 5 and segment_start is not None:
            drop_segments.append((segment_start, len(below_threshold)-1, count))
        
        if drop_segments:
            drop_start, drop_end, _ = drop_segments[0]
            drop_idx = DROP_DETECTION_START + drop_start
            
            drop_phase = trace_arr[drop_idx:min(drop_idx+250, len(trace_arr))]
            min_value = np.min(drop_phase)
            pre_drop_mean = np.mean(trace_arr[max(0, drop_idx-100):drop_idx])
            drop_depth = pre_drop_mean - min_value
            
            drop_results.append({
                'trace_idx': idx,
                'drop_detected': True,
                'drop_idx': drop_idx,
                'drop_depth': drop_depth,
                'min_value': min_value,
                'pre_drop_mean': pre_drop_mean
            })
        else:
            drop_results.append({
                'trace_idx': idx,
                'drop_detected': False,
                'drop_idx': None,
                'drop_depth': None,
                'min_value': None,
                'pre_drop_mean': None
            })
    
    return pd.DataFrame(drop_results)

# Detect anomalies with ADAPTIVE thresholds
def detect_anomalies_adaptive(df, drop_df):
    """Detect anomalies using real data percentiles"""
    
    detected = drop_df[drop_df['drop_detected']]
    
    if len(detected) == 0:
        return pd.DataFrame()
    
    # Calculate thresholds from data
    drop_indices = detected['drop_idx'].values
    drop_depths = detected['drop_depth'].values
    min_values = detected['min_value'].values
    
    # Calculate startup slopes and plateau noise for all traces
    startup_slopes = []
    plateau_stds = []
    
    for trace in df['trace']:
        if len(trace) > 150:
            demarrage = trace[0:150]
            slope = (np.mean(demarrage[-50:]) - np.mean(demarrage[:50])) / 100
            startup_slopes.append(slope)
        
        if len(trace) > 1350:
            plateau = trace[100:1350]
            plateau_stds.append(np.std(plateau))
    
    THRESHOLDS = {
        'drop_too_early': np.percentile(drop_indices, 5),
        'drop_too_late': np.percentile(drop_indices, 95),
        'drop_depth_too_shallow': np.percentile(drop_depths, 5),
        'min_value_too_high': np.percentile(min_values, 95),
        'startup_slope_too_weak': np.percentile(startup_slopes, 5) if startup_slopes else -0.178,
        'plateau_too_noisy': np.percentile(plateau_stds, 95) if plateau_stds else 37.2
    }
    
    anomalies = []
    
    for idx, row in df.iterrows():
        trace = row['trace']
        drop_info = drop_df.iloc[idx]
        
        issues = []
        
        # Issue 1: No drop
        if not drop_info['drop_detected']:
            issues.append("❌ No drop detected")
        else:
            # Issue 2: Drop timing
            if drop_info['drop_idx'] < THRESHOLDS['drop_too_early']:
                issues.append(f"⚠️  Drop too early (idx={drop_info['drop_idx']:.0f})")
            elif drop_info['drop_idx'] > THRESHOLDS['drop_too_late']:
                issues.append(f"⚠️  Drop too late (idx={drop_info['drop_idx']:.0f})")
            
            # Issue 3: Drop depth
            if drop_info['drop_depth'] < THRESHOLDS['drop_depth_too_shallow']:
                issues.append(f"⚠️  Shallow drop (depth={drop_info['drop_depth']:.1f})")
            
            # Issue 4: Min value
            if drop_info['min_value'] > THRESHOLDS['min_value_too_high']:
                issues.append(f"⚠️  Min too high ({drop_info['min_value']:.1f})")
        
        # Issue 5: Shape analysis
        if len(trace) > 150:
            demarrage = trace[0:150]
            plateau = trace[100:1350]
            
            demarrage_slope = (np.mean(demarrage[-50:]) - np.mean(demarrage[:50])) / 100
            if demarrage_slope < THRESHOLDS['startup_slope_too_weak']:
                issues.append("⚠️  Weak startup")
            
            if len(plateau) > 0 and np.std(plateau) > THRESHOLDS['plateau_too_noisy']:
                issues.append("⚠️  Noisy plateau")
        
        anomalies.append({
            'trace_idx': idx,
            'is_anomalous': len(issues) > 0,
            'issue_count': len(issues),
            'issues': issues
        })
    
    return pd.DataFrame(anomalies), THRESHOLDS

# Load data
with st.spinner("📊 Loading and analyzing data..."):
    df = load_and_analyze_data()

if df is None or len(df) == 0:
    st.error("❌ Failed to load data")
    st.stop()

with st.spinner("🔍 Detecting drops and anomalies..."):
    drop_df = detect_drops(df)
    anomaly_df, thresholds = detect_anomalies_adaptive(df, drop_df)

st.success(f"✅ Loaded {len(df)} test runs - Analysis complete!")

# ============================================================
# MAIN TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📚 Educational Overview",
    "📊 Overall Statistics",
    "🔍 Individual Test Analysis",
    "📉 Drop Detection Details",
    "⚠️  Anomalies",
    "🔬 Detailed Zooms"
])

# ============================================================
# TAB 1: EDUCATIONAL OVERVIEW
# ============================================================

with tab1:
    st.markdown("📚 Understanding Aspiration Behavior ")
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.info("""
        **Dataset:**
        - 240 test runs
        - ~1800 samples per run
        - ADC signal 0-256
        """)
    
    st.markdown("---")
    
    # Phase explanation
    st.markdown("### 🔄 The 4 Phases of Aspiration")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        #### Phase 1️⃣ : STARTUP
        **Index: 0-150**
        
        - Signal: 0 → 150
        - Pump **starting**
        - Suction **building up**
        - Normal: Linear ramp
        """)
    
    with col2:
        st.markdown("""
        #### Phase 2️⃣ : HIGH PLATEAU
        **Index: 100-1350**
        
        - Signal: ~150-250
        - Aspiration **active**
        - Liquid being **removed**
        - Normal: Stable with oscillations
        """)
    
    with col3:
        st.markdown("""
        #### Phase 3️⃣ : DROP
        **Index: 1350-1600**
        
        - Signal: 250 → 15
        - **Tube empties**
        - Suction still active
        - Important: Timing & depth
        """)
    
    with col4:
        st.markdown("""
        #### Phase 4️⃣ : LOW PLATEAU
        **Index: 1600-1800**
        
        - Signal: ~10-20
        - Tube **empty**
        - Waiting for next cycle
        - Normal: Stable low
        """)
    
    st.markdown("---")
    
    # Visual diagram
    st.markdown("### 📈 Typical Aspiration Signal Pattern")
    
    # Create example trace
    example_trace = np.concatenate([
        np.linspace(0, 150, 150),                    # Startup
        np.random.normal(200, 15, 1250),            # High plateau
        np.linspace(200, 15, 250),                   # Drop
        np.random.normal(15, 2, 150)                 # Low plateau
    ])
    
    fig_demo = go.Figure()
    
    fig_demo.add_trace(go.Scatter(
        y=example_trace,
        mode='lines',
        name='Signal',
        line=dict(color='blue', width=3),
        fill='tozeroy'
    ))
    
    # Add phase markers
    fig_demo.add_vline(x=150, line_dash="dash", line_color="green", 
                      annotation_text="Phase 1→2", annotation_position="top right")
    fig_demo.add_vline(x=1350, line_dash="dash", line_color="red",
                      annotation_text="Phase 3 Start", annotation_position="top left")
    fig_demo.add_vline(x=1600, line_dash="dash", line_color="orange",
                      annotation_text="Phase 4 Start", annotation_position="top right")
    
    # Add phase background colors
    fig_demo.add_vrect(x0=0, x1=150, fillcolor="green", opacity=0.1, layer="below")
    fig_demo.add_vrect(x0=150, x1=1350, fillcolor="blue", opacity=0.1, layer="below")
    fig_demo.add_vrect(x0=1350, x1=1600, fillcolor="red", opacity=0.1, layer="below")
    fig_demo.add_vrect(x0=1600, x1=1800, fillcolor="orange", opacity=0.1, layer="below")
    
    fig_demo.update_layout(
        title="Typical Aspiration Cycle (Example)",
        xaxis_title="Sample Index",
        yaxis_title="ADC Signal (0-256)",
        height=500,
        template='plotly_white'
    )
    
    st.plotly_chart(fig_demo, use_container_width=True)
    
    st.markdown("---")
    
    # Key metrics explanation - with ADAPTIVE thresholds
    st.markdown("### 🎯 Key Metrics (Adaptive Thresholds from Real Data)")
    
    # Display actual thresholds from the data
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Drop Index Min", f"{thresholds['drop_too_early']:.0f}")
    with col2:
        st.metric("Drop Index Max", f"{thresholds['drop_too_late']:.0f}")
    with col3:
        st.metric("Drop Depth Min", f"{thresholds['drop_depth_too_shallow']:.1f}")
    with col4:
        st.metric("Min Value Max", f"{thresholds['min_value_too_high']:.1f}")
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        #### 📍 Drop Index
        **When does the tube empty?**
        
        - Normal range: {thresholds['drop_too_early']:.0f}-{thresholds['drop_too_late']:.0f}
        - Too early (<{thresholds['drop_too_early']:.0f}): Valve issue?
        - Too late (>{thresholds['drop_too_late']:.0f}): Blockage?
        - Calculated from 5th-95th percentiles
        """)
    
    with col2:
        st.markdown(f"""
        #### 📊 Drop Depth
        **How deep does signal fall?**
        
        - Normal minimum: >{thresholds['drop_depth_too_shallow']:.1f}
        - Too shallow: Tube not emptying
        - Blocked tube or dirty sensor
        - Calculated from 5th percentile
        """)
    
    with col3:
        st.markdown(f"""
        #### 📉 Min Value
        **Lowest signal in drop phase**
        
        - Normal maximum: <{thresholds['min_value_too_high']:.1f}
        - Too high: Incomplete emptying
        - Air bubbles or sensor issue
        - Calculated from 95th percentile
        """)

# ============================================================
# TAB 2: OVERALL STATISTICS
# ============================================================

with tab2:
    st.markdown("## 📊 Overall Dataset Statistics")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Test Runs", len(df))
    with col2:
        st.metric("Drops Detected", drop_df['drop_detected'].sum())
    with col3:
        st.metric("Normal Tests", (~anomaly_df['is_anomalous']).sum())
    with col4:
        st.metric("Anomalies Found", anomaly_df['is_anomalous'].sum())
    
    st.markdown("---")
    
    # Drop statistics
    st.markdown("### 🎯 Drop Detection Statistics")
    
    detected = drop_df[drop_df['drop_detected']]
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        #### Drop Index (When?)
        - **Mean:** {:.0f}
        - **Median:** {:.0f}
        - **Std Dev:** {:.1f}
        - **Range:** {:.0f} - {:.0f}
        - **Normal: {:.0f} - {:.0f}**
        """.format(
            detected['drop_idx'].mean(),
            detected['drop_idx'].median(),
            detected['drop_idx'].std(),
            detected['drop_idx'].min(),
            detected['drop_idx'].max(),
            thresholds['drop_too_early'],
            thresholds['drop_too_late']
        ))
    
    with col2:
        # Histogram
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=detected['drop_idx'],
            nbinsx=40,
            marker=dict(color='rgba(100,150,255,0.7)'),
            name='Drop Index'
        ))
        
        fig_hist.add_vline(x=detected['drop_idx'].mean(), line_dash="dash", line_color="red",
                          annotation_text=f"Mean: {detected['drop_idx'].mean():.0f}")
        
        fig_hist.update_layout(
            title="Drop Index Distribution",
            xaxis_title="Drop Index",
            yaxis_title="Count",
            height=400,
            template='plotly_white'
        )
        
        st.plotly_chart(fig_hist, use_container_width=True)
    
    st.markdown("---")
    
    # Drop depth statistics
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        #### Drop Depth (How Deep?)
        - **Mean:** {:.1f}
        - **Median:** {:.1f}
        - **Std Dev:** {:.1f}
        - **Range:** {:.0f} - {:.0f}
        - **Normal: > {:.1f}**
        """.format(
            detected['drop_depth'].mean(),
            detected['drop_depth'].median(),
            detected['drop_depth'].std(),
            detected['drop_depth'].min(),
            detected['drop_depth'].max(),
            thresholds['drop_depth_too_shallow']
        ))
    
    with col2:
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=detected['drop_depth'],
            nbinsx=40,
            marker=dict(color='rgba(255,150,100,0.7)'),
            name='Drop Depth'
        ))
        
        fig_hist.add_vline(x=thresholds['drop_depth_too_shallow'], line_dash="dash", line_color="red",
                          annotation_text=f"Min: {thresholds['drop_depth_too_shallow']:.1f}")
        
        fig_hist.update_layout(
            title="Drop Depth Distribution",
            xaxis_title="Drop Depth",
            yaxis_title="Count",
            height=400,
            template='plotly_white'
        )
        
        st.plotly_chart(fig_hist, use_container_width=True)
    
    st.markdown("---")
    
    # Min value statistics
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        #### Min Value (How Low?)
        - **Mean:** {:.1f}
        - **Median:** {:.1f}
        - **Std Dev:** {:.1f}
        - **Range:** {:.1f} - {:.1f}
        - **Normal: < {:.1f}**
        """.format(
            detected['min_value'].mean(),
            detected['min_value'].median(),
            detected['min_value'].std(),
            detected['min_value'].min(),
            detected['min_value'].max(),
            thresholds['min_value_too_high']
        ))
    
    with col2:
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=detected['min_value'],
            nbinsx=20,
            marker=dict(color='rgba(150,100,255,0.7)'),
            name='Min Value'
        ))
        
        fig_hist.add_vline(x=thresholds['min_value_too_high'], line_dash="dash", line_color="red",
                          annotation_text=f"Max: {thresholds['min_value_too_high']:.1f}")
        
        fig_hist.update_layout(
            title="Min Value Distribution",
            xaxis_title="Min Value",
            yaxis_title="Count",
            height=400,
            template='plotly_white'
        )
        
        st.plotly_chart(fig_hist, use_container_width=True)

# ============================================================
# TAB 3: INDIVIDUAL TEST ANALYSIS
# ============================================================

with tab3:
    st.markdown("## 🔍 Individual Test Analysis")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        test_idx = st.selectbox(
            "Select Test Run:",
            range(len(df)),
            format_func=lambda x: f"Test {x}" + (" ⚠️ " if anomaly_df.iloc[x]['is_anomalous'] else "")
        )
    
    selected_trace = df.iloc[test_idx]['trace']
    selected_timestamp = df.iloc[test_idx].get('event_timestamp', 'N/A')
    drop_info = drop_df.iloc[test_idx]
    anomaly_info = anomaly_df.iloc[test_idx]
    
    with col2:
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Test ID", test_idx)
        with col_b:
            st.metric("Timestamp", str(selected_timestamp)[:19])
    
    # Anomaly warning
    if anomaly_info['is_anomalous']:
        st.warning(f"""
        ⚠️ **ANOMALIES DETECTED** ({anomaly_info['issue_count']} issue(s)):
        
        {chr(10).join(['  • ' + issue for issue in anomaly_info['issues']])}
        """)
    else:
        st.success("✅ **NORMAL TEST** - No anomalies detected")
    
    st.markdown("---")
    
    # Full trace
    st.markdown("### Full Aspiration Cycle")
    
    fig_full = go.Figure()
    
    fig_full.add_trace(go.Scatter(
        y=selected_trace,
        mode='lines',
        name='Signal',
        line=dict(color='darkblue', width=2),
        fill='tozeroy'
    ))
    
    # Phase markers
    fig_full.add_vline(x=150, line_dash="dot", line_color="green", annotation_text="Phase 1→2")
    fig_full.add_vline(x=1350, line_dash="dot", line_color="red", annotation_text="Drop Start")
    fig_full.add_vline(x=1600, line_dash="dot", line_color="orange", annotation_text="Phase 4 Start")
    
    # Drop marker
    if drop_info['drop_detected'] and drop_info['drop_idx'] is not None:
        fig_full.add_vline(
            x=drop_info['drop_idx'],
            line_dash="dash",
            line_color="purple",
            annotation_text=f"Drop at {int(drop_info['drop_idx'])}",
            annotation_position="top right"
        )
    
    fig_full.update_layout(
        title=f"Full Trace - Test {test_idx}",
        xaxis_title="Sample Index",
        yaxis_title="ADC Signal",
        height=500,
        template='plotly_white'
    )
    
    st.plotly_chart(fig_full, use_container_width=True)
    
    st.markdown("---")
    
    # Phase breakdown
    st.markdown("### Phase-by-Phase Analysis")
    
    fig_phases = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Phase 1: Startup", "Phase 2: High Plateau", "Phase 3: Drop", "Phase 4: Low Plateau"),
        specs=[[{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "scatter"}, {"type": "scatter"}]]
    )
    
    # Phase 1
    phase1 = selected_trace[0:150]
    fig_phases.add_trace(go.Scatter(y=phase1, mode='lines', name='P1', 
                                    line=dict(color='green'), fill='tozeroy'), row=1, col=1)
    
    # Phase 2
    phase2 = selected_trace[100:1350]
    fig_phases.add_trace(go.Scatter(y=phase2, mode='lines', name='P2',
                                    line=dict(color='blue'), fill='tozeroy'), row=1, col=2)
    
    # Phase 3
    phase3 = selected_trace[1350:1600]
    fig_phases.add_trace(go.Scatter(y=phase3, mode='lines', name='P3',
                                    line=dict(color='red'), fill='tozeroy'), row=2, col=1)
    
    # Phase 4
    phase4 = selected_trace[1600:1800]
    fig_phases.add_trace(go.Scatter(y=phase4, mode='lines', name='P4',
                                    line=dict(color='orange'), fill='tozeroy'), row=2, col=2)
    
    fig_phases.update_layout(height=600, showlegend=False, template='plotly_white')
    fig_phases.update_yaxes(title_text="Signal", row=1, col=1)
    fig_phases.update_yaxes(title_text="Signal", row=1, col=2)
    fig_phases.update_yaxes(title_text="Signal", row=2, col=1)
    fig_phases.update_yaxes(title_text="Signal", row=2, col=2)
    
    st.plotly_chart(fig_phases, use_container_width=True)
    
    st.markdown("---")
    
    # Drop metrics
    st.markdown("### Drop Detection Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Drop Detected", "✅ Yes" if drop_info['drop_detected'] else "❌ No")
    with col2:
        st.metric("Drop Index", f"{int(drop_info['drop_idx'])}" if drop_info['drop_idx'] is not None else "N/A")
    with col3:
        st.metric("Drop Depth", f"{drop_info['drop_depth']:.1f}" if drop_info['drop_depth'] is not None else "N/A")
    with col4:
        st.metric("Min Value", f"{drop_info['min_value']:.1f}" if drop_info['min_value'] is not None else "N/A")

# ============================================================
# TAB 4: DROP DETECTION DETAILS
# ============================================================

with tab4:
    st.markdown("## 📉 Drop Detection Analysis")
    
    detected = drop_df[drop_df['drop_detected']]
    
    st.markdown("""
    ### How Drop Detection Works
    
    1. **Hampel Filter:** Remove outliers from signal
    2. **Smooth:** Apply moving median
    3. **Baseline:** Calculate high plateau level (75th percentile)
    4. **Threshold:** Drop = when signal falls below 40% of baseline
    5. **Segment:** Find first sustained drop (≥5 samples below threshold)
    """)
    
    st.markdown("---")
    
    # 4-panel analysis
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Overlay all traces
        st.markdown("### All Traces with Drop Markers")
        
        fig_overlay = go.Figure()
        
        for idx in range(min(100, len(df))):
            trace = df.iloc[idx]['trace']
            drop_idx = drop_df.iloc[idx]['drop_idx']
            
            fig_overlay.add_trace(go.Scatter(
                y=trace,
                mode='lines',
                line=dict(color='rgba(0,0,255,0.2)', width=1),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            if drop_idx is not None:
                fig_overlay.add_vline(x=drop_idx, line_color='red', line_width=0.5)
        
        fig_overlay.update_layout(
            title="Overlay (100 traces) with Drop Markers",
            xaxis_title="Sample Index",
            yaxis_title="Signal",
            height=500,
            template='plotly_white'
        )
        
        st.plotly_chart(fig_overlay, use_container_width=True)
    
    with col2:
        # Drop index scatter
        st.markdown("### Drop Index vs Test #")
        
        fig_scatter = go.Figure()
        
        fig_scatter.add_trace(go.Scatter(
            x=drop_df['trace_idx'],
            y=drop_df['drop_idx'],
            mode='markers',
            marker=dict(size=5, color='blue'),
            name='Drop Index',
            hovertemplate="Test %{x}: Drop at %{y:.0f}<extra></extra>"
        ))
        
        fig_scatter.add_hline(y=thresholds['drop_too_early'], line_dash="dash", 
                             line_color="red", annotation_text=f"Min: {thresholds['drop_too_early']:.0f}")
        fig_scatter.add_hline(y=thresholds['drop_too_late'], line_dash="dash",
                             line_color="red", annotation_text=f"Max: {thresholds['drop_too_late']:.0f}")
        
        fig_scatter.update_layout(
            title="Drop Index Distribution",
            xaxis_title="Test #",
            yaxis_title="Drop Index",
            height=500,
            template='plotly_white'
        )
        
        st.plotly_chart(fig_scatter, use_container_width=True)

# ============================================================
# TAB 5: ANOMALIES
# ============================================================

with tab5:
    st.markdown("## ⚠️  Anomaly Detection Results")
    
    anomalous_df = anomaly_df[anomaly_df['is_anomalous']]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Tests", len(df))
    with col2:
        st.metric("Anomalies", len(anomalous_df))
    with col3:
        st.metric("Anomaly Rate", f"{len(anomalous_df)/len(df)*100:.1f}%")
    
    st.markdown("---")
    
    if len(anomalous_df) > 0:
        st.markdown("### Anomalous Test Runs")
        
        # Create summary table
        anom_summary = []
        for idx, row in anomalous_df.iterrows():
            anom_summary.append({
                'Test #': int(row['trace_idx']),
                'Issues': ' | '.join(row['issues']) if row['issues'] else 'Unknown',
                'Count': int(row['issue_count'])
            })
        
        anom_table_df = pd.DataFrame(anom_summary).sort_values('Count', ascending=False)
        
        st.dataframe(anom_table_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Comparison plots
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### Normal Traces (Sample)")
            
            normal_df = anomaly_df[~anomaly_df['is_anomalous']]
            fig_normal = go.Figure()
            
            for idx in normal_df['trace_idx'].head(50):
                trace = df.iloc[idx]['trace']
                fig_normal.add_trace(go.Scatter(
                    y=trace,
                    mode='lines',
                    line=dict(color='rgba(0,200,0,0.3)', width=1),
                    showlegend=False,
                    hoverinfo='skip'
                ))
            
            fig_normal.update_layout(
                title=f"Normal Traces (n=50)",
                xaxis_title="Index",
                yaxis_title="Signal",
                height=500,
                template='plotly_white'
            )
            st.plotly_chart(fig_normal, use_container_width=True)
        
        with col2:
            st.markdown("### Anomalous Traces (Sample)")
            
            fig_anom = go.Figure()
            
            for idx in anomalous_df['trace_idx'].head(50):
                trace = df.iloc[idx]['trace']
                fig_anom.add_trace(go.Scatter(
                    y=trace,
                    mode='lines',
                    line=dict(color='rgba(255,0,0,0.5)', width=1.5, dash='dash'),
                    showlegend=False,
                    hoverinfo='skip'
                ))
            
            fig_anom.update_layout(
                title=f"Anomalous Traces (n={min(50, len(anomalous_df))})",
                xaxis_title="Index",
                yaxis_title="Signal",
                height=500,
                template='plotly_white'
            )
            st.plotly_chart(fig_anom, use_container_width=True)
    
    else:
        st.success("✅ No anomalies detected! All tests appear normal.")

# ============================================================
# TAB 6: DETAILED ZOOMS
# ============================================================

with tab6:
    st.markdown("## 🔬 Detailed Zoom Analysis - All Tubes")
    
    st.markdown("""
    ### Zoomed Analysis of Critical Phases
    
    This tab shows detailed zooms of all 240 test runs for three critical time windows:
    1. **Startup Phase** - How the pump starts
    2. **Drop Start** - When the signal begins to fall
    3. **Drop End** - Final emptying phase
    """)
    
    st.markdown("---")
    
    # ZOOM 1: STARTUP
    st.markdown("### 🟢 ZOOM 1: Startup Phase (Index 0-150)")
    st.markdown("How the aspiration pump starts - should see linear ramp from 0 to ~150")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig_startup = go.Figure()
        
        for idx in range(len(df)):
            trace = df.iloc[idx]['trace']
            startup = trace[0:150]
            
            fig_startup.add_trace(go.Scatter(
                x=np.arange(len(startup)),
                y=startup,
                mode='lines',
                line=dict(width=1.5),
                opacity=0.4,
                showlegend=False,
                hovertemplate="Index: %{x}<br>Signal: %{y:.0f}<extra></extra>"
            ))
        
        fig_startup.update_layout(
            title="Startup Phase - All 240 Test Runs",
            xaxis_title="Sample Index (0-150)",
            yaxis_title="ADC Signal",
            height=600,
            hovermode='x unified',
            template='plotly_white'
        )
        
        st.plotly_chart(fig_startup, use_container_width=True)
    
    with col1:
        # Statistics
        startup_stats = []
        for trace in df['trace']:
            startup_phase = trace[0:150]
            startup_stats.append({
                'mean': np.mean(startup_phase),
                'max': np.max(startup_phase),
                'slope': (np.mean(startup_phase[-50:]) - np.mean(startup_phase[:50])) / 100
            })
        
        startup_df = pd.DataFrame(startup_stats)
        
        st.markdown("""
        #### Statistics:
        - **Mean Final Signal:** {:.1f}
        - **Mean Max Signal:** {:.1f}
        - **Mean Slope:** {:.3f}
        - **Slope Std Dev:** {:.3f}
        """.format(
            startup_df['mean'].mean(),
            startup_df['max'].mean(),
            startup_df['slope'].mean(),
            startup_df['slope'].std()
        ))
        
        # Slope histogram
        fig_slope = go.Figure()
        fig_slope.add_trace(go.Histogram(
            x=startup_df['slope'],
            nbinsx=30,
            marker=dict(color='rgba(0,200,100,0.7)'),
            name='Startup Slope'
        ))
        
        fig_slope.update_layout(
            title="Startup Slope Distribution",
            xaxis_title="Slope (Signal/Sample)",
            yaxis_title="Count",
            height=400,
            template='plotly_white'
        )
        
        st.plotly_chart(fig_slope, use_container_width=True)
    
    st.markdown("---")
    
    # ZOOM 2: DROP START
    st.markdown("### 🔴 ZOOM 2: Drop Start (Index 1300-1400)")
    st.markdown("Beginning of the drop phase - when signal starts falling from high plateau")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig_drop_start = go.Figure()
        
        for idx in range(len(df)):
            trace = df.iloc[idx]['trace']
            drop_start_phase = trace[1300:1400]
            
            fig_drop_start.add_trace(go.Scatter(
                x=np.arange(1300, 1300 + len(drop_start_phase)),
                y=drop_start_phase,
                mode='lines',
                line=dict(width=1.5),
                opacity=0.4,
                showlegend=False,
                hovertemplate="Index: %{x}<br>Signal: %{y:.0f}<extra></extra>"
            ))
        
        # Add drop detection markers
        for idx in range(min(240, len(drop_df))):
            drop_idx = drop_df.iloc[idx]['drop_idx']
            if drop_idx is not None and 1300 <= drop_idx <= 1400:
                fig_drop_start.add_vline(x=drop_idx, line_color='purple', line_width=0.5)
        
        fig_drop_start.update_layout(
            title="Drop Start Phase - All 240 Test Runs (with detected drop markers)",
            xaxis_title="Sample Index (1300-1400)",
            yaxis_title="ADC Signal",
            height=600,
            hovermode='x unified',
            template='plotly_white'
        )
        
        st.plotly_chart(fig_drop_start, use_container_width=True)
    
    with col2:
        # Statistics
        drop_start_stats = []
        for idx in range(len(df)):
            trace = df.iloc[idx]['trace']
            drop_start_phase = trace[1300:1400]
            drop_start_stats.append({
                'mean': np.mean(drop_start_phase),
                'max': np.max(drop_start_phase),
                'min': np.min(drop_start_phase),
                'slope': (np.min(drop_start_phase) - np.max(drop_start_phase)) / 100
            })
        
        drop_start_df = pd.DataFrame(drop_start_stats)
        
        st.markdown("""
        #### Statistics:
        - **Mean Signal:** {:.1f}
        - **Mean Max:** {:.1f}
        - **Mean Min:** {:.1f}
        - **Mean Slope:** {:.3f}
        """.format(
            drop_start_df['mean'].mean(),
            drop_start_df['max'].mean(),
            drop_start_df['min'].mean(),
            drop_start_df['slope'].mean()
        ))
        
        # Slope histogram
        fig_drop_slope = go.Figure()
        fig_drop_slope.add_trace(go.Histogram(
            x=drop_start_df['slope'],
            nbinsx=30,
            marker=dict(color='rgba(255,100,100,0.7)'),
            name='Drop Slope'
        ))
        
        fig_drop_slope.update_layout(
            title="Drop Start Slope Distribution",
            xaxis_title="Slope (Signal/Sample)",
            yaxis_title="Count",
            height=400,
            template='plotly_white'
        )
        
        st.plotly_chart(fig_drop_slope, use_container_width=True)
    
    st.markdown("---")
    
    # ZOOM 3: DROP END
    st.markdown("### 🟠 ZOOM 3: Drop End (Index 1400-1600)")
    st.markdown("End of drop phase - signal stabilizes at low level (tube emptying complete)")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig_drop_end = go.Figure()
        
        for idx in range(len(df)):
            trace = df.iloc[idx]['trace']
            drop_end_phase = trace[1400:1600]
            
            fig_drop_end.add_trace(go.Scatter(
                x=np.arange(1400, 1400 + len(drop_end_phase)),
                y=drop_end_phase,
                mode='lines',
                line=dict(width=1.5),
                opacity=0.4,
                showlegend=False,
                hovertemplate="Index: %{x}<br>Signal: %{y:.0f}<extra></extra>"
            ))
        
        fig_drop_end.update_layout(
            title="Drop End Phase - All 240 Test Runs",
            xaxis_title="Sample Index (1400-1600)",
            yaxis_title="ADC Signal",
            height=600,
            hovermode='x unified',
            template='plotly_white',
            yaxis_range=[0, 50]  # Zoom in on low values
        )
        
        st.plotly_chart(fig_drop_end, use_container_width=True)
    
    with col2:
        # Statistics
        drop_end_stats = []
        for trace in df['trace']:
            drop_end_phase = trace[1400:1600]
            drop_end_stats.append({
                'mean': np.mean(drop_end_phase),
                'min': np.min(drop_end_phase),
                'max': np.max(drop_end_phase),
                'std': np.std(drop_end_phase)
            })
        
        drop_end_df = pd.DataFrame(drop_end_stats)
        
        st.markdown("""
        #### Statistics:
        - **Mean Signal:** {:.1f}
        - **Mean Min:** {:.1f}
        - **Mean Max:** {:.1f}
        - **Mean Std:** {:.2f}
        """.format(
            drop_end_df['mean'].mean(),
            drop_end_df['min'].mean(),
            drop_end_df['max'].mean(),
            drop_end_df['std'].mean()
        ))
        
        # Box plot
        fig_box = go.Figure()
        fig_box.add_trace(go.Box(
            y=drop_end_df['mean'],
            name='Drop End Signal',
            marker=dict(color='rgba(255,150,100,0.7)')
        ))
        
        fig_box.update_layout(
            title="Drop End Signal Distribution",
            yaxis_title="ADC Signal",
            height=400,
            template='plotly_white'
        )
        
        st.plotly_chart(fig_box, use_container_width=True)
    
    st.markdown("---")
    
    # COMPARISON TABLE
    st.markdown("### 📊 Phase Comparison Summary")
    
    comparison_data = {
        'Phase': ['Startup', 'Drop Start', 'Drop End'],
        'Index Range': ['0-150', '1300-1400', '1400-1600'],
        'Mean Signal': [
            f"{startup_df['mean'].mean():.1f}",
            f"{drop_start_df['mean'].mean():.1f}",
            f"{drop_end_df['mean'].mean():.1f}"
        ],
        'Description': [
            'Pump starting - linear ramp up',
            'Signal falling rapidly',
            'Stable low plateau - tube empty'
        ]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
