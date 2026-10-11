import streamlit as st
import pandas as pd
import ezdxf
import io
import plotly.graph_objects as go
from scipy.spatial import distance_matrix
import numpy as np


st.set_page_config(page_title="Auto Closest Point Connect", layout="wide")

st.title("🚀 Fully Automated Point Connect & DXF Exporter")
st.write("Upload your files. This version bypasses header names entirely and maps data by position (Col 1 = X, Col 2 = Y, Col 3 = Z).")

# Sidebar for file uploads only
st.sidebar.header("1. Upload Data")
original_file = st.sidebar.file_uploader("Upload Original Points (CSV)", type=["csv"], key="orig")
scaled_file = st.sidebar.file_uploader("Upload Scaled Points (CSV)", type=["csv"], key="scale")

def load_csv_safely(uploaded_file):
    if uploaded_file is None:
        return None
    # header=None treats numbers as data rows, not header names
    df = pd.read_csv(uploaded_file, header=None, comment='#')
    return df

if original_file and scaled_file:
    df_orig = load_csv_safely(original_file)
    df_scale = load_csv_safely(scaled_file)
    
    # Check if files have columns to extract
    if df_orig.shape[1] < 2:
        st.error(f"Original CSV must have at least 2 columns. Found only {df_orig.shape[1]}.")
    elif df_scale.shape[1] < 2:
        st.error(f"Scaled CSV must have at least 2 columns. Found only {df_scale.shape[1]}.")
    else:
        st.sidebar.success("📊 File Structure Loaded Successfully!")
        st.sidebar.write(f"**Original Points:** {len(df_orig)}")
        st.sidebar.write(f"**Scaled Points:** {len(df_scale)}")

        # --- EXTRACT COORDINATES BY POSITION INDEX ---
        x_orig_data = df_orig.iloc[:, 0].to_numpy().astype(float)
        y_orig_data = df_orig.iloc[:, 1].to_numpy().astype(float)
        
        if df_orig.shape[1] >= 3:
            z_orig_data = df_orig.iloc[:, 2].to_numpy().astype(float)
            st.sidebar.write("📐 Processing mode: **3D**")
        else:
            z_orig_data = np.zeros(len(df_orig))
            st.sidebar.write("📐 Processing mode: **2D**")
            
        pts_orig = np.column_stack((x_orig_data, y_orig_data, z_orig_data))
        
        x_scale_data = df_scale.iloc[:, 0].to_numpy().astype(float)
        y_scale_data = df_scale.iloc[:, 1].to_numpy().astype(float)
        z_scale_data = df_scale.iloc[:, 2].to_numpy().astype(float) if df_scale.shape[1] >= 3 else np.zeros(len(df_scale))
        pts_scale = np.column_stack((x_scale_data, y_scale_data, z_scale_data))
        
        # --- NEAREST NEIGHBOR CALCULATION ---
        dists = distance_matrix(pts_orig, pts_scale)
        closest_indices = np.argmin(dists, axis=1)
        
        # --- 3D VISUALIZATION SECTION ---
        st.subheader("📊 Visualizing 3D Spike Lines")
        fig = go.Figure()
        
        fig.add_trace(go.Scatter3d(
            x=pts_orig[:, 0], y=pts_orig[:, 1], z=pts_orig[:, 2],
            mode='markers', name='Original Points', marker=dict(color='blue', size=4)
        ))
        fig.add_trace(go.Scatter3d(
            x=pts_scale[:, 0], y=pts_scale[:, 1], z=pts_scale[:, 2],
            mode='markers', name='Scaled Points', marker=dict(color='red', size=4)
        ))
        
        line_x, line_y, line_z = [], [], []
        for i in range(len(pts_orig)):
            nearest_idx = closest_indices[i]
            line_x.extend([pts_orig[i, 0], pts_scale[nearest_idx, 0], None])
            line_y.extend([pts_orig[i, 1], pts_scale[nearest_idx, 1], None])
            line_z.extend([pts_orig[i, 2], pts_scale[nearest_idx, 2], None])
            
        fig.add_trace(go.Scatter3d(
            x=line_x, y=line_y, z=line_z,
            mode='lines', name='Spikes (Closest)', line=dict(color='orange', width=1.5)
        ))
            
        fig.update_layout(
            scene=dict(xaxis_title="Col 0 (X)", yaxis_title="Col 1 (Y)", zaxis_title="Col 2 (Z)"),
            height=700
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # --- DXF GENERATION SECTION ---
        st.subheader("💾 Export Spike Network to DXF")
        doc = ezdxf.new("R2000") 
        msp = doc.modelspace()
        
        for i in range(len(pts_orig)):
            nearest_idx = closest_indices[i]
            p1 = (pts_orig[i, 0], pts_orig[i, 1], pts_orig[i, 2])
            p2 = (pts_scale[nearest_idx, 0], pts_scale[nearest_idx, 1], pts_scale[nearest_idx, 2])
            msp.add_line(p1, p2)
            
        dxf_stream = io.StringIO()
        doc.write(dxf_stream)
        dxf_bytes = dxf_stream.getvalue().encode('utf-8')
        
        st.download_button(
            label="Download Spikes DXF File",
            data=dxf_bytes,
            file_name="closest_point_spikes.dxf",
            mime="application/dxf"
        )
else:
    st.info("💡 Drop both headerless CSV files into the sidebar window to instantly generate closest connection spikes.")
