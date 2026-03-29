import streamlit as st
import plotly.graph_objects as pl
import numpy as np

# --- Set page config ---
st.set_page_config(page_title="3D Point Cloud Viewer", layout="wide")
st.title("3D Point Cloud Visualization (LiDAR-like)")

# --- Sidebar Controls ---
st.sidebar.header("Controls")

# Slider to control number of points
num_points = st.sidebar.slider("Number of Generated Points", min_value=1000, max_value=20000, value=5000, step=1000)

# Optional coloring by semantic classes
num_classes = st.sidebar.selectbox("Number of Semantic Classes", [2, 3, 4, 5], index=2)

# Bonus: Toggle downsampling
enable_downsampling = st.sidebar.checkbox("Enable Downsampling (Random Sampling)", value=False)
downsample_percent = 100
if enable_downsampling:
    downsample_percent = st.sidebar.slider("Downsample Percentage", min_value=10, max_value=100, value=50, step=5)

# --- Data Generation ---
@st.cache_data
def generate_point_cloud(n_points, n_classes):
    """Generates a random point cloud and assigns random semantic classes."""
    # Generate random (x, y, z) coordinates mimicking a roughly centered point cloud
    # Using a combination of normal distribution to make it look a bit more clustered
    points = np.random.randn(n_points, 3) * 10
    
    # Assign random classes to simulate semantic segmentation labels
    classes = np.random.randint(0, n_classes, size=n_points)
    
    return points, classes

points, classes = generate_point_cloud(num_points, num_classes)

# --- Downsampling Logic ---
if enable_downsampling and downsample_percent < 100:
    num_to_sample = int(num_points * (downsample_percent / 100.0))
    # Randomly select indices without replacement
    indices = np.random.choice(num_points, size=num_to_sample, replace=False)
    display_points = points[indices]
    display_classes = classes[indices]
else:
    display_points = points
    display_classes = classes

# --- Performance Warning ---
current_points = len(display_points)
if current_points > 15000:
    st.warning(f"⚠️ Rendering high number of points ({current_points}). You may experience lag when interacting with the plot.")
else:
    st.success(f"Rendering {current_points} points.")

# --- Plotly 3D Render ---
# Create the interactive 3D scatter plot
fig = pl.Figure(
    data=[
        pl.Scatter3d(
            x=display_points[:, 0],
            y=display_points[:, 1],
            z=display_points[:, 2],
            mode='markers',
            marker=dict(
                size=2.5,                     # Keep points small enough to view clearly
                color=display_classes,        # Color array
                colorscale='Turbo',           # Distinct colorscale for classes
                opacity=0.8,                  # Slight transparency 
                colorbar=dict(title="Semantic Class")
            )
        )
    ]
)

# Update layout for better UX
fig.update_layout(
    scene=dict(
        xaxis_title='X Axis',
        yaxis_title='Y Axis',
        zaxis_title='Z Axis',
        aspectmode='data', # Maintains proportional axis scaling based on data
        bgcolor='black'    # Dark background often looks better for point clouds
    ),
    margin=dict(l=0, r=0, b=0, t=0), # Reduce whitespace around plot
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)

# Render the plot in Streamlit taking the full container width
st.plotly_chart(fig, use_container_width=True)
