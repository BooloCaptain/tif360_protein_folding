import numpy as np

try:
    import plotly.graph_objects as go
except ImportError:
    go = None
    print("[WARNING] Plotly is not installed. Run `pip install plotly` for interactive 3D plots.")


def kabsch_align(reference, candidate):
    """
    Rigid-body alignment of candidate coordinates to reference coordinates.
    Both must be arrays of shape (N, 3).
    """
    reference = np.asarray(reference, dtype=np.float32)
    candidate = np.asarray(candidate, dtype=np.float32)

    # Center both point clouds
    ref_center = reference.mean(axis=0)
    cand_center = candidate.mean(axis=0)
    ref_centered = reference - ref_center
    cand_centered = candidate - cand_center

    # Calculate covariance matrix
    covariance = cand_centered.T @ ref_centered
    
    # Singular Value Decomposition
    u, _, vh = np.linalg.svd(covariance)
    
    # THE FIX: Calculate rotation for row vectors (N, 3) 
    # In numpy, vh is already transposed (V^T).
    rotation = u @ vh

    # Correct for reflection (improper rotation)
    if np.linalg.det(rotation) < 0:
        vh[-1, :] *= -1
        rotation = u @ vh

    # Rotate candidate and translate to reference center
    aligned = cand_centered @ rotation + ref_center
    return aligned


def plot_protein_comparison(true_coords, pred_coords, title="True vs Predicted C-alpha Trace", filename="plot.html"):
    """
    Renders an interactive 3D plot and saves it to an HTML file.
    Expects pred_coords to ALREADY be aligned to true_coords.
    """
    if go is None:
        return

    fig = go.Figure()

    # Ground Truth (Blue)
    fig.add_trace(go.Scatter3d(
        x=true_coords[:, 0], y=true_coords[:, 1], z=true_coords[:, 2],
        mode='lines+markers',
        name='Ground Truth',
        line=dict(color='blue', width=4),
        marker=dict(size=3, color='blue')
    ))

    # Prediction (Red)
    fig.add_trace(go.Scatter3d(
        x=pred_coords[:, 0], y=pred_coords[:, 1], z=pred_coords[:, 2],
        mode='lines+markers',
        name='Predicted',
        line=dict(color='red', width=4),
        marker=dict(size=3, color='red')
    ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X (Å)', yaxis_title='Y (Å)', zaxis_title='Z (Å)',
            aspectmode='data' 
        ),
        legend=dict(x=0, y=1)
    )
    
    # Save to file instead of opening a browser
    fig.write_html(filename)
    print(f"Saved interactive plot to {filename}")