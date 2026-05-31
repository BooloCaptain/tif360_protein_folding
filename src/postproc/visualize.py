import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os

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


def plot_protein_comparison(true_coords, pred_coords, node_vars, edge_vars, title, filename):
    import plotly.graph_objects as go
    
    fig = go.Figure()

    # Ground Truth
    fig.add_trace(go.Scatter3d(
        x=true_coords[:, 0], y=true_coords[:, 1], z=true_coords[:, 2],
        mode='lines+markers', name='Ground Truth',
        line=dict(color='blue', width=5), marker=dict(size=4, color='blue')
    ))

    # Prediction: Nodes (Angles)
    fig.add_trace(go.Scatter3d(
        x=pred_coords[:, 0], y=pred_coords[:, 1], z=pred_coords[:, 2],
        mode='markers', name='Nodes (Angles)',
        marker=dict(
            size=4,
            color=node_vars,      # Variance of theta + tau
            colorscale='Viridis', # Green (confident) to Yellow/Purple (uncertain)
            showscale=True, colorbar=dict(title="Angle Uncertainty")
        )
    ))

    # Prediction: Edges (Distances)
    # To plot edges with varying colors, we need to use a trick: 
    # insert None between points, or plot segments individually.
    # Here is a simplified version using a single color scale for simplicity.
    fig.add_trace(go.Scatter3d(
        x=pred_coords[:, 0], y=pred_coords[:, 1], z=pred_coords[:, 2],
        mode='lines', name='Edges (Distances)',
        line=dict(width=5, color=edge_vars, colorscale='Viridis')
    ))

    fig.update_layout(title=title, scene=dict(aspectmode='data'))
    fig.write_html(filename)


def plot_geometry_error_per_residue(pred_1d, target_angles, target_distances, target_ss, mask_1d, save_path="geom_error_per_residue.png"):
    """
    Plots absolute predictions vs targets stacked directly above their Physical Error vs Uncertainty.
    Includes Secondary Structure background shading and vertical delta lines for exact error visualization.
    """
    # 1. Ensure numpy arrays
    if torch.is_tensor(pred_1d):
        pred_1d = pred_1d.detach().cpu().numpy()
        target_angles = target_angles.detach().cpu().numpy()
        target_distances = target_distances.detach().cpu().numpy()
        target_ss = target_ss.detach().cpu().numpy()
        mask_1d = mask_1d.detach().cpu().numpy().astype(bool)

    if pred_1d.ndim == 3:
        pred_1d = pred_1d[0]
        target_angles = target_angles[0]
        target_distances = target_distances[0]
        target_ss = target_ss[0]
        mask_1d = mask_1d[0]

    seq_len = pred_1d.shape[0]
    x = np.arange(seq_len)

    # 2. Extract Predictions & Std Dev
    mu_theta = pred_1d[:, 0:2]
    std_theta = np.exp(pred_1d[:, 2] / 2.0)
    mu_tau = pred_1d[:, 3:5]
    std_tau = np.exp(pred_1d[:, 5] / 2.0)
    mu_d = pred_1d[:, 6]
    std_d = np.exp(pred_1d[:, 7] / 2.0)

    # 3. Extract Targets & Convert to Degrees
    t_theta = target_angles[:, 0:2]
    t_tau = target_angles[:, 2:4]
    t_d = target_distances

    pred_theta_deg = np.degrees(np.arctan2(mu_theta[:, 0], mu_theta[:, 1]))
    true_theta_deg = np.degrees(np.arctan2(t_theta[:, 0], t_theta[:, 1]))
    pred_tau_deg = np.degrees(np.arctan2(mu_tau[:, 0], mu_tau[:, 1]))
    true_tau_deg = np.degrees(np.arctan2(t_tau[:, 0], t_tau[:, 1]))

    # 4. Calculate Errors
    err_theta = np.sum((mu_theta - t_theta)**2, axis=-1)
    err_tau = np.sum((mu_tau - t_tau)**2, axis=-1)
    err_d = np.abs(mu_d - t_d)

    # 5. Apply Kinematic Mask (Replaces invalid tail data with NaN so it doesn't plot)
    for arr in [pred_theta_deg, true_theta_deg, pred_tau_deg, true_tau_deg, mu_d, t_d, 
                err_theta, err_tau, err_d, std_theta, std_tau, std_d]:
        arr[~mask_1d] = np.nan

    # 6. Render 6x1 Stacked Plot (Taller figsize to accommodate 6 rows)
    fig, axs = plt.subplots(6, 1, figsize=(16, 22), sharex=True)

    # --- Helper: Secondary Structure Background Shading ---
    def add_ss_background(ax):
        # 0: Alpha Helix (Pink), 1: Beta Sheet (Yellow/Orange)
        ss_colors = {0: ('#ff9999', 0.25), 1: ('#ffe599', 0.35)}
        
        start_idx = 0
        current_ss = target_ss[0]
        
        for i in range(1, len(target_ss)):
            if target_ss[i] != current_ss:
                if current_ss in ss_colors:
                    c, a = ss_colors[current_ss]
                    ax.axvspan(start_idx - 0.5, i - 0.5, color=c, alpha=a, lw=0, zorder=0)
                start_idx = i
                current_ss = target_ss[i]
                
        # Handle the final trailing block
        if current_ss in ss_colors:
            c, a = ss_colors[current_ss]
            ax.axvspan(start_idx - 0.5, len(target_ss) - 0.5, color=c, alpha=a, lw=0, zorder=0)

    # --- Helper: Values Plot with Vertical Delta Lines ---
    def plot_values_with_deltas(ax, x, pred, true, title, ylabel, is_torsion=False):
        add_ss_background(ax)
        
        # Valid data mask to safely draw vertical lines without NaN warnings
        valid = ~np.isnan(true) & ~np.isnan(pred)
        
        # Draw the target and predicted points as distinct scatter dots
        ax.scatter(x, true, color='black', s=25, label='True Target', alpha=0.7, zorder=3)
        ax.scatter(x, pred, color='darkorange', s=25, label='Predicted', alpha=0.9, zorder=4)
        
        # Draw the vertical lines connecting Pred to True
        if is_torsion:
            # For torsion, if the error spans across the -180/180 boundary, 
            # drawing a straight line looks misleading. We only draw delta lines for non-wrapping errors.
            wrap_mask = np.abs(true - pred) <= 180
            plot_mask = valid & wrap_mask
            ax.vlines(x[plot_mask], np.minimum(true[plot_mask], pred[plot_mask]), 
                      np.maximum(true[plot_mask], pred[plot_mask]), color='gray', alpha=0.5, linewidth=1.5, zorder=2)
            
            ax.set_ylim(-190, 190)
            ax.set_yticks([-180, -90, 0, 90, 180])
        else:
            # Safe to draw delta lines for all valid points
            ax.vlines(x[valid], np.minimum(true[valid], pred[valid]), 
                      np.maximum(true[valid], pred[valid]), color='gray', alpha=0.5, linewidth=1.5, zorder=2)

        ax.set_title(title, fontweight='bold', fontsize=12)
        ax.set_ylabel(ylabel, fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.6)
        # Put legend outside the plot so it doesn't cover data
        ax.legend(loc='upper right', bbox_to_anchor=(1.12, 1.0))

    # --- Helper: Errors Plot ---
    def plot_errors(ax, x, err, std, title, ylabel, is_dist=False):
        add_ss_background(ax)
        
        ax.plot(x, err, color='crimson', linewidth=2, label='True Error')
        ax.plot(x, std, color='royalblue', linestyle='--', linewidth=1.5, label='Predicted Uncertainty (Std)')
        ax.fill_between(x, 0, std, color='royalblue', alpha=0.15)
        
        ax.set_title(title, fontweight='bold', fontsize=12)
        ax.set_ylabel(ylabel, fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.6)
        
        if is_dist:
            ax.axhline(0.1, color='green', linestyle=':', alpha=0.6, linewidth=2, label='0.1 Å Ideal Target')
            ax.set_ylim(bottom=0)
        else:
            ax.set_ylim(0, 4.2)
            
        ax.legend(loc='upper right', bbox_to_anchor=(1.16, 1.0))

    # ==========================
    # Render the 6 Rows
    # ==========================
    
    # Block 1: Theta (Planar Angles)
    plot_values_with_deltas(axs[0], x, pred_theta_deg, true_theta_deg, "1. Theta (θ) Values", "Degrees")
    plot_errors(axs[1], x, err_theta, std_theta, "2. Theta (θ) Error vs Uncertainty", "MSE")

    # Block 2: Tau (Torsion Angles)
    plot_values_with_deltas(axs[2], x, pred_tau_deg, true_tau_deg, "3. Tau (τ) Values (Torsion)", "Degrees", is_torsion=True)
    plot_errors(axs[3], x, err_tau, std_tau, "4. Tau (τ) Error vs Uncertainty", "MSE")

    # Block 3: Distance (Bond Lengths)
    plot_values_with_deltas(axs[4], x, mu_d, t_d, "5. Cα-Cα Distance Values", "Ångströms")
    plot_errors(axs[5], x, err_d, std_d, "6. Cα-Cα Distance Error vs Uncertainty", "Absolute Error (Å)", is_dist=True)

    # --- Create Custom Legend for the Secondary Structure ---
    legend_elements = [
        Patch(facecolor='#ff9999', alpha=0.4, label='Alpha Helix Background'),
        Patch(facecolor='#ffe599', alpha=0.5, label='Beta Sheet Background'),
        Patch(facecolor='white', edgecolor='gray', label='Coil / Loop Background')
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=3, fontsize=13, frameon=False, bbox_to_anchor=(0.5, 0.99))

    axs[5].set_xlabel("Sequence Residue Index", fontweight='bold', fontsize=14)
    
    # Adjust spacing to avoid overlap between titles and x-axis labels
    plt.tight_layout(rect=[0, 0, 0.88, 0.96]) # Leave room on the right for legends, top for SS legend
    plt.subplots_adjust(hspace=0.3) # Add a bit of vertical breathing room between the 6 rows
    
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()