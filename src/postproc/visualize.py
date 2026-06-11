import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

try:
    import plotly.graph_objects as go
except ImportError:
    print("[WARNING] Plotly not installed. 3D traces will fail. Please run: pip install plotly")


# ==========================================
# 1. MATHEMATICAL HELPERS
# ==========================================
def compute_ca_torsions(coords):
    """
    Computes the C-alpha pseudo-torsion angles (dihedrals) for a given trace.
    Returns an array of length L, with padded 0s at the termini.
    """
    L = len(coords)
    torsions = np.zeros(L)
    for i in range(1, L - 2):
        p0, p1, p2, p3 = coords[i-1], coords[i], coords[i+1], coords[i+2]
        
        b0 = -1.0 * (p1 - p0)
        b1 = p2 - p1
        b2 = p3 - p2
        
        b1_norm = np.linalg.norm(b1)
        if b1_norm < 1e-8:
            continue
        b1 /= b1_norm
        
        v = b0 - np.dot(b0, b1) * b1
        w = b2 - np.dot(b2, b1) * b1
        
        x = np.dot(v, w)
        y = np.dot(np.cross(b1, v), w)
        torsions[i] = np.degrees(np.arctan2(y, x))
    return torsions


def create_pairwise_diff_matrix(angles):
    """Creates an L x L matrix of absolute pairwise angle differences."""
    L = len(angles)
    angles_expanded_1 = np.broadcast_to(angles[:, None], (L, L))
    angles_expanded_2 = np.broadcast_to(angles[None, :], (L, L))
    diff = np.abs(angles_expanded_1 - angles_expanded_2)
    return np.minimum(diff, 360.0 - diff)


def kabsch_align(reference, candidate):
    """Rigid-body alignment of candidate coordinates to reference coordinates."""
    reference = np.asarray(reference, dtype=np.float32)
    candidate = np.asarray(candidate, dtype=np.float32)

    ref_center = reference.mean(axis=0)
    cand_center = candidate.mean(axis=0)
    ref_centered = reference - ref_center
    cand_centered = candidate - cand_center

    covariance = cand_centered.T @ ref_centered
    u, _, vh = np.linalg.svd(covariance)
    rotation = u @ vh

    if np.linalg.det(rotation) < 0:
        vh[-1, :] *= -1
        rotation = u @ vh

    aligned = cand_centered @ rotation + ref_center
    return aligned


def calculate_shannon_entropy(logits):
    """Calculates the Shannon Entropy of a probability distribution (e.g., 64 Distogram bins)."""
    # Convert logits to probabilities safely
    probs = F.softmax(torch.tensor(logits), dim=-1).numpy()
    # -Sum(P * log(P))
    entropy = -np.sum(probs * np.log(probs + 1e-8), axis=-1)
    return entropy


# ==========================================
# 2. THE NEW DIAGNOSTIC VISUALIZATIONS
# ==========================================

def plot_protein_comparison(true_coords, pred_coords, angle_error_deg, uncertainty, title="Angle Error vs Uncertainty", filename="plot_tube.html"):
    """
    3-Way Plotly visualization:
    1. Ground Truth (Gray)
    2. Prediction colored by Actual Angle Error in degrees (Blue = Perfect, Red = Wrong)
    3. Prediction colored by Network Uncertainty/log_var (Green = Confident, Purple = Uncertain)
    """
    if 'go' not in globals():
        return

    aligned_pred = kabsch_align(true_coords, pred_coords)
    
    offset_val = np.max(np.abs(true_coords[:, 0])) * 2.0 + 20.0
    error_shifted = aligned_pred + np.array([-offset_val, 0, 0])
    conf_shifted = aligned_pred + np.array([offset_val, 0, 0])

    fig = go.Figure()

    # 1. Prediction colored by ANGLE ERROR (Left)
    fig.add_trace(go.Scatter3d(
        x=error_shifted[:, 0], y=error_shifted[:, 1], z=error_shifted[:, 2],
        mode='lines+markers', name='Actual Angle Error (°)',
        marker=dict(
            size=6, color=angle_error_deg, colorscale='RdBu', reversescale=True, 
            cmin=0.0, cmax=180.0, showscale=True, colorbar=dict(x=0.05, title="Angle Error (°)")
        ),
        line=dict(color='darkgray', width=4)
    ))

    # 2. Ground Truth (Center)
    fig.add_trace(go.Scatter3d(
        x=true_coords[:, 0], y=true_coords[:, 1], z=true_coords[:, 2],
        mode='lines+markers', marker=dict(size=4, color='lightgray'),
        line=dict(color='gray', width=4), name='True Target'
    ))

    # 3. Prediction colored by UNCERTAINTY (Right)
    fig.add_trace(go.Scatter3d(
        x=conf_shifted[:, 0], y=conf_shifted[:, 1], z=conf_shifted[:, 2],
        mode='lines+markers', name='Network Uncertainty',
        marker=dict(
            size=6, color=uncertainty, colorscale='Viridis', reversescale=True,
            showscale=True, colorbar=dict(x=0.95, title="Uncertainty (log_var)")
        ),
        line=dict(color='darkgray', width=4)
    ))

    axis_config = dict(showgrid=False, showbackground=False, visible=False)
    fig.update_layout(
        title=title, scene=dict(xaxis=axis_config, yaxis=axis_config, zaxis=axis_config, aspectmode='data'),
        legend=dict(x=0.4, y=0.98), margin=dict(l=0, r=0, b=0, t=40), paper_bgcolor='white', plot_bgcolor='white'
    )
    fig.write_html(filename)


def plot_gaussian_ramachandran(pred_theta_rad, pred_tau_rad, true_theta_rad, true_tau_rad, log_var_tau, title="Gaussian Ramachandran Map", filename="ramachandran.png"):
    """
    Plots standard Ramachandran angles. Colors predictions by Gaussian Uncertainty.
    Draws connecting lines to show exactly where the 180-degree traps are happening.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Convert to degrees
    pred_x = np.degrees(pred_theta_rad)
    pred_y = np.degrees(pred_tau_rad)
    true_x = np.degrees(true_theta_rad)
    true_y = np.degrees(true_tau_rad)
    
    # Draw standard Ramachandran background quadrants
    ax.axhline(0, color='black', linewidth=1, alpha=0.3)
    ax.axvline(0, color='black', linewidth=1, alpha=0.3)
    
    # Connect True to Pred to show the displacement vector
    for px, py, tx, ty in zip(pred_x, pred_y, true_x, true_y):
        # Don't draw lines that wrap around the -180/180 boundary (messy visually)
        if abs(px - tx) < 180 and abs(py - ty) < 180:
            ax.plot([tx, px], [ty, py], color='gray', alpha=0.4, zorder=1)
    
    # True Targets
    ax.scatter(true_x, true_y, color='black', marker='x', s=30, label='True Target', alpha=0.7, zorder=2)
    
    # Predictions colored by Uncertainty (log_var)
    sc = ax.scatter(pred_x, pred_y, c=log_var_tau, cmap='plasma', s=60, edgecolors='black', zorder=3, label='Prediction')
    
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    
    # Set standard ticks
    ticks = [-180, -90, 0, 90, 180]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    
    ax.set_title(title, fontweight='bold', fontsize=14)
    ax.set_xlabel("Theta (θ) / Phi (°)", fontsize=12)
    ax.set_ylabel("Tau (τ) / Psi (°)", fontsize=12)
    
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Network Uncertainty (log_var_tau)", rotation=270, labelpad=15, fontweight='bold')
    
    ax.legend(loc='upper left')
    ax.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_distogram_analysis(disto_logits, true_coords, pred_coords, title="Distogram Constraint Analysis", filename="distogram_analysis.png"):
    """
    2x2 Diagnostic Grid.
    Row 1: Distogram vs Ground Truth (Checks Outer Product Accuracy)
    Row 2: Distogram vs Predicted 3D (Checks 1D/2D Constraint Agreement)
    """
    L = true_coords.shape[0]
    
    # 1. Distances
    true_dists = np.linalg.norm(true_coords[:, None, :] - true_coords[None, :, :], axis=-1)
    pred_3d_dists = np.linalg.norm(pred_coords[:, None, :] - pred_coords[None, :, :], axis=-1)

    # 1. Clip distances to a reasonable range for better color scaling (e.g., 0-20 Å)
    true_dists = np.clip(true_dists, 0, 22)
    pred_3d_dists = np.clip(pred_3d_dists, 0, 22)
    
    # 2. Expected Distogram Distances & Entropy
    probs = F.softmax(torch.tensor(disto_logits).float(), dim=-1).numpy()
    bin_indices = np.arange(64)
    expected_dists = (np.sum(probs * bin_indices, axis=-1) * (20.0 / 64.0)) + 2.0
    entropy_matrix = -np.sum(probs * np.log(probs + 1e-8), axis=-1)
    
    # 3. Signed Errors (Disto - Target)
    # Positive = Disto predicted a larger distance than the target (Swelling)
    # Negative = Disto predicted a shorter distance than the target (Collapsing)
    err_vs_true = expected_dists - true_dists
    err_vs_pred3d = expected_dists - pred_3d_dists
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 18))
    
    # Helper to plot diagonal split matrices cleanly
    def plot_split(ax, mat_upper, mat_lower, label_upper, label_lower, plot_title, cmap_up, cmap_dn, vmin_up, vmax_up, vmin_dn, vmax_dn):
        combined = np.zeros((L, L))
        upper_mask = np.triu_indices(L, k=1)
        lower_mask = np.tril_indices(L, k=-1)
        
        combined[upper_mask] = mat_upper[upper_mask]
        combined[lower_mask] = mat_lower[lower_mask]
        
        display_up = np.ma.masked_where(np.tril(np.ones((L,L))), combined)
        display_dn = np.ma.masked_where(np.triu(np.ones((L,L))), combined)
        
        im_up = ax.imshow(display_up, cmap=cmap_up, aspect='auto', vmin=vmin_up, vmax=vmax_up)
        im_dn = ax.imshow(display_dn, cmap=cmap_dn, aspect='auto', vmin=vmin_dn, vmax=vmax_dn)
        
        ax.plot([0, L-1], [0, L-1], color='black', linestyle='--', linewidth=1.5)
        
        # Add labels with a semi-transparent white background for readability
        bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.7)
        ax.text(L*0.75, L*0.25, label_upper, color='black', fontsize=12, fontweight='bold', ha='center', bbox=bbox_props)
        ax.text(L*0.25, L*0.75, label_lower, color='black', fontsize=12, fontweight='bold', ha='center', bbox=bbox_props)
        ax.set_title(plot_title, fontweight='bold', fontsize=14, pad=15)
        
        div = make_axes_locatable(ax)
        cax_up = div.append_axes("right", size="5%", pad=0.1)
        cax_dn = div.append_axes("bottom", size="5%", pad=0.15)
        
        cb_up = fig.colorbar(im_up, cax=cax_up, orientation="vertical")
        cb_dn = fig.colorbar(im_dn, cax=cax_dn, orientation="horizontal")
        cb_up.set_label(label_upper, fontweight='bold')
        cb_dn.set_label(label_lower, fontweight='bold')
    
    # ---------------------------------------------------------
    # ROW 1: 2D Track Validation (Distogram vs Ground Truth)
    # ---------------------------------------------------------
    plot_split(
        axes[0, 0], true_dists, expected_dists, 
        "True Target (Å)", "Disto Expected (Å)", "Physical Topology (True vs Disto)", 
        plt.get_cmap('viridis_r'), plt.get_cmap('viridis_r'), 0, 22, 0, 22
    )
    plot_split(
        axes[0, 1], err_vs_true, entropy_matrix, 
        "Signed Error (Disto - True)", "Softmax Entropy", "2D Track Accuracy (Diagnostics)", 
        plt.get_cmap('RdBu_r'), plt.get_cmap('Purples'), -11, 11, 0, 4.16 # 4.16 is max entropy for 64 bins
    )
    
    # ---------------------------------------------------------
    # ROW 2: Cross-Track Constraint Validation (Distogram vs Predicted 3D)
    # ---------------------------------------------------------
    plot_split(
        axes[1, 0], pred_3d_dists, expected_dists, 
        "Pred 3D Coords (Å)", "Disto Expected (Å)", "Constraint Check (Pred 3D vs Disto)", 
        plt.get_cmap('viridis_r'), plt.get_cmap('viridis_r'), 0, 22, 0, 22
    )
    plot_split(
        axes[1, 1], err_vs_pred3d, entropy_matrix, 
        "Signed Error (Disto - Pred3D)", "Softmax Entropy", "1D/2D Conflict Diagnostics", 
        plt.get_cmap('RdBu_r'), plt.get_cmap('Purples'), -11, 11, 0, 4.16
    )

    plt.suptitle(title, fontweight='bold', fontsize=20, y=0.95)
    plt.subplots_adjust(hspace=0.25, wspace=0.25)
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close(fig)


# ==========================================
# 3. EXISTING PLOTTING FUNCTIONS
# ==========================================
def plot_three_way_comparison_clean(true_coords, base_coords, ref_coords, title="3D Comparison", filename="plot.html"):
    if 'go' not in globals(): return
    offset_val = np.max(np.abs(true_coords[:, 0])) * 2.0 + 15.0
    base_shifted = base_coords + np.array([-offset_val, 0, 0])
    ref_shifted = ref_coords + np.array([offset_val, 0, 0])

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=base_shifted[:, 0], y=base_shifted[:, 1], z=base_shifted[:, 2], mode='lines+markers', marker=dict(size=6, color='salmon'), line=dict(color='red', width=8), name='Base (Unrefined)'))
    fig.add_trace(go.Scatter3d(x=true_coords[:, 0], y=true_coords[:, 1], z=true_coords[:, 2], mode='lines+markers', marker=dict(size=6, color='lightblue'), line=dict(color='blue', width=8), name='True Target'))
    fig.add_trace(go.Scatter3d(x=ref_shifted[:, 0], y=ref_shifted[:, 1], z=ref_shifted[:, 2], mode='lines+markers', marker=dict(size=6, color='lightgreen'), line=dict(color='green', width=8), name='L-BFGS Refined'))

    axis_config = dict(showgrid=False, showbackground=False, visible=False)
    fig.update_layout(title=title, scene=dict(xaxis=axis_config, yaxis=axis_config, zaxis=axis_config, aspectmode='data'), legend=dict(x=0.02, y=0.98), margin=dict(l=0, r=0, b=0, t=40), paper_bgcolor='white', plot_bgcolor='white')
    fig.write_html(filename)


def plot_diagonal_split_matrix(matrix_true, matrix_pred, true_ss, pred_ss, title, save_path, cmap, vmax=None):
    fig, ax = plt.subplots(figsize=(8, 8))
    L = matrix_true.shape[0]
    combined = np.tril(matrix_pred, k=-1) + np.triu(matrix_true, k=1)
    im = ax.imshow(combined, cmap=cmap, aspect='auto', vmin=0, vmax=vmax)
    ax.plot([0, L-1], [0, L-1], color='white', linestyle='--', linewidth=1.5)
    ax.tick_params(axis='both', which='major', labelsize=16)
    
    div = make_axes_locatable(ax)
    cax = div.append_axes("top", size="5%", pad=0.15)
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.tick_params(labelsize=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


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
    print(f"Saved 1D Geometry Error Plot to {save_path}")


# ==========================================
# 1. NEW TAU-ONLY PLOT
# ==========================================
def plot_tau_error_per_residue(pred_1d, target_angles, target_ss, mask_1d, save_path="tau_error_per_residue.png"):
    """
    Plots absolute Tau predictions vs targets stacked above Error vs Uncertainty.
    Optimized for single-column publication (larger fonts, tighter layout).
    """
    # 1. Bulletproof conversion: Handles any mix of Tensors and NumPy arrays safely
    def safe_to_numpy(x):
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    pred_1d = safe_to_numpy(pred_1d)
    target_angles = safe_to_numpy(target_angles)
    target_ss = safe_to_numpy(target_ss)
    
    # CRITICAL FIX: Force the mask to be boolean so the '~' operator works
    mask_1d = safe_to_numpy(mask_1d).astype(bool)

    # If a batch dimension was accidentally passed, strip it
    if pred_1d.ndim == 3:
        pred_1d = pred_1d[0]
        target_angles = target_angles[0]
        target_ss = target_ss[0]
        mask_1d = mask_1d[0]

    seq_len = pred_1d.shape[0]
    x = np.arange(seq_len)

    # Extract Tau Predictions & Std Dev
    mu_tau = pred_1d[:, 3:5]
    std_tau = np.exp(pred_1d[:, 5] / 2.0)
    t_tau = target_angles[:, 2:4]

    pred_tau_deg = np.degrees(np.arctan2(mu_tau[:, 0], mu_tau[:, 1]))
    true_tau_deg = np.degrees(np.arctan2(t_tau[:, 0], t_tau[:, 1]))

    # Calculate MSE
    err_tau = np.sum((mu_tau - t_tau)**2, axis=-1)

    # Apply Kinematic Mask (The ~mask_1d will now work perfectly)
    for arr in [pred_tau_deg, true_tau_deg, err_tau, std_tau]:
        arr[~mask_1d] = np.nan

    # Optimized Figure Size for Single Column
    fig, axs = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    # --- Helper: Secondary Structure Background Shading ---
    def add_ss_background(ax):
        ss_colors = {0: ('#ff9999', 0.35), 1: ('#ffe599', 0.45)}
        start_idx = 0
        current_ss = target_ss[0]
        for i in range(1, len(target_ss)):
            if target_ss[i] != current_ss:
                if current_ss in ss_colors:
                    c, a = ss_colors[current_ss]
                    ax.axvspan(start_idx - 0.5, i - 0.5, color=c, alpha=a, lw=0, zorder=0)
                start_idx = i
                current_ss = target_ss[i]
        if current_ss in ss_colors:
            c, a = ss_colors[current_ss]
            ax.axvspan(start_idx - 0.5, len(target_ss) - 0.5, color=c, alpha=a, lw=0, zorder=0)

    # Plot 1: Values with Deltas
    ax0 = axs[0]
    add_ss_background(ax0)
    valid = ~np.isnan(true_tau_deg) & ~np.isnan(pred_tau_deg)
    wrap_mask = np.abs(true_tau_deg - pred_tau_deg) <= 180
    plot_mask = valid & wrap_mask

    ax0.vlines(x[plot_mask], np.minimum(true_tau_deg[plot_mask], pred_tau_deg[plot_mask]), 
               np.maximum(true_tau_deg[plot_mask], pred_tau_deg[plot_mask]), 
               color='gray', alpha=0.5, linewidth=2.0, zorder=2)
    
    ax0.scatter(x, true_tau_deg, color='black', s=40, label='True Target', alpha=0.8, zorder=3)
    ax0.scatter(x, pred_tau_deg, color='darkorange', s=40, label='Predicted', alpha=0.9, zorder=4)

    ax0.set_ylim(-190, 190)
    ax0.set_yticks([-180, -90, 0, 90, 180])
    ax0.set_title("Tau (τ) Values (Torsion)", fontweight='bold', fontsize=14)
    ax0.set_ylabel("Degrees", fontweight='bold', fontsize=12)
    ax0.tick_params(axis='both', labelsize=11)
    ax0.grid(True, linestyle=':', alpha=0.6)
    
    # Legend repositioned to be tight inside the plot area or above
    ax0.legend(loc='lower left', fontsize=11, framealpha=0.9)

    # Plot 2: Error vs Uncertainty
    ax1 = axs[1]
    add_ss_background(ax1)
    
    ax1.plot(x, err_tau, color='crimson', linewidth=2.5, label='True Error')
    ax1.plot(x, std_tau, color='royalblue', linestyle='--', linewidth=2.5, label='Predicted Uncertainty')
    ax1.fill_between(x, 0, std_tau, color='royalblue', alpha=0.2)

    ax1.set_ylim(0, 4.2)
    ax1.set_title("Tau (τ) Error vs Uncertainty", fontweight='bold', fontsize=14)
    ax1.set_ylabel("MSE", fontweight='bold', fontsize=12)
    ax1.set_xlabel("Sequence Residue Index", fontweight='bold', fontsize=14)
    ax1.tick_params(axis='both', labelsize=11)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper right', fontsize=11, framealpha=0.9)

    # SS Background Custom Legend
    legend_elements = [
        Patch(facecolor='#ff9999', alpha=0.5, label='Alpha Helix'),
        Patch(facecolor='#ffe599', alpha=0.6, label='Beta Sheet'),
        Patch(facecolor='white', edgecolor='gray', label='Coil / Loop')
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=3, fontsize=12, frameon=False, bbox_to_anchor=(0.5, 1.05))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


# ==========================================
# 2. UPDATED DISTOGRAM ANALYSIS PLOT
# ==========================================
def plot_distogram_analysis_large_text(disto_logits, true_coords, pred_coords, title="Distogram Constraint Analysis", filename="distogram_analysis_large_text.png"):
    """
    2x2 Diagnostic Grid optimized for single-column rendering.
    """
    L = true_coords.shape[0]
    
    true_dists = np.linalg.norm(true_coords[:, None, :] - true_coords[None, :, :], axis=-1)
    pred_3d_dists = np.linalg.norm(pred_coords[:, None, :] - pred_coords[None, :, :], axis=-1)

    true_dists = np.clip(true_dists, 0, 22)
    pred_3d_dists = np.clip(pred_3d_dists, 0, 22)
    
    probs = F.softmax(torch.tensor(disto_logits).float(), dim=-1).numpy()
    bin_indices = np.arange(64)
    expected_dists = (np.sum(probs * bin_indices, axis=-1) * (20.0 / 64.0)) + 2.0
    entropy_matrix = -np.sum(probs * np.log(probs + 1e-8), axis=-1)
    
    err_vs_true = expected_dists - true_dists
    err_vs_pred3d = expected_dists - pred_3d_dists
    
    # Reduced figsize for denser single-column layout
    fig, axes = plt.subplots(2, 2, figsize=(8, 8)) 
    
    def plot_split(ax, mat_upper, mat_lower, label_upper, label_lower, plot_title, cmap_up, cmap_dn, vmin_up, vmax_up, vmin_dn, vmax_dn):
        combined = np.zeros((L, L))
        upper_mask = np.triu_indices(L, k=1)
        lower_mask = np.tril_indices(L, k=-1)
        
        combined[upper_mask] = mat_upper[upper_mask]
        combined[lower_mask] = mat_lower[lower_mask]
        
        display_up = np.ma.masked_where(np.tril(np.ones((L,L))), combined)
        display_dn = np.ma.masked_where(np.triu(np.ones((L,L))), combined)
        
        im_up = ax.imshow(display_up, cmap=cmap_up, aspect='auto', vmin=vmin_up, vmax=vmax_up)
        im_dn = ax.imshow(display_dn, cmap=cmap_dn, aspect='auto', vmin=vmin_dn, vmax=vmax_dn)
        
        ax.plot([0, L-1], [0, L-1], color='black', linestyle='--', linewidth=1.5)
        
        # Bigger fonts for inside text
        bbox_props = dict(boxstyle="round,pad=0.2", fc="white", ec="black", alpha=0.85)
        ax.text(L*0.75, L*0.25, label_upper, color='black', fontsize=10, fontweight='bold', ha='center', bbox=bbox_props)
        ax.text(L*0.25, L*0.75, label_lower, color='black', fontsize=10, fontweight='bold', ha='center', bbox=bbox_props)
        
        # Shorter padding and bigger font for title
        ax.set_title(plot_title, fontweight='bold', fontsize=12, pad=10)
        ax.tick_params(axis='both', labelsize=10)
        
        div = make_axes_locatable(ax)
        cax_up = div.append_axes("right", size="5%", pad=0.1)
        cax_dn = div.append_axes("bottom", size="5%", pad=0.15)
        
        cb_up = fig.colorbar(im_up, cax=cax_up, orientation="vertical")
        cb_dn = fig.colorbar(im_dn, cax=cax_dn, orientation="horizontal")
        cb_up.set_label(label_upper, fontweight='bold', fontsize=10)
        cb_dn.set_label(label_lower, fontweight='bold', fontsize=10)
        cb_up.ax.tick_params(labelsize=9)
        cb_dn.ax.tick_params(labelsize=9)
    
    # ROW 1
    plot_split(
        axes[0, 0], true_dists, expected_dists, 
        "Target (Å)", "Pred (Å)", "Physical Topology", 
        plt.get_cmap('viridis_r'), plt.get_cmap('viridis_r'), 0, 22, 0, 22
    )
    plot_split(
        axes[0, 1], err_vs_true, entropy_matrix, 
        "Disto Error", "Entropy", "2D Track Accuracy", 
        plt.get_cmap('RdBu_r'), plt.get_cmap('Purples'), -11, 11, 0, 4.16
    )
    
    # ROW 2
    plot_split(
        axes[1, 0], pred_3d_dists, expected_dists, 
        "3D Coords", "Pred Disto", "Constraint Check", 
        plt.get_cmap('viridis_r'), plt.get_cmap('viridis_r'), 0, 22, 0, 22
    )
    plot_split(
        axes[1, 1], err_vs_pred3d, entropy_matrix, 
        "1D/2D Error", "Entropy", "Conflict Diagnostics", 
        plt.get_cmap('RdBu_r'), plt.get_cmap('Purples'), -11, 11, 0, 4.16
    )

    plt.suptitle(title, fontweight='bold', fontsize=16, y=1.02)
    plt.subplots_adjust(hspace=0.45, wspace=0.35)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)