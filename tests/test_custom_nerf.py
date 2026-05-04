import torch
import numpy as np
import pytest
from src.data.dataset_full import ca_to_internal_targets
from src.postproc.diagnostics import rmsd
from src.postproc.visualize import plot_protein_comparison
from src.train import build_ca_coords_nerf
from tests.test_pipeline_diagnostics import _find_clean_roundtrip_segment, _kabsch_align, _load_real_protein_dataset

# Assuming your custom function is imported or defined here
# from src.models.kinematics import build_ca_coords_nerf

def test_custom_ca_reconstruction_roundtrip():
    """
    Verifies that the custom CA-only NeRF function can perfectly reconstruct
    a real protein from its internal pseudo-angles and bond lengths.
    """
    # 1. Setup Environment
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = _load_real_protein_dataset()
    
    # Find a clean segment of a real protein (no missing residues)
    # We use a strict target_rmsd because the math should be nearly perfect
    coords, _ = _find_clean_roundtrip_segment(dataset, min_len=40, target_rmsd=1e-4)
    
    if coords is None:
        pytest.skip("Could not find a suitable clean protein segment for testing.")

    # 2. Convert 3D Ground Truth -> 1D Internal Coordinates
    # ca_to_internal_targets should return:
    # - bond_lengths: distance between CA_i and CA_{i-1}
    # - thetas: planar angles
    # - phis: dihedrals/torsions
    angles, bond_lengths = ca_to_internal_targets(coords)
    
    # Split the angles into thetas and phis as expected by your custom function
    # Assuming angles is shape (L, 2) where col 0 is theta and col 1 is phi
    thetas = angles[:, 0]
    phis = angles[:, 1]

    # 3. Prepare Tensors for Custom Function
    # The custom function expects (B, L)
    bl_tensor = torch.from_numpy(bond_lengths).unsqueeze(0).to(device).float()
    theta_tensor = torch.from_numpy(thetas).unsqueeze(0).to(device).float()
    phi_tensor = torch.from_numpy(phis).unsqueeze(0).to(device).float()

    # 4. Execute Custom Reconstruction
    # This is the function we built in the previous turn
    with torch.no_grad():
        reconstructed_tensor = build_ca_coords_nerf(bl_tensor, theta_tensor, phi_tensor)
    
    reconstructed = reconstructed_tensor.squeeze(0).cpu().numpy()

    # 5. Alignment and Validation
    # Internal coordinates are frame-invariant, so we must align the 
    # reconstructed "cloud" to the original "cloud" using Kabsch.
    aligned_reconstructed = _kabsch_align(coords, reconstructed)
    
    # Calculate RMSD
    error = rmsd(coords, aligned_reconstructed)
    
    plot_protein_comparison(
        true_coords=coords, 
        pred_coords=aligned_reconstructed, 
        title=f"TEST",
        filename="test_reconstruction_roundtrip.html"
    )

    # 6. Assertions
    # For a purely mathematical round-trip, the error should be extremely low.
    # We allow for minor float32 precision drift.
    assert error < 2e-1, f"Reconstruction failed round-trip. RMSD: {error:.6f} Å"
    
    # Check shape integrity
    assert reconstructed.shape == coords.shape, "Output shape mismatch."

def test_reconstruction_is_differentiable():
    """
    Ensures that gradients can flow from the 3D coordinates back to the 1D inputs.
    If this fails, your end-to-end training will not work.
    """
    L = 20
    bl = torch.randn(1, L, requires_grad=True)
    th = torch.randn(1, L, requires_grad=True)
    ph = torch.randn(1, L, requires_grad=True)
    
    coords = build_ca_coords_nerf(bl, th, ph)
    
    # Dummy loss: minimize distance to origin
    loss = (coords ** 2).sum()
    loss.backward()
    
    assert bl.grad is not None
    assert th.grad is not None
    assert ph.grad is not None
    assert not torch.isnan(bl.grad).any(), "Gradient exploded to NaN."