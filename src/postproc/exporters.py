import json

try:
    import trimesh
except Exception:
    trimesh = None


def coords_to_pdb(coords, chain_id='A', resname='ALA'):
    lines = []
    for i, p in enumerate(coords, start=1):
        x, y, z = p.tolist()
        atom_line = (
            "ATOM  {atom:5d}  CA  {res:>3s} {chain:1s}{resi:4d}    "
            "{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
        ).format(atom=i, res=resname, chain=chain_id, resi=i, x=x, y=y, z=z)
        lines.append(atom_line)
    return ''.join(lines)


def write_pdb(path, coords, chain_id='A', resname='ALA'):
    s = coords_to_pdb(coords, chain_id=chain_id, resname=resname)
    with open(path, 'w') as f:
        f.write(s)
    return path


def write_gltf(path, coords):
    """Write protein coordinates to glTF format with explicit backend logging.

    Preferred backend: trimesh (produces standard .glb files)
    Fallback backend: JSON payload (minimal glTF-compatible structure)
    
    Args:
        path: Output file path
        coords: (N, 3) coordinate array
        
    Returns:
        path: Output file path
    """
    if trimesh is not None:
        try:
            cloud = trimesh.points.PointCloud(coords)
            cloud.export(path)
            print(f"[SUCCESS] Exported glTF using trimesh backend to {path}")
            return path
        except Exception as e:
            print(f"[WARNING] trimesh export failed: {e}")
            print(f"[INFO] Falling back to JSON glTF payload")
    else:
        print(f"[INFO] trimesh not available. Using JSON glTF payload fallback.")

    # Fallback: JSON payload
    payload = {
        "asset": {"version": "2.0", "generator": "tif360_protein_folding"},
        "scenes": [{"nodes": [0]}],
        "scene": 0,
        "nodes": [{"name": "protein_coords"}],
        "extras": {
            "coordinate_space": "angstrom",
            "coords": [list(map(float, c)) for c in coords],
            "note": "JSON-only format (no mesh). Install trimesh for standard glTF export.",
        },
    }
    with open(path, "w") as f:
        json.dump(payload, f)
    print(f"[INFO] Exported JSON glTF payload to {path}")
    return path
