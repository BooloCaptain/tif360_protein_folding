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
    """Write a glTF-compatible export.

    Preferred path uses trimesh point-cloud export for tool compatibility.
    Fallback writes a minimal JSON glTF payload containing coordinates in extras.
    """
    if trimesh is not None:
        cloud = trimesh.points.PointCloud(coords)
        cloud.export(path)
        return path

    payload = {
        "asset": {"version": "2.0", "generator": "tif360_protein_folding"},
        "scenes": [{"nodes": [0]}],
        "scene": 0,
        "nodes": [{"name": "protein_coords"}],
        "extras": {
            "coordinate_space": "angstrom",
            "coords": [list(map(float, c)) for c in coords],
            "note": "Install trimesh for richer glTF exports",
        },
    }
    with open(path, "w") as f:
        json.dump(payload, f)
    return path
