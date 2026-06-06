"""
visualize_depth.py — Visualise les depth maps du dataset ViPE
Format réel : depth.tar → {sequence}.zip → {frame:05d}.exr (float16, canal "Z")
"""

import zipfile
import tarfile
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# CONFIG — adapte ce chemin à ton setup
# ============================================================
DATA_DIR = Path("./data/payload/dpsp-012acfb6")


def read_vipe_depth(zip_path):
    """Lit les depth maps d'un ZIP ViPE (format EXR float16, canal Z)."""
    import OpenEXR

    frames = []
    with zipfile.ZipFile(zip_path, "r") as z:
        for file_name in sorted(z.namelist()):
            if not file_name.endswith(".exr"):
                continue
            frame_idx = int(file_name.split(".")[0])
            with z.open(file_name) as f:
                exr = OpenEXR.InputFile(f)
                header = exr.header()
                dw = header["dataWindow"]
                width = dw.max.x - dw.min.x + 1
                height = dw.max.y - dw.min.y + 1
                channels = exr.channels(["Z"])
                depth = np.frombuffer(channels[0], dtype=np.float16).reshape((height, width))
                frames.append((frame_idx, depth.astype(np.float32)))
    return frames


def read_vipe_poses(npz_path):
    """Lit les poses caméra (4x4 SE3 matrices)."""
    data = np.load(npz_path)
    return data["inds"], data["data"]  # inds: frame indices, data: Nx4x4


def read_vipe_intrinsics(npz_path):
    """Lit les intrinsèques [fx, fy, cx, cy]."""
    data = np.load(npz_path)
    return data["inds"], data["data"]  # inds: frame indices, data: Nx4


def main():
    # ----------------------------------------------------------
    # 1. Trouver les fichiers
    # ----------------------------------------------------------
    # Cherche les depth ZIPs (soit déjà extraits, soit dans depth.tar)
    depth_dir = DATA_DIR / "depth"

    if depth_dir.exists():
        zip_files = sorted(depth_dir.glob("*.zip"))
    else:
        # Pas encore extrait ? On cherche depth.tar
        print("Dossier 'depth/' pas trouvé, extraction depuis depth.tar...")
        tar_path = DATA_DIR / "depth.tar"
        if not tar_path.exists():
            print(f"ERREUR: ni {depth_dir} ni {tar_path} trouvés.")
            return
        depth_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(path=depth_dir)
        zip_files = sorted(depth_dir.glob("*.zip"))

    print(f"Trouvé {len(zip_files)} séquences (ZIP)")

    if not zip_files:
        # Peut-être que les ZIPs sont dans un sous-dossier
        zip_files = sorted(depth_dir.rglob("*.zip"))
        print(f"  (recherche récursive: {len(zip_files)} ZIPs)")

    if not zip_files:
        print("ERREUR: aucun fichier .zip trouvé.")
        print(f"  Contenu de {depth_dir}:")
        for f in sorted(depth_dir.iterdir())[:20]:
            print(f"    {f.name}  ({f.stat().st_size} bytes)")
        return

    # ----------------------------------------------------------
    # 2. Lire les poses et intrinsèques
    # ----------------------------------------------------------
    pose_tar = DATA_DIR / "pose.tar"
    intr_tar = DATA_DIR / "intrinsics.tar"

    pose_dir = DATA_DIR / "pose"
    intr_dir = DATA_DIR / "intrinsics"

    if not pose_dir.exists() and pose_tar.exists():
        print("Extraction pose.tar...")
        pose_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(pose_tar, "r") as tar:
            tar.extractall(path=pose_dir)

    if not intr_dir.exists() and intr_tar.exists():
        print("Extraction intrinsics.tar...")
        intr_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(intr_tar, "r") as tar:
            tar.extractall(path=intr_dir)

    # ----------------------------------------------------------
    # 3. Prendre la première séquence et visualiser
    # ----------------------------------------------------------
    first_zip = zip_files[0]
    seq_name = first_zip.stem
    print(f"\nSéquence: {seq_name}")
    print(f"  Depth ZIP: {first_zip} ({first_zip.stat().st_size / 1e6:.1f} MB)")

    # Lire les depth maps
    print("  Lecture des EXR...")
    frames = read_vipe_depth(first_zip)
    print(f"  → {len(frames)} frames chargées")

    if not frames:
        print("ERREUR: aucune frame trouvée dans le ZIP.")
        return

    # Afficher les stats de la première frame
    idx0, depth0 = frames[0]
    valid = np.isfinite(depth0) & (depth0 > 0)
    print(f"  Frame {idx0}: shape={depth0.shape}, dtype=float32")
    if valid.any():
        print(f"    min={depth0[valid].min():.2f}m, max={depth0[valid].max():.2f}m, "
              f"median={np.median(depth0[valid]):.2f}m")

    # Lire les poses si disponibles
    pose_files = sorted(pose_dir.rglob(f"{seq_name}.npz")) if pose_dir.exists() else []
    if pose_files:
        inds, poses = read_vipe_poses(pose_files[0])
        print(f"  Poses: {len(inds)} frames, shape={poses.shape}")

    # Lire les intrinsèques si disponibles
    intr_files = sorted(intr_dir.rglob(f"{seq_name}.npz")) if intr_dir.exists() else []
    if intr_files:
        inds_i, intrinsics = read_vipe_intrinsics(intr_files[0])
        print(f"  Intrinsics: {intrinsics[0]} (fx, fy, cx, cy)")

    # ----------------------------------------------------------
    # 4. Visualisation
    # ----------------------------------------------------------
    # Sélectionner quelques frames régulièrement espacées
    n_show = min(6, len(frames))
    step = max(1, len(frames) // n_show)
    selected = frames[::step][:n_show]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    for i, (frame_idx, depth) in enumerate(selected):
        valid = np.isfinite(depth) & (depth > 0)
        if valid.any():
            vmin, vmax = np.percentile(depth[valid], [2, 98])
        else:
            vmin, vmax = 0, 1

        im = axes[i].imshow(depth, cmap="turbo", vmin=vmin, vmax=vmax)
        axes[i].set_title(f"Frame {frame_idx}", fontsize=10)
        axes[i].axis("off")
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04, label="m")

    for i in range(n_show, len(axes)):
        axes[i].axis("off")

    plt.suptitle(
        f"ViPE Depth Maps — Séquence: {seq_name}\n"
        f"Format: ZIP→EXR float16, canal 'Z', profondeur métrique (mètres)",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()

    out_path = Path(f"depth_viz_{seq_name[:8]}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n✓ Sauvegardé: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()