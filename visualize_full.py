"""
visualize_full_v3.py — 3 figures séparées pour une séquence ViPE :
  1. depth_map.png
  2. intrinsics.png  (matrice K sur la depth map)
  3. trajectory.png
"""

import zipfile
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path("./data/payload/dpsp-012acfb6")


def read_one_depth_frame(zip_path, frame_index=0):
    import OpenEXR
    with zipfile.ZipFile(zip_path, "r") as z:
        names = sorted(z.namelist())
        if frame_index >= len(names):
            frame_index = 0
        fname = names[frame_index]
        with z.open(fname) as f:
            exr = OpenEXR.InputFile(f)
            header = exr.header()
            dw = header["dataWindow"]
            w = dw.max.x - dw.min.x + 1
            h = dw.max.y - dw.min.y + 1
            ch = exr.channels(["Z"])
            depth = np.frombuffer(ch[0], dtype=np.float16).reshape((h, w))
            return int(fname.split(".")[0]), depth.astype(np.float32), len(names)


def main():
    depth_dir = DATA_DIR / "depth"
    pose_dir = DATA_DIR / "pose"
    intr_dir = DATA_DIR / "intrinsics"

    zip_files = sorted(depth_dir.glob("*.zip"))
    if not zip_files:
        zip_files = sorted(depth_dir.rglob("*.zip"))
    first_zip = zip_files[0]
    seq_uuid = first_zip.stem
    print(f"Séquence: {seq_uuid}")

    # Lire les données
    frame_idx, depth, n_frames = read_one_depth_frame(first_zip, frame_index=0)
    valid = np.isfinite(depth) & (depth > 0)
    vmin, vmax = np.percentile(depth[valid], [2, 98]) if valid.any() else (0, 1)

    intr_data = np.load(intr_dir / f"{seq_uuid}.npz")
    intr_vals = intr_data["data"]
    fx, fy, cx, cy = intr_vals[0, :4]
    h, w = depth.shape
    fov_h = 2 * np.degrees(np.arctan(w / (2 * fx)))
    fov_v = 2 * np.degrees(np.arctan(h / (2 * fy)))

    cam_txt = intr_dir / f"{seq_uuid}_camera.txt"
    cam_type = "PINHOLE"
    if cam_txt.exists():
        cam_type = cam_txt.read_text().split(":")[1].strip().split("\n")[0]

    pose_data = np.load(pose_dir / f"{seq_uuid}.npz")
    pose_inds = pose_data["inds"]
    poses = pose_data["data"]
    positions = poses[:, :3, 3]

    print(f"  Depth: {depth.shape}, Intrinsics: fx={fx:.1f}, Poses: {len(poses)} frames")

    # ===========================================================
    # FIGURE 1 : Depth Map
    # ===========================================================
    fig1, ax1 = plt.subplots(figsize=(12, 7))
    im = ax1.imshow(depth, cmap="turbo", vmin=vmin, vmax=vmax)
    ax1.scatter([cx], [cy], c="white", s=80, marker="+", linewidths=2.5, zorder=5)
    ax1.set_title(f"ViPE : Carte de profondeur métrique\n"
                  f"Séquence {seq_uuid[:12]}... | Frame {frame_idx}/{n_frames}",
                  fontsize=14, fontweight="bold")
    ax1.axis("off")
    cb = plt.colorbar(im, ax=ax1, fraction=0.03, pad=0.02)
    cb.set_label("Profondeur (mètres)", fontsize=11)
    fig1.tight_layout()
    fig1.savefig("1_depth_map.png", dpi=150, bbox_inches="tight")
    print("✓ 1_depth_map.png")

    # ===========================================================
    # FIGURE 2 : Intrinsèques — matrice K sur la depth map
    # ===========================================================
    fig2, ax2 = plt.subplots(figsize=(12, 7))
    ax2.imshow(depth, cmap="turbo", vmin=vmin, vmax=vmax)
    ax2.scatter([cx], [cy], c="white", s=120, marker="+", linewidths=3, zorder=5)
    ax2.axis("off")

    K_text = (
        f"         ┌                    ┐\n"
        f"         │ {fx:.1f}    0     {cx:.0f} │\n"
        f"  K  =   │   0    {fy:.1f}  {cy:.0f} │\n"
        f"         │   0      0      1  │\n"
        f"         └                    ┘"
    )
    ax2.text(20, 60, K_text, fontsize=14, fontfamily="monospace", color="white",
             bbox=dict(boxstyle="round,pad=0.6", facecolor="black", alpha=0.7))

    ax2.text(20, h - 30, f"Modèle: {cam_type}  •  FoV: {fov_h:.1f}° × {fov_v:.1f}°",
             fontsize=12, color="white", fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="black", alpha=0.6))

    ax2.set_title("ViPE : Intrinsèques caméra estimées", fontsize=14, fontweight="bold")
    fig2.tight_layout()
    fig2.savefig("2_intrinsics.png", dpi=150, bbox_inches="tight")
    print("✓ 2_intrinsics.png")

    # ===========================================================
    # FIGURE 3 : Trajectoire 3D
    # ===========================================================
    fig3 = plt.figure(figsize=(10, 8))
    ax3 = fig3.add_subplot(111, projection="3d")

    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
    colors = np.linspace(0, 1, len(x))

    ax3.scatter(x, y, z, c=colors, cmap="viridis", s=15, alpha=0.8)
    ax3.plot(x, y, z, color="gray", linewidth=0.8, alpha=0.4)

    ax3.scatter(*positions[0], c="lime", s=120, marker="o", edgecolors="black",
                linewidths=2, zorder=5, label="Début")
    ax3.scatter(*positions[-1], c="red", s=120, marker="s", edgecolors="black",
                linewidths=2, zorder=5, label="Fin")

    n_cams = min(10, len(poses))
    cam_step = max(1, len(poses) // n_cams)
    arrow_len = np.linalg.norm(positions[-1] - positions[0]) * 0.08
    if arrow_len < 0.02:
        arrow_len = 0.05

    for i in range(0, len(poses), cam_step):
        R = poses[i, :3, :3]
        t = poses[i, :3, 3]
        look = R[:, 2] * arrow_len
        ax3.quiver(t[0], t[1], t[2], look[0], look[1], look[2],
                   color="coral", arrow_length_ratio=0.25, linewidth=1.8)

    traj_len = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
    ax3.set_title(f"ViPE : Trajectoire caméra\n"
                  f"{len(pose_inds)} frames | longueur: {traj_len:.2f} m",
                  fontsize=14, fontweight="bold")
    ax3.set_xlabel("X (m)", fontsize=11)
    ax3.set_ylabel("Y (m)", fontsize=11)
    ax3.set_zlabel("Z (m)", fontsize=11)
    ax3.legend(fontsize=10, loc="upper left")
    fig3.tight_layout()
    fig3.savefig("3_trajectory.png", dpi=150, bbox_inches="tight")
    print("✓ 3_trajectory.png")

    plt.show()
    print(f"\nDone — 3 fichiers générés pour la séquence {seq_uuid[:12]}…")


if __name__ == "__main__":
    main()