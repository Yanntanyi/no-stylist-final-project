"""
No Stylist — Framework Evaluation Script
==========================================
Runs the pipeline on multiple photos and generates a comparative
evaluation table for the paper's Framework Evaluation section.

Usage:
    python evaluate.py --photos photo1.jpg photo2.jpg photo3.jpg
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

from pipeline import FaceMeshReconstructor, BodyCompositor, RESULTS_DIR


# ─────────────────────────────────────────────
# EVALUATION METRICS
# ─────────────────────────────────────────────

def evaluate_face_symmetry(landmarks: list, image_size: tuple) -> float:
    """
    Estimates facial symmetry score (0-1) by comparing left/right
    landmark positions around the vertical midline.
    Higher = more symmetry detected = better reconstruction accuracy.
    """
    w, h = image_size
    # Use contour landmarks (indices 0-16 approx for jawline)
    left_xs  = [landmarks[i][0] for i in range(0, 234)]
    right_xs = [landmarks[i][0] for i in range(234, 468)]
    midline  = np.mean([lm[0] for lm in landmarks])

    left_dist  = np.mean([abs(x - midline) for x in left_xs])
    right_dist = np.mean([abs(x - midline) for x in right_xs])

    symmetry = 1.0 - abs(left_dist - right_dist) / max(left_dist, right_dist, 1e-6)
    return round(min(max(symmetry, 0.0), 1.0), 3)


def evaluate_landmark_spread(landmarks: list) -> float:
    """
    Measures how broadly landmarks spread across the face (0-1).
    Low spread may indicate detection failure or partial face visibility.
    """
    xs = [lm[0] for lm in landmarks]
    ys = [lm[1] for lm in landmarks]
    spread_x = max(xs) - min(xs)
    spread_y = max(ys) - min(ys)
    # Normalize against full image: ideal spread is ~0.4-0.7
    return round(min((spread_x + spread_y) / 2 / 0.6, 1.0), 3)


def evaluate_depth_variance(landmarks: list) -> float:
    """
    Measures Z-depth variance in landmarks — a proxy for 3D reconstruction
    quality. Higher variance = more realistic depth modeling.
    """
    zs = [lm[2] for lm in landmarks]
    variance = float(np.var(zs))
    # Normalize to 0-1 range (empirical scale)
    return round(min(variance * 1000, 1.0), 3)


# ─────────────────────────────────────────────
# PER-PHOTO EVALUATION
# ─────────────────────────────────────────────

def evaluate_photo(photo_path: str, photo_id: int) -> dict:
    """
    Runs the full pipeline on one photo and computes evaluation metrics.
    """
    print(f"\n{'─'*50}")
    print(f"  Evaluating photo {photo_id}: {photo_path}")
    print(f"{'─'*50}")

    result = {
        "photo_path": photo_path,
        "photo_id": photo_id,
    }

    # MediaPipe reconstruction
    reconstructor = FaceMeshReconstructor()
    mp_result = reconstructor.reconstruct(photo_path)
    result["mediapipe"] = mp_result

    if mp_result["success"]:
        landmarks   = mp_result["landmarks"]
        image_size  = mp_result["image_size"]

        result["metrics"] = {
            "landmark_count":    mp_result["landmark_count"],
            "processing_time_ms": round(mp_result["processing_time_ms"], 1),
            "symmetry_score":    evaluate_face_symmetry(landmarks, image_size),
            "landmark_spread":   evaluate_landmark_spread(landmarks),
            "depth_variance":    evaluate_depth_variance(landmarks),
        }

        # Composite
        compositor = BodyCompositor()
        comp_result = compositor.composite(
            photo_path, landmarks, image_size
        )
        result["compositor"] = comp_result

        print(f"\n  Metrics for photo {photo_id}:")
        for k, v in result["metrics"].items():
            print(f"    {k}: {v}")

    return result


# ─────────────────────────────────────────────
# COMPARISON REPORT
# ─────────────────────────────────────────────

def generate_comparison_report(evaluations: list):
    """
    Generates a multi-photo comparison grid showing:
    - Original photo
    - MediaPipe mesh
    - Composite avatar
    - Metrics table
    """
    n = len(evaluations)
    if n == 0:
        print("No successful evaluations to report.")
        return

    print("\n[Report] Generating multi-photo comparison report...")

    fig = plt.figure(figsize=(5 * 3, 5 * n + 3))
    fig.suptitle(
        "No Stylist — Multi-Photo Framework Evaluation\nMediaPipe FaceMesh + Body Compositor",
        fontsize=14, fontweight="bold"
    )

    gs = gridspec.GridSpec(n + 1, 3, figure=fig, hspace=0.4, wspace=0.3)

    # Column headers
    header_ax = [fig.add_subplot(gs[0, i]) for i in range(3)]
    for ax, title in zip(header_ax, ["Original Photo", "MediaPipe Mesh", "Composite Avatar"]):
        ax.text(0.5, 0.5, title, ha='center', va='center',
                fontsize=12, fontweight='bold', transform=ax.transAxes)
        ax.set_facecolor('#f0f0f0')
        ax.axis("off")

    # Photo rows
    for row_idx, eval_data in enumerate(evaluations):
        row = row_idx + 1
        photo_path = eval_data["photo_path"]
        pid = eval_data["photo_id"]
        mp_result = eval_data.get("mediapipe", {})
        comp_result = eval_data.get("compositor", {})
        metrics = eval_data.get("metrics", {})

        # Original
        ax_orig = fig.add_subplot(gs[row, 0])
        try:
            orig = Image.open(photo_path)
            ax_orig.imshow(np.array(orig))
        except Exception:
            ax_orig.text(0.5, 0.5, "Load failed", ha='center', va='center')
        ax_orig.set_title(f"Photo {pid}", fontsize=9)
        ax_orig.axis("off")

        # MediaPipe mesh
        ax_mesh = fig.add_subplot(gs[row, 1])
        if mp_result.get("success") and Path(mp_result["annotated_image_path"]).exists():
            mesh_img = cv2.imread(mp_result["annotated_image_path"])
            mesh_rgb = cv2.cvtColor(mesh_img, cv2.COLOR_BGR2RGB)
            ax_mesh.imshow(mesh_rgb)
            if metrics:
                info = (f"Landmarks: {metrics['landmark_count']}\n"
                        f"Time: {metrics['processing_time_ms']}ms\n"
                        f"Symmetry: {metrics['symmetry_score']}\n"
                        f"Depth var: {metrics['depth_variance']}")
                ax_mesh.text(0.02, 0.02, info, transform=ax_mesh.transAxes,
                             fontsize=7, va='bottom',
                             bbox=dict(facecolor='white', alpha=0.85, boxstyle='round'))
        else:
            ax_mesh.text(0.5, 0.5, f"Failed:\n{mp_result.get('error', '?')}",
                         ha='center', va='center', transform=ax_mesh.transAxes, fontsize=8)
        ax_mesh.axis("off")

        # Composite
        ax_comp = fig.add_subplot(gs[row, 2])
        if comp_result.get("success") and Path(comp_result["composite_path"]).exists():
            comp_img = Image.open(comp_result["composite_path"])
            ax_comp.imshow(np.array(comp_img))
            ax_comp.text(0.02, 0.02, "Face from landmarks\non generic body",
                         transform=ax_comp.transAxes, fontsize=7, va='bottom',
                         bbox=dict(facecolor='white', alpha=0.85, boxstyle='round'))
        else:
            ax_comp.text(0.5, 0.5, "Composite failed",
                         ha='center', va='center', transform=ax_comp.transAxes, fontsize=8)
        ax_comp.axis("off")

    report_path = str(RESULTS_DIR / "multi_photo_comparison.png")
    plt.savefig(report_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Report] Saved → {report_path}")

    # Save metrics as JSON
    metrics_path = str(RESULTS_DIR / "evaluation_metrics.json")
    summary = [
        {
            "photo": e["photo_path"],
            "metrics": e.get("metrics", {}),
            "mediapipe_success": e.get("mediapipe", {}).get("success", False),
            "composite_success": e.get("compositor", {}).get("success", False),
        }
        for e in evaluations
    ]
    with open(metrics_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[Report] Metrics JSON saved → {metrics_path}")

    return report_path


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="No Stylist — Multi-photo framework evaluation"
    )
    parser.add_argument(
        "--photos", nargs="+", required=True,
        help="Paths to 1-5 input photos for comparison"
    )
    args = parser.parse_args()

    print("=" * 55)
    print("  No Stylist — Framework Evaluation")
    print("=" * 55)

    evaluations = []
    for i, photo_path in enumerate(args.photos):
        if not Path(photo_path).exists():
            print(f"Warning: {photo_path} not found, skipping.")
            continue
        result = evaluate_photo(photo_path, photo_id=i + 1)
        evaluations.append(result)

    if evaluations:
        generate_comparison_report(evaluations)

    print("\n" + "=" * 55)
    print("  EVALUATION COMPLETE")
    print(f"  {len(evaluations)} photo(s) processed")
    print(f"  Results saved to: {RESULTS_DIR}/")
    print("=" * 55)


if __name__ == "__main__":
    main()
