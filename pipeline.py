"""
No Stylist — Avatar Foundation Pipeline
========================================
Phase 1: MediaPipe face mesh reconstruction
Phase 2: Avaturn full-body avatar generation

Usage:
    python pipeline.py --photo path/to/your/photo.jpg

Outputs are saved to the results/ folder.
"""

import argparse
import os
import sys
import json
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Avaturn free tier API endpoint (sign up at avaturn.me for a free API key)
AVATURN_API_URL = "https://api.avaturn.me/api/v1/avatars"


# ─────────────────────────────────────────────
# PHASE 1: MEDIAPIPE FACE MESH
# ─────────────────────────────────────────────

class FaceMeshReconstructor:
    """
    Uses MediaPipe FaceMesh to detect 468 facial landmarks
    from a single photo and produce a reconstructed face overlay.
    """

    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def reconstruct(self, image_path: str) -> dict:
        """
        Run face mesh reconstruction on a photo.

        Args:
            image_path: Path to input photo (JPG or PNG)

        Returns:
            dict with keys:
                - success (bool)
                - landmarks (list of (x, y, z) tuples, normalized 0-1)
                - annotated_image_path (str)
                - mesh_image_path (str)
                - landmark_count (int)
                - processing_time_ms (float)
        """
        print(f"\n[MediaPipe] Loading image: {image_path}")
        image = cv2.imread(image_path)

        if image is None:
            return {"success": False, "error": f"Could not load image at {image_path}"}

        h, w = image.shape[:2]
        print(f"[MediaPipe] Image size: {w}x{h}px")

        start = time.time()

        with self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,         # includes iris landmarks
            min_detection_confidence=0.5
        ) as face_mesh:

            # MediaPipe expects RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_image)

        elapsed = (time.time() - start) * 1000

        if not results.multi_face_landmarks:
            return {
                "success": False,
                "error": "No face detected. Try a clearer, well-lit frontal photo."
            }

        face_landmarks = results.multi_face_landmarks[0]
        landmarks = [
            (lm.x, lm.y, lm.z)
            for lm in face_landmarks.landmark
        ]

        print(f"[MediaPipe] Detected {len(landmarks)} landmarks in {elapsed:.1f}ms")

        # ── Save annotated image (landmarks drawn on photo) ──
        annotated = image.copy()
        self.mp_drawing.draw_landmarks(
            image=annotated,
            landmark_list=face_landmarks,
            connections=self.mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mp_drawing_styles
                .get_default_face_mesh_tesselation_style()
        )
        self.mp_drawing.draw_landmarks(
            image=annotated,
            landmark_list=face_landmarks,
            connections=self.mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mp_drawing_styles
                .get_default_face_mesh_contours_style()
        )
        annotated_path = str(RESULTS_DIR / "mediapipe_annotated.jpg")
        cv2.imwrite(annotated_path, annotated)
        print(f"[MediaPipe] Saved annotated image → {annotated_path}")

        # ── Save clean mesh visualization (black background) ──
        mesh_canvas = np.zeros_like(image)
        self.mp_drawing.draw_landmarks(
            image=mesh_canvas,
            landmark_list=face_landmarks,
            connections=self.mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_face_mesh_tesselation_style()
        )
        mesh_path = str(RESULTS_DIR / "mediapipe_mesh.jpg")
        cv2.imwrite(mesh_path, mesh_canvas)
        print(f"[MediaPipe] Saved mesh visualization → {mesh_path}")

        # ── Save landmarks as JSON for downstream use ──
        landmarks_path = str(RESULTS_DIR / "mediapipe_landmarks.json")
        with open(landmarks_path, "w") as f:
            json.dump({"landmarks": landmarks, "image_size": [w, h]}, f, indent=2)

        return {
            "success": True,
            "landmarks": landmarks,
            "landmark_count": len(landmarks),
            "annotated_image_path": annotated_path,
            "mesh_image_path": mesh_path,
            "landmarks_json_path": landmarks_path,
            "processing_time_ms": elapsed,
            "image_size": (w, h)
        }


# ─────────────────────────────────────────────
# PHASE 2: FACE-ON-BODY COMPOSITE
# ─────────────────────────────────────────────

class BodyCompositor:
    """
    Places the reconstructed face onto a generic body template.
    This simulates the 'face → avatar' step that tools like Catches
    and Avaturn perform, so we can evaluate the identity fidelity gap.
    """

    def __init__(self):
        self.body_template_path = str(RESULTS_DIR / "body_template.png")
        self._ensure_body_template()

    def _ensure_body_template(self):
        """
        Creates a simple body silhouette template if none exists.
        In a real pipeline you'd replace this with a proper 3D avatar render.
        """
        if Path(self.body_template_path).exists():
            return

        print("[Compositor] Creating body template...")

        # 400x700 canvas — neutral grey background
        W, H = 400, 700
        img = Image.new("RGB", (W, H), color=(230, 230, 230))
        draw = ImageDraw.Draw(img)

        # Simple body silhouette
        # Head placeholder (will be replaced by actual face)
        head_cx, head_cy, head_r = W // 2, 110, 70
        draw.ellipse(
            [head_cx - head_r, head_cy - head_r,
             head_cx + head_r, head_cy + head_r],
            fill=(200, 190, 185), outline=(150, 140, 135), width=2
        )

        # Neck
        draw.rectangle([W//2 - 20, head_cy + head_r - 5, W//2 + 20, head_cy + head_r + 40],
                        fill=(200, 190, 185))

        # Torso
        draw.rectangle([W//2 - 80, 180, W//2 + 80, 420],
                        fill=(100, 110, 140), outline=(80, 90, 120), width=2)

        # Left arm
        draw.rectangle([W//2 - 80 - 30, 180, W//2 - 80, 380],
                        fill=(100, 110, 140), outline=(80, 90, 120), width=2)

        # Right arm
        draw.rectangle([W//2 + 80, 180, W//2 + 80 + 30, 380],
                        fill=(100, 110, 140), outline=(80, 90, 120), width=2)

        # Left leg
        draw.rectangle([W//2 - 70, 420, W//2 - 10, 660],
                        fill=(60, 60, 80), outline=(40, 40, 60), width=2)

        # Right leg
        draw.rectangle([W//2 + 10, 420, W//2 + 70, 660],
                        fill=(60, 60, 80), outline=(40, 40, 60), width=2)

        # Label
        draw.text((10, H - 20), "Body Template v1.0", fill=(150, 150, 150))

        img.save(self.body_template_path)
        print(f"[Compositor] Body template saved → {self.body_template_path}")

    def composite(self, photo_path: str, landmarks: list, image_size: tuple) -> dict:
        """
        Crops the face from the input photo and composites it
        onto the body template.

        Args:
            photo_path: Path to original input photo
            landmarks:  MediaPipe landmarks (normalized 0-1)
            image_size: (width, height) of original photo

        Returns:
            dict with composite image path and metadata
        """
        print(f"\n[Compositor] Building face-on-body composite...")

        original = Image.open(photo_path).convert("RGB")
        orig_w, orig_h = image_size

        # ── Detect face bounding box from landmarks ──
        xs = [lm[0] * orig_w for lm in landmarks]
        ys = [lm[1] * orig_h for lm in landmarks]

        padding = 0.15  # add 15% padding around detected face
        x_min = max(0, int(min(xs) - padding * (max(xs) - min(xs))))
        x_max = min(orig_w, int(max(xs) + padding * (max(xs) - min(xs))))
        y_min = max(0, int(min(ys) - padding * (max(ys) - min(ys))))
        y_max = min(orig_h, int(max(ys) + padding * (max(ys) - min(ys))))

        face_crop = original.crop((x_min, y_min, x_max, y_max))
        face_w = x_max - x_min
        face_h = y_max - y_min
        print(f"[Compositor] Face crop: {face_w}x{face_h}px")

        # ── Load body template ──
        body = Image.open(self.body_template_path).convert("RGB")
        body_w, body_h = body.size

        # ── Scale face to fit the head area on the template (140px diameter) ──
        target_face_size = 140
        scale = target_face_size / max(face_w, face_h)
        new_fw = int(face_w * scale)
        new_fh = int(face_h * scale)
        face_resized = face_crop.resize((new_fw, new_fh), Image.LANCZOS)

        # ── Create circular mask for natural face blending ──
        mask = Image.new("L", (new_fw, new_fh), 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.ellipse([0, 0, new_fw, new_fh], fill=255)

        # Apply feathering by blurring the mask edges
        from PIL import ImageFilter
        mask = mask.filter(ImageFilter.GaussianBlur(radius=8))

        # ── Paste face onto body at head position ──
        head_cx = body_w // 2
        head_cy = 110  # matches template
        paste_x = head_cx - new_fw // 2
        paste_y = head_cy - new_fh // 2

        body.paste(face_resized, (paste_x, paste_y), mask)

        # ── Save composite ──
        composite_path = str(RESULTS_DIR / "composite_avatar.jpg")
        body.save(composite_path, quality=95)
        print(f"[Compositor] Composite saved → {composite_path}")

        return {
            "success": True,
            "composite_path": composite_path,
            "face_crop_size": (face_w, face_h),
            "body_template_size": (body_w, body_h),
            "face_placement": (paste_x, paste_y)
        }


# ─────────────────────────────────────────────
# PHASE 3: AVATURN API (optional, needs API key)
# ─────────────────────────────────────────────

class AvaturnGenerator:
    """
    Calls the Avaturn API to generate a full-body photorealistic avatar.
    Requires a free API key from avaturn.me

    This represents the most advanced end of the pipeline —
    a production-grade avatar platform — which we compare against
    our MediaPipe + compositor approach to evaluate the identity fidelity gap.
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.available = api_key is not None

    def generate(self, photo_path: str) -> dict:
        """
        Submit a photo to Avaturn and retrieve the generated avatar.

        Args:
            photo_path: Path to input photo

        Returns:
            dict with avatar URL or local path, and metadata
        """
        if not self.available:
            print("\n[Avaturn] No API key provided — skipping Avaturn generation.")
            print("[Avaturn] Get a free key at: https://avaturn.me")
            print("[Avaturn] Then run: python pipeline.py --photo photo.jpg --avaturn-key YOUR_KEY")
            return {
                "success": False,
                "skipped": True,
                "reason": "No API key. Get one free at avaturn.me"
            }

        print(f"\n[Avaturn] Submitting photo to Avaturn API...")

        with open(photo_path, "rb") as f:
            photo_bytes = f.read()

        import base64
        photo_b64 = base64.b64encode(photo_bytes).decode("utf-8")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "photo": photo_b64,
            "export_format": "glb",       # 3D model format
            "body_type": "fullbody",
            "quality": "high"
        }

        try:
            import requests
            response = requests.post(
                AVATURN_API_URL,
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            data = response.json()

            avatar_url = data.get("avatar_url") or data.get("url")
            avatar_id = data.get("id") or data.get("avatar_id")

            print(f"[Avaturn] Avatar generated! ID: {avatar_id}")
            print(f"[Avaturn] Download URL: {avatar_url}")

            # Save the avatar URL for download
            result_path = str(RESULTS_DIR / "avaturn_result.json")
            with open(result_path, "w") as f:
                json.dump(data, f, indent=2)

            return {
                "success": True,
                "avatar_id": avatar_id,
                "avatar_url": avatar_url,
                "result_json_path": result_path,
                "raw_response": data
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "note": "Check your API key and that avaturn.me is accessible"
            }


# ─────────────────────────────────────────────
# EVALUATION REPORT
# ─────────────────────────────────────────────

def generate_evaluation_report(mediapipe_result: dict, compositor_result: dict,
                                 avaturn_result: dict, photo_path: str):
    """
    Generates a side-by-side visual comparison report and saves it
    as a PNG. This is what you'll use in your paper's Framework Evaluation section.
    """
    print("\n[Report] Generating evaluation report...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 7))
    fig.suptitle("No Stylist — Framework Evaluation Report", fontsize=16, fontweight="bold", y=1.02)

    # ── Panel 1: Original photo ──
    original = Image.open(photo_path)
    axes[0].imshow(np.array(original))
    axes[0].set_title("Input Photo\n(Original)", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    # ── Panel 2: MediaPipe mesh overlay ──
    if mediapipe_result.get("success"):
        mesh_img = Image.open(mediapipe_result["annotated_image_path"])
        mesh_img_rgb = cv2.cvtColor(np.array(mesh_img), cv2.COLOR_BGR2RGB)
        axes[1].imshow(mesh_img_rgb)

        # Stats annotation
        stats = (
            f"Landmarks: {mediapipe_result['landmark_count']}\n"
            f"Time: {mediapipe_result['processing_time_ms']:.0f}ms\n"
            f"Framework: MediaPipe FaceMesh\n"
            f"Identity fidelity: Technical"
        )
        axes[1].set_title("Phase 1: MediaPipe Reconstruction\n(Face Mesh)", fontsize=12, fontweight="bold")
        axes[1].text(0.02, 0.02, stats, transform=axes[1].transAxes,
                     fontsize=8, verticalalignment='bottom',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        axes[1].text(0.5, 0.5, f"MediaPipe failed:\n{mediapipe_result.get('error', 'Unknown')}",
                     ha='center', va='center', transform=axes[1].transAxes, fontsize=10)
    axes[1].axis("off")

    # ── Panel 3: Composite avatar ──
    if compositor_result.get("success"):
        composite_img = Image.open(compositor_result["composite_path"])
        axes[2].imshow(np.array(composite_img))
        axes[2].set_title("Phase 2: Face-on-Body Composite\n(Avatar Output)", fontsize=12, fontweight="bold")

        notes = "Body: Generic template\nFace: Cropped from landmarks\nBlend: Circular mask + feather"
        axes[2].text(0.02, 0.02, notes, transform=axes[2].transAxes,
                     fontsize=8, verticalalignment='bottom',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        axes[2].text(0.5, 0.5, "Composite failed",
                     ha='center', va='center', transform=axes[2].transAxes, fontsize=10)
    axes[2].axis("off")

    # ── Avaturn status banner ──
    if avaturn_result.get("skipped"):
        fig.text(0.5, -0.02,
                 "Phase 3 (Avaturn API) not run — add --avaturn-key YOUR_KEY to enable",
                 ha='center', fontsize=9, color='gray', style='italic')
    elif avaturn_result.get("success"):
        fig.text(0.5, -0.02,
                 f"Phase 3 (Avaturn): Avatar generated — ID {avaturn_result.get('avatar_id')}",
                 ha='center', fontsize=9, color='green')

    plt.tight_layout()
    report_path = str(RESULTS_DIR / "evaluation_report.png")
    plt.savefig(report_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Report] Saved → {report_path}")
    return report_path


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="No Stylist Avatar Foundation Pipeline"
    )
    parser.add_argument(
        "--photo", required=True,
        help="Path to your input photo (JPG or PNG)"
    )
    parser.add_argument(
        "--avaturn-key", default=None,
        help="Optional: Avaturn API key (free at avaturn.me)"
    )
    args = parser.parse_args()

    if not Path(args.photo).exists():
        print(f"Error: Photo not found at {args.photo}")
        sys.exit(1)

    print("=" * 55)
    print("  No Stylist — Avatar Foundation Pipeline")
    print("=" * 55)

    # Phase 1: MediaPipe face reconstruction
    reconstructor = FaceMeshReconstructor()
    mediapipe_result = reconstructor.reconstruct(args.photo)

    # Phase 2: Face-on-body composite
    compositor_result = {"success": False}
    if mediapipe_result["success"]:
        compositor = BodyCompositor()
        compositor_result = compositor.composite(
            args.photo,
            mediapipe_result["landmarks"],
            mediapipe_result["image_size"]
        )

    # Phase 3: Avaturn (optional)
    avaturn = AvaturnGenerator(api_key=args.avaturn_key)
    avaturn_result = avaturn.generate(args.photo)

    # Generate evaluation report
    if mediapipe_result["success"]:
        report_path = generate_evaluation_report(
            mediapipe_result, compositor_result,
            avaturn_result, args.photo
        )

    # Print summary
    print("\n" + "=" * 55)
    print("  PIPELINE COMPLETE — RESULTS SUMMARY")
    print("=" * 55)
    print(f"  MediaPipe:   {'✓' if mediapipe_result['success'] else '✗'}", end="")
    if mediapipe_result["success"]:
        print(f"  ({mediapipe_result['landmark_count']} landmarks, "
              f"{mediapipe_result['processing_time_ms']:.0f}ms)")
    else:
        print(f"  {mediapipe_result.get('error')}")

    print(f"  Compositor:  {'✓' if compositor_result.get('success') else '✗'}")
    print(f"  Avaturn:     {'skipped (no key)' if avaturn_result.get('skipped') else ('✓' if avaturn_result.get('success') else '✗')}")

    if mediapipe_result["success"]:
        print(f"\n  Report saved → {report_path}")
    print("=" * 55)


if __name__ == "__main__":
    main()
