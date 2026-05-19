# No Stylist — Avatar Foundation Pipeline

This is the code for my DIGS 20006 final project at the University of Chicago.

The project builds a Python pipeline that takes a photo of your face and runs it
through AI frameworks to evaluate how well current technology can generate an
identity-accurate avatar — the foundation layer of a product idea called No Stylist.

---

## What This Does (In Plain English)

You give the program a selfie. It does two things:

1. **Detects 478 points on your face** using Google's MediaPipe AI — things like
   the corners of your eyes, the tip of your nose, your jawline, your lips. It draws
   a mesh connecting all those points and saves it as an image.

2. **Crops your face and places it on a body template** — simulating the same
   approach that production tools like Avaturn and Catches use under the hood.

The output is a side-by-side evaluation report showing your original photo,
the face mesh, and the composite avatar — which together illustrate the
"identity fidelity gap" that the paper argues about.

---

## Before You Start — What You Need

You need two things installed on your Mac before anything else.

### 1. Python 3.10
Check if you have it by opening Terminal and typing:
```
python3 --version
```
If it says Python 3.10.x you are good. If not, download it from:
https://www.python.org/downloads/

### 2. VS Code (optional but recommended)
Download from: https://code.visualstudio.com
This is just a text editor that makes it easy to see your files and run the terminal.

---

## How To Open The Terminal

The Terminal is where you type commands to run the code. Think of it like
texting instructions directly to your computer.

**Option A — Inside VS Code:**
Open VS Code, open your project folder, go to the top menu, click Terminal,
then click New Terminal. A panel appears at the bottom. That is where you type.

**Option B — Mac Terminal app:**
Press Command + Space, type Terminal, hit Enter.
Then navigate to your project folder by typing:
```
cd /Users/yanntanyi/Desktop/DIGS20006/Final-Project
```

---

## Setup — Do This Once

Follow these steps in order. Copy and paste each command into the terminal
exactly as written. Do not skip any steps.

**Step 1 — Go to your project folder:**
```
cd /Users/yanntanyi/Desktop/DIGS20006/Final-Project
```

**Step 2 — Create a virtual environment:**

A virtual environment is like a clean isolated box for this project's code.
It prevents conflicts with other Python stuff on your computer.
```
python3 -m venv venv
```

**Step 3 — Activate the virtual environment:**

You will know it worked when you see (venv) appear at the start of your
terminal line. You need to do this every time you open a new terminal window.
```
source venv/bin/activate
```

**Step 4 — Install the required libraries:**

This downloads everything the code needs to run. Takes about 1-2 minutes.
```
pip install "numpy<2" mediapipe==0.10.9 opencv-python Pillow matplotlib requests
```

You only need to do Steps 1-4 once ever. After that, every time you come back,
just do Step 1 and Step 3 to activate the environment and you are ready.

---

## Adding Your Photos

1. Inside your Final-Project folder there is a folder called photos/
2. Drag any selfie JPG or PNG file into that folder
3. Make sure the filename has no spaces — use something like photo1.JPG

Tips for best results:
- Face clearly visible, looking roughly at the camera
- Good lighting (near a window works great)
- Not too far away — your face should fill a decent portion of the frame

---

## Running The Pipeline

Make sure your virtual environment is active — you should see (venv) at the
start of your terminal line. If you do not see it, run:
```
source venv/bin/activate
```

**Run on a single photo:**
```
python3 pipeline.py --photo photos/photo1.JPG
```
Replace photo1.JPG with whatever your photo file is actually named.

**Run on multiple photos for comparison:**
```
python3 evaluate.py --photos photos/photo1.JPG photos/photo2.JPG photos/photo3.JPG
```

---

## Where To Find Your Results

After running the pipeline, a results/ folder will appear automatically inside
your project folder. Open it in Finder or click it in VS Code's file explorer.

| File | What It Is |
|------|------------|
| mediapipe_annotated.jpg | Your selfie with the AI face mesh drawn on top |
| mediapipe_mesh.jpg | Just the face mesh on a black background |
| mediapipe_landmarks.json | The raw data — all 478 landmark coordinates |
| composite_avatar.jpg | Your face placed onto the body template |
| evaluation_report.png | All three side by side in one image |
| multi_photo_comparison.png | Comparison grid across multiple photos |
| evaluation_metrics.json | Numerical scores for each photo |

---

## If Something Goes Wrong

**"No face detected" error:**
Your photo might be too dark, too far away, or at a sharp angle.
Try a clearer, well-lit photo taken straight on.

**"Module not found" error:**
Your virtual environment is probably not active.
Run this and try again:
```
source venv/bin/activate
```

**"Photo not found" error:**
The filename is case sensitive. photo1.jpg and photo1.JPG are different.
Use exactly the name of your file including the capital letters.

**Any other error:**
Make sure you are in the right folder first. Run:
```
cd /Users/yanntanyi/Desktop/DIGS20006/Final-Project
```
Then activate the venv and try again.

---

## Project File Structure

Here is what every file in this project does:

```
Final-Project/
│
├── pipeline.py          ← The main script. Run this on a single photo.
├── evaluate.py          ← Run this to compare multiple photos side by side.
├── requirements.txt     ← List of libraries the code needs.
├── README.md            ← This file.
│
├── photos/              ← Put your selfies in here.
│   └── photo1.JPG
│
├── results/             ← All outputs appear here after you run the code.
│   ├── mediapipe_annotated.jpg
│   ├── mediapipe_mesh.jpg
│   ├── composite_avatar.jpg
│   ├── evaluation_report.png
│   └── ...
│
├── venv/                ← The virtual environment. Do not touch this folder.
│
└── 3DDFA-V3/            ← Cloned repo from the attempted 3D reconstruction
                            phase. Not used in the final pipeline due to
                            NVIDIA GPU requirements on Mac.
```

---

## What The Project Is Actually Testing

The core question this pipeline investigates:

Can current AI systems produce an avatar that a user looks at
and genuinely feels represents them?

The pipeline tests this at three levels:

**1. Detection**
Can AI accurately map a human face from a single photo?
MediaPipe handles this — and the answer is yes, impressively.
478 landmarks in 93 milliseconds on a regular laptop.

**2. Rendering**
Can that face data be placed on a body in a way that looks right?
The composite avatar output shows the ceiling of this approach.

**3. Identity**
Does the result actually feel like you?
This is where everything currently falls short — and that gap is the finding.

---

## Built With

- Python 3.10
- MediaPipe 0.10.9 (Google)
- OpenCV
- Pillow
- Matplotlib
- NumPy

---

DIGS 20006 — University of Chicago — Spring 2026
Yannick Tanyi
