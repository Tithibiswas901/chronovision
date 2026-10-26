---
title: ChronoVision
emoji: 🛰️
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: "1.66.1"
app_file: app.py
pinned: false
---

# ChronoVision

**Unsupervised change detection on bi-temporal satellite images.**

ChronoVision detects surface changes between two satellite images of the
same location captured at different times, without any labeled training
data. It implements three classical unsupervised change-detection methods,
evaluates them against ground-truth masks from the LEVIR-CD dataset, and
provides an interactive Streamlit app for exploring results.

## Methods

| Method | Idea |
|---|---|
| **Image Difference + Otsu** | Absolute grayscale difference between the two images, thresholded automatically with Otsu's method. |
| **Change Vector Analysis (CVA)** | Euclidean magnitude of the per-pixel difference vector across all 3 RGB channels, thresholded with Otsu's method. |
| **PCA + K-Means (Celik, 2009)** | Builds a difference image, learns a PCA feature space from local 4x4 blocks, projects overlapping neighborhoods into that space, and clusters the resulting feature vectors into "changed" / "unchanged" with K-Means (k=2). This is the primary method used in this project. |

All three methods are followed by a morphological opening + closing pass
(`src/postprocessing.py`) to remove speckle noise and fill small holes in
the predicted change map.

## Dataset

This project uses a subset of **LEVIR-CD**, a public building change
detection dataset for remote sensing (CC0 license). The full dataset can
be obtained from:

- https://justchenhao.github.io/LEVIR/

### Expected layout

Place your LEVIR-CD subset (or any compatible bi-temporal dataset) under
`data/`, with three folders where filenames match across all three:

```
data/
  A/      # image set 1, e.g. test_1.png, test_2.png, ...
  B/      # image set 2, filenames matching A/
  label/  # binary ground-truth change masks, filenames matching A/
```

> **Note on A/B naming:** in this LEVIR-CD mirror, folder `A` is actually
> the *post-change* image and folder `B` the *pre-change* image. None of
> the change-detection math in this project depends on temporal order
> (difference, CVA magnitude, and the PCA feature space are all symmetric
> in A and B), so this has no effect on results. To avoid displaying a
> misleading "before/after" label, the app and figures refer to the two
> inputs neutrally as **"Image 1 (A)"** and **"Image 2 (B)"**.

A small subset (25 pairs) is included directly in this repo under `data/`
so the evaluation and the Streamlit app's sample picker work out of the
box. To use the full dataset instead, replace the contents of `data/A`,
`data/B`, and `data/label` with your own LEVIR-CD split.

## Setup

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Running the evaluation

With your `data/` folder populated as described above:

```bash
python evaluate.py
```

This will:

1. Loop over every matching (A, B, label) filename triple in `data/`.
2. Preprocess each pair (resize to 256x256, histogram-match B onto A).
3. Run all three methods + postprocessing.
4. Compute Precision, Recall, F1, IoU, Overall Accuracy, and Cohen's
   Kappa against the ground-truth label for every (pair, method)
   combination.
5. Write:
   - `results.csv` -- per-pair, per-method metrics.
   - `results_summary.csv` -- mean of each metric, grouped by method.
   - `figures/metrics_bar.png` -- grouped bar chart comparing F1 / IoU / Kappa across methods.
   - `figures/qualitative.png` -- visual comparison grid for 4 sample pairs.
6. Print the summary table to the console.

## Running the interactive app

```bash
streamlit run app.py
```

In the app you can:

- Pick a sample pair from `data/` (metrics are shown automatically if a
  matching ground-truth label exists), or upload your own two images.
- Choose which of the three methods to run.
- View Image 1 (A) and Image 2 (B) side by side, the binary change map,
  a red overlay of detected change on Image 1, and the percent of area
  changed.

## Deploying to Hugging Face Spaces

This repo is already configured for Spaces via the YAML block at the very
top of this file (`sdk: streamlit`, `app_file: app.py`) -- GitHub hides
that block when rendering this page, so it only takes effect on Spaces.
`requirements.txt` also uses `opencv-python-headless` so it runs on
Spaces' Linux containers without extra system packages (regular
`opencv-python` needs `libGL`, which isn't installed there by default).

1. Create a free account at https://huggingface.co if you don't have one.
2. Click **"New Space"** (top-right profile menu, or
   https://huggingface.co/new-space).
3. Set **SDK** to **Streamlit**, pick a name (e.g. `chronovision`), choose
   visibility, and create it. This gives you a git remote URL like
   `https://huggingface.co/spaces/<your-username>/chronovision`.
4. Add it as a git remote and push this repo's `main` branch to it:
   ```bash
   git remote add space https://huggingface.co/spaces/<your-username>/chronovision
   git push space main
   ```
   (The first push will prompt for your Hugging Face username and an
   access token as the password -- create one at
   https://huggingface.co/settings/tokens with "write" access.)
5. The Space builds automatically from `requirements.txt` and serves
   `app.py`. Any later `git push space main` redeploys it.

Since `data/` is tracked in this repo, the sample picker and
ground-truth metrics work immediately on the deployed Space.

## Deploying to Streamlit Community Cloud

Alternatively, the same repo deploys just as easily to Streamlit
Community Cloud:

1. Push this repository to GitHub (including `data/`, which is already
   tracked).
2. Go to https://share.streamlit.io and sign in with your GitHub account.
3. Click **"New app"**, select this repository and the `main` branch.
4. Set **Main file path** to `app.py`.
5. Click **Deploy**. The first build takes a few minutes while it
   installs `requirements.txt`; after that you'll get a public URL for
   the app.

Any time you push new commits to `main`, the deployed app redeploys
automatically.

## Project structure

```
src/
  preprocessing.py   # loading, resizing, grayscale conversion, histogram matching
  methods.py         # the three change-detection algorithms
  postprocessing.py  # morphological opening + closing
  metrics.py         # Precision, Recall, F1, IoU, Accuracy, Kappa
evaluate.py          # batch evaluation over data/, produces CSVs + figures
app.py                # Streamlit demo app
requirements.txt
```
