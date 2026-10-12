# Camouflage Vision — Camouflaged Animal Segmentation

A Python-based AI image-segmentation project using a U-Net model with a ResNet18 encoder to predict pixel-level classes in camouflaged-environment images. The project includes a Flask website for image upload, segmentation-mask visualization, overlay display, class-wise pixel statistics, and browser camera capture.

> **Model quality note:** The project previously reported a test IoU of approximately `0.256`. Inspect prediction outputs carefully before claiming high accuracy.

## Features

- U-Net segmentation model with a ResNet18 encoder.
- Four segmentation classes:
  1. Animal
  2. Masking background
  3. Non-masking background
  4. Foreground attention
- Flask website with image upload and prediction.
- Original image, predicted mask, and blended overlay.
- Class-wise predicted pixel percentages.
- Download controls for generated outputs, where implemented.
- Browser camera capture and prediction, depending on the current implementation.
- Team page with member profiles, roles, contributions, and photos.
- Responsive dark navy interface with cyan and purple accents.

## Technology Stack

- Python, PyTorch
- `segmentation-models-pytorch`
- U-Net / ResNet18
- Flask
- HTML, CSS, JavaScript
- NumPy, Pillow
- Albumentations, Matplotlib

## Project Structure

```text
animal/
├── !Dataset/
│   └── Dataset1/
│       ├── Train/
│       ├── Valid/
│       └── Test/
├── !Results/
│   └── <training-run>/
│       └── model.pth
├── backend/
│   ├── app.py
│   ├── static/
│   │   ├── css/style.css
│   │   ├── js/site.js
│   │   └── js/studio.js
│   └── templates/
│       ├── about.html
│       ├── base.html
│       ├── detect.html
│       ├── home.html
│       └── team.html
├── config/
├── data_model/
├── utils/
├── main.py
├── run_visualization.py
├── requirements.txt
└── README.md
```

Your project may contain additional files. Keep the dataset and model checkpoint in the paths expected by your configuration.

## Installation

Run commands from the project root—the folder containing `main.py` and `requirements.txt`.

### 1. (Recommended) Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use Command Prompt:

```bat
.venv\Scripts\activate.bat
```

### 2. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install flask
```

Use Python and PyTorch versions compatible with the project's dependencies. CPU inference works but may be slow.

## Configure the Model Checkpoint

Open `backend/app.py` and check the `CHECKPOINT` variable. Example:

```python
CHECKPOINT = PROJECT_ROOT / "!Results" / "11-10_18-08-56" / "model.pth"
```

This is only an example path. Change it to the actual checkpoint folder on your computer. Confirm `model.pth` exists before starting the website.

The model architecture, number and ordering of classes, color mapping, and preprocessing in the Flask app must match training. Incorrect resizing or normalization can produce misleading predictions.

## Run the Website

From the project root:

```powershell
python backend/app.py
```

Open **http://127.0.0.1:5000** in your browser. Keep the terminal open while using the site. Press `Ctrl+C` to stop Flask.

## Use the Website

1. Open the AI Detection page.
2. Choose an image and submit it for prediction.
3. Inspect the original image, predicted mask, and overlay.
4. Review the class-wise predicted pixel percentages.
5. Download results if the relevant controls are available.
6. For camera capture, grant browser camera permission, start the camera, capture a frame, and submit it for prediction.
7. Visit the Team page to view profiles and contributions.

Camera access requires browser permission. On a local machine, use `localhost` or `127.0.0.1`; remote camera access generally requires HTTPS.

## Add Team Photos

Place team photos in:

```text
backend/static/images/team/
```

For example:

```text
backend/static/images/team/image1.png
backend/static/images/team/image2.png
backend/static/images/team/image3.png
backend/static/images/team/image4.png
```

The filenames must exactly match the `photo` values in `TEAM_MEMBERS` in `backend/app.py`. If a file is named `.png`, do not configure it as `.jpg`. Ensure the folder is named `images` (plural) and contains a separate `team` folder.

Update each member's name, USN, role, and contribution in `TEAM_MEMBERS` before your final presentation.

## Train the Model

Training is handled by `main.py`. Before starting, verify dataset paths, image-mask pairs, class colors, and device configuration.

```powershell
python main.py
```

Training may take a long time on CPU. Keep the laptop awake and connected to power during long training runs. Checkpoints and run configurations are saved according to the current project configuration.

## Generate Prediction Visualizations

Run from the project root:

```powershell
python run_visualization.py
```

The script loads a trained checkpoint and saves result images in `test_result`, according to the current implementation. Open that folder in VS Code or File Explorer to inspect the images.

## Troubleshooting

### Model checkpoint not found
- Check the `CHECKPOINT` path in `backend/app.py`.
- Confirm the file exists and matches the model architecture.

### Team photos show 404 errors
- Confirm the path is `backend/static/images/team/`.
- Match every `photo` filename and extension exactly.
- Refresh with `Ctrl+F5`.

### Predictions look incorrect
- Confirm preprocessing matches training (resize, normalization, RGB channel order).
- Verify class order and colors match the dataset.
- Inspect multiple test images. Pixel percentages are predicted class coverage, not confidence scores.

### Camera does not start
- Allow camera permission in the browser.
- Check that another application is not using the camera.
- Review browser console errors.

### Slow inference or memory errors
- CPU inference can be slow. Close other resource-heavy programs.
- Do not change the model architecture when loading a checkpoint trained with the existing architecture.

## Limitations

- Quality depends on dataset size, annotation quality, class balance, and training settings.
- The model is not guaranteed to detect every camouflaged animal.
- Class percentages describe predicted pixels; they are not confidence estimates.
- Camera operation may capture and predict one frame at a time rather than perform continuous real-time segmentation.
- Test the website locally before deployment; production hosting and security hardening are not covered here.

## Future Improvements

- Improve dataset coverage and annotation quality.
- Evaluate per-class IoU and inspect predictions.
- Experiment with augmentation, class weights, and encoder settings.
- Improve inference speed and error handling.
- Add automated tests and deployment configuration.

## Team Members

The current team page lists:

- Subiya Anjum — Model Development
- Saniya Tahseen — Backend & Integration
- Sonu — Frontend & UX
- Suhana — Testing & Documentation

Replace placeholder USNs in `backend/app.py` with the correct values before submission.

## Academic Use

This project is intended for educational and demonstration purposes. Check the licensing of the dataset and cite original repositories, pretrained models, datasets, and reused code in your final report.
