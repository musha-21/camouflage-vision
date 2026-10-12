# Camouflage Vision — Premium Flask website

This is a Python/Flask website; no React or Node.js is required.

## Install into your existing project
Copy/merge the included `backend` folder into your existing `animal` project, next to `config/`, `utils/`, `data_model/`, `main.py`, and `!Results/`.

## Run
From the existing project root in PowerShell:
```powershell
python -m pip install flask
python backend/app.py
```
Open http://127.0.0.1:5000

## Model checkpoint
By default `backend/app.py` expects:
`!Results/11-10_18-08-56/model.pth`
Edit `CHECKPOINT` near the top of `backend/app.py` if you want a different checkpoint.

## Team details
Edit the `TEAM_MEMBERS` list near the top of `backend/app.py`. Replace sample names, USNs, roles, initials, and contributions with your actual team details.

## Important: match training preprocessing
The example uses resize to 544×544 and ImageNet normalization. For meaningful predictions, ensure this exactly matches the preprocessing used in `data_model/dataset_class.py` and `utils/augmentation.py`. If training uses letterboxing or a different normalization, update `preprocess_image()` before relying on predictions.

## Live camera
Camera access works on localhost in modern browsers. Press Start camera, then Capture & predict. Each capture sends one frame to the Flask API; this is snapshot-based inference, not continuous real-time video segmentation.

## Security
This is a local project demo. Add authentication, request validation, and deployment security before exposing it publicly.
