# CARA Web UI

This adds an iPad-friendly web frontend while keeping desktop `app.py` intact.

## Run

```bash
cd "1.5: Python Version"
pip install -r webui/requirements-web.txt
python3 webui/main.py
```

Then open:

- On Mac: `http://127.0.0.1:8000`
- On iPad (same Wi-Fi): `http://<MAC_LOCAL_IP>:8000`

## What it does

- **Old Algorithm**: upload images and run local `countCFU` processing.
- **CPSAM Script**: upload images and generate a terminal script for ORCD SSH/sbatch workflow.

## Notes 

- Web jobs are stored in `1.5: Python Version/web_jobs/`.
- Desktop `app.py` continues to run locally as before.
