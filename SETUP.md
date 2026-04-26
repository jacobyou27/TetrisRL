# SETUP.md

This file explains how to install and run the final solo project submission.

## 1. Recommended Python Version

Use **Python 3.11**.

This project was developed on Windows in a virtual environment, but the code is standard Python and should also run on macOS or Linux with the same dependency versions.

## 2. Create and activate a virtual environment

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Verify installation

Run the smoke test:

```bash
python tests/smoke_test_placement_env.py
```

If that succeeds, the environment and core placement logic are installed correctly.

## 5. Main entry points

### Train / tune GA weights

```bash
python -m src.agents.tune_heuristic_ga --population 24 --elite 4 --generations 24 --episodes-per-individual 4 --max-steps 300 --lookahead-depth 0 --workers 12 --verbosity 2 --plot --save-name ga_overnight_l0_v3
```

### Evaluate saved weights with lookahead

```bash
python -m src.agents.heuristic_agent --weights-json runs/ga/ga_overnight_l0_v3/best_weights.json --lookahead-depth 1 --lookahead-weight 1.0 --episodes 20 --max-steps 1000 --plot
```

### Render one episode

```bash
python -m src.agents.heuristic_agent --weights-json runs/ga/ga_overnight_l0_v3/best_weights.json --lookahead-depth 1 --lookahead-weight 1.0 --episodes 1 --max-steps 1000 --render --delay-ms 60
```

### Run simulator benchmark

```bash
python -m scripts.benchmark_simulator
```

### Run simulator profiler

```bash
python scripts/profile_simulator.py
```

## 6. Output locations

The project writes generated outputs under `runs/`, including:

- GA history and checkpoints,
- best saved weights,
- plots,
- and evaluation summaries.

These are generated artifacts and are not required as source code for the submission.

## 7. Troubleshooting

### OpenCV window does not appear

If rendering fails or no window appears:
- make sure `opencv-python` is installed,
- do not use a headless environment,
- or skip rendering and use `--plot` / saved outputs instead.

### GA appears slow or stalls on lookahead

This is expected for one-step lookahead. Benchmarking showed that greedy evaluation is much faster than one-piece lookahead, so the recommended workflow is:
- tune with `--lookahead-depth 0`, then
- evaluate the saved best weights with `--lookahead-depth 1`.

## 8. Final submission notes

For the final cleaned repo:
- remove `__pycache__/` folders and `.pyc` files,
- do not commit large generated outputs from `runs/`,
- and keep only the final source, tests, scripts, and documentation files needed for the project.
