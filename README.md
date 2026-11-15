# Group 90 — XNLI BERT Variant Comparison

## Group members
- 2025AG05424 — Shreeja Mahapatra
- 2025AG05393 — Hari Krishna Moorthy T
- 2025AG05390 — Naveen Ranjolkar

## Files
- `Group90_xnli.ipynb`: full experiment notebook
- `requirements.txt`: Python dependencies
- `results/`: generated CSVs, plots, and run metadata (populated after execution)
- `model_outputs/`: fine-tuned checkpoints (created after execution; not included in the submission ZIP)

## Run instructions
1. Use Python 3.10 or 3.11 and preferably a CUDA-enabled GPU.
2. Open a terminal in this folder.
3. Install dependencies: `pip install -r requirements.txt`
4. Launch Jupyter: `jupyter notebook`
5. Open `Group90_xnli.ipynb` and select **Kernel → Restart & Run All**.
6. Wait for both models to fine-tune and for all language evaluations to finish.
7. Confirm the notebook contains executed outputs and that the `results/` directory contains generated metrics and plots before final submission.

The notebook defaults to the complete English training split and complete multilingual test split. For a quick pilot, reduce `MAX_TRAIN_SAMPLES` and `MAX_EVAL_SAMPLES_PER_LANGUAGE` in Section 4, but disclose the reduced sample sizes in the report.

Important: The notebook is supplied with executable code, but model training and benchmark metrics have not been run in this file-generation environment. Do not submit it as an executed-results notebook until you run all cells successfully and inspect the resulting outputs.
