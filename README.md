# Mentalytics AMM Dashboard

Mentalytics AMM Dashboard is a Streamlit-based demo interface for visualizing structured Artificial Mental Model (AMM) information in rehabilitation exercise assessment.

The dashboard loads structured patient records from a proxy dataset and displays doctor-facing visualizations such as perceived versus actual exercise difficulty, exercise suitability, psychological AMM dimensions, and personality trait radar charts.

## Features

- Patient profile loading from structured JSON records
- Perceived vs actual exercise difficulty visualization
- Exercise comparison chart
- Llama-generated exercise suitability matrix
- Llama-generated Big Five personality scores for AMM visualization
- ICF/BPS-style psychological domain profile
- Personality trait radar chart
- Synthetic proxy dataset for reproducible demo use

## Repository structure

```text
mentalytics-amm-dashboard/
│
├── README.md
├── pyproject.toml
├── .gitignore
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   └── proxy_dataset.json
│
├── scripts/
│   └── download_llama.py
│
├── models/
│   └── Llama-3.1-8B-Instruct/
│
├── docs/
│   └── overview.md
│
└── assets/
    └── screenshots/
```

## Dataset

This repository includes a synthetic proxy dataset:

```text
data/proxy_dataset.json
```

The real research dataset is not included due to privacy and data-sharing restrictions.

The proxy dataset follows the same structure expected by the dashboard, including:

```text
metadata
demographics
health
psychology
lifestyle
exercise_task
additional_exercise_tasks
```

## Model setup

This demo uses:

```text
meta-llama/Llama-3.1-8B-Instruct
```

The model weights are not included in this repository. You must download the model before running the dashboard.

You need:

1. A Hugging Face account
2. Access to `meta-llama/Llama-3.1-8B-Instruct`
3. Hugging Face authentication through the CLI

Login to Hugging Face:

```bash
huggingface-cli login
```

Then download the model into the local `models/` folder:

```bash
python scripts/download_llama.py
```

This will create:

```text
models/Llama-3.1-8B-Instruct/
```

Do not commit this folder to GitHub.

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/mentalytics-amm-dashboard.git
cd mentalytics-amm-dashboard
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install the package:

```bash
pip install -e .
```

If LLM dependencies are listed as optional in `pyproject.toml`, install them with:

```bash
pip install -e ".[llm]"
```

If all dependencies are already listed directly in `pyproject.toml`, then this is enough:

```bash
pip install -e .
```

## Download the model

Before running the app, download Llama:

```bash
huggingface-cli login
python scripts/download_llama.py
```

The model will be saved to:

```text
models/Llama-3.1-8B-Instruct/
```

## Run the dashboard

Start the Streamlit app:

```bash
streamlit run app/streamlit_app.py
```

Then open the local Streamlit URL shown in the terminal.

Usually this is:

```text
http://localhost:8501
```

## Usage

1. Open the dashboard.
2. Go to the Patient Profile page.
3. Enter a patient ID such as:

```text
AMM_0001
```

4. Click **Load Patient**.
5. View the dashboard pages:
   - Dashboard
   - Patient Profile
   - Physical Insights
   - Psychological Insights

## Llama usage

Llama 3.1 8B is used as a bounded scoring module.

It is used to generate structured numeric outputs for:

```text
Big Five personality trait scores
Exercise suitability matrix scores
```

The model is instructed to return JSON only. The dashboard validates and clamps returned values before visualization.

The model does not generate clinical advice, diagnoses, or free-text recommendations.

## Notes on performance

The first model download may take a long time because Llama 3.1 8B is a large model.

After downloading, the model is loaded from the local `models/` folder.

The Streamlit app caches the model and LLM-generated scores to avoid repeated loading and repeated generation for the same patient.

## Privacy

The repository does not include real participant or patient data.

Only a synthetic proxy dataset is included for demonstration purposes.

## Quickstart

```bash
git clone https://github.com/YOUR_USERNAME/mentalytics-amm-dashboard.git
cd mentalytics-amm-dashboard

python -m venv .venv
source .venv/bin/activate

pip install -e .

huggingface-cli login
python scripts/download_llama.py

streamlit run app/streamlit_app.py
```
