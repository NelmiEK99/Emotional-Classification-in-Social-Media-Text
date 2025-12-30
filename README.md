# Emotional Classification in Social Media Text

A Jupyter notebook project that **trains and compares multiple NLP models** for **emotion classification** on short social-media style text.

Models included:

- **Linear SVM + TF‑IDF** (baseline)
- **Bi‑LSTM** (Keras)
- **Text CNN** (Keras)
- **DistilBERT** (Transformers fine‑tuning)
- **RoBERTa‑base** (Transformers fine‑tuning)

---

## Dataset

The notebook expects **three CSV files**:

- `training.csv`
- `validation.csv`
- `test.csv`

Each file must contain:

| column | type | description |
|---|---|---|
| `text` | string | input text (tweet / post style) |
| `label` | int | class id from `0..K-1` |

### Label mapping used in the notebook

```text
0 = sadness
1 = joy
2 = love
3 = anger
4 = fear
5 = surprise
```

> If you use a different mapping, update the `labels_dict` cell (and keep labels as integers).

---

## What’s inside the notebook

- Basic EDA: class balance, missing/duplicate checks, text length & word-count stats, n‑grams
- Text cleaning for classical / Keras models (lowercasing, noise removal, optional stopwords + lemmatization via NLTK)
- Training + evaluation on validation and test sets
- Metrics: **Accuracy** and **Macro‑F1**
- Confusion matrix plots
- A comparison table + charts across all models

---

## Quickstart

### 1) Create an environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
```

### 2) Install dependencies

Minimum packages used in the notebook:

```bash
pip install -U numpy pandas scikit-learn matplotlib nltk tensorflow torch torchvision torchaudio transformers datasets accelerate
```

> Notes  
> - **Transformers fine‑tuning runs much faster on a GPU** (Colab / CUDA).  
> - `tensorflow` can be heavy; if you only run SVM/Transformers you can skip it.

### 3) Add the dataset files

Put the CSVs in a local folder (example):

```text
data/
  training.csv
  validation.csv
  test.csv
```

Then update the config cell in the notebook, e.g.:

```python
TRAIN_PATH = "data/training.csv"
VAL_PATH   = "data/validation.csv"
TEST_PATH  = "data/test.csv"
```

*(The notebook currently contains absolute file paths — you’ll want to replace those before pushing to GitHub.)*

### 4) Run the notebook

- Open `Emotional_Classification_in_SocialMedia_Text.ipynb` in Jupyter/VS Code/Colab
- Run cells top‑to‑bottom

---

## Choosing which models to run

In the config section:

```python
RUN_MODELS = {"svm", "bilstm", "cnn", "distilbert", "roberta"}
```

You can run a subset (faster):

```python
RUN_MODELS = {"svm", "distilbert"}
```

---

## Outputs

The notebook writes results to `outputs/`:

- `outputs/metrics_summary.csv` — validation/test Accuracy & Macro‑F1 for each model
- `outputs/*_val_pred.csv`, `outputs/*_test_pred.csv` — per‑model predictions

Confusion matrices and comparison charts are displayed in the notebook.

---

## Reproducibility

A `set_seed(42)` helper is used to reduce randomness across NumPy / PyTorch / TensorFlow, but deep learning results can still vary slightly between runs and hardware.

---

## Suggested repo structure

```text
.
├── Emotional_Classification_in_SocialMedia_Text.ipynb
├── README.md
├── outputs/                # generated (optional to commit)
└── data/                   # do NOT commit if it contains Kaggle data
```

Tip: add a `.gitignore` entry for `data/`, model checkpoints, and large output files.

---

## Acknowledgements

- Dataset: Kaggle “Emotion Dataset” (search on Kaggle for the dataset owner/slug used in the notebook).
- Models: Hugging Face Transformers (DistilBERT, RoBERTa).

---

## License

Add a license that matches how you want others to use your work (for example: MIT, Apache‑2.0, or GPL‑3.0).
