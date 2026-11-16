# Used-Car Price Prediction Under Dataset Shift

ES 335 course project, Kushal Kumar (24110178) and Tejas Prasad (24110369).

**Question.** How accurately can ML models predict the selling price of a used car in India, and does that accuracy
hold when the model is applied to listings from a data source it was not trained on?

**Website:** `https://tejas-pd.github.io/ES335/` 

## Main result

| MAE (₹ lakh) | Random split | Held-out A | Held-out B | Held-out C | Held-out D |
|---|---|---|---|---|---|
| Mean baseline | 3.97 | 4.11 | 2.88 | 5.94 | 2.70 |
| Model-median lookup | 1.87 | 2.50 | **1.38** | 3.60 | 1.37 |
| Linear (ridge, log price) | 1.28 | 2.41 | 1.42 | **3.31** | **1.06** |
| Random forest (log price) | **1.23** | **2.37** | 1.39 | 3.39 | 1.32 |

On a random split, both ML models beat simple baselines clearly. On a held-out source, errors roughly double and most
of the advantage over a model-median lookup disappears. Held-out A is over-priced by about 24–28% and C under-priced by
about 21–23%. The forest is no more robust than ridge regression. See the website for uncertainty intervals, error
analysis, the age-offset diagnostic and few-shot recalibration.

## Data

[Top 4 Used Car Sales Datasets Combined](https://www.kaggle.com/datasets/satvshr/top-4-used-car-sales-datasets-combined) (Kaggle, uploaded by satvshr; file `output.csv`): 32,200 public Indian
used-car listings with no personal data. **The raw CSV is not committed.** Download it and save it as `data/raw.csv`.
The file has no source column, so `src/prepare.py` reconstructs the four original sources (A–D) from each row's pattern
of missing columns.

## Reproduce

```bash
pip install -r requirements.txt
python src/prepare.py data/raw.csv data/clean.csv   # cleaning + source reconstruction
python src/explore.py data/clean.csv                # per-source summaries, figures/
python src/evaluate.py data/clean.csv               # random split + leave-one-source-out (~10 min on 2 cores)
python src/diagnostics.py data/clean.csv            # error analysis, recalibration, age-offset check
python src/export_web.py data/clean.csv docs/data   # website JSON + compact models
python -m http.server -d docs                       # view at http://localhost:8000
```

`walkthrough.ipynb` runs the same steps and shows the main tables and figures.

- **Random split:** `train_test_split(test_size=0.2, stratify=source, random_state=42)`; tuning by 5-fold CV on the
  training part only.
- **Leave-one-source-out:** train on three sources, tune with inner leave-one-source-out CV over those three, and
  score once on the fourth.
- All learned preprocessing (one-hot categories, scaling, imputation) is inside scikit-learn pipelines fitted on
  training folds only. Seeds are 42 throughout.

## Layout

```
src/prepare.py       cleaning rules, source reconstruction
src/explore.py       data-understanding summaries and figures
src/models.py        baselines, ridge and forest pipelines, tuning grids
src/evaluate.py      both evaluation designs, bootstrap CIs  -> results/metrics.json
src/diagnostics.py   post-hoc error analysis                  -> results/diagnostics.json
src/export_web.py    website data and compact models          -> docs/data/
docs/                static website (GitHub Pages root): index.html, style.css, app.js, models.js
results/, figures/   outputs from the last run
```

## Publishing on GitHub Pages

1. Push this folder to a public GitHub repository. `data/` is git-ignored.
2. Go to **Settings → Pages → Build and deployment**, choose *Deploy from a branch*, branch `main`, folder `/docs`.
3. The site appears at `https://<username>.github.io/<repo>/` within a minute or two.

## AI disclosure

We used Claude (Anthropic, via claude.ai) occasionally throughout the project as an assistance tool, primarily for debugging, analysing our code, and identifying potential issues or improvements in our implementation. It was also used to help restructure and refine written content, including explanations and documentation, to improve clarity, readability, and the overall presentation of our work. The core implementation, experimentation, evaluation, and project decisions remained our responsibility.

## Contributions

**Tejas Prasad**: co-wrote the proposal; owns the data audit, source reconstruction and evaluation design; verified the cleaning and evaluation outputs.

**Kushal Kumar**: co-wrote the proposal; owns the modelling choices and the website; verified the model results and published the site.