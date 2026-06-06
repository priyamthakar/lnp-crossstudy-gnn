# Cross-Study Evaluation of Machine Learning for Lipid Nanoparticle Property Prediction

**A Multi-Modal Benchmark with Interpretable Graph Neural Networks**

**Authors:** Priyam Thakar · Neelesh Kumar Mehra · Tejal A. Mehta

---

## What this is and why it matters

Lipid nanoparticles (LNPs) have moved from niche drug-delivery curiosity to front-page technology — they are the delivery vehicle behind mRNA COVID-19 vaccines and a growing pipeline of RNA therapeutics. Designing an LNP that is stable, potent, and well-tolerated still involves a lot of trial and error, partly because the experimental data that does exist is scattered across dozens of studies with different protocols, different cell lines, and different ways of reporting outcomes.

This repository is the companion code for our paper, which asks a pointed question: **can machine learning models trained on one LNP dataset reliably predict outcomes on another?** We pool formulation data from multiple published studies, harmonize them into a single benchmark, and train a suite of models — from classical fingerprint-based regressors to interpretable graph neural networks — under leave-one-study-out cross-validation. The GNN approach also produces atom-level attribution maps so you can see *which* structural features the model is relying on, rather than treating the prediction as a black box.

If you are a formulation scientist tired of re-running the same optimization experiments, a computational chemist looking for a clean multi-study LNP benchmark, or a methods researcher who wants to stress-test a new architecture on realistic cross-study shift, this repo is built for you.

---

## Folder structure

```
LNP-ML-Benchmark/
├── dataset/          Raw and processed formulation datasets
├── models/           Model definitions (GNN, RF, XGBoost, MLP baselines)
├── notebooks/        End-to-end worked examples and figure generation
├── preprocessing/    Feature engineering, SMILES parsing, dataset harmonization
└── results/          Saved model outputs, metrics tables, attribution maps
```

---

## How to use

1. **Clone or download** this repository and navigate into it.
2. **Install dependencies** (see below).
3. Place your raw dataset files inside `dataset/` following the schema described in `dataset/README.md`.
4. Run `preprocessing/` scripts to featurize and split the data.
5. Train models using the scripts or notebooks in `models/` and `notebooks/`.
6. Outputs land in `results/` — metrics, plots, and GNN attribution maps.

For a guided walkthrough, start with `notebooks/01_quickstart.ipynb` once it is added.

---

## Dependencies

Install everything with:

```bash
pip install -r requirements.txt
```

Core packages: PyTorch, PyTorch Geometric, RDKit, scikit-learn, pandas, numpy, matplotlib, seaborn, Jupyter.

Python 3.9 or later is recommended. A GPU is not required but will speed up GNN training significantly.

---

## Citation

If you use this benchmark or codebase in your work, please cite:

```bibtex
@article{thakar2025lnpbenchmark,
  title   = {Cross-Study Evaluation of Machine Learning for Lipid Nanoparticle
             Property Prediction: A Multi-Modal Benchmark with Interpretable
             Graph Neural Networks},
  author  = {Thakar, Priyam and Mehra, Neelesh Kumar and Mehta, Tejal A.},
  journal = {TBD},
  year    = {2025},
  doi     = {TBD — will be updated after acceptance}
}
```

---

## Contact

Questions, suggestions, or collaboration ideas? Reach out to **Priyam Thakar** at [priyamthakar1@gmail.com](mailto:priyamthakar1@gmail.com).
