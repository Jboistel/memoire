# Inventory Policies — Notebook Edition (M5 / Walmart)

This repo is a **single Jupyter notebook** that simulates simple inventory policies on M5 (Walmart) daily sales. Open it, run all cells, and you get **fill rate** and **average inventory** with a couple of plots.

---

## Files in this repo

* `forecasts_and_replenishment_pol_MASTER_THESIS_BOISTE` — main notebook.
* `data/` — place the M5 CSVs here (at least `sales_train_evaluation.csv` and `calendar.csv`).
* `.gitignore`, `.gitattributes`

Suggested rename:

> `forecasts_and_replenishment_pol_MASTER_THESIS_BOISTEL.ipynb` → **`m5_inventory_policies.ipynb`**

---

## Setup

Python ≥ 3.10

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U jupyter pandas numpy matplotlib scipy
```

Open the notebook with:

```bash
jupyter lab
# or
jupyter notebook
```

---

## Quick start

1. Put the M5 files in `./data/`.
2. Open `m5_inventory_policies.ipynb`.
3. Run all cells. The notebook will:

   * build **very simple forecasts** (moving averages or same‑weekday averages),
   * compute **up‑to‑level** targets with optional **safety stock**,
   * simulate stock with review time `R` and lead time `L`,
   * print **fill rate** and **average inventory**, and draw a few plots.

> Focus is on **policy effects**, not fancy forecasting.

---

## Tuning knobs (inside the notebook)

* **R**: review cadence in days (order every R days).
* **L**: lead time in days.
* All parameter lists in the policies' cells concerning them.
