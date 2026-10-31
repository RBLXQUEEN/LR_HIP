# Health Insurance Cost Prediction

A multiple linear regression model that predicts annual medical costs billed to
health insurance, from a person's age, sex, BMI, number of dependants, smoking
status and region.

The repository contains the trained model, the code that reproduces it, and a
single-page interactive demo that runs the model live in the browser.

**Course:** Predictive Analytics (CSE 3141), Manipal University Jaipur
**Author:** Bhoomi Tiwari

---

## Live demo

Open `index.html` in any browser, or view it through GitHub Pages once enabled
(Settings → Pages → Branch: `main`, folder `/root`).

The demo page is self-contained: all coefficients and all 1,338 data points are
embedded, so it works offline with no server and no build step. It also doubles
as the presentation — press `S` to toggle the speaker notes.

---

## The problem

Given six attributes of a policyholder, estimate their annual medical charges in
US dollars.

The output is a continuous quantity, so this is a **regression** problem rather
than classification. There is no probability and no decision threshold — the
model returns an amount, and we judge it by how close that amount lands.

---

## Dataset

**Medical Cost Personal Dataset** (Kaggle) — 1,338 records, 7 columns, no missing values.

| Column | Type | Description |
|---|---|---|
| `age` | numeric | Age of the primary beneficiary (18–64) |
| `sex` | categorical | `female`, `male` |
| `bmi` | numeric | Body mass index (16–53) |
| `children` | numeric | Number of dependants covered (0–5) |
| `smoker` | categorical | `yes`, `no` |
| `region` | categorical | `northeast`, `northwest`, `southeast`, `southwest` |
| `charges` | numeric | **Target** — annual medical costs billed by health insurance |

The file is included at `data/insurance.csv`, so the project runs without
downloading anything.

---

## Model

**Multiple linear regression with one interaction term, fitted by ordinary least
squares (OLS).**

```
charges = -2223.45
        +  263.62 * age
        +   23.53 * bmi
        +  516.40 * children
        -  500.15 * male
        -20415.61 * smoker
        -  585.48 * northwest
        - 1210.13 * southeast
        - 1231.11 * southwest
        + 1443.10 * (smoker * bmi)
```

Nothing was scaled, so each coefficient is in **dollars per unit of its
variable**. Region is one-hot encoded with `northeast` as the baseline.

### Why the interaction term matters

The two smoking terms only make sense added together:

```
cost of smoking = -20415.61 + 1443.10 * bmi
```

At BMI 32 that is `-20415.61 + 46179.08 = +$25,763`. The large negative base is
a mathematical offset that lets the line pivot upward steeply with body mass —
read in isolation it looks absurd, which is exactly why an interaction must never
be interpreted term by term.

This single term is the difference between a mediocre model and a good one:

| Model | R² |
|---|---|
| Linear regression, no interaction | 0.784 |
| **Linear regression + smoker×BMI** | **0.865** |
| Gradient boosting | 0.879 |

Gradient boosting is only marginally better, and it is not interpretable. The
linear model was kept deliberately: every coefficient can be read and defended.

---

## Results

Measured on the 20% of records held back during training (267 rows).

| Metric | Value | Meaning |
|---|---|---|
| R² | **0.865** | 86.5% of the variation in charges is explained |
| MAE | **$2,757** | A typical prediction misses by about this much |
| RMSE | **$4,574** | Penalises large misses harder than MAE |
| 5-fold CV R² | **0.837** | Confirms the result is not one lucky split |

RMSE is well above MAE, which tells you the errors are not uniform — a handful of
people are predicted badly. That is expected: charges have a long right tail.

---

## Project structure

```
.
├── index.html                 # interactive demo + presentation (self-contained)
├── data/
│   └── insurance.csv          # the dataset
├── notebooks/
│   └── analysis.ipynb         # EDA, modelling and evaluation, end to end
├── src/
│   └── train.py               # reproducible training script
├── model/
│   └── coefficients.json      # exported coefficients used by index.html
├── requirements.txt
└── README.md
```

---

## Running it

```bash
pip install -r requirements.txt
python src/train.py
```

The script prints the metrics and rewrites `model/coefficients.json`. It is
deterministic — `random_state=42` throughout — so the numbers reproduce exactly.

To view the demo, just open `index.html`. No server required.

---

## Limitations

**Wrong country.** These are US insurance charges in dollars. They reflect US
medical pricing and say nothing about Indian premiums.

**No medical history.** The six inputs are what a form captures. They exclude
diagnoses, prescriptions, family history and claims history — the things that
actually drive real underwriting.

**Assumes a straight line.** The relationship between BMI and cost is not linear
indefinitely. Outside the training range (BMI below 16 or above 53) the model is
extrapolating. One concrete example of the danger: the formula implies smoking
becomes *cheaper* below BMI 14.15, which is nonsense, and no record in the data
is anywhere near that.

**Small and observational.** 1,338 rows is enough to demonstrate a method and not
enough to build a pricing engine. The model describes correlation in this sample;
it does not establish that any input causes the charge.

This is a teaching model. It is not a quoting tool and must not be used to price
a real policy.

---

## References

- James, Witten, Hastie & Tibshirani — *An Introduction to Statistical Learning* (Springer, 2017), Chapter 3 on linear regression and Chapter 8 on interactions.
- Géron, A. — *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow*, 3e (O'Reilly, 2022).
- Medical Cost Personal Dataset, Kaggle.
