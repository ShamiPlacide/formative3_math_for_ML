# Formative 3 — Probability Distributions, Bayesian Probability & Gradient Descent

## Group Members
| Member | Part Lead | Manual Iteration |
|---|---|---|
| IGABE Placide | Part 1 — EM Algorithm | Gradient Descent Iteration 2 |
| Francoise RUGERO | Part 2 — Bayesian Probability | Gradient Descent Iteration 3 |
| Kevine UWISANGA | Part 4 — Gradient Descent Code | Gradient Descent Iteration 1 |

---

## Repository Structure
```
├── notebook.ipynb          # Main Jupyter Notebook (all 4 parts)
├── manual_calculations.pdf # Handwritten Part 3 gradient descent
├── contributions.pdf       # Individual contribution breakdown
├── data/
│   ├── galton_heights.csv  # Galton heights dataset (Part 1)
│   └── IMDB Dataset.csv    # IMDb reviews dataset (Part 2)
└── README.md
```

---

## Part 1 — EM Algorithm (Gaussian Mixture Model)

**Dataset:** Galton's 1886 Parent-Child Heights  
**Task:** Treat combined father + child heights as an unlabeled mixture of two Gaussians and recover both distributions using EM.

**Approach:**
- Deduplicated fathers (205 unique) to balance the mixture with 934 child records
- Initialized parameters: μ₁=65.0, μ₂=70.0, equal variances and mixing weights
- Implemented E-step (soft responsibility assignment via Bayes) and M-step (weighted parameter updates) from scratch — no sklearn or mixture libraries

**Converged Results:**

| Parameter | Value |
|---|---|
| μ₁ Children | 64.20 inches |
| μ₂ Fathers | 69.54 inches |
| σ₁ Children | 2.31 inches |
| σ₂ Fathers | 2.45 inches |
| π₁ | 0.4367 |
| π₂ | 0.5633 |
| Final Log-Likelihood | -3046.99 |

**Why not split at the global mean?**  
The global mean (67.21") sits in the middle of the overlap zone between the two distributions. A hard split misclassifies tall children and short fathers with 100% false confidence and ignores each cluster's variance entirely. EM instead assigns soft probabilities to every point — a height of 67" might be 60% child, 40% father — which is both more honest and mathematically correct.

**Live Classification:**  
Given any test height, the model outputs exact posterior probabilities:
```
P(Child  | height) = X%
P(Father | height) = Y%
```

---

## Part 2 — Bayesian Probability (Sentiment Analysis)

**Dataset:** IMDb Movie Reviews (50,000 reviews)  
**Task:** Use Bayes' Theorem to compute P(Positive | keyword) for selected sentiment keywords.

**Keywords Selected:**

| Type | Keywords |
|---|---|
| Positive-indicating | excellent, amazing, wonderful, brilliant |
| Negative-indicating | terrible, awful, boring, waste |

**Formula applied:**

```
P(Positive|kw) = P(kw|Positive) × P(Positive)
                 ─────────────────────────────────────────────────────
                 P(kw|Positive)×P(Positive) + P(kw|Negative)×P(Negative)
```

**Implementation:** Basic Python only (`csv`, `re`) — no ML libraries used.  
Each keyword produces a table of: Prior, Likelihood, Marginal, and Posterior probabilities.

---

## Part 3 — Manual Gradient Descent

**Model:** ŷ = Xm + b (matrix form, no scalar treatment)  
**Cost:** J = 1/2n × ||ŷ - y||²

**Given:**
```
X = [[1, 3], [4, 10]]    y = [5, 6]
m₀ = [-1, 2]             b₀ = [1, 1]      α = 0.01
```

**Gradient formulas (derived via chain rule):**
```
∇ₘJ = (1/n) × Xᵀ(ŷ - y)
∇ᵦJ = (1/n) × (ŷ - y)
```

**Results summary:**

| Iteration | Member | m1 | m2 | b1 | b2 |
|---|---|---|---|---|---|
| 0 (init) | — | -1.0000 | 2.0000 | 1.0000 | 1.0000 |
| 1 | Person A | -1.2250 | 1.4350 | 0.9950 | 0.9450 |
| 2 | Person B | -1.3263 | 1.1841 | 0.9996 | 0.9185 |
| 3 | Person C | -1.3465 | 1.1380 | 1.0085 | 0.9112 |

Full step-by-step working in `manual_calculations.pdf`.

---

## Part 4 — Gradient Descent in Code

**Tools:** Python, NumPy, SciPy (`scipy.misc.derivative`), Matplotlib  
**Implementation follows DRY principle with four modular functions:**

```python
predict(X, m, b)              # ŷ = X@m + b
compute_mse(y_true, y_pred)   # J = (1/2n)||ŷ-y||²
compute_gradients(X, y, m, b) # ∇ₘJ and ∇ᵦJ via matrix ops
update_parameters(m, b, ...)  # gradient descent step
```

**SciPy** is used to numerically verify the derivative of MSE w.r.t m[0], confirming our analytical gradient is correct.

**Plots produced:**
- Parameter plot — how m and b change across all 3 iterations
- Error plot — MSE reduction per iteration with percentage drop labels

**Code output matches manual calculations exactly** (verified in notebook).

---

## How to Run

1. Open `notebook.ipynb` in Google Colab or Jupyter
2. Upload `galton_heights.csv` and `IMDB Dataset.csv` when prompted
3. Run all cells in order — each part is clearly separated by markdown headers
4. For the live EM classification demo (Part 1), update the test height in the `classify_height()` call

---

## Dependencies
```
numpy
pandas
matplotlib
scipy
csv (built-in)
re (built-in)
```
Install via: `pip install numpy pandas matplotlib scipy`
