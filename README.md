# Walkthrough: NBA Rookie career longevity predictor (Logistic Regression)

We have successfully constructed, executed, and tuned the **12-step Logistic Regression pipeline** for predicting NBA rookie career longevity (`5Yrs` $\ge 5$ years).

---

## 1. Model Performance & Comparison

By tuning the inverse regularization strength `C` and the regularization penalty (`L1` vs `L2`) via a Stratified 5-Fold Grid Search, we significantly boosted the baseline performance across every major classification metric:

| Metric | Baseline Model (L2, C=1.0) | Tuned Model (L2, C=0.1624) | Improvement | Description |
|:---|:---:|:---:|:---:|:---|
| **Accuracy** | 67.18% | **69.47%** | **+2.29%** | Overall correct predictions |
| **Precision** | 71.51% | **73.45%** | **+1.94%** | Accuracy of positive longevity predictions |
| **Recall** | 78.53% | **79.75%** | **+1.23%** | Sensitivity (finding all rookies who last 5+ years) |
| **F1-Score** | 74.85% | **76.47%** | **+1.62%** | Harmonic mean of Precision and Recall |
| **ROC-AUC** | 74.36% | **74.74%** | **+0.38%** | Discriminative threshold power |

*The tuned model achieved the optimal balance by applying L2 regularization with $C \approx 0.1624$ (equivalent to a higher penalty strength of $\alpha \approx 6.16$), effectively penalizing large coefficients to prevent overfitting and boost generalization on the test set.*

---

## 2. Visual Performance Artifacts

We generated and saved three visual reports directly in the directory. You can view them below:

### A. Confusion Matrix Comparison
The tuned model shows an increase in True Negatives (correctly identifying players who won't last 5 years) and True Positives, boosting overall accuracy.

### B. ROC Curves Comparison
The ROC curves demonstrate robust model performance, with the Tuned Model showing a higher Area Under the Curve (AUC = 0.7474).

---

## 3. Coefficient Interpretation (Sports Analytics Insights)

Each learned coefficient $w_i$ represents the change in log-odds of the player lasting 5+ years per standard deviation increase in that stat. We exponentiated these coefficients ($e^{w_i}$) to obtain **Odds Ratios (OR)**, making them highly intuitive for scouts and sports coaches:

### Key Analytics Takeaways:

1. **The Three-Point Efficiency Dilemma (`3PM` vs `3PA`)**:
   - **`3PM` (3-Pointers Made)** has a massive positive coefficient of **+1.168** (Odds Ratio = **3.21**). A standard deviation increase in 3-pointers made multiplies a rookie's odds of lasting 5+ years by **3.21x**!
   - **`3PA` (3-Point Attempts)**, however, has a heavy negative coefficient of **-1.188** (Odds Ratio = **0.30**).
   - *Scouting Insight:* This is a classic efficiency indicator. Rookie shooters who shoot high volumes of three-pointers (`3PA`) but fail to make them (`3PM`) have severely degraded career longevity odds. Teams should value three-point efficiency over volume.

2. **Longevity is Built on Games Played (`GP`)**:
   - **`GP`** has a strong coefficient of **+0.623** (Odds Ratio = **1.86**).
   - *Scouting Insight:* Durability, consistency, and earning early rotation spots in the rookie year is a major positive indicator. A player with high games played is **86% more likely** to reach the 5-year career benchmark.

3. **Rebounding & Post Presence (`OREB` vs `DREB`)**:
   - **`OREB` (Offensive Rebounds)** is a strong positive predictor (**+0.506**, OR = **1.66**). 
   - *Scouting Insight:* Offensive rebounding represents hustle, physical size, and active second-chance efforts—all valuable, baseline skills that secure roster spots. Interestingly, defensive rebounds (`DREB`) are slightly negative (-0.154), likely because defensive rebounds are highly correlated with minutes played and general team size rather than independent hustle.

4. **Free Throw Production (`FTM` vs `FTA`)**:
   - **`FTM` (Free Throws Made)** is strongly positive (**+0.480**, OR = **1.62**), while **`FTA` (Attempts)** is negative (**-0.401**, OR = **0.67**).
   - *Scouting Insight:* Similar to the 3-point dilemma, drawing fouls is only valuable if the player actually converts them. A high-attempt free throw shooter who misses often (low FTM, high FTA) is an efficiency liability.
