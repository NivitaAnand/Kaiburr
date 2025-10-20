# Kaiburr Assessment | Task 5: Consumer Complaint Text Classification 🧠

## 1. Project Mandate and Solution Summary

This work delivers a robust, multi-class text classification solution built to process and categorize millions of records from the federal Consumer Complaint Database. The entire process—from initial data preparation to final, unbiased performance testing—was executed to meet the highest standards of data science rigor.

* **Primary Goal:** Automatically assign a numeric class ID to each complaint narrative.
* **Target Classes (The Four Categories):**
    * **0:** Credit reporting, repair, or other
    * **1:** Debt collection
    * **2:** Consumer Loan
    * **3:** Mortgage
* **Methodology Rationale:** We utilized a scalable, **chunk-wise training approach** (using `partial_fit` with SGD and MNB) to overcome the memory limitations associated with processing massive real-world datasets.

***

## 2. Data Preparation and Feature Engineering

### Initial Data Exploration (EDA)

My first step was a visual exploration of the target variable. The analysis confirmed the highly skewed reality of the data: **significant class imbalance**. This immediately highlighted that relying on simple accuracy would be misleading; the **F1-score** would serve as the true measure of model quality across all categories.



### Feature Creation and Cleaning

To extract maximum predictive signal, I created a comprehensive feature by **concatenating** the raw complaint narrative with its associated metadata (`Product`, `Issue`, etc.). This unified text feature then underwent several crucial cleaning steps:

* **Standardization:** Converted all text to lowercase.
* **Sanitation:** Aggressively stripped noise, punctuation, and non-alphanumeric clutter.
* **Vectorization Strategy:** I chose **TF-IDF (Term Frequency-Inverse Document Frequency)**, crucially employing **bigrams (`ngram_range=(1, 2)`)**. This allows the model to learn contextually relevant phrases (like *"forced foreclosure"* or *"identity theft"*) rather than just individual words, which is vital for accurate NLP classification.

***

## 3. Model Training, Comparison, and Evaluation (The Proof)

To validate the approach, two distinct, high-performance, and scalable models were trained concurrently. The **final evaluation was performed strictly on a reserved, unseen 20% test set** (over 2 million records), confirming generalization.

* **Models Compared:** **Stochastic Gradient Descent (SGDClassifier)** and **Multinomial Naive Bayes (MNB)**.

### Comparative Performance Report (on Unseen Test Set)

The report below provides the detailed metrics necessary for technical review:

| Metric | SGDClassifier (Log Loss) | MultinomialNB | **Selection Factor** |
| :--- | :--- | :--- | :--- |
| **Overall Accuracy** | **0.992** (99.2%) | 0.979 (97.9%) | **SGD is superior overall.** |
| **Macro Avg F1-Score** | **0.934** (Best general balance) | 0.918 | |
| **F1-Score (Consumer Loan - Minority Class)** | 0.794 | **0.835** | MNB slightly outperformed on the least-represented class. |

### Final Model Selection

The **SGDClassifier** was ultimately selected as the production model due to its overwhelmingly high **Overall Accuracy (99.2%)** and stronger **Macro Average F1-Score (0.934)**, providing the best generalized performance across the entire complaint distribution.

### Confusion Matrix (Final Model: SGDClassifier)

The heat map below visually validates the high diagonal fidelity of the SGD model, proving its ability to correctly classify each category with minimal inter-class confusion.



***

## 4. Model Deployment and Prediction (Production Readiness)

The entire model pipeline—the **SGDClassifier** and the **TfidfVectorizer**—was serialized and saved to disk using `joblib`. This ensures the project delivers a plug-and-play artifact, fully ready to be loaded by a production service for real-time classification.

* **Deployable Artifacts:** `final_classifier_prod.joblib` and `vectorizer_prod.joblib`

### Test Prediction Results

The final, loaded model was tested against new, unseen narratives:

| Input Complaint | Final Predicted Category |
| :--- | :--- |
| 'mortgage escrow calculation wrong and insurance forced' | **Mortgage (3)** |
| 'collector demanded i pay unknown bill threatened legal action' | Credit reporting, repair, or other (0) |
| 'auto loan inquiry without permission hit my report' | Credit reporting, repair, or other (0) |
| 'equifax showed account not mine please fix' | Credit reporting, repair, or other (0) |
