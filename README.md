# Project Title
- Clustering Student Course Evaluations with PCA, K‑Means & Hierarchical Methods (Türkiye Student Evaluation Dataset)
  ![image1-4-330x220](https://github.com/user-attachments/assets/514040aa-cf30-4979-a6a3-b35419133fed)
# Objective
- Identify natural groupings (clusters) of student course evaluation responses to:
- Uncover distinct student sentiment patterns toward courses/instructors.
- Summarize high‑dimensional Likert‑scale feedback into actionable segments.
- Support data‑driven decisions for teaching improvements and resource allocation.
# Why This Project (Use Cases)
- Instructional improvement: Reveal which question themes drive low vs. high satisfaction.
- Course design: Compare clusters across instr and class to tailor interventions.
- Quality assurance: Detect unusual response patterns (potential outliers or data entry issues).
- Reporting: Provide concise visuals (elbow plot, PCA map, dendrogram) for stakeholders.
# Dataset Overview
- Source: Türkiye Student Evaluation (generic split).
- Each row: one student’s evaluation of a course/instructor.
- Features used for clustering: Likert‑scale questions (columns 5–32 inclusive → 28 questions).
- Categorical/context columns: e.g., instr, class (used for profiling clusters, not for fitting).
- Note: The code you provided uses df.iloc[:, 5:33] as the feature block (X), and applies PCA with n_components=2 for visualization.
# Step‑by‑Step Approach
- Load & Inspect Data
- df.head(), df.info(), df.describe() to verify schema, datatypes, and basic stats.
- Data Quality Checks
- df.isnull().sum() to confirm missing values; handle if present.
# Exploratory Data Analysis (EDA)
- Distribution of instr and class (countplots).
- Per‑question means; overall mean across questions.
- Correlation matrix/heatmap to see redundancy and themes.
- Feature Block Selection
- Use only question columns (Q1–Q28) to avoid leakage from IDs/context.
# Feature Engineering
- (Optional but recommended) standardize features: StandardScaler().
- Compute PCA for visualization and variance analysis; keep 2D for plotting, but consider using more PCs for modeling if needed.
# Modeling — Clustering
- K‑Means: run elbow method (inertia_) over k=1..5 (or wider) to choose k; fit with chosen k (you used k=3) and predict labels.
- Hierarchical (Agglomerative, Ward): fit with n_clusters=2 on PCA space; plot dendrogram on a sample/PCs for interpretability.
- Model Evaluation (Unsupervised)
- Internal metrics: silhouette score, Davies–Bouldin, Calinski–Harabasz.
- Cluster profiling: compare clusters on question means and distribution across instr and class.
- Visualization & Reporting
- Elbow plot, PCA scatter with clusters/centroids, dendrogram.
- Insights & Recommendations
- Translate clusters into student personas and actionable steps for instructors.
# Exploratory Data Analysis (Highlights)
- Category distributions: sns.countplot(df['instr']), sns.countplot(df['class']) to understand sampling across instructors/classes.
- Question means: compute mean per question and overall mean to gauge general satisfaction.
- Correlation heatmap: strong positive correlations are common across Likert items; clusters of questions may indicate latent factors (e.g., instructor performance vs. course organization).
- If multiple questions are highly correlated, PCA helps reduce redundancy and noise.
# Feature Selection
- Included: Question columns df.iloc[:, 5:33] (28 Likert features).
- Excluded from fitting: instr, class (used later for profiling), IDs or metadata fields.
- Rationale: Keep only direct opinion responses for unsupervised grouping.
# Feature Engineering
- Scaling: Use StandardScaler so all questions contribute equally (beneficial for distance‑based methods like K‑Means/Agglomerative).
- Dimensionality Reduction (PCA):
- 2D PCA used for visualization and dendrogram plotting.
- Inspect pca.explained_variance_ratio_.cumsum() to choose the number of PCs covering ~80–95% variance if you also want PCA‑based modeling.
# Code sketch:
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
X = df.iloc[:, 5:33].values
X_sc = StandardScaler().fit_transform(X)
# Visual PCA (2D)
pca2 = PCA(n_components=2, random_state=42)
X_pca = pca2.fit_transform(X_sc)
explained2 = pca2.explained_variance_ratio_.sum()
# Variance analysis (choose PCs)
pca_full = PCA(random_state=42).fit(X_sc)
var_cum = pca_full.explained_variance_ratio_.cumsum()
# Model Training
1) K‑Means (+ Elbow)
from sklearn.cluster import KMeans
# Elbow
distortions = []
for k in range(1, 10):
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    km.fit(X_pca)  # or X_sc / PCs as you prefer
    distortions.append(km.inertia_)
# Fit chosen k (your code used k=3)
km = KMeans(n_clusters=3, init='k-means++', n_init=10, random_state=42)
km.fit(X_pca)
y_kmeans = km.labels_
centers = km.cluster_centers_
2) Hierarchical (Agglomerative, Ward)
from sklearn.cluster import AgglomerativeClustering
agg = AgglomerativeClustering(n_clusters=2, linkage='ward')  # Ward ⇒ Euclidean metric
labels_agg = agg.fit_predict(X_pca)
Environment note: In modern scikit‑learn, remove deprecated args such as n_jobs (K‑Means) and replace affinity with metric (except Ward, which assumes Euclidean and needs no metric arg).
# Model Testing / Evaluation (Unsupervised)
- Use internal validation metrics and stability checks:
- from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
- sil_km = silhouette_score(X_pca, y_kmeans)
- db_km  = davies_bouldin_score(X_pca, y_kmeans)
- ch_km  = calinski_harabasz_score(X_pca, y_kmeans)
- sil_ag = silhouette_score(X_pca, labels_agg)
- db_ag  = davies_bouldin_score(X_pca, labels_agg)
- ch_ag  = calinski_harabasz_score(X_pca, labels_agg)
# Stability tips
- Re‑fit with different random_state (K‑Means) and compare adjusted Rand index on reassignments.
- Bootstrap or sub‑sample rows and check cluster consistency.
- Cluster Profiling (Turning Labels into Insights)
- Summarize cluster characteristics to make them actionable:
- import pandas as pd
# Attach labels
out = df.copy()
out['cluster_k3'] = y_kmeans
out['cluster_ag2'] = labels_agg
# Size per cluster
size_k3 = out['cluster_k3'].value_counts().sort_index()
size_ag2 = out['cluster_ag2'].value_counts().sort_index()
# Mean score per question by cluster
q_cols = df.columns[5:33]
profile_k3 = out.groupby('cluster_k3')[q_cols].mean().round(2)
# Context distributions
by_instr = out.groupby('cluster_k3')['instr'].value_counts(normalize=True).rename('pct').mul(100).round(1)
by_class = out.groupby('cluster_k3')['class'].value_counts(normalize=True).rename('pct').mul(100).round(1)
# Outputs & Deliverables
<img width="840" height="621" alt="Screenshot 2025-08-14 105156" src="https://github.com/user-attachments/assets/1766553f-da3b-4506-9a78-7b6b33f63e1f" />
