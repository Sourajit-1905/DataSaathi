# BACKEND STAGE 1 — Project Setup & Session Foundation

## What this stage is

The scaffold for the entire backend. No analysis yet — just the skeleton, the file system, session management, and the upload/preview/reset endpoints.

## What it does

- Creates the FastAPI app with CORS and all routers registered
- Implements session management (UUID-based, file-keyed)
- Implements the upload endpoint (CSV + XLSX → CSV conversion)
- Implements preview, reset, and info endpoints

## Where it's used

Every single subsequent stage depends on this. The `session_id` created here is the key passed to every API call throughout the app. Without Stage 1, nothing else can run.


# BACKEND STAGE 2 — Data Profiling Service (22 functions)

## What this stage is

The read-only analysis engine that examines the uploaded dataset from every angle before any cleaning or transformation happens.

## What it does

Runs 22 profiling checks — dimensions, column types, missing values, duplicates, outliers, distributions, correlations, cardinality, suspicious values, and more — and combines them into a single quality score and a set of plain English insights. After the algorithmic results are computed, Gemini receives the summary and rewrites the insights panel as a genuine narrative rather than templated strings.

## Where it's used

- **Frontend Profile page** — the first page the user lands on after upload
- **Stage 3 (Cleaning)** — profiling results inform what cleaning is recommended
- **Stage 8 (ML Strategy)** — problem type detection uses profiling output
- **Stage 9 (Report)** — profiling summary is Section 2 of the report
- **Gemini** — profiling results are the first thing sent to Gemini to generate real insights

## Files in this stage

1. `services/profile_service.py` — all 22 functions + Gemini insights
2. `prompts/profiling_insights.txt` — Gemini prompt template
3. `routers/profile.py` — 4 endpoints
4. `models/schemas.py` — updated with profiling response schemas
5. `tests/test_profiling.py` — verification tests

## Features of Data Profiling

The Data Profiling module analyzes the uploaded dataset to understand its structure, quality, characteristics, and potential issues before further analysis.

| No. | Feature | Description |
|---:|---|---|
| 1 | **Dataset Dimensions** | Determines the number of rows, columns, dataset size, and memory usage. |
| 2 | **Column Identification and Meaning** | Identifies columns and estimates their semantic meaning from names and values. |
| 3 | **Numerical Column Detection** | Identifies columns containing numerical data such as measurements, prices, ages, or quantities. |
| 4 | **Categorical Column Detection** | Identifies columns containing discrete categorical values such as gender, city, or department. |
| 5 | **Date/Time Column Detection** | Detects date/time columns and identifies their formats. |
| 6 | **Missing Value Analysis** | Reports the number and percentage of missing values in each column. |
| 7 | **Unique Value and Cardinality Analysis** | Calculates unique values and cardinality to identify low-, medium-, and high-cardinality features. |
| 8 | **Duplicate Row Detection** | Identifies and reports completely duplicated rows. |
| 9 | **Duplicate Column Detection** | Identifies columns containing identical data and potential redundancy. |
| 10 | **Numerical Statistical Summary** | Provides minimum, maximum, mean, median, standard deviation, and range for numerical columns. |
| 11 | **Suspicious and Invalid Value Detection** | Flags potentially invalid or suspicious values, such as negative ages or percentages above 100%. |
| 12 | **Data Type Consistency** | Checks whether column values match their assigned data types and detects inconsistent formats. |
| 13 | **Constant and Near-Constant Column Detection** | Identifies columns with one value or a highly dominant value that may provide little analytical value. |
| 14 | **Identifier Detection** | Identifies columns likely to represent IDs based on names and uniqueness patterns. |
| 15 | **Outlier Detection** | Flags unusual numerical observations using methods such as IQR or standard deviation. |
| 16 | **Distribution Analysis** | Analyzes numerical distributions using skewness and other basic characteristics. |
| 17 | **Categorical Distribution and Class Imbalance** | Analyzes category frequencies and identifies potential class imbalance. |
| 18 | **Date/Time Coverage Analysis** | Reports date ranges, time coverage, and potential temporal gaps or irregularities. |
| 19 | **Initial Relationship Screening** | Performs basic relationship analysis between numerical variables, such as correlation analysis. |
| 20 | **Potential Target Detection** | Identifies possible machine learning target columns based on data type, cardinality, and naming patterns. |
| 21 | **Data Quality Summary** | Combines detected issues into an overall summary of dataset quality. |
| 22 | **Automatic Profiling Insights** | Converts profiling results into human-readable observations and warnings. |

Converts profiling results into human-readable observations and warnings.

The profiling results provide the foundation for Data Cleaning, EDA, Statistical Analysis, Machine Learning, and AI-based Data Insights.


# BACKEND STAGE 3 — Dataset Cleaning

## What this stage is

The first stage where the dataset is actually **modified**. It takes the issues identified during profiling and lets the user apply fixes to the working copy of the data.

## What it does

- Detects 9 categories of data quality issues in the working dataset
- Recommends a strategy for each issue algorithmically (e.g. median for skewed columns, mode for categorical)
- Previews what the data will look like **before** applying any change
- Applies the chosen strategy to the working CSV
- Supports **undo** via snapshot restore
- Maintains a full operation history
- Gemini adds a plain English explanation of **why** each strategy is recommended

## Where it's used

- **Frontend Clean page** — the entire cleaning workflow
- **Stage 4 (Transform)** — cleaning must happen before transformation
- **Stage 9 (Report)** — cleaning history becomes Section 3 of the report
- **Gemini** — receives the issue + chosen strategy and explains the reasoning in plain English

## Files in this stage

1. `services/clean_service.py` — 9 issue detectors + recommend + apply + undo
2. `prompts/clean_recommend.txt` — Gemini prompt template
3. `routers/clean.py` — 6 endpoints
4. `models/schemas.py` — updated with cleaning schemas
5. `tests/test_cleaning.py` — verification tests

## Cleaning Operations

Implements 9 cleaning operations:

| ID | Cleaning Operation | Brief Explanation |
|---|---|---|
| **C1** | Missing values (numerical) | Handles missing numerical values using strategies such as mean, median, or other suitable methods. |
| **C2** | Missing values (categorical) | Fills missing categorical values using methods such as mode or a dedicated category. |
| **C3** | Duplicate rows | Detects and removes completely duplicated records from the dataset. |
| **C4** | Outlier treatment | Identifies extreme values and applies a selected strategy such as removal, capping, or transformation. |
| **C5** | Type conversion | Converts incorrectly typed columns into appropriate numeric or datetime formats. |
| **C6** | String standardisation | Cleans inconsistent text values using trimming and case standardisation. |
| **C7** | Constant / near-constant column removal | Removes columns with little or no variation that provide minimal analytical value. |
| **C8** | Identifier column removal | Removes ID-like columns that are unlikely to provide useful predictive information. |
| **C9** | Negative value treatment | Detects invalid negative values and applies an appropriate correction or treatment strategy. |

## Cleaning Workflow

For each issue:

- `detect_*()` → finds affected rows/columns
- `recommend_*()` → suggests a strategy algorithmically
- `preview_*()` → returns before/after sample **without saving**
- `apply_*()` → applies the strategy to the working CSV (snapshot taken first)

Gemini adds a plain English **"why"** explanation to every recommendation.

**Fallback:** Rule-based explanation if Gemini is unavailable.



# BACKEND STAGE 4 — Transform Service

## What this stage is

The second data modification stage. While cleaning fixes data problems, transformation engineers features by reshaping, scaling, encoding, and creating new columns.

## What it does

- Implements 25 transformation functions across 6 categories: scaling, encoding, binning, mathematical, datetime extraction, and dimensionality reduction
- Previews transformations before applying them
- Applies transformations to the working CSV with undo support
- Maintains a per-session transformation history
- Gemini explains why each transformation is recommended

## Where it's used

- **Frontend Transform page** — column cards, transform configurator, history panel
- **Stage 8 (ML Strategy)** — transform history informs feature engineering recommendations
- **Stage 9 (Report)** — transform history becomes Section 4 of the report
- **Gemini** — receives column stats + chosen transform and explains the reasoning

## Files in this stage

1. `services/transform_service.py` — 25 transformation functions
2. `prompts/transform_recommend.txt` — Gemini prompt template
3. `routers/transform.py` — 5 endpoints
4. `models/schemas.py` — updated with transform schemas
5. `tests/test_transform.py` — verification tests

## Features of Data Transformation

| No. | Transformation | Brief Description |
|---:|---|---|
| **1** | **Normalization** | Scales numerical values to a common range, typically 0–1. |
| **2** | **Standardization** | Transforms numerical features to have a mean near 0 and standard deviation near 1. |
| **3** | **Robust Scaling** | Scales features using median and IQR, making it more resistant to outliers. |
| **4** | **Encoding Categorical Variables** | Converts categorical values into numerical representations using methods such as one-hot, ordinal, label, frequency, or target encoding. |
| **5** | **Date/Time Transformation** | Extracts useful features such as year, month, day, weekday, quarter, and hour from dates. |
| **6** | **Text Transformation** | Converts text into analysis-ready representations using operations such as tokenization, cleaning, TF-IDF, or embeddings. |
| **7** | **Aggregation** | Summarizes records using operations such as sum, mean, count, minimum, or maximum. |
| **8** | **Binning** | Converts continuous numerical values into discrete intervals or groups. |
| **9** | **Log and Mathematical Transformations** | Applies mathematical functions such as log, square-root, reciprocal, or power to modify distributions. |
| **10** | **Power Transformations** | Uses methods such as Box-Cox or Yeo-Johnson to make numerical distributions more suitable for analysis. |
| **11** | **Reshaping Data** | Changes dataset structure using operations such as pivot, melt, stack, and unstack. |
| **12** | **Aggregation by Time** | Converts time-based records into different granularities such as weekly, monthly, quarterly, or yearly. |
| **13** | **Calculated Columns** | Creates new columns using mathematical or logical expressions based on existing columns. |
| **14** | **Feature Engineering** | Creates meaningful features such as ratios, differences, rates, rolling statistics, and lag features. |
| **15** | **Feature Interaction** | Creates features representing relationships between multiple variables, such as Price × Quantity. |
| **16** | **Numerical Feature Transformation** | Applies operations such as addition, subtraction, multiplication, division, square, or percentage change. |
| **17** | **Category Grouping** | Combines multiple categories into broader user-defined groups. |
| **18** | **Dimensionality Reduction** | Reduces numerical features using methods such as PCA, SVD, t-SNE, or UMAP. |
| **19** | **Feature Selection** | Identifies and excludes potentially unnecessary features using statistical, correlation, model-based, or user-defined criteria. |
| **20** | **Transformation Preview** | Shows the expected effect of a transformation before it is applied. |
| **21** | **Transformation History** | Records every transformation applied to the dataset for reproducibility. |
| **22** | **Before-and-After Comparison** | Compares dataset structure and statistics before and after transformation. |
| **23** | **Transformation Validation** | Verifies that the transformed dataset has valid values, types, features, and no unintended data loss. |
| **24** | **User-Controlled Transformation** | Allows users to select columns, configure transformations, preview, apply, reject, undo, or reset changes. |
| **25** | **Transformation Summary** | Generates a summary of transformed columns, created features, encoded variables, scaling operations, and extracted date features. |

## Transform Service

Implements 25 transformation functions across 6 categories:

### Scaling

- **T1** — Min-Max Normalisation
- **T2** — Z-Score Standardisation
- **T3** — Robust Scaler
- **T4** — Log Transform
- **T5** — Power Transform (Box-Cox / Yeo-Johnson)

### Encoding

- **T6** — Label Encoding
- **T7** — One-Hot Encoding
- **T8** — Ordinal Encoding
- **T9** — Frequency Encoding
- **T10** — Target Encoding (mean target per category)

### Binning

- **T11** — Equal-Width Binning
- **T12** — Equal-Frequency Binning (Quantile)
- **T13** — Custom Threshold Binning

### Mathematical

- **T14** — Polynomial Features (degree 2, single column)
- **T15** — Interaction Term (`col_a * col_b`)
- **T16** — Custom Formula Column
- **T17** — Lag Feature (time-series)
- **T18** — Rolling Mean Feature

### Datetime

- **T19** — Extract Year
- **T20** — Extract Month
- **T21** — Extract Day
- **T22** — Extract Day-of-Week
- **T23** — Extract Hour

### Dimensionality

- **T24** — PCA (2 components)
- **T25** — Flag column as excluded from ML

## Data Safety

All `apply_*()` functions return a **new DataFrame** — the original is never mutated.

A snapshot is taken by the router before calling `apply_*()`.



# BACKEND STAGE 5 — EDA Service

## What this stage is

The first pure analysis stage after cleaning and transformation. No data is modified here — the working CSV remains read-only. It explores the data, relationships, patterns, and anomalies before formal statistical testing.

## What it does

- Implements 14 EDA sub-modules covering univariate, bivariate, time-series, multivariate, and outlier analysis
- Supports up to 5 simultaneous user-defined filters without modifying the working CSV
- Generates 8–12 algorithmic insights and rewrites them into a narrative using Gemini
- Answers natural language questions through Gemini and routes them to the appropriate analysis function
- Recommends suitable chart types for column combinations
- Maintains per-session history of analyses, filters, and questions
- Generates an end-of-stage summary of major findings

## Where it's used

- **Frontend EDA page** — 5-tab analysis interface with filter bar
- **Stage 6 (Statistics)** — EDA findings guide statistical test recommendations
- **Stage 8 (ML Strategy)** — correlation findings and target distribution inform model selection
- **Stage 9 (Report)** — EDA results become Section 5 of the final report
- **Gemini** — powers natural language questions and the auto-insights narrative

## Files in this stage

1. `services/eda_service.py` — 14 EDA sub-modules + filter engine + auto-insights + Gemini question answering
2. `prompts/eda_question.txt` — Gemini prompt template for natural language question answering
3. `routers/eda.py` — 5 endpoints (analyze, question, recommend-chart, insights, history)
4. `models/schemas.py` — updated with EDA schemas
5. `tests/test_eda.py` — verification tests

## Full EDA Function Map

| Module | Functions |
|---|---|
| **E1 — Numerical Univariate** | Mean, median, mode, variance, standard deviation, min, max, range, percentiles (P25/50/75/90/95/99), IQR, skewness, kurtosis, distribution label, histogram data, box plot statistics, density data |
| **E2 — Categorical Univariate** | Frequency table, proportions, most/least common categories, category count, Pareto data, rare category detection |
| **E3 — Numerical vs Numerical** | Pearson and Spearman correlation, scatter plot data, trend detection, regression line |
| **E4 — Numerical vs Categorical** | Group-wise mean, median, min, max, standard deviation, percentiles, box plot by category, distribution comparison, group difference detection |
| **E5 — Categorical vs Categorical** | Contingency table, cross-tabulation, category-wise proportions, association analysis, chi-square recommendation |
| **E6 — Time-Based EDA** | Trend over time, daily/weekly/monthly/quarterly/yearly aggregation, rolling statistics, growth rate, period-over-period comparison, peak/low detection, time gap detection |
| **E7 — Multivariate** | Correlation heatmap data, pairwise relationships, grouped multivariate comparisons |
| **E8 — Outlier Investigation** | IQR-based outlier visualization data, z-score analysis, extreme value investigation, outlier contribution |
| **E9 — Auto Insights** | Algorithmically generated insights + Gemini narrative |
| **E10 — NL Question** | Gemini-powered natural language question answering |
| **E11 — Chart Recommendation** | Recommends chart types based on column types and the user's question |
| **E12 — Filters** | Supports up to 5 simultaneous filters without modifying the working CSV |
| **E13 — EDA History** | Maintains per-session history of all analyses performed |
| **E14 — EDA Summary** | Generates an end-of-stage summary of all major findings |




# BACKEND STAGE 6 — Statistics Service

## What this stage is

The formal hypothesis testing stage. While EDA explores patterns informally, Stage 6 applies statistical tests with assumption checks, effect sizes, and plain English explanations.

## What it does

- Recommends appropriate statistical tests based on the research question and data characteristics
- Checks assumptions such as normality, homogeneity of variance, and sample size
- Implements 21 statistical test functions covering parametric, non-parametric, correlation, and categorical tests
- Computes effect sizes alongside test results
- Generates plain English explanations of results using Gemini
- Maintains per-session statistics history

## Where it's used

- **Frontend Stats page** — test builder, assumption checker, results panel, and post-hoc table
- **Stage 8 (ML Strategy)** — statistical significance findings inform feature selection recommendations
- **Stage 9 (Report)** — statistics results become Section 6 of the report
- **Gemini** — generates the plain English explanation for each test result

## Files in this stage

1. `services/stats_service.py` — 21 statistical functions + recommender + assumption checker
2. `prompts/stats_explain.txt` — Gemini prompt for plain English explanations
3. `routers/stats.py` — 4 endpoints
4. `models/schemas.py` — updated with stats schemas
5. `tests/test_stats.py` — verification tests


## Stage 6 Summary — What Was Built and Verified

- **21 statistical test functions** — Parametric (t-tests, ANOVA, regression), non-parametric (Mann-Whitney, Wilcoxon, Kruskal-Wallis), correlation (Pearson, Spearman, Kendall), categorical (Chi-square, Fisher's Exact), and distribution tests (Shapiro-Wilk, KS, Levene, Bartlett)
- **Test recommender** — Routes to the appropriate test based on column types, normality, group counts, and question keywords
- **Assumption checker** — Checks normality, equal variance, and sample size before running tests
- **Effect sizes** — Computes Cohen's d, eta-squared, Cramér's V, r, and R² alongside test results
- **Gemini plain English** — Generates a 2–3 sentence explanation for every result, with a rule-based fallback if Gemini is unavailable
- **Post-hoc Tukey HSD** — Automatically appended to ANOVA results when statistically significant
- **`_sanitize_result`** — Dispatcher-level NumPy type sanitizer that fixes `numpy.bool_` serialization across all 21 tests
- **Stats history** — Logs every test with its p-value and plain English explanation in the in-memory cache



# BACKEND STAGE 7 — NLP Service

## What this stage is

The text analysis stage. It processes free-form text columns using NLP techniques — from basic quality checks to sentiment analysis, keyword extraction, topic modelling, semantic similarity, and text clustering.

## What it does

- Detects text columns and assesses their quality
- Runs sentiment analysis using VADER
- Extracts keywords using TF-IDF
- Generates word clouds from term frequencies
- Detects language per column
- Runs topic modelling using LDA
- Finds similar records using sentence embeddings
- Clusters text records using K-Means on embeddings
- Generates NLP insights via Gemini with rule-based fallback
- Samples large datasets (>2000 rows) for embedding operations

## Where it's used

- **Frontend NLP page** — sentiment cards, keyword bar chart, word cloud, topic cards, similarity results, and cluster cards
- **Stage 9 (Report)** — NLP results become Section 7 of the report
- **Gemini** — generates the insights narrative from NLP results

## Files in this stage

1. `services/nlp_service.py` — 27 NLP functions
2. `prompts/nlp_insights.txt` — Gemini prompt template
3. `routers/nlp.py` — 5 endpoints
4. `models/schemas.py` — updated with NLP schemas
5. `tests/test_nlp.py` — verification tests


# BACKEND STAGE 8 — ML Strategy Service

## What this stage is

The advisory stage. It does not train models; it analyzes the cleaned and transformed dataset to recommend a suitable machine learning approach.

## What it does

- Detects the problem type from the target column and user's objective
- Recommends and ranks suitable models with suitability scores, strengths, and weaknesses
- Detects class imbalance and suggests handling techniques
- Recommends preprocessing and feature engineering
- Suggests evaluation metrics, data splits, and cross-validation
- Generates a modelling roadmap and pipeline architecture
- Uses Gemini for goal-specific reasoning, with rule-based fallback

## Where it's used

- **Frontend ML Strategy page** — problem banner, model cards, comparison table, technique checklist, imbalance warning, architecture diagram, and roadmap
- **Stage 9 (Report)** — ML strategy becomes Section 8 of the report
- **Gemini** — generates reasoning tailored to the user's objective

## Files in this stage

1. `services/ml_service.py` — 19 ML strategy functions
2. `prompts/ml_strategy.txt` — Gemini prompt template
3. `routers/ml.py` — 4 endpoints (recommend, summary, models, roadmap)
4. `models/schemas.py` — updated with ML schemas
5. `tests/test_ml.py` — verification tests

## Full ML Strategy Function Map

| Category | ID | Function | Brief Description |
|---|---|---|---|
| **Analysis** | **M1** | `detect_problem_type()` | Identifies classification, regression, or clustering. |
| **Analysis** | **M2** | `analyze_target()` | Analyzes target characteristics and distribution. |
| **Analysis** | **M3** | `detect_class_imbalance()` | Detects class imbalance. |
| **Analysis** | **M4** | `assess_dataset_readiness()` | Evaluates dataset readiness for ML. |
| **Analysis** | **M5** | `detect_leakage_risk()` | Flags potential data leakage. |
| **Analysis** | **M6** | `detect_multicollinearity()` | Detects highly correlated predictors. |
| **Models** | **M7** | `recommend_models()` | Scores and ranks suitable models. |
| **Models** | **M8** | `build_model_comparison()` | Compares recommended models. |
| **Techniques** | **M9** | `recommend_preprocessing()` | Suggests preprocessing techniques. |
| **Techniques** | **M10** | `recommend_feature_engineering()` | Recommends feature engineering. |
| **Techniques** | **M11** | `recommend_imbalance_handling()` | Suggests imbalance-handling methods. |
| **Evaluation** | **M12** | `recommend_split_strategy()` | Recommends train/test splitting. |
| **Evaluation** | **M13** | `recommend_cross_validation()` | Suggests cross-validation strategies. |
| **Evaluation** | **M14** | `recommend_metrics()` | Selects suitable evaluation metrics. |
| **Plan** | **M15** | `build_architecture()` | Defines the ML pipeline architecture. |
| **Plan** | **M16** | `build_roadmap()` | Generates a step-by-step modelling plan. |
| **Narrative** | **M17** | `generate_reasoning()` | Generates explanations using Gemini or rule-based fallback. |
| **Master** | **M18** | `build_ml_strategy()` | Combines recommendations into a complete strategy. |
| **Master** | **M19** | `ml_summary()` | Produces a compact summary for the report. |

## Important Note

This service is **advisory only** — it recommends ML approaches and does not train or evaluate models.


# BACKEND STAGE 9 — Report Generation

## What this stage is

The packaging stage. It collects results from Stages 2–8, assembles them into a single report, and exports a downloadable PDF or HTML file.

## What it does

- Generates a 10-section report covering profiling, cleaning, transformation, EDA, statistics, NLP, ML strategy, operation history, and appendix, with an Executive Summary
- Lets users select report sections and output format (PDF or HTML)
- Marks unperformed sections as "Not performed"
- Generates the Executive Summary using Gemini, with rule-based fallback; only summaries are sent, never raw rows
- Renders reports using a Jinja2 template with autoescaping
- Provides report downloads and full operation history as JSON
- Returns download URLs instead of filesystem paths

## Where it's used

- **Frontend Report page** — section checklist, completion badges, PDF/HTML toggle, preview, and download options
- **Gemini** — generates only the Executive Summary
- **Earlier stages** — reads their results without modifying them

## Files in this stage

1. `services/report_service.py` — collects section data, generates the Executive Summary, renders, and writes files
2. `prompts/report_summary.txt` — Gemini prompt for the Executive Summary
3. `templates/report_template.html` — Jinja2 report template with print CSS for PDF
4. `routers/report.py` — `generate`, `download/{format}`, and `history-export` endpoints
5. `models/schemas.py` — updated with report schemas
6. `tests/test_report.py` — verification tests

## Report Sections

| No. | Section | Description |
|---:|---|---|
| **1** | Executive Summary | Summarizes key findings and recommendations. |
| **2** | Profiling | Reports dataset structure, quality, and characteristics. |
| **3** | Cleaning | Documents detected issues and applied fixes. |
| **4** | Transformation | Summarizes transformations and feature engineering. |
| **5** | EDA | Presents exploratory analysis findings and patterns. |
| **6** | Statistics | Summarizes statistical tests, significance, and effect sizes. |
| **7** | NLP | Presents text analysis results and insights. |
| **8** | ML Strategy | Documents recommended models, preprocessing, and evaluation. |
| **9** | Full Operation History | Records operations performed throughout the workflow. |
| **10** | Appendix | Provides supporting information and additional details. |


