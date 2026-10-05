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

### 1. Dataset Dimensions

Determines the number of rows, columns, dataset size, and memory usage.

### 2. Column Identification and Meaning

Identifies columns and estimates their semantic meaning from names and values.

### 3. Numerical Column Detection

Identifies columns containing numerical data such as measurements, prices, ages, or quantities.

### 4. Categorical Column Detection

Identifies columns containing discrete categorical values such as gender, city, or department.

### 5. Date/Time Column Detection

Detects date/time columns and identifies their formats.

### 6. Missing Value Analysis

Reports the number and percentage of missing values in each column.

### 7. Unique Value and Cardinality Analysis

Calculates unique values and cardinality to identify low-, medium-, and high-cardinality features.

### 8. Duplicate Row Detection

Identifies and reports completely duplicated rows.

### 9. Duplicate Column Detection

Identifies columns containing identical data and potential redundancy.

### 10. Numerical Statistical Summary

Provides minimum, maximum, mean, median, standard deviation, and range for numerical columns.

### 11. Suspicious and Invalid Value Detection

Flags potentially invalid or suspicious values, such as negative ages or percentages above 100%.

### 12. Data Type Consistency

Checks whether column values match their assigned data types and detects inconsistent formats.

### 13. Constant and Near-Constant Column Detection

Identifies columns with one value or a highly dominant value that may provide little analytical value.

### 14. Identifier Detection

Identifies columns likely to represent IDs based on names and uniqueness patterns.

### 15. Outlier Detection

Flags unusual numerical observations using methods such as IQR or standard deviation.

### 16. Distribution Analysis

Analyzes numerical distributions using skewness and other basic characteristics.

### 17. Categorical Distribution and Class Imbalance

Analyzes category frequencies and identifies potential class imbalance.

### 18. Date/Time Coverage Analysis

Reports date ranges, time coverage, and potential temporal gaps or irregularities.

### 19. Initial Relationship Screening

Performs basic relationship analysis between numerical variables, such as correlation analysis.

### 20. Potential Target Detection

Identifies possible machine learning target columns based on data type, cardinality, and naming patterns.

### 21. Data Quality Summary

Combines detected issues into an overall summary of dataset quality.

### 22. Automatic Profiling Insights

Converts profiling results into human-readable observations and warnings.

The profiling results provide the foundation for Data Cleaning, EDA, Statistical Analysis, Machine Learning, and AI-based Data Insights.