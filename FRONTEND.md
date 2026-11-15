# Stage 1 - Scaffolding

| Foundation | Vite scaffold, Tailwind config, `api.js`, `SessionContext`, `useApi`, `Layout`, `Sidebar`, `Header`, `App.jsx`, `StageStatusBadge`, `LoadingSpinner` |
| ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |


# Stage 2 — Upload Page

Full dataset upload experience with drag-and-drop, client-side validation, file details card, scrollable column tags, preview table with dtypes, and session initialization.

## What it does

- Accepts .csv, .xlsx, .xls up to 50 MB
- Rejects wrong types and oversized files client-side before any network call
- Shows filename, size, row/column count, column name badges after upload
- Renders first 10 rows with column name + dtype headers
- Initializes `SessionContext` and sets `profile` stage to `in_progress`
- Navigates to `/profile` on “Proceed”

## Where used

`/upload` route — the entry point of the entire app.

## Files in this stage

1. `src/context/SessionContext.jsx` — stage statuses now `"pending"/"in_progress"/"completed"`, adds `updateStageStatus`
2. `src/components/StageStatusBadge.jsx` — updated to match new string statuses
3. `src/components/Sidebar.jsx` — Lucide icons, animated stage status indicators
4. `src/components/Header.jsx` — Lucide icons for AI status dot
5. `src/pages/Upload.jsx` — full upload page
6. `src/api/api.js` — single consolidated API file (delete `client.js` and `index.js` after adding this)


# Stage 3 — Profile Page

## What

Runs the backend profiling endpoint and displays a full dataset health report: quality score, missing values, outliers, duplicates, skewness, cardinality, potential targets, and AI insights. Each column opens a detail drawer.

## What it does

- Calls `POST /api/profile/{sid}/run` and shows a loading state
- Quality score ring (0–100) with colour coding
- Summary cards: rows, columns, missing, duplicates
- Per-column table with dtype, missing%, outlier count, skewness, cardinality
- Click any row → slide-in drawer with frequency bar chart and full stats
- AI insights panel (from the profile response)
- Marks stage `completed` and sets `clean` to `in_progress` on success

## Where used

`/profile` route.

## Files in this stage

1. `src/pages/Profile.jsx` — full page
2. `src/components/charts/FrequencyBarChart.jsx` — Recharts bar chart for column value counts
3. `src/components/ColumnDrawer.jsx` — slide-in detail panel
