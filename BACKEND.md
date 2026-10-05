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