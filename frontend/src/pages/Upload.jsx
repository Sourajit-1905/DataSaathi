import { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import { useNavigate } from "react-router-dom";
import { toast } from "react-toastify";
import { datasetAPI } from "../api/api";
import { useSession } from "../context/SessionContext";
import { InlineSpinner } from "../components/LoadingSpinner";
import ErrorMessage from "../components/ErrorMessage";

const ACCEPT = {
  "text/csv": [".csv"],
  "application/vnd.ms-excel": [".xls", ".xlsx"],
  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": [
    ".xlsx",
  ],
};
const MAX_MB = 50;

function Preview({ data }) {
  if (!data) return null;
  const { rows, column_names: cols, column_dtypes: dtypes } = data;
  return (
    <div className="card mt-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="font-semibold text-gray-800">Preview (first 10 rows)</h3>
        <span className="text-xs text-gray-500">{rows?.length} rows shown</span>
      </div>
      <div className="overflow-x-auto">
        <table className="table-base">
          <thead>
            <tr>
              {cols?.map((c) => (
                <th key={c}>
                  <div className="font-medium text-gray-700">{c}</div>
                  <div className="font-normal text-gray-400 text-xs">
                    {dtypes?.[c]}
                  </div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows?.map((row, i) => (
              <tr key={i}>
                {cols?.map((c) => (
                  <td
                    key={c}
                    className={row[c] == null ? "text-gray-300 italic" : ""}
                  >
                    {row[c] == null ? "null" : String(row[c])}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default function Upload() {
  const { startSession } = useSession();
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [preview, setPreview] = useState(null);
  const [meta, setMeta] = useState(null);

  const handleFile = useCallback(
    async (file) => {
      setError(null);
      setPreview(null);
      setMeta(null);
      setLoading(true);
      try {
        const result = await datasetAPI.upload(file);
        setMeta(result);
        const prev = await datasetAPI.preview(result.session_id);
        setPreview({ ...prev, column_names: result.column_names });
        startSession(result.session_id, result);
        toast.success(
          `"${result.filename}" uploaded — ${result.rows.toLocaleString()} rows, ${result.columns} columns.`,
        );
      } catch (err) {
        setError(err.error ?? err.detail ?? "Upload failed.");
      } finally {
        setLoading(false);
      }
    },
    [startSession],
  );

  const onDrop = useCallback(
    (accepted) => {
      if (accepted.length) handleFile(accepted[0]);
    },
    [handleFile],
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: ACCEPT,
    maxSize: MAX_MB * 1024 * 1024,
    maxFiles: 1,
    onDropRejected: (files) => {
      const r = files[0]?.errors[0];
      setError(
        r?.code === "file-too-large"
          ? `File exceeds ${MAX_MB} MB limit.`
          : r?.code === "file-invalid-type"
            ? "Only .csv, .xlsx and .xls files are accepted."
            : (r?.message ?? "File rejected."),
      );
    },
  });

  return (
    <div className="max-w-2xl mx-auto">
      <h2 className="text-xl font-bold text-gray-900 mb-1">
        Upload your dataset
      </h2>
      <p className="text-sm text-gray-500 mb-6">
        Accepts .csv, .xlsx or .xls up to {MAX_MB} MB.
      </p>

      <div
        {...getRootProps()}
        className={`
        border-2 border-dashed rounded-xl p-10 text-center cursor-pointer transition-colors
        ${isDragActive ? "border-blue-400 bg-blue-50" : "border-gray-300 hover:border-blue-400 hover:bg-gray-50"}
      `}
      >
        <input {...getInputProps()} />
        {loading ? (
          <div className="flex flex-col items-center gap-2">
            <InlineSpinner className="h-8 w-8 text-blue-500" />
            <p className="text-sm text-gray-500">Uploading…</p>
          </div>
        ) : (
          <>
            <div className="text-4xl mb-3">📂</div>
            <p className="text-gray-700 font-medium">
              {isDragActive
                ? "Drop it here!"
                : "Drag & drop a file, or click to browse"}
            </p>
            <p className="text-xs text-gray-400 mt-1">
              .csv · .xlsx · .xls · max {MAX_MB} MB
            </p>
          </>
        )}
      </div>

      <ErrorMessage message={error} />

      {meta && (
        <div className="card mt-4 flex items-center gap-4">
          <div className="text-3xl">✅</div>
          <div>
            <p className="font-semibold text-gray-800">{meta.filename}</p>
            <p className="text-sm text-gray-500">
              {meta.rows?.toLocaleString()} rows · {meta.columns} columns ·{" "}
              {meta.file_size_kb < 1024
                ? `${meta.file_size_kb.toFixed(1)} KB`
                : `${(meta.file_size_kb / 1024).toFixed(2)} MB`}
            </p>
          </div>
          <div className="ml-auto">
            <button
              onClick={() => navigate("/profile")}
              className="btn-primary text-sm"
            >
              Continue →
            </button>
          </div>
        </div>
      )}

      <Preview data={preview} />
    </div>
  );
}
