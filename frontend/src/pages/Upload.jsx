import { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import { useNavigate } from "react-router-dom";
import { toast } from "react-toastify";
import {
  UploadCloud,
  FileSpreadsheet,
  CheckCircle2,
  AlertCircle,
  ArrowRight,
  X,
} from "lucide-react";
import { datasetAPI } from "../api/api";
import { useSession } from "../context/SessionContext";
import { InlineSpinner } from "../components/LoadingSpinner";

const ACCEPT = {
  "text/csv": [".csv"],
  "application/vnd.ms-excel": [".xls"],
  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": [
    ".xlsx",
  ],
};
const MAX_MB = 50;
const MAX_B = MAX_MB * 1024 * 1024;

function fmtSize(kb) {
  if (kb < 1024) return `${kb.toFixed(1)} KB`;
  return `${(kb / 1024).toFixed(2)} MB`;
}

function dtypeColor(dtype) {
  if (!dtype) return "bg-gray-100 text-gray-500";
  if (dtype.startsWith("int") || dtype.startsWith("float"))
    return "bg-blue-100 text-blue-700";
  if (dtype.startsWith("object") || dtype.startsWith("string"))
    return "bg-purple-100 text-purple-700";
  if (dtype.startsWith("bool")) return "bg-green-100 text-green-700";
  if (dtype.startsWith("date") || dtype === "datetime64[ns]")
    return "bg-orange-100 text-orange-700";
  return "bg-gray-100 text-gray-500";
}

function UploadError({ message, onDismiss }) {
  if (!message) return null;
  return (
    <div className="mt-4 flex items-start gap-3 rounded-lg bg-red-50 border border-red-200 p-4">
      <AlertCircle className="h-5 w-5 text-red-500 flex-shrink-0 mt-0.5" />
      <p className="text-sm text-red-700 flex-1">{message}</p>
      <button onClick={onDismiss} className="text-red-400 hover:text-red-600">
        <X className="h-4 w-4" />
      </button>
    </div>
  );
}

function MetaCard({ meta, onProceed }) {
  return (
    <div className="card mt-4 flex items-center gap-4">
      <CheckCircle2 className="h-9 w-9 text-green-500 flex-shrink-0" />
      <div className="flex-1 min-w-0">
        <p className="font-semibold text-gray-800 truncate">{meta.filename}</p>
        <p className="text-sm text-gray-500">
          {meta.rows?.toLocaleString()} rows &nbsp;·&nbsp;
          {meta.columns} columns &nbsp;·&nbsp;
          {fmtSize(meta.file_size_kb)}
        </p>
      </div>
      <button
        onClick={onProceed}
        className="btn-primary flex items-center gap-1.5 text-sm"
      >
        Proceed <ArrowRight className="h-4 w-4" />
      </button>
    </div>
  );
}

function ColumnTags({ columns, dtypes }) {
  if (!columns?.length) return null;
  return (
    <div className="card mt-4">
      <h3 className="font-semibold text-gray-800 mb-3">
        Columns{" "}
        <span className="text-sm font-normal text-gray-400">
          ({columns.length})
        </span>
      </h3>
      <div className="flex flex-wrap gap-1.5 max-h-40 overflow-y-auto pr-1">
        {columns.map((col) => (
          <span
            key={col}
            className={`badge ${dtypeColor(dtypes?.[col])}`}
            title={dtypes?.[col] ?? ""}
          >
            {col}
          </span>
        ))}
      </div>
    </div>
  );
}

function PreviewTable({ rows, columns, dtypes }) {
  if (!rows?.length || !columns?.length) return null;
  return (
    <div className="card mt-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="font-semibold text-gray-800">Preview</h3>
        <span className="text-xs text-gray-400">first {rows.length} rows</span>
      </div>
      <div className="overflow-x-auto">
        <table className="table-base">
          <thead>
            <tr>
              {columns.map((col) => (
                <th key={col}>
                  <div className="text-gray-700">{col}</div>
                  {dtypes?.[col] && (
                    <div
                      className={`text-xs font-normal px-1 py-0.5 rounded mt-0.5 inline-block
                                     ${dtypeColor(dtypes[col])}`}
                    >
                      {dtypes[col]}
                    </div>
                  )}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, i) => (
              <tr key={i}>
                {columns.map((col) => (
                  <td
                    key={col}
                    className={
                      row[col] == null ? "text-gray-300 italic text-xs" : ""
                    }
                  >
                    {row[col] == null ? "null" : String(row[col]).slice(0, 60)}
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
  const { startSession, updateStageStatus } = useSession();
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [meta, setMeta] = useState(null);
  const [preview, setPreview] = useState(null);

  const handleFile = useCallback(
    async (file) => {
      setError(null);
      setMeta(null);
      setPreview(null);
      setLoading(true);
      try {
        const result = await datasetAPI.upload(file);
        const prev = await datasetAPI.preview(result.session_id);
        setMeta(result);
        setPreview(prev);
        startSession(result.session_id, result);
        updateStageStatus("profile", "in_progress");
        toast.success(
          `"${result.filename}" uploaded — ${result.rows.toLocaleString()} rows, ${result.columns} columns.`,
        );
      } catch (err) {
        setError(err.error ?? err.detail ?? "Upload failed. Please try again.");
      } finally {
        setLoading(false);
      }
    },
    [startSession, updateStageStatus],
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
    maxSize: MAX_B,
    maxFiles: 1,
    onDropRejected: (rejected) => {
      const code = rejected[0]?.errors[0]?.code;
      setError(
        code === "file-too-large"
          ? `File exceeds the ${MAX_MB} MB limit.`
          : code === "file-invalid-type"
            ? "Only .csv, .xlsx and .xls files are accepted."
            : (rejected[0]?.errors[0]?.message ?? "File rejected."),
      );
    },
  });

  return (
    <div className="max-w-2xl mx-auto">
      <div className="mb-6">
        <h2 className="text-xl font-bold text-gray-900">Upload your dataset</h2>
        <p className="text-sm text-gray-500 mt-1">
          Accepts .csv, .xlsx or .xls up to {MAX_MB} MB. Your file is processed
          locally — nothing is sent to external servers.
        </p>
      </div>

      <div
        {...getRootProps()}
        className={`
        border-2 border-dashed rounded-xl p-12 text-center cursor-pointer transition-colors select-none
        ${isDragActive ? "border-blue-400 bg-blue-50" : "border-gray-300 hover:border-blue-400 hover:bg-gray-50"}
      `}
      >
        <input {...getInputProps()} />
        {loading ? (
          <div className="flex flex-col items-center gap-3">
            <InlineSpinner className="h-10 w-10 text-blue-500" />
            <p className="text-sm text-gray-500">Uploading and reading file…</p>
          </div>
        ) : (
          <>
            {isDragActive ? (
              <UploadCloud className="h-12 w-12 text-blue-400 mx-auto mb-3" />
            ) : (
              <FileSpreadsheet className="h-12 w-12 text-gray-300 mx-auto mb-3" />
            )}
            <p className="text-gray-700 font-medium">
              {isDragActive
                ? "Drop it here!"
                : "Drag & drop a file, or click to browse"}
            </p>
            <p className="text-xs text-gray-400 mt-1">
              .csv &nbsp;·&nbsp; .xlsx &nbsp;·&nbsp; .xls &nbsp;·&nbsp; max{" "}
              {MAX_MB} MB
            </p>
          </>
        )}
      </div>

      <UploadError message={error} onDismiss={() => setError(null)} />
      {meta && <MetaCard meta={meta} onProceed={() => navigate("/profile")} />}
      {meta && preview && (
        <ColumnTags
          columns={meta.column_names}
          dtypes={preview.column_dtypes}
        />
      )}
      {meta && preview && (
        <PreviewTable
          rows={preview.rows}
          columns={meta.column_names}
          dtypes={preview.column_dtypes}
        />
      )}
    </div>
  );
}
