import { X } from "lucide-react";
import FrequencyBarChart from "./charts/FrequencyBarChart";

function Stat({ label, value, sub }) {
  return (
    <div className="bg-gray-50 rounded-lg p-3">
      <p className="text-xs text-gray-500 mb-0.5">{label}</p>
      <p className="font-semibold text-gray-800 text-sm">{value ?? "—"}</p>
      {sub && <p className="text-xs text-gray-400 mt-0.5">{sub}</p>}
    </div>
  );
}

function dtypeBadgeColor(dtype) {
  if (!dtype) return "bg-gray-100 text-gray-500";
  if (dtype.startsWith("int") || dtype.startsWith("float"))
    return "bg-blue-100 text-blue-700";
  if (dtype.startsWith("object") || dtype.startsWith("string"))
    return "bg-purple-100 text-purple-700";
  if (dtype.startsWith("bool")) return "bg-green-100 text-green-700";
  if (dtype.startsWith("date")) return "bg-orange-100 text-orange-700";
  return "bg-gray-100 text-gray-500";
}

export default function ColumnDrawer({ column, data, onClose }) {
  if (!column) return null;

  const isNum = data?.column_type === "numerical";
  const isCat = data?.column_type === "categorical";

  let freqData = [];
  if (isCat && data?.value_counts) {
    freqData = Object.entries(data.value_counts)
      .slice(0, 15)
      .map(([value, count]) => ({ value: String(value).slice(0, 20), count }));
  }
  if (isNum && data?.histogram) {
    freqData = data.histogram.map((b) => ({
      value: `${b.bin_start?.toFixed(1)}`,
      count: b.count,
    }));
  }

  return (
    <>
      <div className="fixed inset-0 z-30 bg-black/20" onClick={onClose} />
      <aside
        className="fixed inset-y-0 right-0 z-40 w-full max-w-sm bg-white shadow-xl
                        flex flex-col overflow-hidden"
      >
        <div className="flex items-center gap-3 px-5 py-4 border-b border-gray-100">
          <div className="flex-1 min-w-0">
            <h3 className="font-semibold text-gray-900 truncate">{column}</h3>
            {data?.dtype && (
              <span className={`badge mt-0.5 ${dtypeBadgeColor(data.dtype)}`}>
                {data.dtype}
              </span>
            )}
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-md text-gray-400 hover:bg-gray-100 hover:text-gray-600"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto px-5 py-4 space-y-5">
          <div className="grid grid-cols-2 gap-2">
            <Stat
              label="Missing"
              value={
                data?.missing_count != null
                  ? `${data.missing_count.toLocaleString()} (${data.missing_pct?.toFixed(1)}%)`
                  : "—"
              }
            />
            <Stat
              label="Unique values"
              value={data?.unique_count?.toLocaleString() ?? "—"}
            />
            {isNum && (
              <>
                <Stat label="Mean" value={data?.mean?.toFixed(4) ?? "—"} />
                <Stat label="Median" value={data?.median?.toFixed(4) ?? "—"} />
                <Stat label="Std dev" value={data?.std?.toFixed(4) ?? "—"} />
                <Stat
                  label="Skewness"
                  value={data?.skewness?.toFixed(3) ?? "—"}
                  sub={
                    Math.abs(data?.skewness ?? 0) > 1
                      ? "High — consider log transform"
                      : undefined
                  }
                />
                <Stat label="Min" value={data?.min?.toFixed(4) ?? "—"} />
                <Stat label="Max" value={data?.max?.toFixed(4) ?? "—"} />
                <Stat
                  label="Outliers (IQR)"
                  value={data?.outlier_count?.toLocaleString() ?? "—"}
                  sub={
                    data?.outlier_pct != null
                      ? `${data.outlier_pct.toFixed(1)}% of rows`
                      : undefined
                  }
                />
              </>
            )}
            {isCat && (
              <>
                <Stat label="Top value" value={data?.top_value ?? "—"} />
                <Stat
                  label="Top frequency"
                  value={data?.top_freq?.toLocaleString() ?? "—"}
                />
              </>
            )}
          </div>

          {freqData.length > 0 && (
            <div>
              <h4 className="text-sm font-medium text-gray-700 mb-2">
                {isCat ? "Top values" : "Distribution"}
              </h4>
              <FrequencyBarChart data={freqData} />
            </div>
          )}

          {isNum && data?.q25 != null && (
            <div>
              <h4 className="text-sm font-medium text-gray-700 mb-2">
                Quartiles
              </h4>
              <div className="grid grid-cols-3 gap-2">
                <Stat label="Q1 (25%)" value={data.q25?.toFixed(3)} />
                <Stat label="Q2 (50%)" value={data.median?.toFixed(3)} />
                <Stat label="Q3 (75%)" value={data.q75?.toFixed(3)} />
              </div>
            </div>
          )}

          {isCat && data?.sample_values?.length > 0 && (
            <div>
              <h4 className="text-sm font-medium text-gray-700 mb-2">
                Sample values
              </h4>
              <div className="flex flex-wrap gap-1">
                {data.sample_values.slice(0, 20).map((v, i) => (
                  <span key={i} className="badge bg-gray-100 text-gray-600">
                    {String(v)}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      </aside>
    </>
  );
}
