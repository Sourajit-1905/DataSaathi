import { useEffect, useState, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { toast } from "react-toastify";
import {
  Play,
  ArrowRight,
  ChevronRight,
  AlertCircle,
  CheckCircle2,
  Info,
  BarChart2,
  Rows3,
  Columns3,
  TrendingUp,
  Lightbulb,
} from "lucide-react";
import { profileAPI } from "../api/api";
import { useSession } from "../context/SessionContext";
import LoadingSpinner from "../components/LoadingSpinner";
import ErrorMessage from "../components/ErrorMessage";
import ColumnDrawer from "../components/ColumnDrawer";

function scoreColor(score) {
  if (score >= 80)
    return {
      ring: "text-green-500",
      bg: "bg-green-50",
      text: "text-green-700",
      label: "Good",
    };
  if (score >= 60)
    return {
      ring: "text-yellow-500",
      bg: "bg-yellow-50",
      text: "text-yellow-700",
      label: "Fair",
    };
  return {
    ring: "text-red-500",
    bg: "bg-red-50",
    text: "text-red-700",
    label: "Poor",
  };
}

function missingColor(pct) {
  if (!pct || pct === 0) return "text-green-600";
  if (pct < 5) return "text-yellow-600";
  return "text-red-600";
}

function skewLabel(sk) {
  if (sk == null) return "—";
  const a = Math.abs(sk);
  if (a < 0.5) return "Normal";
  if (a < 1) return "Moderate";
  return "High";
}

function QualityRing({ score }) {
  const { ring, bg, text, label } = scoreColor(score ?? 0);
  const r = 36,
    cx = 44,
    cy = 44;
  const circ = 2 * Math.PI * r;
  const dash = ((score ?? 0) / 100) * circ;
  return (
    <div className={`card flex items-center gap-5 ${bg}`}>
      <svg width={88} height={88} className="flex-shrink-0 -rotate-90">
        <circle
          cx={cx}
          cy={cy}
          r={r}
          fill="none"
          stroke="#e5e7eb"
          strokeWidth={8}
        />
        <circle
          cx={cx}
          cy={cy}
          r={r}
          fill="none"
          className={ring}
          stroke="currentColor"
          strokeWidth={8}
          strokeDasharray={`${dash} ${circ}`}
          strokeLinecap="round"
          style={{ transition: "stroke-dasharray 0.6s ease" }}
        />
      </svg>
      <div>
        <p className="text-xs font-medium text-gray-500 uppercase tracking-wide">
          Quality score
        </p>
        <p className={`text-4xl font-bold ${text}`}>{score ?? "—"}</p>
        <p className={`text-sm font-medium ${text}`}>{label}</p>
      </div>
    </div>
  );
}

function SummaryCards({ profile }) {
  const dim = profile?.dimensions ?? {};
  const mv = profile?.missing_values ?? {};
  const dup = profile?.duplicate_rows ?? {};
  const totalMissing = Object.values(mv).reduce(
    (s, c) => s + (c?.count ?? 0),
    0,
  );
  const dupCount = dup?.count ?? 0;
  const cards = [
    {
      icon: Rows3,
      label: "Rows",
      value: dim.rows?.toLocaleString() ?? "—",
      color: "text-blue-600",
    },
    {
      icon: Columns3,
      label: "Columns",
      value: dim.columns?.toLocaleString() ?? "—",
      color: "text-purple-600",
    },
    {
      icon: AlertCircle,
      label: "Missing cells",
      value: totalMissing.toLocaleString(),
      color: totalMissing > 0 ? "text-yellow-600" : "text-green-600",
    },
    {
      icon: CheckCircle2,
      label: "Duplicate rows",
      value: dupCount.toLocaleString(),
      color: dupCount > 0 ? "text-orange-600" : "text-green-600",
    },
  ];
  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
      {cards.map(({ icon: Icon, label, value, color }) => (
        <div key={label} className="card flex items-center gap-3">
          <Icon className={`h-7 w-7 flex-shrink-0 ${color}`} />
          <div>
            <p className="text-xs text-gray-500">{label}</p>
            <p className={`text-lg font-bold ${color}`}>{value}</p>
          </div>
        </div>
      ))}
    </div>
  );
}

function ColumnTable({ columns, profile, onSelectColumn }) {
  const mv = profile?.missing_values ?? {};
  const out = profile?.outlier_detection ?? {};
  const dist = profile?.distribution_analysis ?? {};
  const card = profile?.cardinality ?? {};
  const numCols = new Set(profile?.numerical_columns ?? []);
  const catCols = new Set(profile?.categorical_columns ?? []);
  return (
    <div className="card overflow-hidden p-0">
      <div className="px-5 py-3 border-b border-gray-100 flex items-center gap-2">
        <BarChart2 className="h-4 w-4 text-gray-400" />
        <h3 className="font-semibold text-gray-800">Column summary</h3>
        <span className="text-xs text-gray-400 ml-auto">
          click a row for details
        </span>
      </div>
      <div className="overflow-x-auto">
        <table className="table-base">
          <thead>
            <tr>
              <th>Column</th>
              <th>Type</th>
              <th>Missing</th>
              <th>Outliers</th>
              <th>Skewness</th>
              <th>Cardinality</th>
              <th className="w-6" />
            </tr>
          </thead>
          <tbody>
            {columns.map((col) => {
              const mvData = mv[col] ?? {};
              const outData = out[col] ?? {};
              const distCol = dist[col] ?? {};
              const skVal = distCol?.skewness ?? null;
              const cardObj = card[col] ?? {};
              const cardVal =
                cardObj?.unique_count ?? cardObj?.cardinality ?? null;
              const pct = mvData.pct ?? 0;
              const dtype = numCols.has(col)
                ? "numerical"
                : catCols.has(col)
                  ? "categorical"
                  : "";
              return (
                <tr
                  key={col}
                  className="cursor-pointer hover:bg-blue-50 transition-colors"
                  onClick={() => onSelectColumn(col)}
                >
                  <td className="font-medium text-gray-800">{col}</td>
                  <td>
                    {dtype ? (
                      <span className="badge bg-gray-100 text-gray-600 text-xs">
                        {dtype}
                      </span>
                    ) : (
                      <span className="text-gray-300">—</span>
                    )}
                  </td>
                  <td className={`font-medium ${missingColor(pct)}`}>
                    {pct > 0 ? `${pct.toFixed(1)}%` : "—"}
                  </td>
                  <td
                    className={
                      outData.outlier_count > 0
                        ? "text-orange-600 font-medium"
                        : "text-gray-400"
                    }
                  >
                    {outData.outlier_count ?? "—"}
                  </td>
                  <td
                    className={
                      Math.abs(skVal ?? 0) > 1
                        ? "text-yellow-600 font-medium"
                        : "text-gray-500"
                    }
                  >
                    {skVal != null
                      ? `${skVal.toFixed(2)} (${skewLabel(skVal)})`
                      : "—"}
                  </td>
                  <td className="text-gray-500">
                    {cardVal?.toLocaleString() ?? "—"}
                  </td>
                  <td>
                    <ChevronRight className="h-4 w-4 text-gray-300" />
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function AIInsights({ insights }) {
  if (!insights?.length) return null;
  return (
    <div className="card">
      <div className="flex items-center gap-2 mb-3">
        <Lightbulb className="h-4 w-4 text-yellow-500" />
        <h3 className="font-semibold text-gray-800">AI Insights</h3>
      </div>
      <ul className="space-y-2">
        {insights.map((ins, i) => (
          <li key={i} className="flex items-start gap-2 text-sm text-gray-700">
            <span className="mt-1 h-1.5 w-1.5 rounded-full bg-blue-400 flex-shrink-0" />
            {ins}
          </li>
        ))}
      </ul>
    </div>
  );
}

function PotentialTargets({ targets }) {
  if (!targets?.length) return null;
  return (
    <div className="card">
      <div className="flex items-center gap-2 mb-3">
        <TrendingUp className="h-4 w-4 text-blue-500" />
        <h3 className="font-semibold text-gray-800">Potential ML targets</h3>
      </div>
      <div className="flex flex-wrap gap-2">
        {targets.map((t) => (
          <div
            key={t.column}
            className="rounded-lg border border-blue-100 bg-blue-50 px-3 py-2"
          >
            <p className="text-sm font-medium text-blue-800">{t.column}</p>
            <p className="text-xs text-blue-500">
              {t.suitability ?? t.reasoning ?? t.reason ?? ""}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}

function buildColumnData(col, profile) {
  if (!profile) return null;
  const mv = profile.missing_values?.[col] ?? {};
  const out = profile.outlier_detection?.[col] ?? {};
  const dist = profile.distribution_analysis?.[col] ?? {};
  const skVal = dist?.skewness ?? null;
  const card = profile.cardinality?.[col] ?? {};
  const freq =
    profile.frequency_tables?.[col] ??
    profile.categorical_imbalance?.[col] ??
    {};
  const numCols = new Set(profile?.numerical_columns ?? []);
  const catCols = new Set(profile?.categorical_columns ?? []);
  const dtype = numCols.has(col)
    ? "numerical"
    : catCols.has(col)
      ? "categorical"
      : "";
  const isNum = numCols.has(col);
  return {
    dtype,
    column_type: isNum ? "numerical" : "categorical",
    missing_count: mv.count,
    missing_pct: mv.pct,
    unique_count: card.unique_count ?? card.cardinality,
    mean: dist.mean,
    median: dist.median ?? dist.p50,
    std: dist.std,
    min: dist.min,
    max: dist.max,
    q25: dist.p25 ?? dist.q25,
    q75: dist.p75 ?? dist.q75,
    skewness: skVal,
    outlier_count: out.outlier_count,
    outlier_pct: out.pct ?? out.outlier_pct,
    histogram: dist.histogram,
    top_value: freq.top_value ?? freq.most_common,
    top_freq: freq.top_freq ?? freq.most_common_count,
    value_counts: freq.value_counts,
    sample_values: card.sample_values ?? card.samples,
  };
}

export default function Profile() {
  const { sessionId, uploadMeta, updateStageStatus } = useSession();
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [runResult, setRunResult] = useState(null);
  const [profile, setProfile] = useState(null);
  const [selCol, setSelCol] = useState(null);

  const runProfile = useCallback(async () => {
    if (!sessionId) return;
    setLoading(true);
    setError(null);
    try {
      const run = await profileAPI.run(sessionId);
      setRunResult(run);
      const full = await profileAPI.getResult(sessionId);
      setProfile(full?.profile ?? full);
      updateStageStatus("profile", "completed");
      updateStageStatus("clean", "in_progress");
      toast.success("Profiling complete.");
    } catch (err) {
      setError(err.error ?? err.detail ?? "Profiling failed.");
      updateStageStatus("profile", "pending");
    } finally {
      setLoading(false);
    }
  }, [sessionId, updateStageStatus]);

  useEffect(() => {
    if (sessionId && !profile && !loading) runProfile();
  }, [sessionId]); // eslint-disable-line react-hooks/exhaustive-deps

  if (!sessionId) {
    return (
      <div className="max-w-xl mx-auto text-center py-20">
        <Info className="h-10 w-10 text-gray-300 mx-auto mb-3" />
        <p className="text-gray-500">
          Upload a dataset first to run profiling.
        </p>
        <button
          onClick={() => navigate("/upload")}
          className="btn-primary mt-4"
        >
          Go to Upload
        </button>
      </div>
    );
  }

  const qualityScore =
    runResult?.quality_summary?.score ?? profile?.quality_summary?.score;
  const insights = profile?.insights ?? [];

  return (
    <div className="max-w-5xl mx-auto space-y-5">
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div>
          <h2 className="text-xl font-bold text-gray-900">Dataset Profile</h2>
          {uploadMeta && (
            <p className="text-sm text-gray-500 mt-0.5">
              {uploadMeta.filename}
            </p>
          )}
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={runProfile}
            disabled={loading}
            className="btn-secondary flex items-center gap-1.5 text-sm"
          >
            <Play className="h-4 w-4" />
            {loading ? "Running…" : profile ? "Re-run" : "Run Profile"}
          </button>
          {profile && (
            <button
              onClick={() => navigate("/clean")}
              className="btn-primary flex items-center gap-1.5 text-sm"
            >
              Proceed to Clean <ArrowRight className="h-4 w-4" />
            </button>
          )}
        </div>
      </div>

      {loading && (
        <LoadingSpinner label="Analysing all columns — this takes a few seconds…" />
      )}
      {error && <ErrorMessage message={error} onRetry={runProfile} />}

      {profile && !loading && (
        <>
          <div className="grid md:grid-cols-2 gap-5">
            <QualityRing score={qualityScore} />
            {insights.length > 0 && <AIInsights insights={insights} />}
          </div>
          <SummaryCards profile={profile} />
          <ColumnTable
            columns={
              uploadMeta?.column_names ??
              Object.keys(profile.missing_values ?? {})
            }
            profile={profile}
            onSelectColumn={setSelCol}
          />
          <PotentialTargets targets={profile.potential_targets} />
        </>
      )}

      <ColumnDrawer
        column={selCol}
        data={selCol ? buildColumnData(selCol, profile) : null}
        onClose={() => setSelCol(null)}
      />
    </div>
  );
}
