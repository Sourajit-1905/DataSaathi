import { NavLink, useNavigate } from "react-router-dom";
import { useState } from "react";
import {
  Upload,
  ScanSearch,
  Eraser,
  Shuffle,
  BarChart2,
  FlaskConical,
  MessageSquare,
  Bot,
  FileText,
  Sparkles,
  RotateCcw,
  CheckCircle,
  Loader2,
} from "lucide-react";
import { useSession } from "../context/SessionContext";
import ConfirmModal from "./ConfirmModal";
import { datasetAPI } from "../api/api";
import { toast } from "react-toastify";

const NAV = [
  { to: "/upload", Icon: Upload, label: "Upload", always: true },
  { to: "/profile", Icon: ScanSearch, label: "Profile", stage: "profile" },
  { to: "/clean", Icon: Eraser, label: "Clean", stage: "clean" },
  { to: "/transform", Icon: Shuffle, label: "Transform", stage: "transform" },
  { to: "/eda", Icon: BarChart2, label: "Explore", stage: "eda" },
  { to: "/stats", Icon: FlaskConical, label: "Statistics", stage: "stats" },
  { to: "/nlp", Icon: MessageSquare, label: "NLP", stage: "nlp" },
  { to: "/ml", Icon: Bot, label: "ML Strategy", stage: "ml" },
  { to: "/report", Icon: FileText, label: "Report", stage: "report" },
  { to: "/ai", Icon: Sparkles, label: "AI Assistant", always: true },
];

function StageIcon({ status }) {
  if (status === "completed")
    return <CheckCircle className="h-3.5 w-3.5 text-green-500 flex-shrink-0" />;
  if (status === "in_progress")
    return (
      <Loader2 className="h-3.5 w-3.5 text-blue-400 animate-spin flex-shrink-0" />
    );
  return null;
}

export default function Sidebar({ open, onClose }) {
  const { sessionId, uploadMeta, stageStatus, clearSession } = useSession();
  const [showReset, setShowReset] = useState(false);
  const navigate = useNavigate();

  const handleReset = async () => {
    setShowReset(false);
    try {
      await datasetAPI.reset(sessionId);
      clearSession();
      navigate("/upload");
      toast.success("Dataset reset to original.");
    } catch {
      toast.error("Could not reset the dataset.");
    }
  };

  return (
    <>
      {open && (
        <div
          className="fixed inset-0 z-20 bg-black/30 md:hidden"
          onClick={onClose}
        />
      )}

      <aside
        className={`
        fixed inset-y-0 left-0 z-30 w-56 bg-white border-r border-gray-200
        flex flex-col transform transition-transform duration-200
        ${open ? "translate-x-0" : "-translate-x-full"}
        md:translate-x-0 md:static md:z-auto
      `}
      >
        <div className="h-14 flex items-center px-4 border-b border-gray-100 flex-shrink-0">
          <span className="text-lg font-bold text-blue-600">DataSaathi</span>
          <span className="ml-1 text-xs text-gray-400 font-medium">AI</span>
        </div>

        {uploadMeta && (
          <div className="mx-3 mt-3 mb-1 px-3 py-2 bg-blue-50 rounded-lg">
            <p className="text-xs font-medium text-blue-700 truncate">
              {uploadMeta.filename}
            </p>
            <p className="text-xs text-blue-500">
              {uploadMeta.rows?.toLocaleString()} rows · {uploadMeta.columns}{" "}
              cols
            </p>
          </div>
        )}

        <nav className="flex-1 overflow-y-auto py-2 px-2 space-y-0.5">
          {NAV.map(({ to, Icon, label, stage, always }) => {
            const locked = !always && !sessionId;
            const status = stage ? stageStatus[stage] : null;
            return (
              <NavLink
                key={to}
                to={locked ? "#" : to}
                onClick={locked ? (e) => e.preventDefault() : onClose}
                aria-disabled={locked}
                className={({ isActive }) => `
                  flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm font-medium
                  transition-colors select-none
                  ${
                    locked
                      ? "opacity-40 cursor-not-allowed text-gray-400"
                      : isActive
                        ? "bg-blue-50 text-blue-700"
                        : "text-gray-600 hover:bg-gray-100 hover:text-gray-900"
                  }
                `}
              >
                <Icon className="h-4 w-4 flex-shrink-0" />
                <span className="flex-1">{label}</span>
                {status && <StageIcon status={status} />}
              </NavLink>
            );
          })}
        </nav>

        {sessionId && (
          <div className="border-t border-gray-100 px-3 py-3 flex-shrink-0">
            <button
              onClick={() => setShowReset(true)}
              className="w-full flex items-center gap-2 text-xs text-red-500
                         hover:text-red-700 py-1.5 px-2 rounded hover:bg-red-50 transition-colors"
            >
              <RotateCcw className="h-3.5 w-3.5" />
              Reset dataset
            </button>
          </div>
        )}
      </aside>

      <ConfirmModal
        open={showReset}
        title="Reset to original?"
        message="This will undo all cleaning and transforms and clear all analysis results."
        onConfirm={handleReset}
        onCancel={() => setShowReset(false)}
        danger
      />
    </>
  );
}
