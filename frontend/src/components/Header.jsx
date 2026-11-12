import { useEffect, useState } from "react";
import { Menu, Wifi, WifiOff, Loader2 } from "lucide-react";
import { useSession } from "../context/SessionContext";
import StageStatusBadge from "./StageStatusBadge";
import { aiAPI } from "../api/api";

function AIStatusDot() {
  const [status, setStatus] = useState(null);
  useEffect(() => {
    aiAPI
      .status()
      .then(setStatus)
      .catch(() => {});
  }, []);
  if (!status) return null;
  if (status.ai_enabled)
    return (
      <span
        className="flex items-center gap-1 text-xs text-green-600"
        title="AI enabled"
      >
        <Wifi className="h-3.5 w-3.5" />
        <span className="hidden sm:inline">AI on</span>
      </span>
    );
  if (status.cooling_down)
    return (
      <span
        className="flex items-center gap-1 text-xs text-yellow-600"
        title={`Cooling down — retry in ${status.retry_after_seconds}s`}
      >
        <Loader2 className="h-3.5 w-3.5 animate-spin" />
        <span className="hidden sm:inline">{status.retry_after_seconds}s</span>
      </span>
    );
  return (
    <span
      className="flex items-center gap-1 text-xs text-gray-400"
      title="Rule-based mode"
    >
      <WifiOff className="h-3.5 w-3.5" />
      <span className="hidden sm:inline">Rule-based</span>
    </span>
  );
}

export default function Header({ onMenuClick }) {
  const { uploadMeta } = useSession();
  return (
    <header className="h-14 bg-white border-b border-gray-200 flex items-center px-4 gap-3 flex-shrink-0">
      <button
        onClick={onMenuClick}
        className="md:hidden p-1.5 rounded-md text-gray-500 hover:bg-gray-100"
        aria-label="Open menu"
      >
        <Menu className="h-5 w-5" />
      </button>
      <h1 className="text-sm font-semibold text-gray-800 truncate flex-shrink-0">
        {uploadMeta ? uploadMeta.filename : "DataSaathi"}
      </h1>
      <div className="flex-1" />
      <StageStatusBadge />
      <AIStatusDot />
    </header>
  );
}
