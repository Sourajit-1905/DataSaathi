import { useEffect, useState } from 'react';
import { useSession } from '../context/SessionContext';
import StageStatusBadge from './StageStatusBadge';
import { aiAPI } from '../api/api';

function AIStatusDot() {
  const [status, setStatus] = useState(null);
  useEffect(() => { aiAPI.status().then(setStatus).catch(() => {}); }, []);
  if (!status) return null;
  const color = status.ai_enabled ? 'bg-green-400' : status.cooling_down ? 'bg-yellow-400' : 'bg-gray-300';
  const label = status.ai_enabled ? 'AI enabled'
    : status.cooling_down ? `AI cooling down (${status.retry_after_seconds}s)` : 'Rule-based mode';
  return (
    <span className="flex items-center gap-1 text-xs text-gray-500" title={label}>
      <span className={`inline-block h-2 w-2 rounded-full ${color}`} />
      {label}
    </span>
  );
}

export default function Header({ onMenuClick }) {
  const { uploadMeta } = useSession();
  return (
    <header className="h-14 bg-white border-b border-gray-200 flex items-center px-4 gap-3 flex-shrink-0">
      <button onClick={onMenuClick}
        className="md:hidden p-1.5 rounded-md text-gray-500 hover:bg-gray-100" aria-label="Open menu">
        <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
        </svg>
      </button>
      <h1 className="text-sm font-semibold text-gray-800 truncate flex-shrink-0">
        {uploadMeta ? uploadMeta.filename : 'DataSaathi'}
      </h1>
      <div className="flex-1" />
      <StageStatusBadge />
      <AIStatusDot />
    </header>
  );
}