export default function ErrorMessage({ message, onRetry }) {
  if (!message) return null;
  return (
    <div className="rounded-lg bg-red-50 border border-red-200 p-4 flex items-start gap-3">
      <span className="text-red-500 text-lg flex-shrink-0">⚠</span>
      <div className="flex-1 min-w-0">
        <p className="text-sm text-red-700">{message}</p>
        {onRetry && (
          <button onClick={onRetry} className="mt-2 text-xs font-medium text-red-600 underline hover:no-underline">
            Try again
          </button>
        )}
      </div>
    </div>
  );
}