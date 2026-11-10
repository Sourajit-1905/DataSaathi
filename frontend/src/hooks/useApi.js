import { useState, useCallback } from 'react';
import { toast } from 'react-toastify';

export function useApi(apiFn, { onSuccess, onError, successMessage } = {}) {
  const [data,    setData]    = useState(null);
  const [loading, setLoading] = useState(false);
  const [error,   setError]   = useState(null);

  const run = useCallback(async (...args) => {
    setLoading(true);
    setError(null);
    try {
      const result = await apiFn(...args);
      setData(result);
      if (successMessage) toast.success(successMessage);
      onSuccess?.(result);
      return result;
    } catch (err) {
      const message = err.error ?? err.detail ?? 'An unexpected error occurred.';
      setError(message);
      if (onError) onError(err);
      else toast.error(message);
      return null;
    } finally {
      setLoading(false);
    }
  }, [apiFn, onSuccess, onError, successMessage]);

  const reset = useCallback(() => {
    setData(null); setError(null); setLoading(false);
  }, []);

  return { data, loading, error, run, reset, setData };
}