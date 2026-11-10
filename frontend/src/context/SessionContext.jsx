import { createContext, useContext, useState, useCallback } from 'react';

const SessionContext = createContext(null);

export function SessionProvider({ children }) {
  const [sessionId, setSessionId] = useState(() => sessionStorage.getItem('ds_session') ?? null);
  const [uploadMeta, setUploadMeta] = useState(() => {
    try { return JSON.parse(sessionStorage.getItem('ds_meta') ?? 'null'); }
    catch { return null; }
  });
  const [stageStatus, setStageStatus] = useState({
    profile: false, clean: false, transform: false,
    eda: false, stats: false, nlp: false, ml: false, report: false,
  });

  const startSession = useCallback((id, meta) => {
    setSessionId(id);
    setUploadMeta(meta);
    sessionStorage.setItem('ds_session', id);
    sessionStorage.setItem('ds_meta', JSON.stringify(meta));
    setStageStatus({
      profile: false, clean: false, transform: false,
      eda: false, stats: false, nlp: false, ml: false, report: false,
    });
  }, []);

  const clearSession = useCallback(() => {
    setSessionId(null);
    setUploadMeta(null);
    sessionStorage.removeItem('ds_session');
    sessionStorage.removeItem('ds_meta');
    setStageStatus({
      profile: false, clean: false, transform: false,
      eda: false, stats: false, nlp: false, ml: false, report: false,
    });
  }, []);

  const markStage = useCallback((stage, done = true) => {
    setStageStatus((prev) => ({ ...prev, [stage]: done }));
  }, []);

  return (
    <SessionContext.Provider value={{
      sessionId, uploadMeta, stageStatus,
      startSession, clearSession, markStage,
    }}>
      {children}
    </SessionContext.Provider>
  );
}

export function useSession() {
  const ctx = useContext(SessionContext);
  if (!ctx) throw new Error('useSession must be used inside <SessionProvider>');
  return ctx;
}