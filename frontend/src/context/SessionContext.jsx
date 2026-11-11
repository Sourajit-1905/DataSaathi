import { createContext, useContext, useState, useCallback } from "react";

const INITIAL_STAGES = {
  profile: "pending",
  clean: "pending",
  transform: "pending",
  eda: "pending",
  stats: "pending",
  nlp: "pending",
  ml: "pending",
  report: "pending",
};

const SessionContext = createContext(null);

export function SessionProvider({ children }) {
  const [sessionId, setSessionId] = useState(
    () => sessionStorage.getItem("ds_session") ?? null,
  );
  const [uploadMeta, setUploadMeta] = useState(() => {
    try {
      return JSON.parse(sessionStorage.getItem("ds_meta") ?? "null");
    } catch {
      return null;
    }
  });
  const [stageStatus, setStageStatus] = useState({ ...INITIAL_STAGES });

  const startSession = useCallback((id, meta) => {
    setSessionId(id);
    setUploadMeta(meta);
    sessionStorage.setItem("ds_session", id);
    sessionStorage.setItem("ds_meta", JSON.stringify(meta));
    setStageStatus({ ...INITIAL_STAGES });
  }, []);

  const clearSession = useCallback(() => {
    setSessionId(null);
    setUploadMeta(null);
    sessionStorage.removeItem("ds_session");
    sessionStorage.removeItem("ds_meta");
    setStageStatus({ ...INITIAL_STAGES });
  }, []);

  const updateStageStatus = useCallback((stage, status) => {
    setStageStatus((prev) => ({ ...prev, [stage]: status }));
  }, []);

  const markStage = useCallback(
    (stage, done = true) => {
      updateStageStatus(stage, done ? "completed" : "pending");
    },
    [updateStageStatus],
  );

  return (
    <SessionContext.Provider
      value={{
        sessionId,
        uploadMeta,
        stageStatus,
        startSession,
        clearSession,
        updateStageStatus,
        markStage,
      }}
    >
      {children}
    </SessionContext.Provider>
  );
}

export function useSession() {
  const ctx = useContext(SessionContext);
  if (!ctx) throw new Error("useSession must be used inside <SessionProvider>");
  return ctx;
}
