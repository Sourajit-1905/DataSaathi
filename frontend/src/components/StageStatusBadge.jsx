import { useSession } from "../context/SessionContext";

const STAGE_LABELS = {
  profile: "Profiled",
  clean: "Cleaned",
  transform: "Transformed",
  eda: "Explored",
  stats: "Tested",
  nlp: "NLP",
  ml: "ML ready",
  report: "Reported",
};

export default function StageStatusBadge() {
  const { stageStatus } = useSession();
  const completed = Object.entries(stageStatus).filter(
    ([, v]) => v === "completed",
  );
  if (!completed.length) return null;
  return (
    <div className="hidden md:flex items-center gap-1 flex-wrap">
      {completed.map(([stage]) => (
        <span key={stage} className="badge bg-green-100 text-green-700">
          {STAGE_LABELS[stage] ?? stage}
        </span>
      ))}
    </div>
  );
}
