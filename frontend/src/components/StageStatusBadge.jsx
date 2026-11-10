import { useSession } from '../context/SessionContext';

const STAGE_LABELS = {
  profile: 'Profiled', clean: 'Cleaned', transform: 'Transformed',
  eda: 'Explored', stats: 'Tested', nlp: 'NLP done', ml: 'ML ready', report: 'Reported',
};

export default function StageStatusBadge() {
  const { stageStatus } = useSession();
  const done = Object.entries(stageStatus).filter(([, v]) => v);
  if (!done.length) return null;
  return (
    <div className="hidden md:flex items-center gap-1 flex-wrap">
      {done.map(([stage]) => (
        <span key={stage} className="badge bg-green-100 text-green-700">
          {STAGE_LABELS[stage] ?? stage}
        </span>
      ))}
    </div>
  );
}