interface Props {
  score: number;
}

const LEVELS = {
  HIGH: {
    label: 'HIGH',
    classes: 'border-emerald-300/20 bg-emerald-300/10 text-emerald-200/80',
  },
  MEDIUM: {
    label: 'MEDIUM',
    classes: 'border-amber-300/20 bg-amber-300/10 text-amber-200/80',
  },
  LOW: {
    label: 'LOW',
    classes: 'border-rose-300/20 bg-rose-300/10 text-rose-200/80',
  },
} as const;

export function ConfidenceBadge({ score }: Props) {
  const key = score >= 0.75 ? 'HIGH' : score >= 0.5 ? 'MEDIUM' : 'LOW';
  const { label, classes } = LEVELS[key];

  return (
    <span
      className={`inline-flex items-center border px-2 py-0.5 font-mono text-[9px] font-medium uppercase tracking-[0.18em] ${classes}`}
    >
      {label} {(score * 100).toFixed(0)}%
    </span>
  );
}
