'use client';

interface Props {
  sectionId: string;
  docName: string;
  jurisdiction: string;
  onRemove: () => void;
}

export function ContextPill({ sectionId, docName, jurisdiction, onRemove }: Props) {
  return (
    <div className="flex h-10 shrink-0 items-center justify-between bg-[#141414] px-3">
      <span className="truncate text-[12px] text-white/60">
        Sec {sectionId} / {docName} / {jurisdiction}
      </span>
      <button
        onClick={onRemove}
        className="text-white/30 transition-[color,transform] duration-150 ease-out hover:text-white/70 active:scale-[0.97]"
        aria-label="Remove section context"
        type="button"
      >
        x
      </button>
    </div>
  );
}
