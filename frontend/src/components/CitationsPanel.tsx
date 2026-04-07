'use client';

import { useState } from 'react';
import { JURISDICTION_COLORS, JURISDICTION_LABELS } from '@/lib/constants';
import type { Citation } from '@/lib/types';

interface Props {
  citations: Citation[];
}

export function CitationsPanel({ citations }: Props) {
  const [open, setOpen] = useState(false);

  return (
    <div className="mt-3">
      <button
        type="button"
        onClick={() => setOpen(value => !value)}
        className="inline-flex items-center gap-2 font-mono text-[10px] uppercase tracking-[0.2em] text-white/30 transition-colors hover:text-[#4f98a3] active:scale-[0.98]"
      >
        <span className={`transition-transform duration-150 ease-out ${open ? 'rotate-180' : ''}`}>&#x25BC;</span>
        {citations.length} citation{citations.length === 1 ? '' : 's'}
      </button>

      {open ? (
        <div className="mt-3 space-y-2">
          {citations.map(citation => {
            const color = JURISDICTION_COLORS[citation.jurisdiction] ?? '#888';
            return (
              <article key={citation.chunk_id} className="border border-white/[0.07] bg-white/[0.025] p-3">
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0 space-y-1">
                    <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-white/30">
                      Sec {citation.section_id}
                    </p>
                    <p className="truncate text-[12px] font-medium text-white/70">{citation.doc_name}</p>
                  </div>
                  <span
                    className="shrink-0 px-1.5 py-0.5 font-mono text-[9px] uppercase tracking-[0.14em]"
                    style={{ backgroundColor: `${color}24`, color }}
                  >
                    {JURISDICTION_LABELS[citation.jurisdiction] ?? citation.jurisdiction}
                  </span>
                </div>

                <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-2 font-mono text-[9px] uppercase tracking-[0.16em] text-white/30">
                  <span>Effective {citation.effective_date ?? 'not listed'}</span>
                  <a
                    href={citation.source_url}
                    target="_blank"
                    rel="noreferrer"
                    className="text-[#4f98a3] transition-colors hover:text-[#8ad2de]"
                  >
                    Source
                  </a>
                </div>
              </article>
            );
          })}
        </div>
      ) : null}
    </div>
  );
}
