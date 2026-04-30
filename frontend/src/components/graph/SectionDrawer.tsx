'use client';

import { JURISDICTION_COLORS, JURISDICTION_LABELS } from '@/lib/constants';
import type { SectionContent } from '@/lib/types';

interface Props {
  content: SectionContent | null;
  isLoading: boolean;
  onClose: () => void;
  onNodeNavigate: (sectionId: string, jurisdiction: string) => void;
  onChatAboutSection: (sectionId: string, title: string, docName: string, jurisdiction: string) => void;
}

export function SectionDrawer({
  content,
  isLoading,
  onClose,
  onNodeNavigate,
  onChatAboutSection,
}: Props) {
  const isOpen = isLoading || content !== null;
  const color = content ? JURISDICTION_COLORS[content.jurisdiction] ?? '#888' : '#888';

  return (
    <div
      className={`absolute z-20 flex min-h-0 flex-col overflow-hidden bg-[#141414]/95 shadow-[0_22px_70px_rgba(0,0,0,0.4)] backdrop-blur transition-[opacity,transform] duration-300 ease-out
        max-lg:inset-x-0 max-lg:bottom-0 max-lg:h-[70%] max-lg:rounded-t-[24px] max-lg:border-t max-lg:border-white/10
        lg:inset-x-3 lg:bottom-3 lg:max-h-[42%] lg:border lg:border-white/[0.07]
        ${isOpen ? 'pointer-events-auto translate-y-0 opacity-100' : 'pointer-events-none translate-y-full lg:translate-y-3 opacity-0'}
      `}
      aria-hidden={!isOpen}
    >
      <div className="h-1.5 w-12 shrink-0 self-center rounded-full bg-zinc-800 my-3 lg:hidden" />
      {content ? (
        <header className="flex shrink-0 items-start justify-between gap-3 border-b border-white/[0.06] px-4 py-3">
          <div className="min-w-0 flex-1">
            <div className="flex flex-wrap items-center gap-2">
              <span className="font-mono text-[10px] uppercase tracking-[0.22em] text-white/30">Selected statute</span>
              <span
                className="px-1.5 py-0.5 font-mono text-[9px] uppercase tracking-[0.14em]"
                style={{ backgroundColor: `${color}24`, color }}
              >
                {JURISDICTION_LABELS[content.jurisdiction] ?? content.jurisdiction}
              </span>
              {content.effective_date ? (
                <span className="font-mono text-[9px] uppercase tracking-[0.14em] text-white/25">
                  Effective {content.effective_date}
                </span>
              ) : null}
            </div>
            <p className="mt-1 truncate text-sm font-semibold text-white/90">
              Sec {content.section_id} / {content.title}
            </p>
            <p className="mt-0.5 truncate text-[11px] text-white/30">{content.doc_name}</p>
          </div>

          <div className="flex shrink-0 items-center gap-3">
            {content.source_url ? (
              <a
                href={content.source_url}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-block font-mono text-[9px] uppercase tracking-[0.18em] text-white/30 transition-[color,transform] duration-150 ease-out hover:text-[#4f98a3] active:scale-[0.97]"
              >
                View PDF
              </a>
            ) : null}
            <button
              onClick={onClose}
              className="text-white/30 transition-[color,transform] duration-150 ease-out hover:text-white/70 active:scale-[0.97]"
              aria-label="Close section drawer"
              type="button"
            >
              x
            </button>
          </div>
        </header>
      ) : null}

      {isLoading ? (
        <div className="flex flex-1 items-center justify-center">
          <div className="flex items-center gap-2 font-mono text-[10px] uppercase tracking-[0.24em] text-white/30">
            {[0, 1, 2].map(i => (
              <span
                key={i}
                className="h-1.5 w-1.5 rounded-full bg-white/20 animate-pulse"
                style={{ animationDelay: `${i * 150}ms` }}
              />
            ))}
            Loading section
          </div>
        </div>
      ) : null}

      {!isLoading && content ? (
        <>
          <div className="ledger-scroll min-h-0 flex-1 overflow-y-auto px-4 py-3 text-[13px] leading-6 text-white/70">
            {content.chunks.map((chunk, index) => (
              <article key={chunk.chunk_id} className={index > 0 ? 'mt-4 border-t border-white/[0.05] pt-4' : ''}>
                <p className="whitespace-pre-wrap">{chunk.text}</p>
              </article>
            ))}
          </div>

          <footer className="flex shrink-0 items-center gap-3 border-t border-white/[0.06] px-4 py-2.5">
            <div className="ledger-scroll flex min-w-0 flex-1 gap-1.5 overflow-x-auto">
              {content.connected_sections.slice(0, 10).map((section, index) => (
                <button
                  key={`${section.section_id}-${index}`}
                  onClick={() => onNodeNavigate(section.section_id, section.jurisdiction)}
                  className="shrink-0 border border-white/[0.08] bg-white/[0.02] px-2 py-1 font-mono text-[9px] uppercase tracking-[0.14em] text-white/40 transition-[background-color,border-color,color,transform] duration-150 ease-out hover:border-white/20 hover:bg-white/[0.05] hover:text-white/75 active:scale-[0.97]"
                  type="button"
                  style={{
                    borderColor: section.edge_type.startsWith('DERIVED') ? 'rgba(232,175,52,0.28)' : undefined,
                  }}
                >
                  Sec {section.section_id}
                </button>
              ))}
            </div>

            <button
              onClick={() =>
                onChatAboutSection(content.section_id, content.title, content.doc_name, content.jurisdiction)
              }
              className="shrink-0 border border-[#4f98a3]/35 bg-[#4f98a3]/10 px-3 py-1.5 text-[11px] font-medium text-[#9ed4dc] transition-[background-color,border-color,transform] duration-150 ease-out hover:border-[#4f98a3]/70 hover:bg-[#4f98a3]/16 active:scale-[0.97]"
              type="button"
            >
              Chat about section
            </button>
          </footer>
        </>
      ) : null}
    </div>
  );
}
