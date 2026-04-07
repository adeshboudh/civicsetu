'use client';

import { JURISDICTION_COLORS, JURISDICTION_LABELS } from '@/lib/constants';
import type { GraphNode, SectionContent } from '@/lib/types';

interface Props {
  node: GraphNode;
  sectionContent: SectionContent | null;
  isSectionLoading: boolean;
  onChatAboutSection: (sectionId: string, title: string, docName: string, jurisdiction: string) => void;
  onClose: () => void;
}

export function NodeInfoCard({
  node,
  sectionContent,
  isSectionLoading,
  onChatAboutSection,
  onClose,
}: Props) {
  const color = JURISDICTION_COLORS[node.jurisdiction] ?? '#888';
  const refsOut = sectionContent?.connected_sections.filter(s => s.edge_type === 'REFERENCES_OUT') ?? [];
  const refsIn = sectionContent?.connected_sections.filter(s => s.edge_type === 'REFERENCES_IN') ?? [];
  const derivedOut = sectionContent?.connected_sections.filter(s => s.edge_type === 'DERIVED_FROM_OUT') ?? [];
  const derivedIn = sectionContent?.connected_sections.filter(s => s.edge_type === 'DERIVED_FROM_IN') ?? [];
  const hasRelationships = refsOut.length > 0 || refsIn.length > 0 || derivedOut.length > 0 || derivedIn.length > 0;

  return (
    <aside className="absolute right-3 top-3 z-20 w-72 border border-white/[0.07] bg-[#141414]/95 text-xs text-white/60 shadow-[0_22px_70px_rgba(0,0,0,0.36)] backdrop-blur">
      <div className="border-b border-white/[0.06] px-3 py-3">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0 flex-1">
            <div className="mb-1 flex items-center gap-2">
              <span className="font-mono text-[10px] uppercase tracking-[0.2em] text-white/30">Sec</span>
              <span className="truncate text-[13px] font-semibold text-white/90">{node.section_id}</span>
              <span
                className="shrink-0 px-1.5 py-0.5 font-mono text-[9px] uppercase tracking-[0.14em]"
                style={{ backgroundColor: `${color}24`, color }}
              >
                {JURISDICTION_LABELS[node.jurisdiction] ?? node.jurisdiction}
              </span>
            </div>
            <p className="line-clamp-2 text-[12px] leading-5 text-white/70">{node.title}</p>
            <p className="mt-1 truncate font-mono text-[9px] uppercase tracking-[0.14em] text-white/30">
              {node.doc_name}
            </p>
          </div>
          <button
            onClick={onClose}
            className="text-white/30 transition-colors hover:text-white/70 active:scale-95"
            aria-label="Close section summary"
            type="button"
          >
            x
          </button>
        </div>
      </div>

      <div className="grid grid-cols-2 border-b border-white/[0.05]">
        <div className="border-r border-white/[0.05] px-3 py-2">
          <p className="font-mono text-[9px] uppercase tracking-[0.18em] text-white/25">Links</p>
          <p className="mt-0.5 text-sm text-white/75">{node.connection_count}</p>
        </div>
        <div className="px-3 py-2">
          <p className="font-mono text-[9px] uppercase tracking-[0.18em] text-white/25">State</p>
          <p className="mt-0.5 text-sm text-white/75">{node.is_active ? 'Active' : 'Archived'}</p>
        </div>
      </div>

      <div className="ledger-scroll max-h-40 space-y-2 overflow-y-auto px-3 py-3">
        {isSectionLoading ? (
          <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-white/30">Loading relationships</p>
        ) : null}

        {!isSectionLoading && refsOut.length > 0 ? (
          <RelationshipGroup label="References" values={refsOut.map(s => s.section_id)} />
        ) : null}
        {!isSectionLoading && refsIn.length > 0 ? (
          <RelationshipGroup label="Referenced by" values={refsIn.map(s => s.section_id)} />
        ) : null}
        {!isSectionLoading && derivedOut.length > 0 ? (
          <RelationshipGroup label="Derives from" values={derivedOut.map(s => s.section_id)} />
        ) : null}
        {!isSectionLoading && derivedIn.length > 0 ? (
          <RelationshipGroup label="Derived by" values={derivedIn.map(s => s.section_id)} />
        ) : null}
        {!isSectionLoading && !hasRelationships ? (
          <p className="text-[11px] leading-5 text-white/25">No relationship detail loaded for this node.</p>
        ) : null}
      </div>

      <div className="border-t border-white/[0.05] px-3 py-3">
        <button
          onClick={() => onChatAboutSection(node.section_id, node.title, node.doc_name, node.jurisdiction)}
          className="w-full border border-white/[0.09] bg-white/[0.03] px-3 py-2 text-[11px] font-medium text-white/70 transition-[background-color,border-color,transform] duration-150 ease-out hover:border-[#4f98a3]/50 hover:bg-white/[0.06] hover:text-white active:scale-[0.98]"
          type="button"
        >
          Chat about this section
        </button>
      </div>
    </aside>
  );
}

function RelationshipGroup({ label, values }: { label: string; values: string[] }) {
  return (
    <div>
      <p className="mb-1 font-mono text-[9px] uppercase tracking-[0.18em] text-white/25">{label}</p>
      <p className="truncate text-[11px] leading-5 text-white/50">{values.map(value => `Sec ${value}`).join(', ')}</p>
    </div>
  );
}
