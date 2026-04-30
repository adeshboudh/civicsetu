'use client';

import { useState } from 'react';
import { ALL_JURISDICTIONS, JURISDICTION_LABELS } from '@/lib/constants';
import type { GraphFilters, GraphTopology } from '@/lib/types';

interface Props {
  filters: GraphFilters;
  onFiltersChange: (f: GraphFilters) => void;
  topology: GraphTopology | null;
}

const JURISDICTION_DOT_CLASSES = {
  CENTRAL: {
    on: 'border-[#2db7a3] bg-[#2db7a3] opacity-100',
    off: 'border-[#2db7a3] bg-transparent opacity-50',
  },
  MAHARASHTRA: {
    on: 'border-[#4e7cff] bg-[#4e7cff] opacity-100',
    off: 'border-[#4e7cff] bg-transparent opacity-50',
  },
  UTTAR_PRADESH: {
    on: 'border-[#f4b63f] bg-[#f4b63f] opacity-100',
    off: 'border-[#f4b63f] bg-transparent opacity-50',
  },
  KARNATAKA: {
    on: 'border-[#78b94d] bg-[#78b94d] opacity-100',
    off: 'border-[#78b94d] bg-transparent opacity-50',
  },
  TAMIL_NADU: {
    on: 'border-[#ff7a59] bg-[#ff7a59] opacity-100',
    off: 'border-[#ff7a59] bg-transparent opacity-50',
  },
} as const;

export function GraphFilterSidebar({ filters, onFiltersChange, topology }: Props) {
  const [collapsed, setCollapsed] = useState(true);

  function toggleJurisdiction(j: (typeof ALL_JURISDICTIONS)[number]) {
    const next = new Set(filters.jurisdictions);
    if (next.has(j)) {
      next.delete(j);
    } else {
      next.add(j);
    }
    onFiltersChange({ ...filters, jurisdictions: next });
  }

  function toggleEdgeType(t: 'REFERENCES' | 'DERIVED_FROM') {
    const next = new Set(filters.edgeTypes);
    if (next.has(t)) {
      next.delete(t);
    } else {
      next.add(t);
    }
    onFiltersChange({ ...filters, edgeTypes: next });
  }

  return (
    <div className="absolute left-3 top-3 z-10 w-48 border border-white/[0.07] bg-[#141414]/95 text-xs text-white/60 shadow-[0_18px_60px_rgba(0,0,0,0.32)] backdrop-blur">
      <button
        className="flex w-full items-center gap-2 px-3 py-2.5 text-left font-mono text-[10px] uppercase tracking-[0.22em] text-white/40 transition-[background-color,color,transform] duration-150 ease-out hover:text-white/70 active:scale-[0.97]"
        onClick={() => setCollapsed(value => !value)}
        aria-label="Toggle graph filters"
        type="button"
      >
        <span className="h-1.5 w-1.5 rounded-full bg-[#4f98a3]" />
        Filters
        <span className={`ml-auto transition-transform duration-150 ease-out origin-center ${collapsed ? '-rotate-90' : ''}`}>
          <svg suppressHydrationWarning xmlns="http://www.w3.org/2000/svg" height="20px" viewBox="0 -960 960 960" width="20px" fill="#e3e3e3"><path d="M480-384 288-576h384L480-384Z"/></svg>
        </span>
      </button>

      {!collapsed ? (
        <div className="space-y-4 border-t border-white/[0.05] px-3 pb-3 pt-3">
          <section>
            <p className="mb-2 font-mono text-[9px] uppercase tracking-[0.22em] text-white/30">Jurisdictions</p>
            <div className="space-y-1">
              {ALL_JURISDICTIONS.map(j => (
                <label
                  key={j}
                  className="flex cursor-pointer items-center gap-2 py-0.5 transition-[color,transform] duration-150 ease-out hover:text-white/80 active:scale-[0.98]"
                >
                  <input
                    type="checkbox"
                    checked={filters.jurisdictions.has(j)}
                    onChange={() => toggleJurisdiction(j)}
                    className="sr-only"
                  />
                  <span
                    className={`h-2.5 w-2.5 shrink-0 rounded-full border transition-[background-color,opacity] duration-150 ease-out ${
                      filters.jurisdictions.has(j) ? JURISDICTION_DOT_CLASSES[j].on : JURISDICTION_DOT_CLASSES[j].off
                    }`}
                  />
                  <span>{JURISDICTION_LABELS[j]}</span>
                </label>
              ))}
            </div>
          </section>

          <section>
            <p className="mb-2 font-mono text-[9px] uppercase tracking-[0.22em] text-white/30">Relationship</p>
            {(['REFERENCES', 'DERIVED_FROM'] as const).map(t => (
              <label
                key={t}
                className="flex cursor-pointer items-center gap-2 py-0.5 transition-[color,transform] duration-150 ease-out hover:text-white/80 active:scale-[0.98]"
              >
                <input
                  type="checkbox"
                  checked={filters.edgeTypes.has(t)}
                  onChange={() => toggleEdgeType(t)}
                  className="sr-only"
                />
                <span
                  className={`h-px w-4 shrink-0 transition-opacity ${
                    filters.edgeTypes.has(t) ? 'opacity-100' : 'opacity-25'
                  } ${
                    t === 'REFERENCES'
                      ? 'bg-white/40'
                      : 'border-t border-dashed border-[#e8af34]/80 bg-transparent'
                  }`}
                />
                <span>{t === 'REFERENCES' ? 'References' : 'Derived from'}</span>
              </label>
            ))}
          </section>

          <section>
            <p className="mb-2 font-mono text-[9px] uppercase tracking-[0.22em] text-white/30">Node scale</p>
            {(['connections', 'uniform'] as const).map(mode => (
              <label
                key={mode}
                className="flex cursor-pointer items-center gap-2 py-0.5 transition-[color,transform] duration-150 ease-out hover:text-white/80 active:scale-[0.98]"
              >
                <input
                  type="radio"
                  name="nodeSizeMode"
                  checked={filters.nodeSizeMode === mode}
                  onChange={() => onFiltersChange({ ...filters, nodeSizeMode: mode })}
                  className="sr-only"
                />
                <span
                  className={`h-2.5 w-2.5 shrink-0 rounded-full border border-white/40 transition-colors duration-150 ease-out ${
                    filters.nodeSizeMode === mode ? 'bg-white/90' : 'bg-transparent'
                  }`}
                />
                <span>{mode === 'connections' ? 'By connections' : 'Uniform'}</span>
              </label>
            ))}
          </section>
        </div>
      ) : null}
    </div>
  );
}
