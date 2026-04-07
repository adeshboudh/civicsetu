'use client';

import { useEffect, useRef, useState } from 'react';
import type { Jurisdiction } from '@/lib/types';

const JURISDICTIONS = [
  { value: '' as const, label: 'All Jurisdictions', shortLabel: 'All', dotClass: 'bg-white/35' },
  { value: 'CENTRAL' as const, label: 'Central', shortLabel: 'Central', dotClass: 'bg-[#2db7a3]' },
  { value: 'MAHARASHTRA' as const, label: 'Maharashtra', shortLabel: 'Maharashtra', dotClass: 'bg-[#4e7cff]' },
  { value: 'UTTAR_PRADESH' as const, label: 'Uttar Pradesh', shortLabel: 'Uttar Pradesh', dotClass: 'bg-[#f4b63f]' },
  { value: 'KARNATAKA' as const, label: 'Karnataka', shortLabel: 'Karnataka', dotClass: 'bg-[#78b94d]' },
  { value: 'TAMIL_NADU' as const, label: 'Tamil Nadu', shortLabel: 'Tamil Nadu', dotClass: 'bg-[#ff7a59]' },
];

interface Props {
  onSend: (query: string, jurisdiction: Jurisdiction | '') => void;
  disabled: boolean;
  pendingQuery?: string;
}

export function InputBar({ onSend, disabled, pendingQuery }: Props) {
  const [query, setQuery] = useState('');
  const [jurisdiction, setJurisdiction] = useState<Jurisdiction | ''>('');
  const [isJurisdictionOpen, setIsJurisdictionOpen] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const jurisdictionMenuRef = useRef<HTMLDivElement>(null);
  const MAX_VISIBLE_LINES = 9;

  const selectedJurisdiction = JURISDICTIONS.find(item => item.value === jurisdiction) ?? JURISDICTIONS[0];

  useEffect(() => {
    if (!pendingQuery) {
      return;
    }

    setQuery(pendingQuery);
    requestAnimationFrame(() => {
      textareaRef.current?.focus();
      autoResize();
    });
  }, [pendingQuery]);

  useEffect(() => {
    if (!isJurisdictionOpen) {
      return;
    }

    function handlePointerDown(event: PointerEvent) {
      if (!jurisdictionMenuRef.current?.contains(event.target as Node)) {
        setIsJurisdictionOpen(false);
      }
    }

    window.addEventListener('pointerdown', handlePointerDown);
    return () => window.removeEventListener('pointerdown', handlePointerDown);
  }, [isJurisdictionOpen]);

  function autoResize() {
    const element = textareaRef.current;
    if (!element) {
      return;
    }

    const computed = window.getComputedStyle(element);
    const lineHeight = Number.parseFloat(computed.lineHeight) || 24;
    const maxHeight = lineHeight * MAX_VISIBLE_LINES;
    element.style.height = 'auto';
    element.style.height = `${Math.min(element.scrollHeight, maxHeight)}px`;
    element.style.overflowY = element.scrollHeight > maxHeight ? 'auto' : 'hidden';
  }

  function handleInput(event: React.ChangeEvent<HTMLTextAreaElement>) {
    setQuery(event.target.value);
    autoResize();
  }

  function handleSend() {
    const text = query.trim();
    if (!text || disabled) {
      return;
    }

    onSend(text, jurisdiction);
    setQuery('');
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.overflowY = 'hidden';
    }
  }

  function handleKeyDown(event: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      handleSend();
    }
  }

  return (
    <div className="shrink-0 bg-[#0d0d0d] p-3">
      <div ref={jurisdictionMenuRef} className="relative mb-1 w-fit">
        <button
          type="button"
          onClick={() => setIsJurisdictionOpen(value => !value)}
          className="inline-flex h-7 items-center gap-2 rounded-[7px] border border-white/[0.07] bg-[#141414] px-2.5 font-mono text-[10px] uppercase tracking-[0.2em] text-white/50 transition-[background-color,border-color,color,transform] duration-150 ease-out hover:border-white/15 hover:bg-[#1a1a1a] hover:text-white/70 active:scale-[0.98]"
          aria-haspopup="listbox"
          aria-expanded={isJurisdictionOpen}
        >
          <span className={`h-1.5 w-1.5 rounded-full ${selectedJurisdiction.dotClass}`} />
          {selectedJurisdiction.shortLabel}
          <span className={`text-white/25 transition-transform duration-150 ease-out ${isJurisdictionOpen ? 'rotate-180' : ''}`}>
            &#x25BC;
          </span>
        </button>

        {isJurisdictionOpen ? (
          <div
            role="listbox"
            className="absolute bottom-8 left-0 z-30 w-52 border border-white/[0.08] bg-[#141414]/98 p-1 shadow-[0_18px_50px_rgba(0,0,0,0.45)] backdrop-blur"
          >
            {JURISDICTIONS.map(item => (
              <button
                key={item.value}
                type="button"
                role="option"
                aria-selected={jurisdiction === item.value}
                onClick={() => {
                  setJurisdiction(item.value);
                  setIsJurisdictionOpen(false);
                }}
                className={`flex w-full items-center justify-between gap-3 rounded-[6px] px-2.5 py-2 text-left text-[12px] transition-[background-color,color] duration-150 ease-out ${
                  jurisdiction === item.value
                    ? 'bg-[#4f98a3]/12 text-[#9ed4dc]'
                    : 'text-white/60 hover:bg-white/[0.04] hover:text-white/80'
                }`}
              >
                <span className="flex min-w-0 items-center gap-2">
                  <span className={`h-1.5 w-1.5 shrink-0 rounded-full ${item.dotClass}`} />
                  <span className="truncate">{item.label}</span>
                </span>
                {jurisdiction === item.value ? (
                  <span className={`h-1.5 w-1.5 shrink-0 rounded-full ${item.dotClass}`} />
                ) : null}
              </button>
            ))}
          </div>
        ) : null}
      </div>

      <div className="relative flex items-end">
        <textarea
          ref={textareaRef}
          value={query}
          onChange={handleInput}
          onKeyDown={handleKeyDown}
          rows={1}
          placeholder="Ask about RERA regulations..."
          className="ledger-scroll min-h-[40px] w-full resize-none overflow-hidden rounded-[10px] bg-[#1a1a1a] py-2 pl-3 pr-10 text-[14px] leading-6 text-white/85 outline-none transition-[border-color,height] duration-150 ease-out placeholder:text-white/25 focus:ring-1 focus:ring-[#4f98a3]/50"
        />
        <button
          type="button"
          onClick={handleSend}
          disabled={disabled || !query.trim()}
          aria-label="Send message"
          className="absolute bottom-2 right-3 text-[#4f98a3] transition-opacity hover:opacity-80 disabled:opacity-25"
        >
          -&gt;
        </button>
      </div>
    </div>
  );
}
