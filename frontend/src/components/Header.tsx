'use client';

import { useTheme } from 'next-themes';
import { useEffect, useState } from 'react';

interface Props {
  onNewConversation: () => void;
}

export function Header({ onNewConversation: _onNewConversation }: Props) {
  const { resolvedTheme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  return (
    <header className="z-20 flex h-12 shrink-0 items-center justify-between bg-[#0d0d0d] px-6">
      <div className="flex w-full items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-[#4f98a3]">CS</span>
          <h1 className="ledger-brand text-xl">CivicSetu</h1>
        </div>

        <div className="flex items-center gap-4">
          {mounted ? (
            <button
              type="button"
              onClick={() => setTheme(resolvedTheme === 'dark' ? 'light' : 'dark')}
              aria-label={resolvedTheme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
              className="grid h-8 w-8 place-items-center rounded-[6px] text-zinc-400 transition-[background-color,color,transform] duration-150 ease-out hover:bg-white/5 hover:text-white/70 active:scale-95"
            >
              {resolvedTheme === 'dark' ? <MoonIcon /> : <SunIcon />}
            </button>
          ) : null}
        </div>
      </div>
    </header>
  );
}

function MoonIcon() {
  return (
    <svg
      aria-hidden="true"
      className="h-4 w-4"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M20.2 14.8A8.5 8.5 0 0 1 9.2 3.8 8 8 0 1 0 20.2 14.8Z" />
    </svg>
  );
}

function SunIcon() {
  return (
    <svg
      aria-hidden="true"
      className="h-4 w-4"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <circle cx="12" cy="12" r="3.5" />
      <path d="M12 2.8v2.4" />
      <path d="M12 18.8v2.4" />
      <path d="m4.7 4.7 1.7 1.7" />
      <path d="m17.6 17.6 1.7 1.7" />
      <path d="M2.8 12h2.4" />
      <path d="M18.8 12h2.4" />
      <path d="m4.7 19.3 1.7-1.7" />
      <path d="m17.6 6.4 1.7-1.7" />
    </svg>
  );
}
