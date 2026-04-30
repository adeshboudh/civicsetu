'use client';

import { useTheme } from 'next-themes';
import { useEffect, useState } from 'react';

interface Props {
  onNewConversation: () => void;
  onMenuClick?: () => void;
}

export function Header({ onNewConversation: _onNewConversation, onMenuClick }: Props) {
  const { resolvedTheme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  return (
    <header className="z-20 flex h-12 shrink-0 items-center justify-between bg-[#0d0d0d] px-6">
      <div className="flex w-full items-center justify-between">
        <div className="flex items-center gap-2">
          {/* Use standard img instead of next/image to allow suppressHydrationWarning to work on the img tag directly,
              preventing hydration mismatches caused by browser extensions like Dark Reader. */}
          <img
            src="/logo.png"
            alt="CivicSetu Logo"
            width={24}
            height={24}
            className="rounded-sm"
            style={{ color: 'transparent' }}
            suppressHydrationWarning
          />
          <h1 className="ledger-brand text-xl">CivicSetu</h1>
        </div>

        <div className="flex items-center gap-4">
          {onMenuClick && (
            <button
              type="button"
              onClick={onMenuClick}
              aria-label="Open menu"
              className="grid h-8 w-8 place-items-center rounded-[6px] text-zinc-400 transition-[background-color,color,transform] duration-150 ease-out hover:bg-white/5 hover:text-white/70 active:scale-95 lg:hidden"
            >
              <MenuIcon />
            </button>
          )}
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

function MenuIcon() {
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
      <line x1="3" y1="12" x2="21" y2="12" />
      <line x1="3" y1="6" x2="21" y2="6" />
      <line x1="3" y1="18" x2="21" y2="18" />
    </svg>
  );
}
