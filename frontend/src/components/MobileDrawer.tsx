'use client';

interface Props {
  isOpen: boolean;
  onClose: () => void;
  onNewConversation: () => void;
}

const railItems = [
  { label: 'Research', glyph: 'R' },
  { label: 'Legal tools', glyph: 'L' },
  { label: 'Library', glyph: 'B' },
  { label: 'History', glyph: 'H' },
] as const;

export function MobileDrawer({ isOpen, onClose, onNewConversation }: Props) {
  return (
    <>
      <div
        className={`fixed inset-0 z-[60] bg-black/60 backdrop-blur-sm transition-opacity duration-300 ${isOpen ? 'opacity-100' : 'pointer-events-none opacity-0'}`}
        onClick={onClose}
      />
      <aside
        className={`fixed inset-y-0 left-0 z-[70] w-[280px] bg-[#141414] shadow-2xl transition-transform duration-300 ease-out ${isOpen ? 'translate-x-0' : '-translate-x-full'}`}
      >
        <div className="flex h-16 items-center px-6 border-b border-white/5">
          <img
            src="/logo.png"
            alt="CivicSetu Logo"
            width={28}
            height={28}
            className="rounded-sm"
          />
        </div>
        
        <div className="flex h-[calc(100%-4rem)] flex-col p-4">
          <div className="space-y-6 flex-1">
            <button
              onClick={() => {
                onNewConversation();
                onClose();
              }}
              className="flex w-full items-center gap-2 rounded-[8px] bg-[#1a1a1a] px-4 py-3 text-sm text-white/80 transition-[background-color,color,transform] duration-150 ease-out hover:bg-[#222222] hover:text-white active:scale-[0.97]"
            >
              <span className="text-lg font-medium">+</span>
              New Conversation
            </button>

            <nav className="space-y-1">
              <p className="px-4 mb-2 text-[10px] uppercase tracking-[0.2em] text-white/20 font-semibold">Navigation</p>
              {railItems.map(item => (
                <button
                  key={item.label}
                  onClick={onClose}
                  className="flex w-full items-center gap-4 px-4 py-3 rounded-lg text-sm text-zinc-400 transition-[background-color,color,transform] duration-150 ease-out hover:bg-white/5 hover:text-white active:scale-[0.97]"
                >
                  <span className="grid h-6 w-6 place-items-center rounded bg-white/5 text-[10px] font-bold text-[#4f98a3]">
                    {item.glyph}
                  </span>
                  {item.label}
                </button>
              ))}
            </nav>
          </div>

          <div className="space-y-4 pb-4">
            <button
              onClick={onClose}
              className="flex w-full items-center gap-4 px-4 py-3 rounded-lg text-sm text-zinc-400 transition-[background-color,color,transform] duration-150 ease-out hover:bg-white/5 hover:text-white active:scale-[0.97]"
            >
              <span className="grid h-6 w-6 place-items-center rounded bg-white/5 text-[10px] font-bold text-zinc-500">
                S
              </span>
              Settings
            </button>
            <div className="flex items-center gap-2 px-4 text-[10px] font-mono uppercase tracking-tight text-zinc-600">
              <span className="h-1 w-1 rounded-full bg-[#4f98a3]" />
              System ready
            </div>
          </div>
        </div>
      </aside>
    </>
  );
}
