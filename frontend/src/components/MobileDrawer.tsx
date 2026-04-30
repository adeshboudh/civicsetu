'use client';

import logo from '@frontend/assets/logo.svg';

interface Props {
  isOpen: boolean;
  onClose: () => void;
  onNewConversation: () => void;
}

export function MobileDrawer({ isOpen, onClose, onNewConversation }: Props) {
  return (
    <div className="lg:hidden">
      <div
        className={`fixed inset-0 z-[60] bg-black/60 backdrop-blur-sm transition-opacity duration-300 ${isOpen ? 'opacity-100' : 'pointer-events-none opacity-0'}`}
        onClick={onClose}
      />
      <aside
        className={`fixed inset-y-0 left-0 z-[70] w-[280px] bg-[#141414] shadow-2xl transition-transform duration-300 ease-out ${isOpen ? 'translate-x-0' : '-translate-x-full'}`}
      >
        <div className="flex h-16 items-center px-6 border-b border-white/5">
          <img
            src={logo.src}
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
              <svg suppressHydrationWarning xmlns="http://www.w3.org/2000/svg" height="18px" viewBox="0 -960 960 960" width="18px" fill="currentColor">
                <path d="M440-440H200v-80h240v-240h80v240h240v80H520v240h-80v-240Z"/>
              </svg>
              New Conversation
            </button>
          </div>

          <div className="space-y-4 pb-4">
            <button
              onClick={onClose}
              className="flex w-full items-center gap-4 px-4 py-3 rounded-lg text-sm text-zinc-400 transition-[background-color,color,transform] duration-150 ease-out hover:bg-white/5 hover:text-white active:scale-[0.97]"
            >
              <span className="material-symbols-outlined text-[20px]">settings</span>
              Settings
            </button>
            <div className="flex items-center gap-2 px-4 text-[10px] font-mono uppercase tracking-tight text-zinc-600">
              <span className="h-1 w-1 rounded-full bg-[#4f98a3]" />
              System ready
            </div>
          </div>
        </div>
      </aside>
    </div>
  );
}
