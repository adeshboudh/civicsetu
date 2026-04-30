'use client';

interface Props {
  activeTab: 'chat' | 'graph';
  onTabChange: (tab: 'chat' | 'graph') => void;
}

export function BottomNavBar({ activeTab, onTabChange }: Props) {
  return (
    <nav className="fixed bottom-0 left-0 z-50 flex h-20 w-full items-center justify-around bg-[#141414] pb-safe shadow-[0_-10px_30px_rgba(0,0,0,0.5)] lg:hidden">
      <button
        onClick={() => onTabChange('chat')}
        className={`flex flex-col items-center gap-1 transition-transform duration-150 ease-out active:scale-[0.97] ${activeTab === 'chat' ? 'text-[#4f98a3]' : 'text-zinc-500'}`}
      >
        <span className="text-xl">
          <span className="material-symbols-outlined">
            chat_bubble
          </span>
        </span>
        <span className="text-[10px] font-medium uppercase tracking-wider">Chat</span>
      </button>
      <button
        onClick={() => onTabChange('graph')}
        className={`flex flex-col items-center gap-1 transition-transform duration-150 ease-out active:scale-[0.97] ${activeTab === 'graph' ? 'text-[#4f98a3]' : 'text-zinc-500'}`}
      >
        <span className="text-xl">
          <span className="material-symbols-outlined">
            graph_4
          </span>
        </span>
        <span className="text-[10px] font-medium uppercase tracking-wider">Graph</span>
      </button>
    </nav>
  );
}
