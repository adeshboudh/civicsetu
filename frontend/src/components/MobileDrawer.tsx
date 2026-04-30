'use client';

interface Props {
  isOpen: boolean;
  onClose: () => void;
}

export function MobileDrawer({ isOpen, onClose }: Props) {
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
           <h2 className="font-['Instrument_Serif'] italic text-2xl text-[#4f98a3]">CivicSetu</h2>
        </div>
        <nav className="p-4 space-y-2">
           {['Research', 'Legal tools', 'Library', 'History', 'Settings'].map(item => (
             <button key={item} className="flex w-full items-center gap-3 px-4 py-3 rounded-lg text-sm text-zinc-400 hover:bg-white/5 hover:text-white transition-colors">
               {item}
             </button>
           ))}
        </nav>
      </aside>
    </>
  );
}
