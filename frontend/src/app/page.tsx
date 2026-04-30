'use client';

import { useCallback, useState } from 'react';
import logo from '@frontend/assets/logo.svg';
import { BottomNavBar } from '@/components/BottomNavBar';
import { ChatThread } from '@/components/ChatThread';
import { Header } from '@/components/Header';
import { InputBar } from '@/components/InputBar';
import { MobileDrawer } from '@/components/MobileDrawer';
import { ContextPill } from '@/components/graph/ContextPill';
import { GraphExplorer } from '@/components/graph/GraphExplorer';
import { useChat } from '@/hooks/useChat';
import { useGraphExplorer } from '@/hooks/useGraphExplorer';
import type { Jurisdiction, SectionContext } from '@/lib/types';

export default function Home() {
  const { messages, isLoading, sendMessage, sendSectionMessage, newConversation } = useChat();
  const graphState = useGraphExplorer();

  const [pendingQuery, setPendingQuery] = useState<string | undefined>();
  const [sectionContext, setSectionContext] = useState<SectionContext | null>(null);
  const [activeTab, setActiveTab] = useState<'chat' | 'graph'>('chat');
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);

  const handleChatAboutSection = useCallback(
    (sectionId: string, title: string, docName: string, jurisdiction: string) => {
      setSectionContext({ sectionId, title, docName, jurisdiction });
      setPendingQuery(`Explain ${title} of ${docName}`);
      setActiveTab('chat');
      setTimeout(() => setPendingQuery(undefined), 0);
    },
    [],
  );

  const handleExampleClick = useCallback((query: string) => {
    setPendingQuery(query);
    setTimeout(() => setPendingQuery(undefined), 0);
  }, []);

  function handleSend(text: string, jurisdiction: Jurisdiction | '') {
    if (sectionContext) {
      void sendSectionMessage(text, sectionContext.sectionId, sectionContext.jurisdiction);
      setSectionContext(null);
      return;
    }

    void sendMessage(text, jurisdiction);
  }

  return (
    <div className="flex h-screen w-full bg-[#0d0d0d] text-[#e5e2e1] overflow-hidden">
      {/* Persistent Desktop Sidebar (Full Height) */}
      <aside
        className={`group relative flex shrink-0 flex-col border-r border-white/5 bg-[#111111] transition-[width] duration-300 ease-out max-lg:hidden ${
          isSidebarOpen ? 'w-[260px]' : 'w-[64px]'
        }`}
      >
        {/* Clickable area to expand when collapsed - covers everything */}
        {!isSidebarOpen && (
          <button
            onClick={() => setIsSidebarOpen(true)}
            className="absolute inset-0 z-20 h-full w-full cursor-pointer bg-transparent transition-colors hover:bg-white/[0.02]"
            aria-label="Expand sidebar"
            title="Expand sidebar"
          />
        )}

        <div className="flex h-12 items-center justify-between px-[18px]">
          <img src={logo.src} alt="CivicSetu Logo" width={28} height={28} className="rounded-sm shrink-0" />
          
          {isSidebarOpen && (
            <button
              onClick={() => setIsSidebarOpen(false)}
              className="relative z-30 grid h-8 w-8 place-items-center rounded-[6px] text-zinc-500 transition-colors hover:bg-white/5 hover:text-white/80 active:scale-95"
              title="Collapse sidebar"
            >
              <svg suppressHydrationWarning role="img" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="shrink-0">
                <rect width="18" height="18" x="3" y="3" rx="2" ry="2"/>
                <line x1="9" x2="9" y1="3" y2="21"/>
              </svg>
            </button>
          )}
        </div>

        <div className="relative z-10 flex flex-1 flex-col p-3 overflow-hidden">
          <button
            type="button"
            onClick={newConversation}
            disabled={!isSidebarOpen}
            className={`flex items-center gap-3 rounded-[8px] bg-[#1a1a1a] p-2.5 text-xs text-white/80 transition-[background-color,color,transform] duration-150 ease-out hover:bg-[#222222] hover:text-white active:scale-[0.97] ${
              !isSidebarOpen ? 'justify-center' : ''
            }`}
            title={isSidebarOpen ? "New Conversation" : ""}
          >
            <span className="shrink-0">
              <svg suppressHydrationWarning xmlns="http://www.w3.org/2000/svg" height="20px" viewBox="0 -960 960 960" width="20px" fill="currentColor">
                <path d="M440-440H200v-80h240v-240h80v240h240v80H520v240h-80v-240Z"/>
              </svg>
            </span>
            {isSidebarOpen && <span className="truncate">New Conversation</span>}
          </button>
        </div>

        <div className="relative z-10 p-3 space-y-4 overflow-hidden">
          <button
            type="button"
            aria-label="Settings"
            disabled={!isSidebarOpen}
            className={`flex items-center gap-3 rounded-[6px] p-2.5 text-xs text-zinc-400 transition-[background-color,color,transform] duration-150 ease-out hover:bg-white/5 hover:text-white active:scale-[0.97] ${
              !isSidebarOpen ? 'justify-center' : ''
            }`}
            title={isSidebarOpen ? "Settings" : ""}
          >
            <span className="material-symbols-outlined text-[22px] shrink-0">settings</span>
            {isSidebarOpen && <span className="truncate">Settings</span>}
          </button>
          
          <div className={`flex items-center gap-3 px-2.5 text-[10px] font-mono uppercase tracking-tight text-zinc-600 ${!isSidebarOpen ? 'justify-center' : ''}`}>
            <span className="h-1.5 w-1.5 rounded-full bg-[#4f98a3] shrink-0" />
            {isSidebarOpen && <span className="truncate">System ready</span>}
          </div>
        </div>
      </aside>

      {/* Main Content Area */}
      <div className="flex flex-1 flex-col min-w-0 overflow-hidden">
        <Header onNewConversation={newConversation} onMenuClick={() => setIsSidebarOpen(true)} />

        <main className="flex min-h-0 flex-1 overflow-hidden">
          {/* Chat Panel */}
          <aside
            className={`flex min-w-0 flex-1 flex-col bg-[#141414] max-lg:w-full ${
              activeTab !== 'chat' ? 'max-lg:hidden' : ''
            }`}
          >
            <div className="flex min-h-0 flex-1 flex-col">
              {sectionContext ? (
                <ContextPill
                  sectionId={sectionContext.sectionId}
                  docName={sectionContext.docName}
                  jurisdiction={sectionContext.jurisdiction}
                  onRemove={() => setSectionContext(null)}
                />
              ) : null}

              <div className="min-h-0 flex-1 overflow-hidden">
                <ChatThread messages={messages} isLoading={isLoading} onExampleClick={handleExampleClick} />
              </div>

              <InputBar onSend={handleSend} disabled={isLoading} pendingQuery={pendingQuery} />
            </div>
          </aside>

          {/* Graph Explorer Panel */}
          <section
            className={`flex w-[55%] min-w-0 flex-col bg-[#0d0d0d] max-lg:w-full ${
              activeTab !== 'graph' ? 'max-lg:hidden' : ''
            }`}
          >
            <GraphExplorer {...graphState} onChatAboutSection={handleChatAboutSection} />
          </section>
        </main>

        <footer className="flex h-6 shrink-0 items-center justify-between bg-[#0d0d0d] px-4 font-mono text-[10px] uppercase tracking-tight text-zinc-600 lg:flex">
          <div className="flex items-center gap-4">
            {/* Removed static DB_LATENCY and CORPUS block */}
          </div>
          <div className="flex items-center gap-1.5 text-[#4f98a3]">
            <span className="h-1 w-1 rounded-full bg-[#4f98a3]" />
            System ready
          </div>
        </footer>
      </div>

      <BottomNavBar activeTab={activeTab} onTabChange={setActiveTab} />
      <MobileDrawer isOpen={isSidebarOpen} onClose={() => setIsSidebarOpen(false)} onNewConversation={newConversation} />
    </div>
  );
}
