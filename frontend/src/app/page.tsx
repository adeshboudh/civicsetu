'use client';

import { useCallback, useState } from 'react';
import { ChatThread } from '@/components/ChatThread';
import { Header } from '@/components/Header';
import { InputBar } from '@/components/InputBar';
import { ContextPill } from '@/components/graph/ContextPill';
import { GraphExplorer } from '@/components/graph/GraphExplorer';
import { useChat } from '@/hooks/useChat';
import { useGraphExplorer } from '@/hooks/useGraphExplorer';
import type { Jurisdiction, SectionContext } from '@/lib/types';

const railItems = [
  { label: 'Research', glyph: 'R', active: true },
  { label: 'Legal tools', glyph: 'L', active: false },
  { label: 'Library', glyph: 'B', active: false },
  { label: 'History', glyph: 'H', active: false },
] as const;

export default function Home() {
  const { messages, isLoading, sendMessage, sendSectionMessage, newConversation } = useChat();
  const graphState = useGraphExplorer();

  const [pendingQuery, setPendingQuery] = useState<string | undefined>();
  const [sectionContext, setSectionContext] = useState<SectionContext | null>(null);
  const [activeTab, setActiveTab] = useState<'chat' | 'graph'>('chat');
  const [isDrawerOpen, setIsDrawerOpen] = useState(false);

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
    <div className="flex h-screen flex-col overflow-hidden bg-[#0d0d0d] text-[#e5e2e1]">
      <Header onNewConversation={newConversation} />

      <main className="flex min-h-0 flex-1 overflow-hidden">
        <aside
          className={`flex w-[45%] min-w-[420px] flex-col bg-[#141414] max-lg:w-full max-lg:min-w-0 ${
            activeTab !== 'chat' ? 'max-lg:hidden' : ''
          }`}
        >
          <div className="flex min-h-0 flex-1">
            <nav className="flex w-14 shrink-0 flex-col items-center gap-5 bg-[#111111] py-4">
              {railItems.map(item => (
                <button
                  key={item.label}
                  type="button"
                  aria-label={item.label}
                  className={`grid h-8 w-8 place-items-center rounded-[6px] text-sm transition-[background-color,color,transform] duration-150 ease-out active:scale-95 ${
                    item.active
                      ? 'bg-[#222222] text-[#4f98a3]'
                      : 'text-white/30 hover:bg-[#222222] hover:text-white/70'
                  }`}
                >
                  {item.glyph}
                </button>
              ))}
              <button
                type="button"
                aria-label="Settings"
                className="mt-auto grid h-8 w-8 place-items-center rounded-[6px] text-sm text-white/30 transition-[background-color,color,transform] duration-150 ease-out hover:bg-[#222222] hover:text-white/70 active:scale-95"
              >
                S
              </button>
            </nav>

            <div className="flex min-w-0 flex-1 flex-col">
              <div className="flex h-14 shrink-0 items-center justify-between bg-[#141414] px-4">
                <button
                  type="button"
                  onClick={newConversation}
                  className="inline-flex items-center gap-2 rounded-[6px] bg-[#1a1a1a] px-3 py-1.5 text-xs text-white/60 transition-[background-color,color,transform] duration-150 ease-out hover:bg-[#222222] hover:text-white/80 active:scale-[0.98]"
                >
                  <span className="text-sm">+</span>
                  New Conversation
                </button>
                <span className="font-mono text-[10px] uppercase tracking-[0.28em] text-white/25">
                  Active session
                </span>
              </div>

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
          </div>
        </aside>

        <section
          className={`flex w-[55%] min-w-0 flex-col bg-[#0d0d0d] max-lg:w-full ${
            activeTab !== 'graph' ? 'max-lg:hidden' : ''
          }`}
        >
          <GraphExplorer {...graphState} onChatAboutSection={handleChatAboutSection} />
        </section>
      </main>

      <footer className="flex h-6 shrink-0 items-center justify-between bg-[#0d0d0d] px-4 font-mono text-[10px] uppercase tracking-tight text-zinc-600">
        <div className="flex items-center gap-4">
          <span>DB_LATENCY: 12ms</span>
          <span>CORPUS: RERA_INDIA_2026</span>
        </div>
        <div className="flex items-center gap-1.5 text-[#4f98a3]">
          <span className="h-1 w-1 rounded-full bg-[#4f98a3]" />
          System ready
        </div>
      </footer>
    </div>
  );
}
