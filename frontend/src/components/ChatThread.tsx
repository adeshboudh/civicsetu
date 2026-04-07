'use client';

import { useEffect, useRef } from 'react';
import { MessageBubble } from '@/components/MessageBubble';
import type { ChatMessage } from '@/lib/types';

const EXAMPLE_QUERIES = [
  'What are promoter obligations under RERA?',
  'What penalty applies for delayed possession?',
  'How does agent registration work under Maharashtra rules?',
  'What is the complaint filing process for a buyer?',
] as const;

interface Props {
  messages: ChatMessage[];
  isLoading: boolean;
  onExampleClick: (query: string) => void;
}

export function ChatThread({ messages, isLoading, onExampleClick }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  if (messages.length === 0 && !isLoading) {
    return (
      <div className="ledger-scroll flex h-full flex-col justify-center overflow-y-auto px-4 py-8">
        <div className="max-w-md">
          <p className="font-mono text-[10px] uppercase tracking-[0.28em] text-white/25">Research desk</p>
          <h2 className="mt-3 text-2xl font-semibold tracking-[-0.02em] text-white/84">
            Ask the ledger a legal question.
          </h2>
          <p className="mt-3 text-sm leading-6 text-white/50">
            Query Indian RERA provisions, compare jurisdictions, and route selected graph sections directly into the chat.
          </p>
        </div>

        <div className="mt-8 grid gap-2">
          {EXAMPLE_QUERIES.map(query => (
            <button
              key={query}
              type="button"
              onClick={() => onExampleClick(query)}
              className="rounded-[10px] bg-[#1a1a1a] px-3.5 py-3 text-left text-[13px] leading-5 text-white/60 transition-colors hover:bg-[#222222] hover:text-white/80"
            >
              {query}
            </button>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="ledger-scroll flex h-full flex-col gap-6 overflow-y-auto p-4">
      {messages.map(message => (
        <MessageBubble key={message.id} message={message} />
      ))}

      {isLoading ? (
        <div className="flex gap-1.5 self-start p-2">
          <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-white/20" />
          <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-white/20 [animation-delay:200ms]" />
          <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-white/20 [animation-delay:400ms]" />
        </div>
      ) : null}

      <div ref={bottomRef} />
    </div>
  );
}
