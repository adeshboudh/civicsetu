'use client';

import { useCallback, useRef, useState } from 'react';
import { queryRera, querySectionContext } from '@/lib/api';
import type { ApiResponse, ChatMessage, Jurisdiction } from '@/lib/types';

const SESSION_KEY = 'civicsetu_session_id';

export interface UseChatReturn {
  messages: ChatMessage[];
  isLoading: boolean;
  sendMessage: (text: string, jurisdiction: Jurisdiction | '') => Promise<void>;
  sendSectionMessage: (text: string, sectionId: string, jurisdiction: string) => Promise<void>;
  newConversation: () => void;
}

export function useChat(): UseChatReturn {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const sessionIdRef = useRef<string | null>(
    typeof window !== 'undefined' ? window.localStorage.getItem(SESSION_KEY) : null,
  );

  const _handleResponse = useCallback((data: ApiResponse) => {
    if (data.session_id) {
      sessionIdRef.current = data.session_id;
      if (typeof window !== 'undefined') {
        window.localStorage.setItem(SESSION_KEY, data.session_id);
      }
    }
    setMessages(prev => [
      ...prev,
      { id: crypto.randomUUID(), role: 'assistant', text: data.answer, data },
    ]);
  }, []);

  const _handleError = useCallback((error: unknown) => {
    setMessages(prev => [
      ...prev,
      {
        id: crypto.randomUUID(),
        role: 'error',
        text: error instanceof Error ? error.message : 'Request failed',
      },
    ]);
  }, []);

  const sendMessage = useCallback(
    async (text: string, jurisdiction: Jurisdiction | '') => {
      setMessages(prev => [...prev, { id: crypto.randomUUID(), role: 'user', text }]);
      setIsLoading(true);
      try {
        const data = await queryRera({
          query: text,
          top_k: 5,
          ...(jurisdiction ? { jurisdiction_filter: jurisdiction } : {}),
          ...(sessionIdRef.current ? { session_id: sessionIdRef.current } : {}),
        });
        _handleResponse(data);
      } catch (error) {
        _handleError(error);
      } finally {
        setIsLoading(false);
      }
    },
    [_handleResponse, _handleError],
  );

  const sendSectionMessage = useCallback(
    async (text: string, sectionId: string, jurisdiction: string) => {
      setMessages(prev => [...prev, { id: crypto.randomUUID(), role: 'user', text }]);
      setIsLoading(true);
      try {
        const data = await querySectionContext({
          query: text,
          section_id: sectionId,
          jurisdiction,
          ...(sessionIdRef.current ? { session_id: sessionIdRef.current } : {}),
        });
        _handleResponse(data);
      } catch (error) {
        _handleError(error);
      } finally {
        setIsLoading(false);
      }
    },
    [_handleResponse, _handleError],
  );

  const newConversation = useCallback(() => {
    sessionIdRef.current = null;
    if (typeof window !== 'undefined') {
      window.localStorage.removeItem(SESSION_KEY);
    }
    setMessages([]);
  }, []);

  return { messages, isLoading, sendMessage, sendSectionMessage, newConversation };
}