import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { CitationsPanel } from '@/components/CitationsPanel';
import { ConfidenceBadge } from '@/components/ConfidenceBadge';
import type { ChatMessage } from '@/lib/types';
import { isCivicSetuResponse } from '@/lib/types';

interface Props {
  message: ChatMessage;
}

export function MessageBubble({ message }: Props) {
  if (message.role === 'user') {
    return (
      <div className="flex justify-end">
        <div className="max-w-[85%] rounded-[12px] bg-white/5 px-3.5 py-2.5 text-[13px] leading-6 text-white/85">
          <p className="whitespace-pre-wrap">{message.text}</p>
        </div>
      </div>
    );
  }

  if (message.role === 'error') {
    return (
      <div className="max-w-[95%] self-start rounded-[10px] bg-red-950/25 px-3.5 py-3 text-[13px] leading-6 text-red-200/80">
        {message.text}
      </div>
    );
  }

  const data = message.data;
  const isRichResponse = data !== undefined && isCivicSetuResponse(data);

  return (
    <div className="flex max-w-[95%] flex-col gap-3 self-start">
      <div className="text-[14px] leading-7 text-white/70">
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          components={{
            p: ({ children }) => <p className="mb-4 last:mb-0">{children}</p>,
            h1: ({ children }) => <h1 className="mb-3 text-xl font-semibold leading-7 text-white/90">{children}</h1>,
            h2: ({ children }) => <h2 className="mb-3 text-lg font-semibold leading-7 text-white/90">{children}</h2>,
            h3: ({ children }) => <h3 className="mb-2 text-base font-semibold leading-6 text-white/85">{children}</h3>,
            ul: ({ children }) => <ul className="mb-4 list-disc space-y-1 pl-5 last:mb-0">{children}</ul>,
            ol: ({ children }) => <ol className="mb-4 list-decimal space-y-1 pl-5 last:mb-0">{children}</ol>,
            li: ({ children }) => <li className="pl-1">{children}</li>,
            strong: ({ children }) => <strong className="font-semibold text-white/90">{children}</strong>,
            blockquote: ({ children }) => (
              <blockquote className="mb-4 border-l-2 border-[#4f98a3]/50 pl-4 text-white/55 last:mb-0">
                {children}
              </blockquote>
            ),
            hr: () => <hr className="my-5 border-white/10" />,
            table: ({ children }) => (
              <div className="ledger-scroll mb-4 overflow-x-auto last:mb-0">
                <table className="min-w-full border-collapse text-left text-[13px] leading-6">
                  {children}
                </table>
              </div>
            ),
            thead: ({ children }) => <thead className="border-b border-white/15 text-white/85">{children}</thead>,
            tbody: ({ children }) => <tbody className="divide-y divide-white/10">{children}</tbody>,
            th: ({ children }) => <th className="px-3 py-2 font-semibold">{children}</th>,
            td: ({ children }) => <td className="px-3 py-2 align-top text-white/65">{children}</td>,
            code: ({ children, className }) => (
              <code className={`rounded bg-white/10 px-1.5 py-0.5 font-mono text-[0.92em] text-white/80 ${className ?? ''}`}>
                {children}
              </code>
            ),
            pre: ({ children }) => (
              <pre className="ledger-scroll mb-4 overflow-x-auto rounded-[10px] bg-[#1a1a1a] px-4 py-3 font-mono text-xs text-white/70 last:mb-0">
                {children}
              </pre>
            ),
            a: ({ children, href }) => (
              <a
                href={href}
                target="_blank"
                rel="noreferrer"
                className="inline-block font-medium text-[#4f98a3] underline decoration-[#4f98a3]/40 underline-offset-4 transition-[color,transform] duration-150 ease-out hover:text-[#72bdc6] active:scale-[0.97]"
              >
                {children}
              </a>
            ),
          }}
        >
          {message.text}
        </ReactMarkdown>
      </div>

      {isRichResponse && data.citations.length > 0 ? <CitationsPanel citations={data.citations} /> : null}

      {isRichResponse ? (
        <div className="flex flex-col gap-1.5">
          <div className="flex items-center gap-2">
            <span className="text-[12px] text-white/30">Confidence:</span>
            <ConfidenceBadge score={data.confidence_score} />
          </div>
          {data.conflict_warnings.length > 0 ? (
            <p className="text-xs leading-5 text-amber-200/70">Conflict warning: {data.conflict_warnings.join(', ')}</p>
          ) : null}
          {data.amendment_notice ? (
            <p className="text-xs leading-5 text-[#8ad2de]/75">Amendment notice: {data.amendment_notice}</p>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
