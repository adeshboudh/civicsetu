'use client';

import { useEffect, useRef, useState } from 'react';
import { ForceGraph } from './ForceGraph';
import { GraphFilterSidebar } from './GraphFilterSidebar';
import { SectionDrawer } from './SectionDrawer';
import type { UseGraphExplorerReturn } from '@/hooks/useGraphExplorer';

interface Props extends UseGraphExplorerReturn {
  onChatAboutSection: (
    sectionId: string,
    title: string,
    docName: string,
    jurisdiction: string,
  ) => void;
}

export function GraphExplorer({
  topology,
  isLoading,
  error,
  selectedNode,
  hoveredNode,
  sectionContent,
  isSectionLoading,
  filters,
  setSelectedNode,
  setHoveredNode,
  setFilters,
  navigateToNode,
  onChatAboutSection,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [dims, setDims] = useState({ width: 0, height: 0 });

  useEffect(() => {
    if (!containerRef.current) {
      return;
    }

    const ro = new ResizeObserver(entries => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        setDims({ width: Math.floor(width), height: Math.floor(height) });
      }
    });

    ro.observe(containerRef.current);
    return () => ro.disconnect();
  }, []);

  const nodes = topology?.nodes ?? [];
  const edges = topology?.edges ?? [];

  return (
    <div className="flex h-full min-h-0 flex-col bg-[#0d0d0d]">
      <div className="flex h-12 shrink-0 items-center justify-between border-b border-white/[0.06] px-4">
        <div>
          <p className="font-mono text-[10px] uppercase tracking-[0.24em] text-white/30">Graph Explorer</p>
          <p className="mt-0.5 text-[12px] text-white/40">
            {nodes.length} sections / {edges.length} relationships
          </p>
        </div>
        <div className="flex items-center gap-2 font-mono text-[10px] uppercase tracking-[0.2em] text-white/30">
          <span className="h-1.5 w-1.5 rounded-full bg-[#4f98a3]" />
          Live topology
        </div>
      </div>

      <div ref={containerRef} className="relative min-h-0 flex-1 overflow-hidden bg-[#0d0d0d]">
        <div className="pointer-events-none absolute inset-0 opacity-[0.14] [background-image:radial-gradient(circle_at_center,rgba(229,226,225,0.72)_1px,transparent_1.5px)] [background-position:0_0] [background-size:24px_24px]" />

        {isLoading ? (
          <div className="absolute inset-0 z-10 flex items-center justify-center">
            <div className="flex items-center gap-2 font-mono text-[10px] uppercase tracking-[0.24em] text-white/30">
              {[0, 1, 2].map(i => (
                <span
                  key={i}
                  className="h-1.5 w-1.5 rounded-full bg-white/25 animate-pulse"
                  style={{ animationDelay: `${i * 140}ms` }}
                />
              ))}
              Loading graph
            </div>
          </div>
        ) : null}

        {error ? (
          <div className="absolute inset-0 z-10 flex items-center justify-center">
            <div className="max-w-xs border border-red-300/15 bg-red-950/10 p-4 text-center">
              <p className="text-sm text-red-200/80">Graph unavailable</p>
              <p className="mt-1 text-xs leading-5 text-white/30">{error}</p>
            </div>
          </div>
        ) : null}

        {!isLoading && !error && nodes.length === 0 ? (
          <div className="absolute inset-0 z-10 flex items-center justify-center">
            <div className="border border-white/[0.06] bg-[#141414] p-5 text-center">
              <p className="text-sm text-white/60">No section relationships found</p>
              <p className="mt-1 text-xs text-white/30">Run ingestion to populate the graph.</p>
            </div>
          </div>
        ) : null}

        {dims.width > 0 && dims.height > 0 && nodes.length > 0 ? (
          <ForceGraph
            nodes={nodes}
            edges={edges}
            filters={filters}
            selectedNode={selectedNode}
            hoveredNode={hoveredNode}
            onNodeClick={node => setSelectedNode(node)}
            onNodeHover={setHoveredNode}
            width={dims.width}
            height={dims.height}
          />
        ) : null}

        <GraphFilterSidebar filters={filters} onFiltersChange={setFilters} topology={topology} />

        <SectionDrawer
          content={sectionContent}
          isLoading={isSectionLoading}
          onClose={() => setSelectedNode(null)}
          onNodeNavigate={navigateToNode}
          onChatAboutSection={onChatAboutSection}
        />
      </div>
    </div>
  );
}
