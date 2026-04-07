'use client';

import { useCallback, useEffect, useState } from 'react';
import { fetchGraphTopology, fetchSectionContent } from '@/lib/api';
import { ALL_JURISDICTIONS } from '@/lib/constants';
import type { GraphFilters, GraphNode, GraphTopology, SectionContent } from '@/lib/types';

export interface UseGraphExplorerReturn {
  topology: GraphTopology | null;
  isLoading: boolean;
  error: string | null;
  selectedNode: GraphNode | null;
  hoveredNode: GraphNode | null;
  sectionContent: SectionContent | null;
  isSectionLoading: boolean;
  filters: GraphFilters;
  setSelectedNode: (node: GraphNode | null) => void;
  setHoveredNode: (node: GraphNode | null) => void;
  setFilters: (filters: GraphFilters) => void;
  navigateToNode: (sectionId: string, jurisdiction: string) => void;
}

export function useGraphExplorer(): UseGraphExplorerReturn {
  const [topology, setTopology] = useState<GraphTopology | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [selectedNode, setSelectedNodeRaw] = useState<GraphNode | null>(null);
  const [hoveredNode, setHoveredNode] = useState<GraphNode | null>(null);
  const [sectionContent, setSectionContent] = useState<SectionContent | null>(null);
  const [isSectionLoading, setIsSectionLoading] = useState(false);

  const [filters, setFilters] = useState<GraphFilters>({
    jurisdictions: new Set(ALL_JURISDICTIONS),
    edgeTypes: new Set(['REFERENCES', 'DERIVED_FROM']),
    nodeSizeMode: 'connections',
  });

  // Fetch topology once on mount
  useEffect(() => {
    let cancelled = false;
    setIsLoading(true);
    fetchGraphTopology()
      .then(data => {
        if (!cancelled) {
          setTopology(data);
          setError(null);
        }
      })
      .catch(err => {
        if (!cancelled) setError((err as Error).message);
      })
      .finally(() => {
        if (!cancelled) setIsLoading(false);
      });
    return () => { cancelled = true; };
  }, []);

  // Fetch section content when a node is selected
  const setSelectedNode = useCallback((node: GraphNode | null) => {
    setSelectedNodeRaw(node);
    if (!node) {
      setSectionContent(null);
      return;
    }
    setIsSectionLoading(true);
    fetchSectionContent(node.section_id, node.jurisdiction, node.chunk_id)
      .then(data => setSectionContent(data))
      .catch(err => {
        console.error('Section content fetch failed:', err);
        setSectionContent(null);
      })
      .finally(() => setIsSectionLoading(false));
  }, []);

  // Navigate to a node by section_id + jurisdiction (called from SectionDrawer chips)
  const navigateToNode = useCallback(
    (sectionId: string, jurisdiction: string) => {
      if (!topology) return;
      const target = topology.nodes.find(
        n => n.section_id === sectionId && n.jurisdiction === jurisdiction,
      );
      if (target) setSelectedNode(target);
    },
    [topology, setSelectedNode],
  );

  return {
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
  };
}
