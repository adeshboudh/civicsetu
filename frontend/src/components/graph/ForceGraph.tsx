'use client';

import * as d3 from 'd3';
import { useEffect, useRef } from 'react';
import { EDGE_STYLES, JURISDICTION_COLORS, NODE_RADIUS } from '@/lib/constants';
import type { GraphEdge, GraphFilters, GraphNode } from '@/lib/types';

interface Props {
  nodes: GraphNode[];
  edges: GraphEdge[];
  filters: GraphFilters;
  selectedNode: GraphNode | null;
  hoveredNode: GraphNode | null;
  onNodeClick: (node: GraphNode | null) => void;
  onNodeHover: (node: GraphNode | null) => void;
  width: number;
  height: number;
}

export function ForceGraph({
  nodes,
  edges,
  filters,
  selectedNode,
  hoveredNode,
  onNodeClick,
  onNodeHover,
  width,
  height,
}: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const simRef = useRef<d3.Simulation<GraphNode, GraphEdge> | null>(null);
  const zoomRef = useRef<d3.ZoomBehavior<SVGSVGElement, unknown> | null>(null);

  const filteredNodes = nodes.filter(node => filters.jurisdictions.has(node.jurisdiction));
  const filteredNodeIds = new Set(filteredNodes.map(node => node.chunk_id));
  const filteredEdges = edges.filter(
    edge =>
      filters.edgeTypes.has(edge.edge_type) &&
      filteredNodeIds.has(typeof edge.source === 'string' ? edge.source : edge.source.chunk_id) &&
      filteredNodeIds.has(typeof edge.target === 'string' ? edge.target : edge.target.chunk_id),
  );

  useEffect(() => {
    if (!svgRef.current || width === 0 || height === 0) {
      return;
    }

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const defs = svg.append('defs');
    const filter = defs.append('filter').attr('id', 'node-glow');
    filter.append('feGaussianBlur').attr('stdDeviation', '3').attr('result', 'blur');
    const merge = filter.append('feMerge');
    merge.append('feMergeNode').attr('in', 'blur');
    merge.append('feMergeNode').attr('in', 'SourceGraphic');

    const container = svg.append('g').attr('class', 'zoom-container');
    const edgeGroup = container.append('g').attr('class', 'edges');
    const nodeGroup = container.append('g').attr('class', 'nodes');

    const simNodes: GraphNode[] = filteredNodes.map(node => ({ ...node }));
    const chunkIdToNode = new Map(simNodes.map(node => [node.chunk_id, node]));
    const simEdges: GraphEdge[] = filteredEdges.map(edge => ({
      ...edge,
      source: typeof edge.source === 'string' ? edge.source : edge.source.chunk_id,
      target: typeof edge.target === 'string' ? edge.target : edge.target.chunk_id,
    }));

    const connectionCounts = simNodes.map(node => node.connection_count);
    const minConn = Math.min(...connectionCounts, 1);
    const maxConn = Math.max(...connectionCounts, 2);
    const radiusScale = d3.scaleSqrt().domain([minConn, maxConn]).range([NODE_RADIUS.MIN, NODE_RADIUS.MAX]);
    const getRadius = (node: GraphNode) =>
      filters.nodeSizeMode === 'uniform' ? NODE_RADIUS.DEFAULT : radiusScale(node.connection_count);

    const sim = d3
      .forceSimulation<GraphNode>(simNodes)
      .force(
        'link',
        d3
          .forceLink<GraphNode, GraphEdge>(simEdges)
          .id(node => node.chunk_id)
          .distance(82)
          .strength(0.38),
      )
      .force('charge', d3.forceManyBody<GraphNode>().strength(-64).distanceMax(310))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collide', d3.forceCollide<GraphNode>(node => getRadius(node) + 3))
      .alphaDecay(0.02)
      .velocityDecay(0.42);

    simRef.current = sim;

    const link = edgeGroup
      .selectAll<SVGLineElement, GraphEdge>('line')
      .data(simEdges)
      .join('line')
      .attr('stroke', edge => EDGE_STYLES[edge.edge_type]?.color ?? 'rgba(255,255,255,0.14)')
      .attr('stroke-width', edge => EDGE_STYLES[edge.edge_type]?.width ?? 1)
      .attr('stroke-dasharray', edge => EDGE_STYLES[edge.edge_type]?.dashArray ?? 'none')
      .attr('stroke-opacity', 0.82);

    const nodeEl = nodeGroup
      .selectAll<SVGGElement, GraphNode>('g.node')
      .data(simNodes, node => node.chunk_id)
      .join('g')
      .attr('class', 'node')
      .style('cursor', 'pointer')
      .call(
        d3
          .drag<SVGGElement, GraphNode>()
          .on('start', (event: d3.D3DragEvent<SVGGElement, GraphNode, GraphNode>, node: GraphNode) => {
            if (!event.active) {
              sim.alphaTarget(0.3).restart();
            }
            node.fx = node.x;
            node.fy = node.y;
          })
          .on('drag', (event: d3.D3DragEvent<SVGGElement, GraphNode, GraphNode>, node: GraphNode) => {
            node.fx = event.x;
            node.fy = event.y;
          })
          .on('end', (event: d3.D3DragEvent<SVGGElement, GraphNode, GraphNode>, node: GraphNode) => {
            if (!event.active) {
              sim.alphaTarget(0);
            }
            node.fx = null;
            node.fy = null;
          }),
      );

    nodeEl
      .append('circle')
      .attr('class', 'glow-ring')
      .attr('r', node => getRadius(node) + 4)
      .attr('fill', 'none')
      .attr('stroke', node => JURISDICTION_COLORS[node.jurisdiction] ?? '#888')
      .attr('stroke-width', 1.5)
      .attr('filter', 'url(#node-glow)')
      .attr('opacity', 0);

    nodeEl
      .append('circle')
      .attr('class', 'main-circle')
      .attr('r', node => getRadius(node))
      .attr('fill', node => JURISDICTION_COLORS[node.jurisdiction] ?? '#888')
      .attr('stroke', '#0d0d0d')
      .attr('stroke-width', 1.25)
      .attr('stroke-opacity', 0.75);

    nodeEl
      .append('text')
      .attr('class', 'node-label')
      .attr('dy', node => -(getRadius(node) + 6))
      .attr('text-anchor', 'middle')
      .attr('fill', 'rgba(229,226,225,0.86)')
      .attr('font-size', '10px')
      .attr('font-family', 'var(--font-inter), sans-serif')
      .attr('pointer-events', 'none')
      .text(node => `Sec ${node.section_id} / ${node.title.slice(0, 28)}${node.title.length > 28 ? '...' : ''}`)
      .attr('opacity', 0);

    nodeEl
      .on('mouseenter', (_event: MouseEvent, node: GraphNode) => {
        onNodeHover(chunkIdToNode.get(node.chunk_id) ?? null);
      })
      .on('mouseleave', () => {
        onNodeHover(null);
      })
      .on('click', (event: MouseEvent, node: GraphNode) => {
        event.stopPropagation();
        onNodeClick(chunkIdToNode.get(node.chunk_id) ?? node);
      });

    svg.on('click', () => onNodeClick(null));

    const zoom = d3
      .zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.3, 5])
      .on('zoom', (event: d3.D3ZoomEvent<SVGSVGElement, unknown>) => {
        container.attr('transform', event.transform.toString());
        nodeEl.select<SVGTextElement>('.node-label').attr('opacity', event.transform.k > 2.5 ? 0.9 : 0);
      });

    zoomRef.current = zoom;
    svg.call(zoom);

    sim.on('tick', () => {
      link
        .attr('x1', edge => (edge.source as GraphNode).x ?? 0)
        .attr('y1', edge => (edge.source as GraphNode).y ?? 0)
        .attr('x2', edge => (edge.target as GraphNode).x ?? 0)
        .attr('y2', edge => (edge.target as GraphNode).y ?? 0);

      nodeEl.attr('transform', node => `translate(${node.x ?? 0},${node.y ?? 0})`);
    });

    return () => {
      sim.stop();
      svg.on('.zoom', null);
    };
    // Rebuilding here keeps D3 ownership simple when the filter set changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [filteredNodes.length, filteredEdges.length, width, height, filters.nodeSizeMode]);

  useEffect(() => {
    if (!svgRef.current) {
      return;
    }

    const svg = d3.select(svgRef.current);
    const nodeEl = svg.selectAll<SVGGElement, GraphNode>('g.node');

    if (!hoveredNode) {
      nodeEl.selectAll('.main-circle').attr('opacity', 1);
      nodeEl.selectAll<SVGTextElement, GraphNode>('.node-label').each(function() {
        const zoom = zoomRef.current;
        if (zoom && svgRef.current) {
          const transform = d3.zoomTransform(svgRef.current);
          d3.select(this).attr('opacity', transform.k > 2.5 ? 0.9 : 0);
        } else {
          d3.select(this).attr('opacity', 0);
        }
      });
      return;
    }

    const neighborIds = new Set<string>([hoveredNode.chunk_id]);
    svg.selectAll<SVGLineElement, GraphEdge>('line').each((edge: GraphEdge) => {
      const src = typeof edge.source === 'string' ? edge.source : edge.source.chunk_id;
      const tgt = typeof edge.target === 'string' ? edge.target : edge.target.chunk_id;
      if (src === hoveredNode.chunk_id) {
        neighborIds.add(tgt);
      }
      if (tgt === hoveredNode.chunk_id) {
        neighborIds.add(src);
      }
    });

    nodeEl.each(function(this: SVGGElement, node: GraphNode) {
      const isNeighbor = neighborIds.has(node.chunk_id);
      d3.select(this).select('.main-circle').attr('opacity', isNeighbor ? 1 : 0.12);
      d3.select(this)
        .select('.node-label')
        .attr('opacity', node.chunk_id === hoveredNode.chunk_id ? 1 : 0);
    });
  }, [hoveredNode]);

  useEffect(() => {
    if (!svgRef.current) {
      return;
    }

    const svg = d3.select(svgRef.current);
    svg.selectAll<SVGCircleElement, GraphNode>('.glow-ring').each(function(this: SVGCircleElement, node: GraphNode) {
      const isSelected = selectedNode?.chunk_id === node.chunk_id;
      d3.select(this)
        .attr('opacity', isSelected ? 0.8 : 0)
        .classed('node-selected', isSelected);
    });
  }, [selectedNode]);

  return (
    <svg
      ref={svgRef}
      width={width}
      height={height}
      className="h-full w-full bg-[#0d0d0d]"
      style={{ display: 'block' }}
    />
  );
}
