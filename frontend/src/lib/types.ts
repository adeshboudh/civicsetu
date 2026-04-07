export type Jurisdiction =
  | 'CENTRAL'
  | 'MAHARASHTRA'
  | 'UTTAR_PRADESH'
  | 'KARNATAKA'
  | 'TAMIL_NADU';

export type QueryType =
  | 'fact_lookup'
  | 'cross_reference'
  | 'conflict_detection'
  | 'temporal'
  | 'penalty_lookup';

export interface Citation {
  section_id: string;
  doc_name: string;
  jurisdiction: Jurisdiction;
  effective_date: string | null;
  source_url: string;
  chunk_id: string;
}

export interface CivicSetuResponse {
  answer: string;
  citations: Citation[];
  confidence_score: number;
  query_type_resolved: QueryType;
  conflict_warnings: string[];
  amendment_notice: string | null;
  session_id: string | null;
  disclaimer: string;
  confidence_level: 'HIGH' | 'MEDIUM' | 'LOW';
}

export interface InsufficientInfoResponse {
  answer: string;
  searched_query: string;
  session_id: string | null;
  disclaimer: string;
}

export type ApiResponse = CivicSetuResponse | InsufficientInfoResponse;

export function isCivicSetuResponse(r: ApiResponse): r is CivicSetuResponse {
  return 'citations' in r && Array.isArray(r.citations);
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'error';
  text: string;
  data?: ApiResponse;
}

// ── Graph Types ───────────────────────────────────────────────────────────────

export interface GraphNode {
  chunk_id: string;
  section_id: string;
  title: string;
  jurisdiction: Jurisdiction;
  doc_name: string;
  is_active: boolean;
  connection_count: number;
  // D3 simulation fields, mutated in place by force simulation.
  x?: number;
  y?: number;
  vx?: number;
  vy?: number;
  fx?: number | null;
  fy?: number | null;
}

export interface GraphEdge {
  source: string | GraphNode;
  target: string | GraphNode;
  edge_type: 'REFERENCES' | 'DERIVED_FROM';
}

export interface GraphTopology {
  nodes: GraphNode[];
  edges: GraphEdge[];
  stats: Record<string, number>;
}

export interface SectionChunk {
  chunk_id: string;
  text: string;
  page_number: number;
}

export interface ConnectedSection {
  section_id: string;
  title: string;
  jurisdiction: Jurisdiction;
  edge_type: 'REFERENCES_OUT' | 'REFERENCES_IN' | 'DERIVED_FROM_OUT' | 'DERIVED_FROM_IN';
}

export interface SectionContent {
  section_id: string;
  title: string;
  doc_name: string;
  jurisdiction: Jurisdiction;
  effective_date: string | null;
  source_url: string;
  chunks: SectionChunk[];
  connected_sections: ConnectedSection[];
}

export interface GraphFilters {
  jurisdictions: Set<Jurisdiction>;
  edgeTypes: Set<'REFERENCES' | 'DERIVED_FROM'>;
  nodeSizeMode: 'connections' | 'uniform';
}

export interface SectionContext {
  sectionId: string;
  title: string;
  docName: string;
  jurisdiction: string;
}
