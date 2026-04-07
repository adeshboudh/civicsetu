import type {
  ApiResponse,
  GraphTopology,
  Jurisdiction,
  SectionContent,
} from './types';

const API_BASE_URL = process.env.NEXT_PUBLIC_BACKEND_URL ?? '';

function apiUrl(path: string): string {
  return `${API_BASE_URL}${path}`;
}

export interface QueryPayload {
  query: string;
  session_id?: string;
  jurisdiction_filter?: Jurisdiction;
  top_k?: number;
}

export async function queryRera(payload: QueryPayload): Promise<ApiResponse> {
  const res = await fetch(apiUrl('/api/v1/query'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const err = (await res.json()) as { detail?: string };
      if (err.detail) detail = err.detail;
    } catch {}
    throw new Error(`API error ${res.status}: ${detail}`);
  }
  return res.json() as Promise<ApiResponse>;
}

export async function fetchGraphTopology(): Promise<GraphTopology> {
  const res = await fetch(apiUrl('/api/v1/graph/topology'));
  if (!res.ok) {
    throw new Error(`Topology fetch failed: ${res.status}`);
  }
  return res.json() as Promise<GraphTopology>;
}

export async function fetchSectionContent(
  sectionId: string,
  jurisdiction: string,
  chunkId?: string,
): Promise<SectionContent> {
  const params = new URLSearchParams({ jurisdiction });
  if (chunkId) {
    params.set('chunk_id', chunkId);
  }
  const url = apiUrl(`/api/v1/graph/section/${encodeURIComponent(sectionId)}?${params.toString()}`);
  const res = await fetch(url);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const err = (await res.json()) as { detail?: string };
      if (err.detail) detail = err.detail;
    } catch {}
    throw new Error(`Section fetch failed: ${res.status} ${detail}`);
  }
  return res.json() as Promise<SectionContent>;
}

export interface SectionContextPayload {
  query: string;
  section_id: string;
  jurisdiction: string;
  session_id?: string;
}

export async function querySectionContext(
  payload: SectionContextPayload,
): Promise<ApiResponse> {
  const res = await fetch(apiUrl('/api/v1/query/section-context'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const err = (await res.json()) as { detail?: string };
      if (err.detail) detail = err.detail;
    } catch {}
    throw new Error(`API error ${res.status}: ${detail}`);
  }
  return res.json() as Promise<ApiResponse>;
}
