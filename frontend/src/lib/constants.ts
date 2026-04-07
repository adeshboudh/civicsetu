import type { Jurisdiction } from './types';

export const JURISDICTION_COLORS: Record<Jurisdiction, string> = {
  CENTRAL:       '#2db7a3',
  MAHARASHTRA:   '#4e7cff',
  UTTAR_PRADESH: '#f4b63f',
  KARNATAKA:     '#78b94d',
  TAMIL_NADU:    '#ff7a59',
};

export const EDGE_STYLES = {
  REFERENCES: {
    color: 'rgba(255,255,255,0.15)',
    dashArray: 'none',
    width: 1,
  },
  DERIVED_FROM: {
    color: '#e8af34',
    dashArray: '6,3',
    width: 1.5,
  },
} as const;

export const NODE_RADIUS = {
  MIN: 4,
  MAX: 18,
  DEFAULT: 6,
} as const;

export const ALL_JURISDICTIONS: Jurisdiction[] = [
  'CENTRAL',
  'MAHARASHTRA',
  'UTTAR_PRADESH',
  'KARNATAKA',
  'TAMIL_NADU',
];

export const JURISDICTION_LABELS: Record<Jurisdiction, string> = {
  CENTRAL:       'Central',
  MAHARASHTRA:   'Maharashtra',
  UTTAR_PRADESH: 'Uttar Pradesh',
  KARNATAKA:     'Karnataka',
  TAMIL_NADU:    'Tamil Nadu',
};
