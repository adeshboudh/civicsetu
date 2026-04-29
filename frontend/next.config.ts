import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  output: 'export',
  images: {
    unoptimized: true,
  },
  // We remove rewrites because they don't work with output: 'export'
  // The frontend will now call the API relatively or via NEXT_PUBLIC_BACKEND_URL
};

export default nextConfig;
