/** @type {import('next').NextConfig} */
const nextConfig = {
  eslint: {
    ignoreDuringBuilds: true,
  },
  typescript: {
    ignoreBuildErrors: true,
  },
  images: {
    unoptimized: true,
  },
  experimental: {
    allowedDevOrigins: ['192.168.29.205', '100.80.88.3', 'localhost']
  },
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: process.env.NEXT_PUBLIC_API_URL ? `${process.env.NEXT_PUBLIC_API_URL}/api/ml/:path*` : 'http://localhost:8080/api/ml/:path*', // Proxy to Spring Boot Backend
      },
    ]
  },
}

export default nextConfig
