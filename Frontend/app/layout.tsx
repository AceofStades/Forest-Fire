import type React from "react"
import type { Metadata } from "next"
import { Inter } from "next/font/google"
import "./globals.css"
import { ThemeProvider } from "@/components/theme-provider"
import { Toaster } from "@/components/ui/toaster"
// import type { Metadata } from 'next'
// import { GeistSans } from 'geist/font/sans'
// import { GeistMono } from 'geist/font/mono'
import { Analytics } from '@vercel/analytics/next'

const inter = Inter({ subsets: ["latin"] })

export const metadata: Metadata = {
  title: "Forest Fire Prediction System",
  description: "AI/ML-powered forest fire spread simulation and prediction platform",
  generator: 'v0.dev'
}


export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>){
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={inter.className}>
        <ThemeProvider attribute="class" defaultTheme="system" enableSystem disableTransitionOnChange>
          {children}
          <Analytics />
          <Toaster />
        </ThemeProvider>
      </body>
    </html>
  )
}
