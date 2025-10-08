"use client"

import dynamic from "next/dynamic"
import { Navbar } from "@/src/components/navbar"
import { useAuth } from "@/src/contexts/auth-context"
import { useRouter } from "next/navigation"
import { useEffect } from "react"

// ✅ Load the globe scene client-side only
const GlobeScene = dynamic(() => import("@/src/components/globe-scene"), {
  ssr: false,
})

export default function GlobePage() {
  const { isAuthenticated } = useAuth()
  const router = useRouter()

  useEffect(() => {
    if (!isAuthenticated) {
      router.push("/login")
    }
  }, [isAuthenticated, router])

  if (!isAuthenticated) return null

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-800">
      <Navbar />
      <GlobeScene />
    </div>
  )
}
