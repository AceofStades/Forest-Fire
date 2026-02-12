"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { useState } from "react"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import { Sheet, SheetContent, SheetTrigger } from "@/components/ui/sheet"
import { Home, Map, Upload, Brain, BarChart3, Download, Info, Menu, Layers3 } from "lucide-react"

const navigation = [
  { name: "Home", href: "/", icon: Home },
  { name: "Dashboard", href: "/dashboard", icon: Map },
  { name: "Upload & Predict", href: "/upload", icon: Upload },
  { name: "Simulation", href: "/simulation", icon: Layers3 },
  { name: "ML Insights", href: "/ml-insights", icon: Brain },
  { name: "Analytics", href: "/analytics", icon: BarChart3 },
  // { name: "Download Report", href: "/report", icon: Download },
  { name: "About", href: "/about", icon: Info },
]

export function Navigation() {
  const pathname = usePathname()
  const [isOpen, setIsOpen] = useState(false)

  // Split navigation for desktop layout
  const mainNavigation = navigation.filter(item => item.name !== "About")
  const aboutItem = navigation.find(item => item.name === "About")

  return (
    <nav className="border-b border-white/[0.05] bg-black/80 backdrop-blur-xl sticky top-0 z-50">
      <div className="container mx-auto px-4">
        {/* Added 'relative' to the container to support absolute centering */}
        <div className="flex h-16 items-center justify-between relative">
          
          {/* 1. Logo (Left) */}
          <div className="flex items-center shrink-0">
            <Link href="/" className="flex items-center space-x-2 group">
              <div className="h-8 w-8 rounded-lg bg-orange-800 flex items-center justify-center transition-transform duration-300 group-hover:scale-105">
                <span className="text-white font-bold text-xs">🔥</span>
              </div>
              <span className="font-bold text-md text-white tracking-tighter uppercase">Fire Prediction</span>
            </Link>
          </div>

          {/* 2. Main Navigation (Centered) */}
          {/* Uses absolute centering to stay truly in the middle regardless of logo width */}
          <div className="hidden lg:flex absolute left-1/2 -translate-x-1/2 items-center space-x-1">
            {mainNavigation.map((item) => {
              const Icon = item.icon
              const isActive = pathname === item.href
              return (
                <Link
                  key={item.name}
                  href={item.href}
                  className={cn(
                    "flex items-center space-x-1.5 px-3 py-1.5 rounded-md text-[13px] font-medium transition-all duration-200",
                    isActive
                      ? "text-orange-600 bg-orange-950/20"
                      : "text-slate-400 hover:text-white hover:bg-white/5",
                  )}
                >
                  <Icon className="h-3.5 w-3.5" />
                  <span>{item.name}</span>
                </Link>
              )
            })}
          </div>

          {/* 3. Right Corner Items (About + Mobile Toggle) */}
          <div className="flex items-center space-x-4">
            {/* Desktop About Link */}
            {aboutItem && (
              <Link
                href={aboutItem.href}
                className={cn(
                  "hidden lg:flex items-center space-x-1.5 px-3 py-1.5 rounded-md text-[13px] font-medium transition-all",
                  pathname === aboutItem.href
                    ? "text-orange-600 bg-orange-950/20"
                    : "text-slate-400 hover:text-white hover:bg-white/5"
                )}
              >
                <aboutItem.icon className="h-3.5 w-3.5" />
                <span>{aboutItem.name}</span>
              </Link>
            )}

            {/* Mobile Menu Trigger */}
            <Sheet open={isOpen} onOpenChange={setIsOpen}>
              <SheetTrigger asChild>
                <Button variant="ghost" size="icon" className="lg:hidden text-slate-400 hover:text-white hover:bg-white/5">
                  <Menu className="h-5 w-5" />
                  <span className="sr-only">Toggle menu</span>
                </Button>
              </SheetTrigger>
              <SheetContent side="right" className="w-[300px] bg-[#050505] border-white/5">
                <div className="flex flex-col space-y-4 mt-8">
                  <div className="flex items-center space-x-2 mb-8 px-3">
                    <div className="h-8 w-8 rounded-lg bg-orange-800 flex items-center justify-center">
                      <span className="text-white font-bold text-xs">🔥</span>
                    </div>
                    <span className="font-bold text-lg text-white">Fire Prediction</span>
                  </div>

                  {navigation.map((item) => {
                    const Icon = item.icon
                    const isActive = pathname === item.href
                    return (
                      <Link
                        key={item.name}
                        href={item.href}
                        onClick={() => setIsOpen(false)}
                        className={cn(
                          "flex items-center space-x-3 px-3 py-3 rounded-lg text-sm font-medium transition-all",
                          isActive
                            ? "bg-orange-950/30 text-orange-600"
                            : "text-slate-400 hover:text-white hover:bg-white/5",
                        )}
                      >
                        <Icon className="h-5 w-5" />
                        <span>{item.name}</span>
                      </Link>
                    )
                  })}
                </div>
              </SheetContent>
            </Sheet>
          </div>
        </div>
      </div>
    </nav>
  )
}