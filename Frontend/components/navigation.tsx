"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { useState } from "react"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import { Sheet, SheetContent, SheetTrigger } from "@/components/ui/sheet"
import { Home, Map, Upload, Brain, BarChart3, Download, Info, Moon, Sun, Menu, Layers3 } from "lucide-react"
import { useTheme } from "next-themes"

const navigation = [
  { name: "Home", href: "/", icon: Home },
  { name: "Dashboard", href: "/dashboard", icon: Map },
  { name: "Upload & Predict", href: "/upload", icon: Upload },
  { name: "Simulation", href: "/simulation", icon: Layers3 },
  { name: "ML Insights", href: "/ml-insights", icon: Brain },
  { name: "Analytics", href: "/analytics", icon: BarChart3 },
  { name: "Download Report", href: "/report", icon: Download },
  { name: "About", href: "/about", icon: Info },
]

export function Navigation() {
  const pathname = usePathname()
  const { theme, setTheme } = useTheme()
  const [isOpen, setIsOpen] = useState(false)

  const toggleTheme = () => {
    setTheme(theme === "light" ? "dark" : "light")
  }

  return (
    <nav className="border-b border-white/[0.06] bg-black/70 backdrop-blur-2xl sticky top-0 z-50">
      <div className="container mx-auto px-4">
        <div className="flex h-16 items-center justify-between">
          <div className="flex items-center space-x-8">
            <Link href="/" className="flex items-center space-x-2 group">
              <div className="h-9 w-9 rounded-xl bg-gradient-to-br from-orange-500 to-red-600 flex items-center justify-center shadow-lg shadow-orange-500/20 group-hover:shadow-orange-500/40 transition-shadow duration-300">
                <span className="text-white font-bold text-sm">🔥</span>
              </div>
              <span className="font-bold text-lg text-white">FirePredict</span>
            </Link>

            {/* Desktop Navigation */}
            <div className="hidden lg:flex items-center space-x-1">
              {navigation.map((item) => {
                const Icon = item.icon
                const isActive = pathname === item.href
                return (
                  <Link
                    key={item.name}
                    href={item.href}
                    className={cn(
                      "flex items-center space-x-1.5 px-3 py-2 rounded-lg text-sm font-medium transition-all duration-200",
                      isActive
                        ? "bg-orange-500/15 text-orange-400 shadow-sm shadow-orange-500/10"
                        : "text-gray-400 hover:text-white hover:bg-white/[0.06]",
                    )}
                  >
                    <Icon className="h-4 w-4" />
                    <span>{item.name}</span>
                  </Link>
                )
              })}
            </div>
          </div>

          <div className="flex items-center space-x-2">
            {/* Theme Toggle */}
            <Button variant="ghost" size="icon" onClick={toggleTheme} className="text-gray-400 hover:text-white hover:bg-white/[0.06]">
              <Sun className="h-4 w-4 rotate-0 scale-100 transition-all dark:-rotate-90 dark:scale-0" />
              <Moon className="absolute h-4 w-4 rotate-90 scale-0 transition-all dark:rotate-0 dark:scale-100" />
              <span className="sr-only">Toggle theme</span>
            </Button>

            {/* Mobile Menu */}
            <Sheet open={isOpen} onOpenChange={setIsOpen}>
              <SheetTrigger asChild>
                <Button variant="ghost" size="icon" className="lg:hidden text-gray-400 hover:text-white">
                  <Menu className="h-5 w-5" />
                  <span className="sr-only">Toggle menu</span>
                </Button>
              </SheetTrigger>
              <SheetContent side="right" className="w-[300px] sm:w-[400px] bg-[#0a0a10] border-white/[0.06]">
                <div className="flex flex-col space-y-4 mt-6">
                  <div className="flex items-center space-x-2 mb-4">
                    <div className="h-9 w-9 rounded-xl bg-gradient-to-br from-orange-500 to-red-600 flex items-center justify-center">
                      <span className="text-white font-bold text-sm">🔥</span>
                    </div>
                    <span className="font-bold text-xl text-white">FirePredict</span>
                  </div>

                  {navigation.map((item) => {
                    const Icon = item.icon
                    return (
                      <Link
                        key={item.name}
                        href={item.href}
                        onClick={() => setIsOpen(false)}
                        className={cn(
                          "flex items-center space-x-3 px-3 py-3 rounded-lg text-sm font-medium transition-all duration-200",
                          pathname === item.href
                            ? "bg-orange-500/15 text-orange-400"
                            : "text-gray-400 hover:text-white hover:bg-white/[0.06]",
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
