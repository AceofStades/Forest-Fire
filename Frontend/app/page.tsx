"use client" // Directive MUST be on Line 1

import { useEffect } from "react"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Navigation } from "@/components/navigation"
import Hero1 from "@/components/hero1"
import LandingFeatures from "@/components/LandingFeatures"
import Techstack from "@/components/techstack"
import { Brain, Activity, ArrowRight, Zap, Flame, Map as MapIcon, ShieldCheck, Target } from "lucide-react"

export default function HomePage() {
  
  useEffect(() => {
    console.log("Forest Fire Prediction System: Operational");
  }, []);

  return (
    <div className="min-h-screen bg-[#050505] text-slate-300 selection:bg-orange-500/30 font-sans tracking-tight">
      
      {/* 1. Project Title - Minimalist Top Bar */}

      <Navigation />
      
      {/* 2. Hero Section */}
      <Hero1 />

      {/* 3. NEW: Mission Statement Section (Harnessing AI) */}
      <section className="container mx-auto px-6 py-24 text-center border-b border-white/5">
        <div className="max-w-3xl mx-auto space-y-6">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-orange-950/30 border border-orange-900/50 text-orange-500 text-[10px] uppercase tracking-widest font-bold">
            <Target className="h-3 w-3" />
            Core Mission
          </div>
          <h2 className="text-3xl md:text-5xl font-extralight text-white leading-tight tracking-tighter">
            Harnessing <span className="text-orange-800 italic">Artificial Intelligence</span> to Predict, Prevent, and Visualize Forest Fires.
          </h2>
          <p className="text-slate-500 text-sm md:text-base font-light max-w-xl mx-auto leading-relaxed">
            Turning complex environmental data into actionable insights for a safer, more resilient tomorrow.
          </p>
        </div>
      </section>

      {/* 4. Performance Metrics */}
      <section className="border-b border-white/5 bg-[#080808]">
        <div className="container mx-auto px-6 py-12">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-12">
            {[
              { label: "Precision", value: "30m", icon: MapIcon },
              { label: "Lead Time", value: "24h", icon: Zap },
              { label: "Variables", value: "15+", icon: Activity },
              { label: "Confidence", value: "89%", icon: ShieldCheck },
            ].map((stat) => (
              <div key={stat.label} className="flex flex-col gap-1">
                <div className="flex items-center gap-2 text-orange-900/60">
                  <stat.icon className="h-3 w-3" />
                  <span className="text-[9px] uppercase tracking-widest font-bold">{stat.label}</span>
                </div>
                <div className="text-3xl font-light text-white tracking-tighter">{stat.value}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* 5. Core Features */}
      <LandingFeatures />

      {/* 6. Neural Architecture */}
      <section className="container mx-auto px-6 py-32">
        <div className="max-w-xl mb-24">
          <h2 className="text-5xl font-extralight text-white mb-8 tracking-tighter">
            Analytical <span className="text-orange-800 italic">Ecosystem.</span>
          </h2>
          <p className="text-slate-500 text-lg font-light leading-relaxed">
            Our platform utilizes advanced neural segmentation to transform atmospheric data into real-time threat intelligence.
          </p>
        </div>

        <div className="grid lg:grid-cols-3 gap-px bg-white/5 border border-white/5">
          {[
            { title: "Spatial Intelligence", icon: Brain, desc: "U-Net segmentation maps terrain using 3x3 Convolutional filters." },
            { title: "Temporal Sequence", icon: Activity, desc: "LSTM memory cells analyze moisture depletion and heat accumulation over 72-hour windows." },
            { title: "Risk Projection", icon: Flame, desc: "Cellular Automata simulations model fire spread across complex topographical gradients." },
          ].map((item, i) => (
            <div key={i} className="p-16 bg-[#050505] hover:bg-[#0a0a0a] transition-all group relative">
              <item.icon className="h-8 w-8 text-orange-950 group-hover:text-orange-700 transition-colors mb-12" />
              <h3 className="text-xl font-bold text-white mb-6 tracking-tight">{item.title}</h3>
              <p className="text-slate-500 text-sm leading-relaxed font-light">{item.desc}</p>
              <div className="absolute bottom-0 left-0 h-[2px] w-0 bg-orange-900 group-hover:w-full transition-all duration-1000" />
            </div>
          ))}
        </div>
      </section>

      <Techstack />

      {/* 7. Action CTA */}
      <section className="container mx-auto px-6 py-48 text-center">
        <div className="max-w-3xl mx-auto border border-white/5 p-20 bg-gradient-to-b from-[#0a0a0a] to-black">
          <h2 className="text-6xl font-bold text-white mb-12 tracking-tighter">
            Predict. <span className="text-orange-900">Prevent.</span> Protect.
          </h2>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Button asChild size="lg" className="bg-white text-black hover:bg-slate-200 rounded-none px-12 h-16 font-bold uppercase tracking-widest text-[10px]">
              <Link href="/dashboard">Launch Analysis</Link>
            </Button>
            <Button asChild variant="outline" size="lg" className="border-white/10 hover:bg-white/5 text-white rounded-none px-12 h-16 uppercase tracking-widest text-[10px]">
              <Link href="/simulation">Live Engine <ArrowRight className="ml-2 h-3 w-3" /></Link>
            </Button>
          </div>
        </div>
      </section>
    </div>
  )
}