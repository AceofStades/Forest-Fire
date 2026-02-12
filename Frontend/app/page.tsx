import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Navigation } from "@/components/navigation"
import Hero1 from "@/components/hero1"
import LandingFeatures from "@/components/LandingFeatures"
import Techstack from "@/components/techstack"

export default function HomePage() {
  return (
    <div className="min-h-screen bg-[#060609]">
      <Navigation />
      <Hero1 />
      <LandingFeatures />
      <Techstack />

      {/* CTA Section */}
      <section className="relative container mx-auto px-4 py-24 text-center">
        <div className="absolute inset-0 bg-gradient-to-t from-orange-600/5 via-transparent to-transparent pointer-events-none" />
        <div className="relative max-w-3xl mx-auto glass-card p-12">
          <h2 className="text-3xl md:text-4xl font-bold mb-4 bg-gradient-to-r from-orange-400 to-red-400 bg-clip-text text-transparent">
            Ready to Explore Fire Prediction?
          </h2>
          <p className="text-lg text-gray-400 mb-8 max-w-2xl mx-auto leading-relaxed">
            Start by exploring our interactive dashboard, convert your 2D maps to 3D, or upload your own data for custom
            predictions.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Button
              asChild
              size="lg"
              className="bg-gradient-to-r from-orange-500 to-red-600 hover:from-orange-400 hover:to-red-500 text-white border-0 shadow-lg shadow-orange-500/20 hover:shadow-orange-500/40"
            >
              <Link href="/dashboard">Start Exploring</Link>
            </Button>
            <Button asChild variant="outline" size="lg" className="border-orange-500/30 text-orange-300 hover:bg-orange-500/10 hover:border-orange-400/50">
              <Link href="/simulation">Try Simulation</Link>
            </Button>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-white/[0.06] py-8 text-center text-sm text-gray-500">
        <p>&copy; 2026 FirePredict — Forest Fire Prediction System</p>
      </footer>
    </div>
  )
}
