import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Navigation } from "@/components/navigation"
import { Zap, Globe, Layers3 } from "lucide-react"
import Hero1 from "@/components/hero1"
import LandingFeatures from "@/components/LandingFeatures"

export default function HomePage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-orange-50 via-red-50 to-yellow-50 dark:from-gray-900 dark:via-gray-800 dark:to-gray-900">
      <Navigation />
      <Hero1 />
      <LandingFeatures />

      {/* Technology Stack */}
      <section className="container mx-auto px-4 py-16 bg-white/50 dark:bg-gray-800/50 rounded-lg mx-4">
        <h2 className="text-3xl font-bold text-center mb-12">Powered by Advanced Technology</h2>
        <div className="grid md:grid-cols-3 gap-8 text-center">
          <div>
            <Zap className="h-12 w-12 text-orange-600 mx-auto mb-4" />
            <h3 className="text-xl font-semibold mb-2">Machine Learning</h3>
            <p className="text-muted-foreground">U-Net, LSTM, and Cellular Automata models for accurate predictions</p>
          </div>
          <div>
            <Globe className="h-12 w-12 text-blue-600 mx-auto mb-4" />
            <h3 className="text-xl font-semibold mb-2">Geospatial Analysis</h3>
            <p className="text-muted-foreground">
              Advanced GIS processing with Leaflet.js and geospatial data handling
            </p>
          </div>
          <div>
            <Layers3 className="h-12 w-12 text-green-600 mx-auto mb-4" />
            <h3 className="text-xl font-semibold mb-2">3D Innovation</h3>
            <p className="text-muted-foreground">
              Three.js powered 3D terrain conversion and immersive fire simulation
            </p>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="container mx-auto px-4 py-16 text-center">
        <h2 className="text-3xl font-bold mb-6">Ready to Explore Fire Prediction?</h2>
        <p className="text-xl text-muted-foreground mb-8 max-w-2xl mx-auto">
          Start by exploring our interactive dashboard, convert your 2D maps to 3D, or upload your own data for custom
          predictions
        </p>
        <div className="flex flex-col sm:flex-row gap-4 justify-center">
          <Button
            asChild
            size="lg"
            className="bg-gradient-to-r from-orange-500 to-red-600 hover:from-orange-600 hover:to-red-700"
          >
            <Link href="/dashboard">Start Exploring</Link>
          </Button>
          <Button asChild variant="outline" size="lg">
            <Link href="/2d-to-3d">Try 2D to 3D</Link>
          </Button>
        </div>
      </section>
    </div>
  )
}
