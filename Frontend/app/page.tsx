import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Card, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Navigation } from "@/components/navigation"
import { Map, Upload, Brain, BarChart3, Box, Zap, Globe, Layers3 } from "lucide-react"

export default function HomePage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-orange-50 via-red-50 to-yellow-50 dark:from-gray-900 dark:via-gray-800 dark:to-gray-900">
      <Navigation />

      {/* Hero Section for landing page */}
      <section className="container mx-auto px-4 py-16">
        <div className="text-center max-w-4xl mx-auto">
          <h1 className="text-4xl md:text-6xl font-bold bg-gradient-to-r from-orange-600 to-red-600 bg-clip-text text-transparent mb-6">
            Forest Fire Prediction System
          </h1>
          <p className="text-xl text-muted-foreground mb-8 leading-relaxed">
            Advanced AI/ML-powered platform for simulating and modeling forest fire spread. Get next-day fire
            probability maps, real-time spread animations, 2D to 3D terrain conversion, and custom predictions using
            cutting-edge U-Net, LSTM, and Cellular Automata models.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Button
              asChild
              size="lg"
              className="bg-gradient-to-r from-orange-500 to-red-600 hover:from-orange-600 hover:to-red-700"
            >
              <Link href="/dashboard">
                <Map className="mr-2 h-5 w-5" />
                Explore Dashboard
              </Link>
            </Button>
            <Button asChild variant="outline" size="lg">
              <Link href="/2d-to-3d">
                <Layers3 className="mr-2 h-5 w-5" />
                Convert 2D to 3D
              </Link>
            </Button>
          </div>
        </div>
      </section>

      {/* Features Grid */}
      <section className="container mx-auto px-4 py-16">
        <h2 className="text-3xl font-bold text-center mb-12">Key Features</h2>
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          <Card className="border-orange-200 hover:shadow-lg transition-shadow">
            <CardHeader>
              <Map className="h-10 w-10 text-orange-600 mb-2" />
              <CardTitle>Interactive Map Dashboard</CardTitle>
              <CardDescription>
                Real-time fire probability maps with 30m resolution, historical data visualization, and animated fire
                spread simulation for Uttarakhand region
              </CardDescription>
            </CardHeader>
          </Card>

          <Card className="border-red-200 hover:shadow-lg transition-shadow">
            <CardHeader>
              <Upload className="h-10 w-10 text-red-600 mb-2" />
              <CardTitle>Custom Data Upload</CardTitle>
              <CardDescription>
                Upload GeoTIFF or GeoJSON files to get personalized fire risk predictions for your area of interest
              </CardDescription>
            </CardHeader>
          </Card>

          <Card className="border-purple-200 hover:shadow-lg transition-shadow">
            <CardHeader>
              <Layers3 className="h-10 w-10 text-purple-600 mb-2" />
              <CardTitle>2D to 3D Conversion</CardTitle>
              <CardDescription>
                Revolutionary feature to convert 2D terrain maps into interactive 3D fire simulation models with
                real-time rendering
              </CardDescription>
            </CardHeader>
          </Card>

          <Card className="border-yellow-200 hover:shadow-lg transition-shadow">
            <CardHeader>
              <Brain className="h-10 w-10 text-yellow-600 mb-2" />
              <CardTitle>AI/ML Models</CardTitle>
              <CardDescription>
                Advanced U-Net segmentation, LSTM time-series prediction, and Cellular Automata for accurate fire spread
                modeling
              </CardDescription>
            </CardHeader>
          </Card>

          <Card className="border-green-200 hover:shadow-lg transition-shadow">
            <CardHeader>
              <Box className="h-10 w-10 text-green-600 mb-2" />
              <CardTitle>3D Visualization</CardTitle>
              <CardDescription>
                Immersive 3D terrain rendering with animated fire spread overlays, forest ecosystems, and interactive
                camera controls
              </CardDescription>
            </CardHeader>
          </Card>

          <Card className="border-blue-200 hover:shadow-lg transition-shadow">
            <CardHeader>
              <BarChart3 className="h-10 w-10 text-blue-600 mb-2" />
              <CardTitle>Advanced Analytics</CardTitle>
              <CardDescription>
                Comprehensive statistical analysis, risk zone mapping, and detailed performance metrics visualization
              </CardDescription>
            </CardHeader>
          </Card>
        </div>
      </section>

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
