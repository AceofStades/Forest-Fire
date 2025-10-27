import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Navigation } from "@/components/navigation"
// import { Zap, Globe, Layers3 } from "lucide-react"
import Hero1 from "@/components/hero1"
import LandingFeatures from "@/components/LandingFeatures"
import axios from "axios";
import Techstack from "@/components/techstack"
// import CTA from "@/components/CTA"

export default function HomePage() {



  const fetchPrediction = async () => {
    try {
      const response = await axios.post("http://localhost:8000/predict", {
        temperature: 25,
        humidity: 70,
        wind_speed: 5,
      });
      console.log(response.data);
    } catch (error) {
      console.error("Error fetching prediction:", error);
    }
  };



  return (
    <div className="min-h-screen bg-gradient-to-br from-orange-50 via-red-50 to-yellow-50 dark:from-gray-900 dark:via-gray-800 dark:to-gray-900">
      <Navigation />
      <Hero1 />
      <LandingFeatures />
      <Techstack />


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
