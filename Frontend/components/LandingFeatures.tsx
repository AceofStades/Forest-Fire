import { Card, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Map, Upload, Brain, BarChart3, Box, Layers3 } from "lucide-react"

const features = [
    {
        icon: Map,
        title: "Interactive Map Dashboard",
        description: "Real-time fire probability maps with 30m resolution, historical data visualization, and animated fire spread simulation for Uttarakhand region.",
        accent: "from-orange-500 to-amber-400",
        iconColor: "text-orange-400",
    },
    {
        icon: Upload,
        title: "Custom Data Upload",
        description: "Upload GeoTIFF or GeoJSON files to get personalized fire risk predictions for your area of interest.",
        accent: "from-red-500 to-orange-400",
        iconColor: "text-red-400",
    },
    {
        icon: Layers3,
        title: "Simulation",
        description: "Used Cellular Automata to Simulate the Forest Fire on Map for better understanding and planning.",
        accent: "from-purple-500 to-pink-400",
        iconColor: "text-purple-400",
    },
    {
        icon: Brain,
        title: "AI/ML Models",
        description: "Advanced U-Net segmentation, LSTM time-series prediction, and Cellular Automata for accurate fire spread modeling.",
        accent: "from-amber-500 to-yellow-400",
        iconColor: "text-amber-400",
    },
    {
        icon: Box,
        title: "3D Visualization",
        description: "Immersive 3D terrain rendering with animated fire spread overlays, forest ecosystems, and interactive camera controls.",
        accent: "from-emerald-500 to-teal-400",
        iconColor: "text-emerald-400",
    },
    {
        icon: BarChart3,
        title: "Advanced Analytics",
        description: "Comprehensive statistical analysis, risk zone mapping, and detailed performance metrics visualization.",
        accent: "from-cyan-500 to-blue-400",
        iconColor: "text-cyan-400",
    },
]

const LandingFeatures = () => {
    return (
        <section className="relative py-24 overflow-hidden">
            {/* Subtle dark background with radial glow */}
            <div className="absolute inset-0 bg-gradient-to-b from-[#060609] via-[#0a0a12] to-[#060609]" />
            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-orange-600/[0.03] rounded-full blur-[120px]" />

            <div className="relative container mx-auto px-4">
                <h2 className="text-3xl md:text-4xl font-bold text-center mb-4 bg-gradient-to-r from-white to-gray-400 bg-clip-text text-transparent">
                    Key Features
                </h2>
                <p className="text-gray-500 text-center mb-14 max-w-xl mx-auto">
                    Powered by cutting-edge AI and geospatial technology for comprehensive fire prediction.
                </p>
                <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
                    {features.map((f) => {
                        const Icon = f.icon
                        return (
                            <Card
                                key={f.title}
                                className="group glass-card fire-glow-card border-white/[0.06] hover:border-orange-500/20 bg-white/[0.02] cursor-default"
                            >
                                <CardHeader>
                                    <div className={`w-12 h-12 rounded-xl bg-gradient-to-br ${f.accent} flex items-center justify-center mb-4 shadow-lg group-hover:scale-110 transition-transform duration-300`}>
                                        <Icon className="h-6 w-6 text-white" />
                                    </div>
                                    <CardTitle className="text-white text-lg">{f.title}</CardTitle>
                                    <CardDescription className="text-gray-400 leading-relaxed">
                                        {f.description}
                                    </CardDescription>
                                </CardHeader>
                            </Card>
                        )
                    })}
                </div>
            </div>
        </section>
    )
}

export default LandingFeatures
