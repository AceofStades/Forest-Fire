import { Card, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Map, Upload, Brain, BarChart3, Box, Layers3 } from "lucide-react"

const LandingFeatures = () => {
    return (
        <section className="bg-[#050505] py-24 border-b border-white/5">
            <div className="container mx-auto px-6">
                <h2 className="text-xs font-bold text-slate-500 uppercase tracking-[0.4em] mb-16 text-center">Core Capabilities</h2>
                <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-1">
                    {[
                        { icon: Map, title: "Interactive Dashboard", desc: "30m resolution probability maps and historical data visualization for the Uttarakhand region.", border: "border-white/5" },
                        { icon: Upload, title: "Custom Data", desc: "Upload GeoTIFF or GeoJSON files for localized, personalized fire risk predictions.", border: "border-white/5" },
                        { icon: Layers3, title: "Cellular Simulation", desc: "Advanced fire spread simulations powered by Cellular Automata models.", border: "border-white/5" },
                        { icon: Brain, title: "AI/ML Modeling", desc: "U-Net segmentation and LSTM time-series prediction for accurate threat assessment.", border: "border-white/5" },
                        { icon: Box, title: "Safe Path Navigation", desc: "Advanced AI-powered platform for next-day fire probability mapping and safe path navigation to safeguard forest ecosystems.", border: "border-white/5" },
                        { icon: BarChart3, title: "Advanced Analytics", desc: "Comprehensive statistical analysis and performance metrics visualization for risk monitoring.", border: "border-white/5" },
                    ].map((feature, i) => (
                        <Card key={i} className="bg-transparent border border-white/5 rounded-none hover:bg-white/[0.02] transition-colors shadow-none">
                            <CardHeader className="p-10">
                                <feature.icon className="h-8 w-8 text-orange-700 mb-6" />
                                <CardTitle className="text-white font-bold tracking-tight mb-4">{feature.title}</CardTitle>
                                <CardDescription className="text-slate-500 text-sm leading-relaxed">{feature.desc}</CardDescription>
                            </CardHeader>
                        </Card>
                    ))}
                </div>
            </div>
        </section>
    )
}

export default LandingFeatures