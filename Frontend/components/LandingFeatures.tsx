import { Card, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Map, Upload, Brain, BarChart3, Box, Layers3 } from "lucide-react"



const LandingFeatures = () => {
    return (
        <div className="bg-gradient-to-r from-orange-500 to-red-600 hover:from-orange-600 hover:to-red-700 h-[70vh]">
            <section className="container mx-auto px-4 py-16 ">
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
                            <CardTitle>Simulation </CardTitle>
                            <CardDescription>
                                Used Cellular Automata to Simulate the Forest Fire on Map For better understanding.
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

        </div>
    )
}

export default LandingFeatures
