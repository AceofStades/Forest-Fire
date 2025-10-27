import { Zap, Globe, Layers3 } from 'lucide-react'
import React from 'react'

const Techstack = () => {
    return (
        <div>
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
        </div>
    )
}

export default Techstack
