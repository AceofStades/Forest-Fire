import { Zap, Globe, Layers3 } from "lucide-react"

const stacks = [
    {
        icon: Zap,
        title: "Machine Learning",
        desc: "U-Net, LSTM, and Cellular Automata models for accurate predictions",
        accent: "from-orange-500 to-amber-400",
    },
    {
        icon: Globe,
        title: "Geospatial Analysis",
        desc: "Advanced GIS processing with Leaflet.js and geospatial data handling",
        accent: "from-cyan-500 to-blue-400",
    },
    {
        icon: Layers3,
        title: "3D Innovation",
        desc: "Three.js powered 3D terrain conversion and immersive fire simulation",
        accent: "from-emerald-500 to-lime-400",
    },
]

const Techstack = () => {
    return (
        <section className="relative py-24">
            <div className="absolute inset-0 bg-gradient-to-b from-[#060609] via-[#08080e] to-[#060609]" />
            <div className="relative container mx-auto px-4">
                <h2 className="text-3xl md:text-4xl font-bold text-center mb-4 bg-gradient-to-r from-white to-gray-400 bg-clip-text text-transparent">
                    Powered by Advanced Technology
                </h2>
                <p className="text-gray-500 text-center mb-14 max-w-lg mx-auto">
                    Industry-leading frameworks and state-of-the-art models.
                </p>
                <div className="grid md:grid-cols-3 gap-8 text-center">
                    {stacks.map((s) => {
                        const Icon = s.icon
                        return (
                            <div key={s.title} className="group glass-card fire-glow-card p-8 flex flex-col items-center">
                                <div className={`w-16 h-16 rounded-2xl bg-gradient-to-br ${s.accent} flex items-center justify-center mb-5 shadow-lg group-hover:scale-110 transition-transform duration-300`}>
                                    <Icon className="h-8 w-8 text-white" />
                                </div>
                                <h3 className="text-xl font-semibold mb-2 text-white">{s.title}</h3>
                                <p className="text-gray-400 leading-relaxed">{s.desc}</p>
                            </div>
                        )
                    })}
                </div>
            </div>
        </section>
    )
}

export default Techstack
