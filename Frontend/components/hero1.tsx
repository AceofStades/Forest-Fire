"use client"

import { useEffect, useRef } from "react"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Map, Layers3 } from "lucide-react"
import BlurText from "./Blurtext"
import dynamic from "next/dynamic"

const FireParticles3D = dynamic(() => import("./FireParticles3D"), { ssr: false })

const Hero1 = () => {
    const videoRef = useRef<HTMLVideoElement>(null)

    useEffect(() => {
        if (videoRef.current) {
            videoRef.current.playbackRate = 0.5
        }
    }, [])

    return (
        <div className="relative min-h-[100vh] flex items-center justify-center overflow-hidden">
            {/* Background Video */}
            <video
                ref={videoRef}
                autoPlay
                loop
                muted
                playsInline
                className="absolute inset-0 w-full h-full object-cover z-0"
            >
                <source src="/videos/hero1.mp4" type="video/mp4" />
                Your browser does not support the video tag.
            </video>

            {/* Dark Overlay */}
            <div className="absolute inset-0 bg-black/65 z-10" />

            {/* 3D Ember Particles */}
            <div className="absolute inset-0 z-[15]">
                <FireParticles3D />
            </div>

            {/* Bottom Fade to blend into next section */}
            <div className="absolute bottom-0 left-0 right-0 h-32 bg-gradient-to-t from-[#060609] to-transparent z-[16]" />

            {/* Foreground Content */}
            <section className="relative z-20 container mx-auto px-4 py-16 text-center text-white">
                <div className="max-w-6xl mx-auto">
                    <BlurText
                        text="Forest Fire Prediction System"
                        delay={150}
                        animateBy="words"
                        direction="top"
                        className="text-5xl md:text-7xl font-bold bg-gradient-to-r from-orange-400 via-red-400 to-amber-300 bg-clip-text  mb-6 pl-14"
                    />
                    
                    <h3 className="text-xl md:text-2xl font-medium mb-4 text-gray-200/90 max-w-4xl mx-auto leading-relaxed">
                        Harnessing AI to Predict, Prevent, and Visualize Forest Fires — Turning Data into Actionable Insights for a Safer Tomorrow.
                    </h3>

                    <p className="text-base text-gray-400 mb-10 leading-relaxed max-w-3xl mx-auto">
                        Advanced AI/ML-powered platform for simulating and modeling forest fire spread. Get next-day fire
                        probability maps, real-time spread animations, 2D terrain, and custom predictions using
                        cutting-edge U-Net, LSTM, and Cellular Automata models.
                    </p>
                    

                    <div className="flex flex-col sm:flex-row gap-4 justify-center">
                        <Button
                            asChild
                            size="lg"
                            className="bg-gradient-to-r from-orange-500 to-red-600 hover:from-orange-400 hover:to-red-500 text-white text-base px-8 py-3 shadow-lg shadow-orange-600/30 hover:shadow-orange-500/50 border-0"
                        >
                            <Link href="/dashboard">
                                <Map className="mr-2 h-5 w-5" />
                                Explore Dashboard
                            </Link>
                        </Button>

                        <Button
                            asChild
                            variant="outline"
                            size="lg"
                            className="border-orange-500/40 text-orange-200 hover:bg-orange-500/10 hover:border-orange-400/60 text-base px-8 py-3"
                        >
                            <Link href="/simulation">
                                <Layers3 className="mr-2 h-5 w-5" />
                                Simulation
                            </Link>
                        </Button>
                    </div>
                </div>
            </section>
        </div>
    )
}

export default Hero1
