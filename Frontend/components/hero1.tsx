"use client"

import { useEffect, useRef } from "react"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Map, Layers3 } from "lucide-react"
import BlurText from "./Blurtext"

const Hero1 = () => {
    const videoRef = useRef<HTMLVideoElement>(null)

    useEffect(() => {
        // Set video playback speed to 0.5x
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

            {/* Overlay for better text contrast */}
            <div className="absolute inset-0 bg-black/40 z-10" />

            {/* Foreground Content */}
            <section className="relative z-20 container mx-auto px-4 py-16 text-center text-white">
                <div className="max-w-6xl mx-auto ">
                    <BlurText
                        text="Forest Fire Prediction System"
                        delay={150}
                        animateBy="words"
                        direction="top"
                        className="text-5xl md:text-7xl font-bold bg-gradient-to-r from-orange-400 to-red-500 bg-clip-text mb-6 pl-14"
                    />
                    <h3 className="text-2xl md:text-3xl font-semibold mb-4">
                        Harnessing AI to Predict, Prevent, and Visualize Forest Fires — Turning Data into Actionable Insights for a Safer Tomorrow.
                    </h3>

                    <p className="text-lg text-gray-200 mb-8 leading-relaxed">
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

                        <Button asChild variant="outline" size="lg" className="text-white border-white hover:bg-white/10">
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
