"use client"

import { useState, useEffect } from "react"
import { motion, AnimatePresence } from "framer-motion"

interface SplashScreenProps {
  onComplete: () => void
}

interface Particle {
  left: string
  top: string
  duration: number
  delay: number
}

export default function SplashScreen({ onComplete }: SplashScreenProps) {
  const [isVisible, setIsVisible] = useState(true)
  const [currentPhase, setCurrentPhase] = useState(0)
  const [particles, setParticles] = useState<Particle[]>([])

  useEffect(() => {
    // Generate random particle positions only on the client
    const generatedParticles: Particle[] = Array.from({ length: 20 }).map(() => ({
      left: `${Math.random() * 100}%`,
      top: `${Math.random() * 100}%`,
      duration: Math.random() * 3 + 2,
      delay: Math.random() * 2
    }))
    setParticles(generatedParticles)
  }, [])

  useEffect(() => {
    // Phase progression: 0 -> 1 -> 2 -> exit
    const phaseTimer1 = setTimeout(() => setCurrentPhase(1), 1000)
    const phaseTimer2 = setTimeout(() => setCurrentPhase(2), 2000)
    
    const timer = setTimeout(() => {
      setIsVisible(false)
      setTimeout(onComplete, 800) // Wait for exit animation
    }, 4000)

    return () => {
      clearTimeout(phaseTimer1)
      clearTimeout(phaseTimer2)
      clearTimeout(timer)
    }
  }, [onComplete])

  return (
    <AnimatePresence>
      {isVisible && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0, scale: 0.95 }}
          transition={{ duration: 0.8, ease: "easeInOut" }}
          className="fixed inset-0 z-50 flex items-center justify-center overflow-hidden"
        >
          {/* Dynamic gradient background */}
          <motion.div 
            className="absolute inset-0"
            animate={{
              background: [
                "linear-gradient(135deg, #1e3a8a 0%, #1e40af 25%, #0891b2 50%, #0891b2 75%, #06b6d4 100%)",
                "linear-gradient(135deg, #0c4a6e 0%, #1e40af 25%, #0369a1 50%, #0891b2 75%, #22d3ee 100%)",
                "linear-gradient(135deg, #1e3a8a 0%, #1e40af 25%, #0891b2 50%, #0891b2 75%, #06b6d4 100%)"
              ]
            }}
            transition={{ duration: 4, ease: "linear" }}
          />

          {/* Floating particles */}
          <div className="absolute inset-0 pointer-events-none">
            {particles.map((p, i) => (
              <motion.div
                key={i}
                className="absolute w-1 h-1 bg-white/30 rounded-full"
                style={{ left: p.left, top: p.top }}
                animate={{
                  y: [-20, -100, -20],
                  opacity: [0, 1, 0],
                  scale: [0, 1, 0]
                }}
                transition={{
                  duration: p.duration,
                  repeat: Infinity,
                  delay: p.delay,
                  ease: "easeInOut"
                }}
              />
            ))}
          </div>

          {/* Main content container */}
          <div className="relative z-10 text-center text-white px-8">
            {/* Logo/Icon container */}
            <motion.div
              initial={{ scale: 0, rotate: -180 }}
              animate={{
                scale: currentPhase >= 0 ? 1 : 0,
                rotate: 0
              }}
              transition={{
                duration: 1.2,
                ease: "easeOut",
                type: "spring",
                stiffness: 100
              }}
              className="mb-12"
            >
              <div className="relative">
                {/* Glow effect */}
                <motion.div
                  className="absolute inset-0 bg-cyan-400/20 rounded-full blur-xl"
                  animate={{
                    scale: [1, 1.2, 1],
                    opacity: [0.3, 0.6, 0.3]
                  }}
                  transition={{ duration: 2, repeat: Infinity }}
                />

                {/* Main icon container */}
                <div className="relative w-28 h-28 mx-auto bg-white/10 rounded-full flex items-center justify-center backdrop-blur-lg border border-white/20 shadow-2xl">
                  {/* Ocean wave icon */}
                  <motion.svg
                    className="w-14 h-14 text-white"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                    initial={{ pathLength: 0 }}
                    animate={{ pathLength: 1 }}
                    transition={{ duration: 2, ease: "easeInOut" }}
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={1.5}
                      d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"
                    />
                  </motion.svg>
                </div>
              </div>
            </motion.div>

            {/* Title */}
            <motion.div
              initial={{ y: 50, opacity: 0 }}
              animate={{
                y: currentPhase >= 1 ? 0 : 50,
                opacity: currentPhase >= 1 ? 1 : 0
              }}
              transition={{ duration: 0.8, ease: "easeOut" }}
            >
              <h1 className="text-5xl md:text-7xl font-bold mb-6 relative">
                <motion.span
                  className="bg-gradient-to-r from-white via-cyan-200 to-blue-200 bg-clip-text text-transparent"
                  animate={{
                    backgroundPosition: ["0% 50%", "100% 50%", "0% 50%"]
                  }}
                  transition={{ duration: 3, repeat: Infinity }}
                  style={{ backgroundSize: "200% 200%" }}
                >
                  FloatChat
                </motion.span>

                {/* Underline effect */}
                <motion.div
                  className="absolute -bottom-2 left-1/2 transform -translate-x-1/2 h-1 bg-gradient-to-r from-transparent via-cyan-400 to-transparent"
                  initial={{ width: 0 }}
                  animate={{ width: currentPhase >= 1 ? "100%" : 0 }}
                  transition={{ duration: 1, delay: 0.5 }}
                />
              </h1>

              <motion.p
                initial={{ y: 30, opacity: 0 }}
                animate={{
                  y: currentPhase >= 2 ? 0 : 30,
                  opacity: currentPhase >= 2 ? 1 : 0
                }}
                transition={{ duration: 0.6, delay: 0.2 }}
                className="text-xl md:text-2xl text-cyan-100/90 font-light mb-8 tracking-wide"
              >
                AI-Powered Ocean Data Interface
              </motion.p>
            </motion.div>

            {/* Enhanced loading animation */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{
                opacity: currentPhase >= 2 ? 1 : 0,
                y: currentPhase >= 2 ? 0 : 20
              }}
              transition={{ duration: 0.5, delay: 0.4 }}
              className="flex justify-center items-center space-x-3"
            >
              {/* Wave-like loading dots */}
              {[0, 1, 2, 3, 4].map((i) => (
                <motion.div
                  key={i}
                  className="w-2 h-2 bg-white/80 rounded-full"
                  animate={{
                    scale: [1, 1.5, 1],
                    opacity: [0.4, 1, 0.4],
                    y: [0, -8, 0]
                  }}
                  transition={{
                    duration: 1.2,
                    repeat: Infinity,
                    delay: i * 0.15,
                    ease: "easeInOut"
                  }}
                />
              ))}
            </motion.div>

            {/* Progress indicator */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: currentPhase >= 2 ? 1 : 0 }}
              transition={{ duration: 0.5, delay: 0.6 }}
              className="mt-8"
            >
              <div className="w-48 h-1 mx-auto bg-white/20 rounded-full overflow-hidden">
                <motion.div
                  className="h-full bg-gradient-to-r from-cyan-400 to-blue-400 rounded-full"
                  initial={{ width: 0 }}
                  animate={{ width: "100%" }}
                  transition={{ duration: 3, ease: "easeInOut" }}
                />
              </div>
              <motion.p
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.5, delay: 1 }}
                className="text-cyan-200/60 text-sm mt-3 font-medium"
              >
                Diving into the depths...
              </motion.p>
            </motion.div>
          </div>

          {/* Corner decorations */}
          <div className="absolute top-8 right-8 opacity-20">
            <motion.div
              animate={{ rotate: 360 }}
              transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
              className="w-16 h-16 border-2 border-white/30 rounded-full"
            />
          </div>
          <div className="absolute bottom-8 left-8 opacity-20">
            <motion.div
              animate={{ rotate: -360 }}
              transition={{ duration: 25, repeat: Infinity, ease: "linear" }}
              className="w-12 h-12 border border-white/20 rounded-full"
            />
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}