"use client"

import React, { useRef, useMemo } from "react"
import { Canvas, useFrame } from "@react-three/fiber"
import * as THREE from "three"

const PARTICLE_COUNT = 600

function EmberParticles() {
    const meshRef = useRef<THREE.Points>(null!)

    const { positions, colors, sizes, speeds } = useMemo(() => {
        const positions = new Float32Array(PARTICLE_COUNT * 3)
        const colors = new Float32Array(PARTICLE_COUNT * 3)
        const sizes = new Float32Array(PARTICLE_COUNT)
        const speeds = new Float32Array(PARTICLE_COUNT)

        for (let i = 0; i < PARTICLE_COUNT; i++) {
            // Scatter across a wide area
            positions[i * 3] = (Math.random() - 0.5) * 20   // x
            positions[i * 3 + 1] = (Math.random() - 0.5) * 12   // y
            positions[i * 3 + 2] = (Math.random() - 0.5) * 8    // z

            // Fire palette: orange to deep red
            const t = Math.random()
            colors[i * 3] = 1.0                              // R
            colors[i * 3 + 1] = 0.15 + t * 0.45                  // G (0.15–0.6)
            colors[i * 3 + 2] = t * 0.1                          // B

            sizes[i] = 1.5 + Math.random() * 3.5
            speeds[i] = 0.2 + Math.random() * 0.8
        }
        return { positions, colors, sizes, speeds }
    }, [])

    useFrame((_, delta) => {
        if (!meshRef.current) return
        const posArr = meshRef.current.geometry.attributes.position.array as Float32Array
        for (let i = 0; i < PARTICLE_COUNT; i++) {
            // Drift upward
            posArr[i * 3 + 1] += speeds[i] * delta * 0.8
            // Gentle sway
            posArr[i * 3] += Math.sin(Date.now() * 0.001 + i) * delta * 0.04

            // Reset when too high
            if (posArr[i * 3 + 1] > 7) {
                posArr[i * 3 + 1] = -6
                posArr[i * 3] = (Math.random() - 0.5) * 20
                posArr[i * 3 + 2] = (Math.random() - 0.5) * 8
            }
        }
        meshRef.current.geometry.attributes.position.needsUpdate = true
    })

    return (
        <points ref={meshRef}>
            <bufferGeometry>
                <bufferAttribute
                    attach="attributes-position"
                    args={[positions, 3]}
                />
                <bufferAttribute
                    attach="attributes-color"
                    args={[colors, 3]}
                />
                <bufferAttribute
                    attach="attributes-size"
                    args={[sizes, 1]}
                />
            </bufferGeometry>
            <pointsMaterial
                size={0.06}
                vertexColors
                transparent
                opacity={0.7}
                depthWrite={false}
                blending={THREE.AdditiveBlending}
                sizeAttenuation
            />
        </points>
    )
}

export default function FireParticles3D() {
    return (
        <div className="absolute inset-0 z-0 pointer-events-none" style={{ opacity: 0.5 }}>
            <Canvas
                camera={{ position: [0, 0, 6], fov: 60 }}
                gl={{ alpha: true, antialias: false }}
                style={{ background: "transparent" }}
            >
                <ambientLight intensity={0.2} />
                <EmberParticles />
            </Canvas>
        </div>
    )
}
