"use client"

import { useTexture } from "@react-three/drei"
import { useFrame } from "@react-three/fiber"
import { useRef } from "react"
import type * as THREE from "three"

export function EnhancedEarthGlobe({
  isHovered,
  setHovered,
}: {
  isHovered: boolean
  setHovered: (h: boolean) => void
}) {
  const earthRef = useRef<THREE.Mesh>(null)

  // const earthTexture = useTexture("/assets/3d/texture_earth.jpg")
  const earthTexture = useTexture("/textures/earth_daymap_2k.jpg")


  // Rotate Earth slowly
  useFrame(() => {
    if (earthRef.current) {
      earthRef.current.rotation.y += 0.002
    }
  })

  return (
    <mesh
      ref={earthRef}
      scale={isHovered ? 2.2 : 2}
      onPointerOver={() => setHovered(true)}
      onPointerOut={() => setHovered(false)}
    >
      {/* geometry */}
      <sphereGeometry args={[1, 64, 64]} />

      <meshStandardMaterial map={earthTexture} roughness={0.8} metalness={0.1} bumpScale={0.02} />
    </mesh>
  )
}
