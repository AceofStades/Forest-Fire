"use client"

import { useRef, useState } from "react"
import { Html } from "@react-three/drei"
import type * as THREE from "three"

interface FloatData {
  Latitude: string
  Longitude: string
  Location: string
}

export function EnhancedFloatMarker({
  float,
  position,
  onClick,
}: {
  float: FloatData
  position: [number, number, number]
  onClick: () => void
}) {
  const markerRef = useRef<THREE.Group>(null)
  const [isHovered, setIsHovered] = useState(false)

  return (
    <group position={position} ref={markerRef}>
      {/* Red Pin (sphere + cone) */}
      <group
        onClick={(e) => {
          e.stopPropagation()
          onClick()
        }}
        onPointerOver={(e) => {
          e.stopPropagation()
          setIsHovered(true)
          document.body.style.cursor = "pointer"
        }}
        onPointerOut={() => {
          setIsHovered(false)
          document.body.style.cursor = "default"
        }}
      >
        {/* Pin head (sphere) */}
        <mesh position={[0, 0.12, 0]}>
          <sphereGeometry args={[0.06, 16, 16]} />
          <meshStandardMaterial
            color={isHovered ? "#ff0000" : "#cc0000"}
            emissive={isHovered ? "#ff0000" : "#550000"}
            emissiveIntensity={isHovered ? 1.0 : 0.5}
          />
        </mesh>

        {/* Pin body (cone) */}
        <mesh rotation={[Math.PI, 0, 0]} position={[0, -0.05, 0]}>
          <coneGeometry args={[0.03, 0.12, 16]} />
          <meshStandardMaterial
            color={isHovered ? "#ff0000" : "#cc0000"}
            emissive={isHovered ? "#ff0000" : "#550000"}
            emissiveIntensity={isHovered ? 1.0 : 0.5}
          />
        </mesh>
      </group>

      {/* Tooltip */}
      {isHovered && (
        <Html distanceFactor={12} center>
          <div className="bg-black/80 text-white px-2 py-1 rounded text-xs whitespace-nowrap pointer-events-none">
            {float.Location}
          </div>
        </Html>
      )}
    </group>
  )
}
