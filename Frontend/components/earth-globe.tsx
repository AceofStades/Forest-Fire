"use client"

import { useRef, useMemo } from "react"
import { Canvas, useFrame, useLoader } from "@react-three/fiber"
import { OrbitControls, Sphere, Environment, Stars } from "@react-three/drei"
import * as THREE from "three"

function Earth() {
  const meshRef = useRef<THREE.Mesh>(null)
  const cloudsRef = useRef<THREE.Mesh>(null)

  const [earthTexture, normalTexture, cloudsTexture] = useLoader(THREE.TextureLoader, [
    "/textures/earth_daymap_enhanced.jpg",
    "/textures/earth_normal.jpg",
    "/textures/earth_clouds.jpg",
  ])

  const earthMaterial = useMemo(() => {
    return new THREE.MeshPhongMaterial({
      map: earthTexture,
      normalMap: normalTexture,
      normalScale: new THREE.Vector2(0.3, 0.3),
      shininess: 200,
      specular: new THREE.Color(0x222222),
      transparent: false,
    })
  }, [earthTexture, normalTexture])

  const cloudsMaterial = useMemo(() => {
    return new THREE.MeshLambertMaterial({
      map: cloudsTexture,
      transparent: true,
      opacity: 0.4,
      alphaTest: 0.1,
      side: THREE.DoubleSide,
    })
  }, [cloudsTexture])

  useFrame((state, delta) => {
    if (meshRef.current) {
      meshRef.current.rotation.y += delta * 0.03
    }
    if (cloudsRef.current) {
      cloudsRef.current.rotation.y += delta * 0.05
    }
  })

  return (
    <group>
      <Sphere ref={meshRef} args={[2, 256, 256]} material={earthMaterial} />

      <Sphere ref={cloudsRef} args={[2.005, 128, 128]} material={cloudsMaterial} />
    </group>
  )
}

export function EarthGlobe() {
  return (
    <div className="w-full h-[500px] rounded-lg overflow-hidden bg-gradient-to-b from-indigo-900 via-blue-900 to-black">
      <Canvas camera={{ position: [0, 0, 5], fov: 45 }}>
        <ambientLight intensity={0.15} />
        <directionalLight
          position={[5, 3, 5]}
          intensity={2}
          castShadow
          shadow-mapSize-width={4096}
          shadow-mapSize-height={4096}
          shadow-camera-near={0.1}
          shadow-camera-far={50}
          shadow-camera-left={-10}
          shadow-camera-right={10}
          shadow-camera-top={10}
          shadow-camera-bottom={-10}
        />
        <pointLight position={[-10, -10, -5]} intensity={0.3} color="#1e40af" />

        <Stars radius={300} depth={60} count={25000} factor={8} saturation={0} fade speed={0.5} />

        <Earth />
        <OrbitControls
          enablePan={false}
          enableZoom={true}
          enableRotate={true}
          minDistance={3}
          maxDistance={8}
          autoRotate={true}
          autoRotateSpeed={0.3}
          dampingFactor={0.05}
          enableDamping={true}
        />

        <Environment preset="night" />
      </Canvas>
    </div>
  )
}
