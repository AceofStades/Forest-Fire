"use client"

import { Suspense, useRef, useState, useMemo, useEffect } from "react"
import { Canvas, useFrame } from "@react-three/fiber"
import { OrbitControls, Environment, Html } from "@react-three/drei"
import { Navigation } from "@/components/navigation"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Slider } from "@/components/ui/slider"
import { Switch } from "@/components/ui/switch"
import { Badge } from "@/components/ui/badge"
import { Play, Pause, RotateCcw, Camera, Maximize, Settings, Smartphone } from "lucide-react"
import type * as THREE from "three"

// AR Viewer Component for 3D Visualization
function ARViewer3D({ onClose }: { onClose: () => void }) {
  const [isARSupported, setIsARSupported] = useState(false)
  const [isARActive, setIsARActive] = useState(false)

  useEffect(() => {
    // Check if WebXR is supported
    if (navigator.xr) {
      navigator.xr.isSessionSupported("immersive-ar").then((supported) => {
        setIsARSupported(supported)
      })
    }
  }, [])

  const startAR = async () => {
    if (!navigator.xr || !isARSupported) {
      alert("AR is not supported on this device. Please use a compatible mobile device with AR capabilities.")
      return
    }

    try {
      setIsARActive(true)
      // In a real implementation, this would start the AR session
      setTimeout(() => {
        alert(
          "AR session would start here. You would see the 3D forest fire simulation overlaid on your real environment.",
        )
        setIsARActive(false)
      }, 2000)
    } catch (error) {
      console.error("Failed to start AR session:", error)
      setIsARActive(false)
    }
  }

  return (
    <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50">
      <div className="bg-white dark:bg-gray-800 p-6 rounded-lg max-w-md w-full mx-4">
        <h3 className="text-lg font-semibold mb-4">AR Forest Fire Visualization</h3>

        {!isARSupported ? (
          <div className="space-y-4">
            <div className="bg-orange-50 dark:bg-orange-900/20 p-4 rounded-lg">
              <div className="flex items-center space-x-2 mb-2">
                <Smartphone className="h-5 w-5 text-orange-600" />
                <span className="font-medium text-orange-800 dark:text-orange-200">AR Not Available</span>
              </div>
              <p className="text-sm text-orange-700 dark:text-orange-300">
                AR viewing requires a compatible mobile device with WebXR support.
              </p>
            </div>
            <div className="text-sm text-muted-foreground space-y-2">
              <p>
                <strong>Supported devices:</strong>
              </p>
              <ul className="list-disc list-inside space-y-1">
                <li>Android devices with ARCore</li>
                <li>iOS devices with ARKit (iOS 12+)</li>
                <li>Chrome or Safari browser</li>
              </ul>
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg">
              <p className="text-sm text-green-700 dark:text-green-300">
                🎉 Your device supports AR! The 3D forest fire simulation will appear in your real environment.
              </p>
            </div>
            <p className="text-sm text-muted-foreground">
              Make sure you're in a well-lit area with enough space to move around safely.
            </p>
            <Button onClick={startAR} disabled={isARActive} className="w-full">
              {isARActive ? (
                <>Starting AR Session...</>
              ) : (
                <>
                  <Smartphone className="mr-2 h-4 w-4" />
                  Start AR Experience
                </>
              )}
            </Button>
          </div>
        )}

        <div className="flex space-x-2 mt-4">
          <Button variant="outline" onClick={onClose} className="flex-1 bg-transparent">
            Close
          </Button>
        </div>
      </div>
    </div>
  )
}

// 3D Forest Component
function Forest({ fireIntensity }: { fireIntensity: number }) {
  const groupRef = useRef<THREE.Group>(null)

  const trees = useMemo(() => {
    const treePositions = []
    for (let i = 0; i < 50; i++) {
      treePositions.push({
        x: (Math.random() - 0.5) * 18,
        z: (Math.random() - 0.5) * 18,
        height: 1 + Math.random() * 2,
        burning: Math.random() < fireIntensity * 0.3,
      })
    }
    return treePositions
  }, [])

  useFrame((state) => {
    if (groupRef.current) {
      groupRef.current.children.forEach((tree, index) => {
        if (trees[index]?.burning) {
          tree.rotation.z = Math.sin(state.clock.elapsedTime * 2 + index) * 0.1
        }
      })
    }
  })

  return (
    <group ref={groupRef}>
      {trees.map((tree, index) => (
        <group key={index} position={[tree.x, -2 + tree.height / 2, tree.z]}>
          {/* Tree trunk */}
          <mesh position={[0, -tree.height / 2, 0]}>
            <cylinderGeometry args={[0.1, 0.15, tree.height * 0.6]} />
            <meshStandardMaterial color="#8B4513" />
          </mesh>
          {/* Tree foliage */}
          <mesh position={[0, tree.height * 0.2, 0]}>
            <coneGeometry args={[0.5, tree.height * 0.8]} />
            <meshStandardMaterial
              color={tree.burning ? "#ff4500" : "#228B22"}
              emissive={tree.burning ? "#ff2200" : "#000000"}
              emissiveIntensity={tree.burning ? 0.3 : 0}
            />
          </mesh>
        </group>
      ))}
    </group>
  )
}

// 3D Terrain Component
function Terrain({ fireIntensity = 0, wireframe = false }: { fireIntensity?: number; wireframe?: boolean }) {
  const meshRef = useRef<THREE.Mesh>(null)

  useFrame((state) => {
    if (meshRef.current) {
      meshRef.current.rotation.z = Math.sin(state.clock.elapsedTime * 0.1) * 0.01
    }
  })

  return (
    <mesh ref={meshRef} position={[0, -2, 0]} rotation={[-Math.PI / 2, 0, 0]}>
      <planeGeometry args={[20, 20, 50, 50]} />
      <meshStandardMaterial color="#4a5d23" wireframe={wireframe} roughness={0.8} metalness={0.1} />
    </mesh>
  )
}

// Fire Particles Component
function FireParticles({ intensity = 0.5, visible = true }: { intensity?: number; visible?: boolean }) {
  const pointsRef = useRef<THREE.Points>(null)
  const maxParticles = 1000
  const activeParticles = Math.floor(intensity * maxParticles)

  const positions = useMemo(() => {
    const pos = new Float32Array(maxParticles * 3)
    for (let i = 0; i < maxParticles; i++) {
      pos[i * 3] = (Math.random() - 0.5) * 10
      pos[i * 3 + 1] = Math.random() * 5
      pos[i * 3 + 2] = (Math.random() - 0.5) * 10
    }
    return pos
  }, [])

  const colors = useMemo(() => {
    const col = new Float32Array(maxParticles * 3)
    for (let i = 0; i < maxParticles; i++) {
      col[i * 3] = 1
      col[i * 3 + 1] = Math.random() * 0.5
      col[i * 3 + 2] = 0
    }
    return col
  }, [])

  const opacities = useMemo(() => {
    const op = new Float32Array(maxParticles)
    for (let i = 0; i < maxParticles; i++) {
      op[i] = i < activeParticles ? 1.0 : 0.0
    }
    return op
  }, [activeParticles])

  useFrame((state) => {
    if (pointsRef.current && visible) {
      pointsRef.current.rotation.y = state.clock.elapsedTime * 0.1

      const positionAttribute = pointsRef.current.geometry.attributes.position
      const opacityAttribute = pointsRef.current.geometry.attributes.opacity

      for (let i = 0; i < activeParticles; i++) {
        const index = i * 3
        positions[index + 1] += 0.02 * (1 + intensity)
        if (positions[index + 1] > 5) {
          positions[index + 1] = 0
          positions[index] = (Math.random() - 0.5) * 10
          positions[index + 2] = (Math.random() - 0.5) * 10
        }
      }

      for (let i = 0; i < maxParticles; i++) {
        opacities[i] = i < activeParticles ? 0.5 + Math.random() * 0.5 : 0.0
      }

      positionAttribute.needsUpdate = true
      opacityAttribute.needsUpdate = true
    }
  })

  if (!visible) return null

  return (
    <points ref={pointsRef}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[positions, 3]} count={maxParticles} array={positions} itemSize={3} />
        <bufferAttribute attach="attributes-color" args={[colors, 3]} count={maxParticles} array={colors} itemSize={3} />
        <bufferAttribute attach="attributes-opacity" args={[opacities, 1]} count={maxParticles} array={opacities} itemSize={1} />
      </bufferGeometry>
      <pointsMaterial size={0.15} vertexColors transparent opacity={0.8} alphaTest={0.1} />
    </points>
  )
}

// 3D Scene Component
function Scene3D({
  fireIntensity,
  showFire,
  wireframe,
  showGrid,
  ambientLighting,
}: {
  fireIntensity: number
  showFire: boolean
  wireframe: boolean
  showGrid: boolean
  ambientLighting: boolean
}) {
  return (
    <>
      {ambientLighting && <ambientLight intensity={0.4} />}
      <directionalLight position={[10, 10, 5]} intensity={1} />
      <pointLight position={[0, 5, 0]} intensity={fireIntensity * 2} color="#ff4500" />

      {showGrid && <gridHelper args={[20, 20, "#666666", "#444444"]} position={[0, -1.9, 0]} />}

      <Terrain fireIntensity={fireIntensity} wireframe={wireframe} />
      <Forest fireIntensity={fireIntensity} />
      <FireParticles intensity={fireIntensity} visible={showFire} />

      <Html position={[0, 3, 0]} center>
        <div className="bg-black/50 text-white px-3 py-1 rounded text-sm">
          Fire Intensity: {Math.round(fireIntensity * 100)}% | Trees Burning: {Math.round(fireIntensity * 15)}
        </div>
      </Html>
    </>
  )
}

export default function Visualization3DPage() {
  const [isPlaying, setIsPlaying] = useState(false)
  const [fireIntensity, setFireIntensity] = useState([0.3])
  const [showFire, setShowFire] = useState(true)
  const [wireframe, setWireframe] = useState(false)
  const [showGrid, setShowGrid] = useState(true)
  const [ambientLighting, setAmbientLighting] = useState(true)
  const [cameraMode, setCameraMode] = useState("orbit")
  const [showAR, setShowAR] = useState(false)

  // Auto-play animation
  useEffect(() => {
    let interval: NodeJS.Timeout
    if (isPlaying) {
      interval = setInterval(() => {
        setFireIntensity((prev) => {
          const newValue = prev[0] + 0.01
          return newValue >= 1 ? [0] : [newValue]
        })
      }, 100)
    }
    return () => clearInterval(interval)
  }, [isPlaying])

  return (
    <div className="min-h-screen bg-background">
      <Navigation />

      <div className="container mx-auto px-4 py-6">
        <div className="mb-6">
          <h1 className="text-3xl font-bold mb-2">3D Fire Visualization</h1>
          <p className="text-muted-foreground">Immersive 3D terrain rendering with animated fire spread simulation</p>
        </div>

        <div className="grid lg:grid-cols-4 gap-6">
          {/* 3D Canvas */}
          <div className="lg:col-span-3">
            <Card>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle>3D Forest Fire Simulation</CardTitle>
                    <CardDescription>Interactive 3D fire spread with forest terrain</CardDescription>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Badge variant="outline" className="bg-orange-50 text-orange-700 border-orange-200">
                      Real-time Rendering
                    </Badge>
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <div className="h-[600px] w-full bg-gradient-to-b from-sky-200 to-sky-100 rounded-lg overflow-hidden">
                  <Canvas camera={{ position: [10, 8, 10], fov: 60 }}>
                    <Suspense fallback={null}>
                      <Scene3D
                        fireIntensity={fireIntensity[0]}
                        showFire={showFire}
                        wireframe={wireframe}
                        showGrid={showGrid}
                        ambientLighting={ambientLighting}
                      />
                      <OrbitControls
                        enablePan={true}
                        enableZoom={true}
                        enableRotate={true}
                        maxPolarAngle={Math.PI / 2}
                      />
                      <Environment preset="sunset" />
                    </Suspense>
                  </Canvas>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Controls Panel */}
          <div className="space-y-6">
            {/* Animation Controls */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center">
                  <Play className="mr-2 h-5 w-5" />
                  Simulation Controls
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center space-x-2">
                  <Button
                    variant={isPlaying ? "default" : "outline"}
                    size="sm"
                    onClick={() => setIsPlaying(!isPlaying)}
                  >
                    {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => {
                      setFireIntensity([0])
                      setIsPlaying(false)
                    }}
                  >
                    <RotateCcw className="h-4 w-4" />
                  </Button>
                </div>

                <div className="space-y-2">
                  <label className="text-sm font-medium">Fire Intensity: {Math.round(fireIntensity[0] * 100)}%</label>
                  <Slider
                    value={fireIntensity}
                    onValueChange={setFireIntensity}
                    max={1}
                    step={0.01}
                    className="w-full"
                  />
                </div>

                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">Show Fire Particles</span>
                  <Switch checked={showFire} onCheckedChange={setShowFire} />
                </div>
              </CardContent>
            </Card>

            {/* Camera Controls */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center">
                  <Camera className="mr-2 h-5 w-5" />
                  Camera Controls
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 gap-2">
                  <Button
                    variant={cameraMode === "orbit" ? "default" : "outline"}
                    size="sm"
                    onClick={() => setCameraMode("orbit")}
                  >
                    Orbit
                  </Button>
                  <Button
                    variant={cameraMode === "fly" ? "default" : "outline"}
                    size="sm"
                    onClick={() => setCameraMode("fly")}
                  >
                    Fly
                  </Button>
                </div>

                <div className="text-xs text-muted-foreground space-y-1">
                  <p>• Left click + drag: Rotate</p>
                  <p>• Right click + drag: Pan</p>
                  <p>• Scroll: Zoom in/out</p>
                </div>
              </CardContent>
            </Card>

            {/* View Options */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center">
                  <Settings className="mr-2 h-5 w-5" />
                  View Options
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-sm">Wireframe Mode</span>
                    <Switch checked={wireframe} onCheckedChange={setWireframe} />
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm">Show Grid</span>
                    <Switch checked={showGrid} onCheckedChange={setShowGrid} />
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm">Ambient Lighting</span>
                    <Switch checked={ambientLighting} onCheckedChange={setAmbientLighting} />
                  </div>
                </div>

                <div className="space-y-2">
                  <Button variant="outline" size="sm" className="w-full bg-transparent">
                    <Maximize className="mr-2 h-4 w-4" />
                    Fullscreen
                  </Button>
                  <Button variant="outline" size="sm" className="w-full bg-transparent" onClick={() => setShowAR(true)}>
                    <Smartphone className="mr-2 h-4 w-4" />
                    View in AR
                  </Button>
                </div>
              </CardContent>
            </Card>

            {/* Scene Info */}
            <Card>
              <CardHeader>
                <CardTitle>Scene Information</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-muted-foreground">Forest Area</span>
                  <span className="font-medium">20km²</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-muted-foreground">Trees</span>
                  <span className="font-medium">50</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-muted-foreground">Fire Particles</span>
                  <span className="font-medium">{Math.floor(fireIntensity[0] * 1000)}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-muted-foreground">Burning Trees</span>
                  <Badge variant="destructive">{Math.round(fireIntensity[0] * 15)}</Badge>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>

        {/* AR Viewer Modal */}
        {showAR && <ARViewer3D onClose={() => setShowAR(false)} />}
      </div>
    </div>
  )
}
