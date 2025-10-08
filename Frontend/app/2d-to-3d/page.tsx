"use client"

import type React from "react"

import { useState, useRef, Suspense, useEffect } from "react"
import { Canvas, useFrame } from "@react-three/fiber"
import { OrbitControls, Environment, Html } from "@react-three/drei"
import { Navigation } from "@/components/navigation"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Progress } from "@/components/ui/progress"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Slider } from "@/components/ui/slider"
import {
  Upload,
  FileImage,
  Layers3,
  Download,
  RotateCcw,
  Zap,
  CheckCircle,
  AlertCircle,
  Settings,
  Play,
  Pause,
  Smartphone,
  Clock,
  MapPin,
  TrendingUp,
} from "lucide-react"
import { useToast } from "@/hooks/use-toast"
import * as THREE from "three"


// Fire Spread Map Component
function FireSpreadMap({
  hour,
  imageData,
  fireSpots,
}: {
  hour: number
  imageData: string | null
  fireSpots: Array<{ x: number; y: number; intensity: number; id: string }>
}) {
  return (
    <div className="relative w-full h-64 bg-gradient-to-br from-green-200 to-green-400 rounded-lg overflow-hidden border-2 border-gray-200">
      {/* Base terrain map */}
      {imageData ? (
        <img
          src={imageData || "/placeholder.svg"}
          alt={`Terrain Hour ${hour}`}
          className="w-full h-full object-cover opacity-70"
        />
      ) : (
        <div className="w-full h-full bg-gradient-to-br from-green-300 via-yellow-200 to-orange-200" />
      )}

      {/* Fire spots overlay */}
      <div className="absolute inset-0">
        {fireSpots.map((spot) => (
          <div
            key={spot.id}
            className="absolute transform -translate-x-1/2 -translate-y-1/2 animate-pulse"
            style={{
              left: `${spot.x}%`,
              top: `${spot.y}%`,
            }}
          >
            <div
              className={`rounded-full border-2 ${
                spot.intensity > 0.8
                  ? "bg-red-500 border-red-700"
                  : spot.intensity > 0.5
                    ? "bg-orange-500 border-orange-700"
                    : "bg-yellow-500 border-yellow-700"
              }`}
              style={{
                width: `${20 + spot.intensity * 30}px`,
                height: `${20 + spot.intensity * 30}px`,
              }}
            />
            <div
              className="absolute inset-0 rounded-full bg-red-400 opacity-30 animate-ping"
              style={{
                width: `${30 + spot.intensity * 40}px`,
                height: `${30 + spot.intensity * 40}px`,
                left: `${-5 - spot.intensity * 5}px`,
                top: `${-5 - spot.intensity * 5}px`,
              }}
            />
          </div>
        ))}
      </div>

      {/* Hour label */}
      <div className="absolute top-2 left-2 bg-black/70 text-white px-2 py-1 rounded text-sm font-medium">
        Hour-{hour.toString().padStart(2, "0")}
      </div>

      {/* North arrow */}
      <div className="absolute top-2 right-2 bg-white/90 p-2 rounded">
        <div className="text-xs font-bold">N</div>
        <div className="text-lg">↑</div>
      </div>
    </div>
  )
}

// Risk Legend Component
function RiskLegend() {
  const riskLevels = [
    { label: "Very Low", color: "bg-green-500", textColor: "text-green-700" },
    { label: "Low", color: "bg-green-300", textColor: "text-green-600" },
    { label: "Moderate", color: "bg-yellow-400", textColor: "text-yellow-700" },
    { label: "High", color: "bg-orange-500", textColor: "text-orange-700" },
    { label: "Very High", color: "bg-red-500", textColor: "text-red-700" },
  ]

  return (
    <div className="bg-white/90 p-3 rounded-lg border">
      <h4 className="font-medium text-sm mb-2">Fire Risk Legend</h4>
      <div className="space-y-1">
        {riskLevels.map((level) => (
          <div key={level.label} className="flex items-center space-x-2">
            <div className={`w-4 h-4 ${level.color} rounded`} />
            <span className={`text-xs ${level.textColor} font-medium`}>{level.label}</span>
          </div>
        ))}
      </div>
      <div className="mt-2 pt-2 border-t">
        <div className="flex items-center space-x-2">
          <div className="w-4 h-4 bg-red-500 rounded-full animate-pulse" />
          <span className="text-xs text-red-700 font-medium">Active Fire</span>
        </div>
      </div>
    </div>
  )
}

// 3D Terrain from 2D heightmap
function Generated3DTerrain({
  heightmapData,
  wireframe = false,
  showFire = false,
}: {
  heightmapData: ImageData | null
  wireframe?: boolean
  showFire?: boolean
}) {
  const meshRef = useRef<THREE.Mesh>(null)
  const [geometry, setGeometry] = useState<THREE.PlaneGeometry | null>(null)

  useEffect(() => {
    if (heightmapData) {
      const newGeometry = new THREE.PlaneGeometry(10, 10, 63, 63)
      const vertices = newGeometry.attributes.position.array as Float32Array

      // Apply height data from the uploaded image
      for (let i = 0; i < vertices.length; i += 3) {
        const x = Math.floor(((vertices[i] + 5) / 10) * 63)
        const z = Math.floor(((vertices[i + 2] + 5) / 10) * 63)
        const pixelIndex = (z * 64 + x) * 4

        if (pixelIndex < heightmapData.data.length) {
          // Use grayscale value for height (average RGB for better results)
          const r = heightmapData.data[pixelIndex]
          const g = heightmapData.data[pixelIndex + 1]
          const b = heightmapData.data[pixelIndex + 2]
          const grayscale = (r + g + b) / 3
          const height = (grayscale / 255) * 4 // Scale height more dramatically
          vertices[i + 1] = height
        }
      }

      newGeometry.attributes.position.needsUpdate = true
      newGeometry.computeVertexNormals()
      setGeometry(newGeometry)
    }
  }, [heightmapData])

  useFrame((state) => {
    if (meshRef.current && showFire) {
      const material = meshRef.current.material as THREE.MeshStandardMaterial
      material.emissive.setHex(0x331100)
      material.emissiveIntensity = 0.1 + Math.sin(state.clock.elapsedTime * 2) * 0.05
    }
  })

  if (!heightmapData || !geometry) {
    return (
      <mesh position={[0, -2, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[10, 10, 32, 32]} />
        <meshStandardMaterial color="#4a5d23" wireframe={wireframe} />
      </mesh>
    )
  }

  return (
    <group>
      <mesh ref={meshRef} position={[0, -2, 0]} rotation={[-Math.PI / 2, 0, 0]} geometry={geometry}>
        <meshStandardMaterial
          color={showFire ? "#8B4513" : "#4a5d23"}
          wireframe={wireframe}
          roughness={0.8}
          metalness={0.1}
        />
      </mesh>

      {showFire && (
        <Html position={[0, 2, 0]} center>
          <div className="bg-red-500/80 text-white px-3 py-1 rounded text-sm animate-pulse">
            🔥 Fire Simulation Active
          </div>
        </Html>
      )}
    </group>
  )
}

// Fire particles for 3D scene
function FireParticles3D({ visible = true }: { visible?: boolean }) {
  const pointsRef = useRef<THREE.Points>(null)

  useFrame((state) => {
    if (pointsRef.current && visible) {
      pointsRef.current.rotation.y = state.clock.elapsedTime * 0.1

      const positionAttribute = pointsRef.current.geometry.attributes.position
      const positions = positionAttribute.array as Float32Array

      for (let i = 0; i < positions.length; i += 3) {
        positions[i + 1] += 0.02
        if (positions[i + 1] > 4) {
          positions[i + 1] = 0
          positions[i] = (Math.random() - 0.5) * 8
          positions[i + 2] = (Math.random() - 0.5) * 8
        }
      }

      positionAttribute.needsUpdate = true
    }
  })

  if (!visible) return null

  const particleCount = 200
  const positions = new Float32Array(particleCount * 3)
  const colors = new Float32Array(particleCount * 3)

  for (let i = 0; i < particleCount; i++) {
    positions[i * 3] = (Math.random() - 0.5) * 8
    positions[i * 3 + 1] = Math.random() * 4
    positions[i * 3 + 2] = (Math.random() - 0.5) * 8

    colors[i * 3] = 1
    colors[i * 3 + 1] = Math.random() * 0.5
    colors[i * 3 + 2] = 0
  }

  return (
    <points ref={pointsRef}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          args={[positions, 3]}
          count={particleCount}
          array={positions}
          itemSize={3}
        />
        <bufferAttribute
          attach="attributes-color"
          args={[colors, 3]}
          count={particleCount}
          array={colors}
          itemSize={3}
        />
      </bufferGeometry>
      <pointsMaterial size={0.1} vertexColors transparent opacity={0.6} />
    </points>
  )
}

// AR Viewer Component
function ARViewer({ heightmapData, onClose }: { heightmapData: ImageData | null; onClose: () => void }) {
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
        alert("AR session would start here. This is a demo implementation.")
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
        <h3 className="text-lg font-semibold mb-4">AR Viewer</h3>

        {!isARSupported ? (
          <div className="space-y-4">
            <Alert>
              <Smartphone className="h-4 w-4" />
              <AlertDescription>
                AR viewing requires a compatible mobile device with WebXR support. Please try on a modern smartphone or
                tablet.
              </AlertDescription>
            </Alert>
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
            <p className="text-sm text-muted-foreground">
              Your 3D terrain model is ready for AR viewing. Make sure you're in a well-lit area with enough space.
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

export default function Map2Dto3DPage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const [progress, setProgress] = useState(0)
  const [heightmapData, setHeightmapData] = useState<ImageData | null>(null)
  const [previewImage, setPreviewImage] = useState<string | null>(null)
  const [wireframe, setWireframe] = useState(false)
  const [showFire, setShowFire] = useState(false)
  const [isAnimating, setIsAnimating] = useState(false)
  const [showAR, setShowAR] = useState(false)
  const [activeTab, setActiveTab] = useState("upload")

  // Fire spread analysis state
  const [currentHour, setCurrentHour] = useState([1])
  const [isSpreadPlaying, setIsSpreadPlaying] = useState(false)
  const [spreadSpeed, setSpreadSpeed] = useState([1])

  const { toast } = useToast()

  // Generate fire spread data for each hour
  const generateFireSpreadData = (hour: number) => {
    const baseSpots = [
      { x: 30, y: 40, intensity: 0.9, id: "fire-1" },
      { x: 60, y: 30, intensity: 0.7, id: "fire-2" },
    ]

    const spots: { intensity: number; x: number; y: number; id: string }[] = []

    // Add base fire spots with increasing intensity
    baseSpots.forEach((spot, index) => {
      spots.push({
        ...spot,
        intensity: Math.min(spot.intensity + (hour - 1) * 0.1, 1),
      })

      // Add spreading fire spots based on hour
      if (hour > 2) {
        const spreadRadius = (hour - 2) * 15
        const numNewSpots = Math.floor(hour / 2)

        for (let i = 0; i < numNewSpots; i++) {
          const angle = (i / numNewSpots) * Math.PI * 2
          const distance = Math.random() * spreadRadius
          spots.push({
            x: Math.max(5, Math.min(95, spot.x + Math.cos(angle) * distance)),
            y: Math.max(5, Math.min(95, spot.y + Math.sin(angle) * distance)),
            intensity: Math.max(0.3, 1 - (hour - 2) * 0.1),
            id: `fire-${index}-spread-${i}-h${hour}`,
          })
        }
      }
    })

    return spots
  }

  // Auto-play fire spread animation
  useEffect(() => {
    let interval: NodeJS.Timeout
    if (isSpreadPlaying) {
      interval = setInterval(() => {
        setCurrentHour((prev) => {
          const newHour = prev[0] + 1
          return newHour > 12 ? [1] : [newHour]
        })
      }, 1000 / spreadSpeed[0]) // Speed control
    }
    return () => clearInterval(interval)
  }, [isSpreadPlaying, spreadSpeed])

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      if (!file.type.startsWith("image/")) {
        toast({
          title: "Invalid file type",
          description: "Please upload an image file (PNG, JPG, etc.)",
          variant: "destructive",
        })
        return
      }

      if (file.size > 10 * 1024 * 1024) {
        toast({
          title: "File too large",
          description: "Please upload an image smaller than 10MB",
          variant: "destructive",
        })
        return
      }

      setSelectedFile(file)

      // Create preview
      const reader = new FileReader()
      reader.onload = (e) => {
        setPreviewImage(e.target?.result as string)
      }
      reader.readAsDataURL(file)

      toast({
        title: "File selected",
        description: `${file.name} is ready for conversion`,
      })
    }
  }

  const processImageToHeightmap = async () => {
    if (!selectedFile || !previewImage) return

    setIsProcessing(true)
    setProgress(0)

    // Simulate processing steps
    const steps = [
      { message: "Loading image...", progress: 20 },
      { message: "Analyzing terrain data...", progress: 40 },
      { message: "Generating heightmap...", progress: 60 },
      { message: "Creating 3D mesh...", progress: 80 },
      { message: "Finalizing 3D model...", progress: 100 },
    ]

    for (const step of steps) {
      await new Promise((resolve) => setTimeout(resolve, 800))
      setProgress(step.progress)
    }

    // Process the image to create heightmap data
    const img = new Image()
    img.onload = () => {
      const canvas = document.createElement("canvas")
      const ctx = canvas.getContext("2d")

      if (ctx) {
        canvas.width = 64
        canvas.height = 64
        ctx.drawImage(img, 0, 0, 64, 64)

        const imageData = ctx.getImageData(0, 0, 64, 64)
        setHeightmapData(imageData)

        // Auto-switch to preview tab
        setActiveTab("preview")
      }
    }
    img.src = previewImage

    setIsProcessing(false)
    toast({
      title: "Conversion complete!",
      description: "Your 2D map has been converted to 3D terrain. Check the Preview tab!",
    })
  }

  const downloadModel = () => {
    if (!heightmapData) {
      toast({
        title: "No 3D model available",
        description: "Please convert a 2D map first",
        variant: "destructive",
      })
      return
    }

    // Simulate download
    toast({
      title: "Download started",
      description: "Your 3D model is being prepared for download",
    })
  }

  return (
    <div className="min-h-screen bg-background">
      <Navigation />

      <div className="container mx-auto px-4 py-6">
        <div className="mb-6">
          <h1 className="text-3xl font-bold mb-2">2D to 3D Map Conversion</h1>
          <p className="text-muted-foreground">
            Upload your 2D terrain maps and convert them into interactive 3D fire simulation models
          </p>
        </div>

        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="upload">Upload & Convert</TabsTrigger>
            <TabsTrigger value="preview">3D Preview</TabsTrigger>
            <TabsTrigger value="simulation">Fire Simulation</TabsTrigger>
            <TabsTrigger value="spread-analysis">Fire Spread Analysis</TabsTrigger>
          </TabsList>

          <TabsContent value="upload" className="space-y-6">
            <div className="grid lg:grid-cols-2 gap-6">
              {/* Upload Section */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center">
                    <Upload className="mr-2 h-5 w-5" />
                    Upload 2D Terrain Map
                  </CardTitle>
                  <CardDescription>Supported formats: PNG, JPG, JPEG, BMP (Max size: 10MB)</CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                  <div className="space-y-2">
                    <Label htmlFor="terrain-upload">Terrain Image</Label>
                    <div className="border-2 border-dashed border-muted-foreground/25 rounded-lg p-6 text-center hover:border-muted-foreground/50 transition-colors">
                      <Input
                        id="terrain-upload"
                        type="file"
                        accept="image/*"
                        onChange={handleFileUpload}
                        className="hidden"
                      />
                      <Label htmlFor="terrain-upload" className="cursor-pointer">
                        <div className="space-y-2">
                          <FileImage className="h-10 w-10 mx-auto text-muted-foreground" />
                          <div className="text-sm">
                            {selectedFile ? (
                              <div className="space-y-1">
                                <p className="font-medium">{selectedFile.name}</p>
                                <p className="text-muted-foreground">
                                  {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                                </p>
                              </div>
                            ) : (
                              <>
                                <p className="font-medium">Click to upload or drag and drop</p>
                                <p className="text-muted-foreground">PNG, JPG, JPEG, BMP files</p>
                              </>
                            )}
                          </div>
                        </div>
                      </Label>
                    </div>
                  </div>

                  {previewImage && (
                    <div className="space-y-2">
                      <Label>2D Preview</Label>
                      <div className="border rounded-lg p-2">
                        <img
                          src={previewImage || "/placeholder.svg"}
                          alt="2D Terrain Preview"
                          className="w-full h-48 object-cover rounded"
                        />
                      </div>
                    </div>
                  )}

                  {isProcessing && (
                    <div className="space-y-2">
                      <div className="flex items-center justify-between text-sm">
                        <span>Converting to 3D...</span>
                        <span>{progress}%</span>
                      </div>
                      <Progress value={progress} />
                    </div>
                  )}

                  <Button onClick={processImageToHeightmap} disabled={!selectedFile || isProcessing} className="w-full">
                    {isProcessing ? (
                      <>Processing...</>
                    ) : (
                      <>
                        <Layers3 className="mr-2 h-4 w-4" />
                        Convert to 3D
                      </>
                    )}
                  </Button>
                </CardContent>
              </Card>

              {/* Instructions */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center">
                    <Settings className="mr-2 h-5 w-5" />
                    Conversion Instructions
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <Alert>
                    <AlertCircle className="h-4 w-4" />
                    <AlertDescription>
                      Upload a 2D terrain map to generate an interactive 3D fire simulation model
                    </AlertDescription>
                  </Alert>

                  <div className="space-y-3">
                    <h4 className="font-medium">Best Results Tips:</h4>
                    <ul className="text-sm space-y-1 text-muted-foreground">
                      <li>• Use grayscale images for height data</li>
                      <li>• Higher contrast = more dramatic terrain</li>
                      <li>• Square images work best (512x512, 1024x1024)</li>
                      <li>• Avoid images with text or overlays</li>
                    </ul>
                  </div>

                  <div className="space-y-3">
                    <h4 className="font-medium">Available Features:</h4>
                    <ul className="text-sm space-y-1 text-muted-foreground">
                      <li>1. 3D terrain visualization</li>
                      <li>2. Interactive fire simulation</li>
                      <li>3. Hour-by-hour fire spread analysis</li>
                      <li>4. AR viewing capabilities</li>
                      <li>5. Export and download options</li>
                    </ul>
                  </div>

                  <div className="pt-4 border-t">
                    <div className="flex items-center space-x-2">
                      <CheckCircle className="h-4 w-4 text-green-600" />
                      <span className="text-sm">Real-time 3D rendering</span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <CheckCircle className="h-4 w-4 text-green-600" />
                      <span className="text-sm">2D fire spread mapping</span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <CheckCircle className="h-4 w-4 text-green-600" />
                      <span className="text-sm">AR viewing capabilities</span>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="preview" className="space-y-6">
            <div className="grid lg:grid-cols-4 gap-6">
              <div className="lg:col-span-3">
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center">
                      <Layers3 className="mr-2 h-5 w-5" />
                      3D Terrain Preview
                    </CardTitle>
                    <CardDescription>
                      {heightmapData ? "Generated from your uploaded image" : "Upload an image to see 3D preview"}
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="h-[500px] w-full bg-gradient-to-b from-sky-200 to-sky-100 rounded-lg overflow-hidden">
                      <Canvas camera={{ position: [8, 6, 8], fov: 60 }}>
                        <Suspense fallback={null}>
                          <ambientLight intensity={0.4} />
                          <directionalLight position={[10, 10, 5]} intensity={1} />
                          <Generated3DTerrain heightmapData={heightmapData} wireframe={wireframe} showFire={showFire} />
                          <OrbitControls enablePan enableZoom enableRotate />
                          <Environment preset="sunset" />
                        </Suspense>
                      </Canvas>
                    </div>
                  </CardContent>
                </Card>
              </div>

              <div className="space-y-4">
                <Card>
                  <CardHeader>
                    <CardTitle>View Controls</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="flex items-center justify-between">
                      <span className="text-sm">Wireframe Mode</span>
                      <Button
                        variant={wireframe ? "default" : "outline"}
                        size="sm"
                        onClick={() => setWireframe(!wireframe)}
                      >
                        {wireframe ? "On" : "Off"}
                      </Button>
                    </div>

                    <div className="flex items-center justify-between">
                      <span className="text-sm">Fire Preview</span>
                      <Button
                        variant={showFire ? "destructive" : "outline"}
                        size="sm"
                        onClick={() => setShowFire(!showFire)}
                      >
                        {showFire ? "On" : "Off"}
                      </Button>
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle>Export Options</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-2">
                    <Button variant="outline" size="sm" className="w-full bg-transparent" onClick={downloadModel}>
                      <Download className="mr-2 h-4 w-4" />
                      Download 3D Model
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      className="w-full bg-transparent"
                      onClick={() => setShowAR(true)}
                      disabled={!heightmapData}
                    >
                      <Smartphone className="mr-2 h-4 w-4" />
                      View in AR
                    </Button>
                  </CardContent>
                </Card>

                {heightmapData && (
                  <Card>
                    <CardHeader>
                      <CardTitle>Model Info</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-2">
                      <div className="flex justify-between text-sm">
                        <span className="text-muted-foreground">Status</span>
                        <Badge variant="outline" className="bg-green-50 text-green-700">
                          Ready
                        </Badge>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span className="text-muted-foreground">Vertices</span>
                        <span>4,096</span>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span className="text-muted-foreground">Size</span>
                        <span>10km²</span>
                      </div>
                    </CardContent>
                  </Card>
                )}
              </div>
            </div>
          </TabsContent>

          <TabsContent value="simulation" className="space-y-6">
            <div className="grid lg:grid-cols-4 gap-6">
              <div className="lg:col-span-3">
                <Card>
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <div>
                        <CardTitle className="flex items-center">
                          <Zap className="mr-2 h-5 w-5" />
                          Fire Simulation
                        </CardTitle>
                        <CardDescription>Interactive fire spread on your 3D terrain</CardDescription>
                      </div>
                      <Badge variant="outline" className="bg-red-50 text-red-700 border-red-200">
                        {isAnimating ? "Active" : "Paused"}
                      </Badge>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <div className="h-[500px] w-full bg-gradient-to-b from-orange-100 to-red-100 rounded-lg overflow-hidden">
                      <Canvas camera={{ position: [8, 6, 8], fov: 60 }}>
                        <Suspense fallback={null}>
                          <ambientLight intensity={0.3} />
                          <directionalLight position={[10, 10, 5]} intensity={1} />
                          <pointLight position={[0, 5, 0]} intensity={1} color="#ff4500" />

                          <Generated3DTerrain heightmapData={heightmapData} wireframe={false} showFire={true} />
                          <FireParticles3D visible={isAnimating} />

                          <OrbitControls enablePan enableZoom enableRotate />
                          <Environment preset="sunset" />
                        </Suspense>
                      </Canvas>
                    </div>
                  </CardContent>
                </Card>
              </div>

              <div className="space-y-4">
                <Card>
                  <CardHeader>
                    <CardTitle>Simulation Controls</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="flex items-center space-x-2">
                      <Button
                        variant={isAnimating ? "default" : "outline"}
                        size="sm"
                        onClick={() => setIsAnimating(!isAnimating)}
                        disabled={!heightmapData}
                      >
                        {isAnimating ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
                      </Button>
                      <Button variant="outline" size="sm" onClick={() => setIsAnimating(false)}>
                        <RotateCcw className="h-4 w-4" />
                      </Button>
                    </div>

                    <div className="text-xs text-muted-foreground space-y-1">
                      <p>• Click Play to start fire simulation</p>
                      <p>• Use mouse to rotate and zoom</p>
                      <p>• Fire spreads based on terrain height</p>
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle>AR Experience</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-2">
                    <Button
                      variant="outline"
                      size="sm"
                      className="w-full bg-transparent"
                      onClick={() => setShowAR(true)}
                      disabled={!heightmapData}
                    >
                      <Smartphone className="mr-2 h-4 w-4" />
                      View Fire in AR
                    </Button>
                    <p className="text-xs text-muted-foreground">Experience the fire simulation in augmented reality</p>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle>Simulation Stats</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-muted-foreground">Terrain Size</span>
                      <span className="font-medium">10km²</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-muted-foreground">Fire Particles</span>
                      <span className="font-medium">{isAnimating ? "200" : "0"}</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-muted-foreground">Status</span>
                      <Badge variant={isAnimating ? "destructive" : "outline"}>
                        {isAnimating ? "Burning" : "Safe"}
                      </Badge>
                    </div>
                  </CardContent>
                </Card>
              </div>
            </div>
          </TabsContent>

          <TabsContent value="spread-analysis" className="space-y-6">
            <div className="grid lg:grid-cols-4 gap-6">
              <div className="lg:col-span-3 space-y-6">
                {/* Prediction Map */}
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center">
                      <MapPin className="mr-2 h-5 w-5" />
                      Fire Risk Prediction Map
                    </CardTitle>
                    <CardDescription>
                      Initial fire risk assessment based on terrain and environmental factors
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="relative">
                      <FireSpreadMap hour={0} imageData={previewImage} fireSpots={[]} />
                      <div className="absolute top-2 left-2 bg-blue-600 text-white px-2 py-1 rounded text-sm font-medium">
                        Prediction Map
                      </div>
                      <div className="absolute bottom-2 right-2">
                        <RiskLegend />
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* Fire Spread Timeline */}
                <Card>
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <div>
                        <CardTitle className="flex items-center">
                          <TrendingUp className="mr-2 h-5 w-5" />
                          Fire Spread Timeline
                        </CardTitle>
                        <CardDescription>Hour-by-hour fire spread progression over 12 hours</CardDescription>
                      </div>
                      <Badge variant="outline" className="bg-orange-50 text-orange-700 border-orange-200">
                        Hour {currentHour[0].toString().padStart(2, "0")}
                      </Badge>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                      {[1, 2, 3, 4, 5, 6].map((hour) => (
                        <div
                          key={hour}
                          className={`relative ${
                            currentHour[0] === hour ? "ring-2 ring-orange-500 ring-offset-2" : ""
                          }`}
                        >
                          <FireSpreadMap
                            hour={hour}
                            imageData={previewImage}
                            fireSpots={generateFireSpreadData(hour)}
                          />
                        </div>
                      ))}
                    </div>

                    <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mt-4">
                      {[7, 8, 9, 10, 11, 12].map((hour) => (
                        <div
                          key={hour}
                          className={`relative ${
                            currentHour[0] === hour ? "ring-2 ring-orange-500 ring-offset-2" : ""
                          }`}
                        >
                          <FireSpreadMap
                            hour={hour}
                            imageData={previewImage}
                            fireSpots={generateFireSpreadData(hour)}
                          />
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              </div>

              <div className="space-y-4">
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center">
                      <Clock className="mr-2 h-5 w-5" />
                      Animation Controls
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="flex items-center space-x-2">
                      <Button
                        variant={isSpreadPlaying ? "default" : "outline"}
                        size="sm"
                        onClick={() => setIsSpreadPlaying(!isSpreadPlaying)}
                      >
                        {isSpreadPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
                      </Button>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => {
                          setCurrentHour([1])
                          setIsSpreadPlaying(false)
                        }}
                      >
                        <RotateCcw className="h-4 w-4" />
                      </Button>
                    </div>

                    <div className="space-y-2">
                      <label className="text-sm font-medium">Current Hour: {currentHour[0]}</label>
                      <Slider
                        value={currentHour}
                        onValueChange={setCurrentHour}
                        max={12}
                        min={1}
                        step={1}
                        className="w-full"
                      />
                    </div>

                    <div className="space-y-2">
                      <label className="text-sm font-medium">Animation Speed: {spreadSpeed[0]}x</label>
                      <Slider
                        value={spreadSpeed}
                        onValueChange={setSpreadSpeed}
                        max={5}
                        min={0.5}
                        step={0.5}
                        className="w-full"
                      />
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle>Spread Statistics</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-muted-foreground">Current Hour</span>
                      <Badge variant="outline">{currentHour[0]}</Badge>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-muted-foreground">Active Fire Spots</span>
                      <span className="font-medium">{generateFireSpreadData(currentHour[0]).length}</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-muted-foreground">Burned Area</span>
                      <span className="font-medium">{(currentHour[0] * 0.8).toFixed(1)} km²</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-muted-foreground">Spread Rate</span>
                      <span className="font-medium">{(currentHour[0] * 0.2).toFixed(1)} km/h</span>
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle>Analysis Tools</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-2">
                    <Button variant="outline" size="sm" className="w-full bg-transparent">
                      <Download className="mr-2 h-4 w-4" />
                      Export Spread Data
                    </Button>
                    <Button variant="outline" size="sm" className="w-full bg-transparent">
                      <MapPin className="mr-2 h-4 w-4" />
                      Generate Report
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      className="w-full bg-transparent"
                      onClick={() => setActiveTab("simulation")}
                    >
                      <Layers3 className="mr-2 h-4 w-4" />
                      View in 3D
                    </Button>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle>Environmental Factors</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span className="text-muted-foreground">Wind Speed</span>
                      <span>15 km/h</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-muted-foreground">Humidity</span>
                      <span>35%</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-muted-foreground">Temperature</span>
                      <span>32°C</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-muted-foreground">Fuel Moisture</span>
                      <span>8%</span>
                    </div>
                  </CardContent>
                </Card>
              </div>
            </div>
          </TabsContent>
        </Tabs>

        {/* AR Viewer Modal */}
        {showAR && <ARViewer heightmapData={heightmapData} onClose={() => setShowAR(false)} />}
      </div>
    </div>
  )
}
