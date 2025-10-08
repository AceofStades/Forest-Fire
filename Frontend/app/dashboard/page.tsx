"use client"

import { useState, useEffect } from "react"
import dynamic from "next/dynamic"
import { Navigation } from "@/components/navigation"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Slider } from "@/components/ui/slider"
import { Switch } from "@/components/ui/switch"
import { Badge } from "@/components/ui/badge"
import {
  Play,
  Pause,
  RotateCcw,
  Layers,
  Thermometer,
  Droplets,
  Mountain,
  TreePine,
  Users,
  RouteIcon as Road,
} from "lucide-react"

// Dynamically import map component to avoid SSR issues
const MapComponent = dynamic(() => import("@/components/map-component"), {
  ssr: false,
  loading: () => <div className="h-[600px] bg-muted animate-pulse rounded-lg" />,
})

export default function DashboardPage() {
  const [isPlaying, setIsPlaying] = useState(false)
  const [timeStep, setTimeStep] = useState([0])
  const [layers, setLayers] = useState({
    fireProbability: true,
    terrain: false,
    temperature: false,
    humidity: false,
    vegetation: false,
    settlements: false,
    roads: false,
  })

  const timeLabels = ["Current", "1hr", "2hr", "3hr", "6hr", "12hr", "24hr"]

  useEffect(() => {
    let interval: NodeJS.Timeout
    if (isPlaying) {
      interval = setInterval(() => {
        setTimeStep((prev) => {
          const newValue = prev[0] + 1
          return newValue >= timeLabels.length ? [0] : [newValue]
        })
      }, 1000)
    }
    return () => clearInterval(interval)
  }, [isPlaying, timeLabels.length])

  const toggleLayer = (layerName: keyof typeof layers) => {
    setLayers((prev) => ({ ...prev, [layerName]: !prev[layerName] }))
  }

  return (
    <div className="min-h-screen bg-background">
      <Navigation />

      <div className="container mx-auto px-4 py-6">
        <div className="mb-6">
          <h1 className="text-3xl font-bold mb-2">Fire Prediction Dashboard</h1>
          <p className="text-muted-foreground">Real-time forest fire probability maps and spread simulation</p>
        </div>

        <div className="grid lg:grid-cols-4 gap-6">
          {/* Map Area */}
          <div className="lg:col-span-3">
            <Card>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle>Interactive Fire Map</CardTitle>
                    <CardDescription>Current prediction: {timeLabels[timeStep[0]]}</CardDescription>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Badge variant="outline" className="bg-red-50 text-red-700 border-red-200">
                      High Risk: 15.2%
                    </Badge>
                    <Badge variant="outline" className="bg-yellow-50 text-yellow-700 border-yellow-200">
                      Medium Risk: 32.8%
                    </Badge>
                    <Badge variant="outline" className="bg-green-50 text-green-700 border-green-200">
                      Low Risk: 52.0%
                    </Badge>
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <MapComponent layers={layers} timeStep={timeStep[0]} />
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
                  Animation Controls
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
                      setTimeStep([0])
                      setIsPlaying(false)
                    }}
                  >
                    <RotateCcw className="h-4 w-4" />
                  </Button>
                </div>

                <div className="space-y-2">
                  <label className="text-sm font-medium">Time: {timeLabels[timeStep[0]]}</label>
                  <Slider
                    value={timeStep}
                    onValueChange={setTimeStep}
                    max={timeLabels.length - 1}
                    step={1}
                    className="w-full"
                  />
                </div>
              </CardContent>
            </Card>

            {/* Layer Controls */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center">
                  <Layers className="mr-2 h-5 w-5" />
                  Map Layers
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <div className="w-3 h-3 bg-gradient-to-r from-red-500 to-orange-500 rounded"></div>
                      <span className="text-sm">Fire Probability</span>
                    </div>
                    <Switch checked={layers.fireProbability} onCheckedChange={() => toggleLayer("fireProbability")} />
                  </div>

                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <Mountain className="w-3 h-3 text-gray-600" />
                      <span className="text-sm">Terrain</span>
                    </div>
                    <Switch checked={layers.terrain} onCheckedChange={() => toggleLayer("terrain")} />
                  </div>

                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <Thermometer className="w-3 h-3 text-red-500" />
                      <span className="text-sm">Temperature</span>
                    </div>
                    <Switch checked={layers.temperature} onCheckedChange={() => toggleLayer("temperature")} />
                  </div>

                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <Droplets className="w-3 h-3 text-blue-500" />
                      <span className="text-sm">Humidity</span>
                    </div>
                    <Switch checked={layers.humidity} onCheckedChange={() => toggleLayer("humidity")} />
                  </div>

                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <TreePine className="w-3 h-3 text-green-600" />
                      <span className="text-sm">Vegetation</span>
                    </div>
                    <Switch checked={layers.vegetation} onCheckedChange={() => toggleLayer("vegetation")} />
                  </div>

                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <Users className="w-3 h-3 text-purple-600" />
                      <span className="text-sm">Settlements</span>
                    </div>
                    <Switch checked={layers.settlements} onCheckedChange={() => toggleLayer("settlements")} />
                  </div>

                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <Road className="w-3 h-3 text-gray-800" />
                      <span className="text-sm">Roads</span>
                    </div>
                    <Switch checked={layers.roads} onCheckedChange={() => toggleLayer("roads")} />
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Current Conditions */}
            <Card>
              <CardHeader>
                <CardTitle>Current Conditions</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-muted-foreground">Temperature</span>
                  <span className="font-medium">28°C</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-muted-foreground">Humidity</span>
                  <span className="font-medium">45%</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-muted-foreground">Wind Speed</span>
                  <span className="font-medium">12 km/h</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-muted-foreground">Fire Weather Index</span>
                  <Badge variant="destructive">High</Badge>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  )
}
