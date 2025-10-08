"use client"

import { useEffect, useRef } from "react"
import L from "leaflet"
import "leaflet/dist/leaflet.css"

// Fix for default markers in Leaflet
delete (L.Icon.Default.prototype as any)._getIconUrl
L.Icon.Default.mergeOptions({
  iconRetinaUrl: "/placeholder.svg?height=25&width=25",
  iconUrl: "/placeholder.svg?height=25&width=25",
  shadowUrl: "/placeholder.svg?height=25&width=25",
})

interface MapComponentProps {
  layers: {
    fireProbability: boolean
    terrain: boolean
    temperature: boolean
    humidity: boolean
    vegetation: boolean
    settlements: boolean
    roads: boolean
  }
  timeStep: number
}

export default function MapComponent({ layers, timeStep }: MapComponentProps) {
  const mapRef = useRef<HTMLDivElement>(null)
  const mapInstanceRef = useRef<L.Map | null>(null)
  const layersRef = useRef<{ [key: string]: L.Layer }>({})

  useEffect(() => {
    if (!mapRef.current || mapInstanceRef.current) return

    // Initialize map centered on Uttarakhand, India
    const map = L.map(mapRef.current).setView([30.0668, 79.0193], 9)
    mapInstanceRef.current = map

    // Add base tile layer
    const baseLayer = L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      attribution: "© OpenStreetMap contributors",
    })
    baseLayer.addTo(map)

    // Create different layer types
    const fireProbabilityLayer = L.tileLayer(
      "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
      {
        opacity: 0.6,
        attribution: "Fire Probability Data",
      },
    )
    layersRef.current.fireProbability = fireProbabilityLayer

    const terrainLayer = L.tileLayer(
      "https://server.arcgisonline.com/ArcGIS/rest/services/World_Terrain_Base/MapServer/tile/{z}/{y}/{x}",
      {
        opacity: 0.7,
        attribution: "Terrain Data",
      },
    )
    layersRef.current.terrain = terrainLayer

    const temperatureLayer = L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      opacity: 0.4,
      attribution: "Temperature Data",
    })
    layersRef.current.temperature = temperatureLayer

    const humidityLayer = L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      opacity: 0.3,
      attribution: "Humidity Data",
    })
    layersRef.current.humidity = humidityLayer

    const vegetationLayer = L.tileLayer(
      "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
      {
        opacity: 0.5,
        attribution: "Vegetation Data",
      },
    )
    layersRef.current.vegetation = vegetationLayer

    // Add sample fire markers for Uttarakhand region
    const fireMarkers = [
      { lat: 30.0868, lng: 79.0393, risk: "high", name: "Dehradun Forest" },
      { lat: 29.9468, lng: 78.1642, risk: "medium", name: "Haridwar Region" },
      { lat: 30.2165, lng: 78.7809, risk: "low", name: "Rishikesh Area" },
      { lat: 29.8543, lng: 79.2021, risk: "high", name: "Nainital Forest" },
      { lat: 30.3165, lng: 78.0322, risk: "medium", name: "Mussoorie Hills" },
    ]

    fireMarkers.forEach((marker) => {
      const color = marker.risk === "high" ? "#ef4444" : marker.risk === "medium" ? "#f97316" : "#22c55e"
      const radius = marker.risk === "high" ? 12 : marker.risk === "medium" ? 10 : 8

      L.circleMarker([marker.lat, marker.lng], {
        radius: radius,
        fillColor: color,
        color: color,
        weight: 2,
        opacity: 0.8,
        fillOpacity: 0.6,
      })
        .addTo(map)
        .bindPopup(`
          <div>
            <strong>${marker.name}</strong><br/>
            Risk Level: <span style="color: ${color}; font-weight: bold;">${marker.risk.toUpperCase()}</span><br/>
            Time Step: ${timeStep + 1}
          </div>
        `)
    })

    return () => {
      if (mapInstanceRef.current) {
        mapInstanceRef.current.remove()
        mapInstanceRef.current = null
      }
    }
  }, [])

  // Update layers based on props
  useEffect(() => {
    if (!mapInstanceRef.current) return

    const map = mapInstanceRef.current

    // Handle all layer toggles
    Object.keys(layers).forEach((layerKey) => {
      const layer = layersRef.current[layerKey]
      const isEnabled = layers[layerKey as keyof typeof layers]

      if (layer) {
        if (isEnabled && !map.hasLayer(layer)) {
          map.addLayer(layer)
        } else if (!isEnabled && map.hasLayer(layer)) {
          map.removeLayer(layer)
        }
      }
    })

    // Update fire probability opacity based on time step (simulate animation)
    if (layersRef.current.fireProbability) {
      const opacity = 0.3 + timeStep * 0.1
      ;(layersRef.current.fireProbability as L.TileLayer).setOpacity(Math.min(opacity, 0.8))
    }
  }, [layers, timeStep])

  return (
    <div className="relative">
      <div ref={mapRef} className="h-[600px] w-full rounded-lg overflow-hidden" />
      <div className="absolute top-4 right-4 bg-white/90 backdrop-blur-sm p-3 rounded-lg shadow-lg dark:bg-gray-800/90">
        <div className="text-sm font-medium mb-2">Legend - Uttarakhand, India</div>
        <div className="space-y-1 text-xs">
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-red-500 rounded-full"></div>
            <span>High Risk</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-orange-500 rounded-full"></div>
            <span>Medium Risk</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-green-500 rounded-full"></div>
            <span>Low Risk</span>
          </div>
        </div>
      </div>
    </div>
  )
}
