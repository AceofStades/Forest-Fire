"use client";
import React, { useState, useEffect } from 'react';
import { MapContainer, TileLayer, Polyline, Marker, Popup, useMapEvents } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';

import { gridToGPS, gpsToGrid, getMapBounds } from '../../lib/geoUtils';
import FireOverlay from './FireOverlay';

// Fix Leaflet Marker Icons in Next.js
// @ts-ignore
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

// Click Handler Component
const MapClickHandler = ({ onMapClick }: { onMapClick: (lat: number, lng: number) => void }) => {
  useMapEvents({
    click(e: { latlng: { lat: number; lng: number; }; }) {
      onMapClick(e.latlng.lat, e.latlng.lng);
    },
  });
  return null;
};

export default function RealTimeMap() {
  const [grid, setGrid] = useState<number[][] | null>(null);
  const [path, setPath] = useState<[number, number][]>([]);
  const [userPos, setUserPos] = useState<[number, number]>([30.1000, 79.1000]); 
  const [rescueCenter] = useState<[number, number]>([30.1400, 79.1800]); 

  // 1. Fetch Fire Grid from Backend
  useEffect(() => {
    fetch('http://localhost:8000/fire-grid')
      .then(res => res.json())
      .then(data => setGrid(data.grid))
      .catch(err => console.error("Backend Error:", err));
  }, []);

  // 2. Handle Map Click -> Calculate Safe Path
  const handleMapClick = async (lat: number, lng: number) => {
    setUserPos([lat, lng]);

    // Convert Real GPS -> Grid Coordinates
    const startGrid = gpsToGrid(lat, lng);
    const goalGrid = gpsToGrid(rescueCenter[0], rescueCenter[1]);

    try {
      const res = await fetch('http://localhost:8000/get-safe-path', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ start: startGrid, goal: goalGrid })
      });
      
      const data = await res.json();
      
      // Convert Grid Path -> Real GPS Path
      const gpsPath = data.path.map((p: number[]) => gridToGPS(p[0], p[1]));
      setPath(gpsPath);
      
    } catch (e) {
      console.error("Pathfinding failed:", e);
    }
  };

  const mapBounds = getMapBounds(); 
  const mapCenter: [number, number] = [30.1000, 79.1000];

  return (
    <div className="w-full h-[600px] rounded-xl overflow-hidden shadow-2xl border-4 border-slate-800 relative">
      <MapContainer 
        center={mapCenter} 
        zoom={12} 
        style={{ height: '100%', width: '100%' }}
        maxBounds={mapBounds}
      >
       <TileLayer 
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" 
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        />
        
        {/* Fire Simulation Overlay */}
        {grid && <FireOverlay probGrid={grid} />}

        {/* Safe Path */}
        {path.length > 0 && (
          <Polyline 
            positions={path} 
            pathOptions={{ 
              color: "#22c55e", 
              weight: 5, 
              opacity: 0.9, 
              dashArray: "10, 10",
              lineCap: 'round'
            }} 
          />
        )}

        <Marker position={userPos}>
          <Popup>You (Click map to move)</Popup>
        </Marker>
        <Marker position={rescueCenter}>
          <Popup>Rescue Center</Popup>
        </Marker>

        <MapClickHandler onMapClick={handleMapClick} />
      </MapContainer>
      
      {/* Legend */}
      <div className="absolute bottom-5 right-5 bg-slate-900/90 p-4 rounded-lg text-white z-[1000] border border-slate-700">
        <h4 className="font-bold mb-2 text-sm uppercase tracking-wider">Live Status</h4>
        <div className="flex items-center gap-2 mb-1">
          <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
          <span className="text-xs">Safe Path</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 bg-orange-600 rounded-full"></div>
          <span className="text-xs">Active Fire</span>
        </div>
      </div>
    </div>
  );
}