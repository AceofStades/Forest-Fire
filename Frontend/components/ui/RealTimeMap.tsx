"use client";
import { MapContainer, TileLayer, Polyline, Marker, Popup, useMap } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import { useState } from 'react';
import { gridToGPS, gpsToGrid } from '@/lib/geoUtils';

export default function RealTimeMap() {
  const [path, setPath] = useState<[number, number][]>([]);
  const [userPos, setUserPos] = useState<[number, number]>([30.0668, 79.0193]); // Pauri, Uttarakhand
  const rescueCenter: [number, number] = [30.1500, 79.3000];

  const getSafePath = async () => {
    const startGrid = gpsToGrid(userPos[0], userPos[1]);
    const goalGrid = gpsToGrid(rescueCenter[0], rescueCenter[1]);

    const res = await fetch('http://localhost:8000/get-safe-path', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ start: startGrid, goal: goalGrid })
    });
    
    const data = await res.json();
    // Convert the grid indices back to Lat/Lng for Leaflet
    const gpsPath = data.path.map((point: [number, number]) => gridToGPS(point[0], point[1]));
    setPath(gpsPath);
  };

  return (
    <div className="w-full h-[600px] rounded-xl overflow-hidden shadow-2xl border-4 border-slate-800">
      <MapContainer center={userPos} zoom={9} style={{ height: '100%', width: '100%' }}>
        <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
        
        {/* The Safe Path drawn from D* Lite output */}
        {path.length > 0 && (
          <Polyline positions={path} pathOptions={{ color: "#22c55e", weight: 5, opacity: 0.8, dashArray: "10, 10" }} />
        )}

        <Marker position={userPos}><Popup>You are here</Popup></Marker>
        <Marker position={rescueCenter}><Popup>Rescue Center</Popup></Marker>
      </MapContainer>

      <button 
        onClick={getSafePath}
        className="absolute bottom-10 left-1/2 -translate-x-1/2 z-[1000] bg-green-600 text-white px-8 py-3 rounded-full font-bold hover:bg-green-500 shadow-xl transition-all"
      >
        Calculate Safe Escape Route
      </button>
    </div>
  );
}