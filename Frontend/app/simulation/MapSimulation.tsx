"use client";
import React, { useRef, useEffect, useState } from 'react';
import { MapContainer, TileLayer, ImageOverlay, Polyline, useMapEvents } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';

import { BOUNDS, ROWS, COLS, gridToLatLng, latLngToGrid } from './simulationUtils';



interface SimulationProps {
    probGrid: number[][] | null; // 320 rows x 400 cols
    fireState?: Uint8Array | null; // 1D array of size w*h
    width?: number;
    height?: number;
}

const MapEvents = ({ onMapClick }: { onMapClick: (lat: number, lon: number) => void }) => {
    useMapEvents({
        click(e) {
            onMapClick(e.latlng.lat, e.latlng.lng);
        },
    });
    return null;
};

const MapSimulation: React.FC<SimulationProps> = ({ probGrid, fireState, width = COLS, height = ROWS }) => {
    const [imageUrl, setImageUrl] = useState<string>("");
    const [fireUrl, setFireUrl] = useState<string>("");
    const [path, setPath] = useState<[number, number][]>([]);
    const canvasRef = useRef<HTMLCanvasElement | null>(null);
    const fireCanvasRef = useRef<HTMLCanvasElement | null>(null);

    useEffect(() => {
        // Fix for default marker icon in Next.js
        // @ts-ignore
        delete L.Icon.Default.prototype._getIconUrl;
        L.Icon.Default.mergeOptions({
            iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
            iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
            shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
        });

        const canvas = document.createElement('canvas');
        canvas.width = COLS;
        canvas.height = ROWS;
        canvasRef.current = canvas;

        const fCanvas = document.createElement('canvas');
        fCanvas.width = width;
        fCanvas.height = height;
        fireCanvasRef.current = fCanvas;
    }, [width, height]);

    // Render Static Probability Grid
    useEffect(() => {
        if (!canvasRef.current || !probGrid || probGrid.length === 0) return;
        const ctx = canvasRef.current.getContext('2d');
        if (!ctx) return;

        const imageData = ctx.createImageData(COLS, ROWS);
        const data = imageData.data;

        for (let r = 0; r < ROWS; r++) {
            for (let c = 0; c < COLS; c++) {
                const prob = probGrid[r][c];
                const idx = (r * COLS + c) * 4;
                if (prob > 0.0) {
                    data[idx] = 255;
                    data[idx + 1] = 50;
                    data[idx + 2] = 0;
                    data[idx + 3] = Math.floor(prob * 200);
                } else {
                    data[idx + 3] = 0;
                }
            }
        }

        ctx.putImageData(imageData, 0, 0);
        setImageUrl(canvasRef.current.toDataURL());
    }, [probGrid]);


    // Render Dynamic Fire State
    useEffect(() => {
        if (!fireCanvasRef.current || !fireState) return;
        const ctx = fireCanvasRef.current.getContext('2d');
        if (!ctx) return;

        const imageData = ctx.createImageData(width, height);
        const data = imageData.data;

        for (let i = 0; i < fireState.length; i++) {
            const state = fireState[i];
            const idx = i * 4;
            if (state === 1) { // Burning
                data[idx] = 255; // R
                data[idx + 1] = 69; // G
                data[idx + 2] = 0;  // B
                data[idx + 3] = 200; // A
            } else if (state === 2) { // Burnt
                data[idx] = 20;
                data[idx + 1] = 20;
                data[idx + 2] = 20;
                data[idx + 3] = 150;
            } else {
                data[idx + 3] = 0;
            }
        }
        ctx.putImageData(imageData, 0, 0);
        setFireUrl(fireCanvasRef.current.toDataURL());

    }, [fireState, width, height]);


    const handleMapClick = async (lat: number, lon: number) => {
        const [r, c] = latLngToGrid(lat, lon);
        console.log(`Clicked at Lat: ${lat}, Lon: ${lon} -> Grid: [${r}, ${c}]`);

        const goal: [number, number] = [Math.floor(ROWS / 2), Math.floor(COLS / 2)];
        const start: [number, number] = [r, c];

        try {
            const response = await fetch('http://127.0.0.1:8000/get-safe-path', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ start, goal })
            });

            if (response.ok) {
                const data = await response.json();
                const latLngPath = data.path.map(([pr, pc]: [number, number]) => gridToLatLng(pr, pc));
                setPath(latLngPath);
            }
        } catch (err) {
            console.error("Pathfinding error:", err);
        }
    };

    return (
        <div className="w-full h-[700px] rounded-2xl overflow-hidden border border-orange-500/30 relative glow-border shadow-lg shadow-orange-500/5">
            <MapContainer
                center={[30.00, 80.00]}
                zoom={13}
                style={{ height: '100%', width: '100%' }}
                scrollWheelZoom={true}
            >
                {/* Realistic Satellite Map for vegetation visibility */}
                <TileLayer
                    attribution='Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community'
                    url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
                />

                {/* Fire Probability Grid Overlay */}
                {imageUrl && (
                    <ImageOverlay
                        url={imageUrl}
                        bounds={BOUNDS}
                        opacity={0.4}
                    />
                )}

                {/* Active Fire Simulation Overlay */}
                {fireUrl && (
                    <ImageOverlay
                        url={fireUrl}
                        bounds={BOUNDS}
                        opacity={0.9}
                        zIndex={100}
                    />
                )}

                {/* Safe Path */}
                {path.length > 0 && (
                    <Polyline
                        positions={path}
                        color="#4ade80"
                        weight={5}
                        opacity={0.9}
                    />
                )}

                <MapEvents onMapClick={handleMapClick} />
            </MapContainer>

            {/* Info Overlay - Glassmorphism */}
            <div className="absolute top-4 right-4 bg-black/80 backdrop-blur-xl text-white p-4 rounded-xl z-[1000] max-w-xs border border-white/10 shadow-xl">
                <h3 className="font-bold mb-2 text-sm bg-gradient-to-r from-orange-400 to-amber-300 bg-clip-text text-transparent">
                    Uttarakhand Forest Safe Route
                </h3>
                <p className="text-xs text-gray-400 leading-relaxed">
                    Click anywhere on the map to find a safe path to the sector center avoiding fire zones.
                </p>
                <div className="mt-3 space-y-1.5">
                    <div className="flex items-center gap-2">
                        <div className="w-3.5 h-3.5 bg-gradient-to-br from-red-500 to-orange-500 opacity-80 rounded-sm shadow-sm shadow-red-500/30"></div>
                        <span className="text-xs text-gray-300">Fire Risk Zone</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <div className="w-3.5 h-1 bg-gradient-to-r from-green-400 to-emerald-300 rounded-full"></div>
                        <span className="text-xs text-gray-300">Safe Route (D* Lite)</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <div className="w-3.5 h-3.5 bg-red-600 rounded-sm"></div>
                        <span className="text-xs text-gray-300">Active Fire</span>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default MapSimulation;
