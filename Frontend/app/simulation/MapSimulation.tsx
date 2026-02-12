"use client";
import React, { useRef, useEffect, useState, useMemo } from 'react';
import { MapContainer, TileLayer, ImageOverlay, Polyline, useMapEvents } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';

// Fix for default marker icon in Next.js
// @ts-ignore
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
    iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
    iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
    shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

interface SimulationProps {
    probGrid: number[][]; // 320 rows x 400 cols
}

// Bounding Box for a region in Uttarakhand (e.g., near Jim Corbett)
// Top-Left: 29.60, 78.85
// Bottom-Right: 29.45, 79.10
// Width: ~0.25 deg, Height: ~0.15 deg
const BOUNDS: [[number, number], [number, number]] = [
    [29.5600, 78.9000], // South-West (Min Lat, Min Lon)
    [29.5800, 78.9400]  // North-East (Max Lat, Max Lon)
];

const ROWS = 320;
const COLS = 400;

// Helper to convert Grid Index to Lat/Lon
const gridToLatLng = (r: number, c: number) => {
    const latSpan = BOUNDS[1][0] - BOUNDS[0][0];
    const lonSpan = BOUNDS[1][1] - BOUNDS[0][1];

    // Grid (0,0) is Top-Left, which is Max Lat, Min Lon
    // Grid (320, 400) is Bottom-Right, which is Min Lat, Max Lon

    const lat = BOUNDS[1][0] - (r / ROWS) * latSpan;
    const lon = BOUNDS[0][1] + (c / COLS) * lonSpan;

    return [lat, lon] as [number, number];
};

// Helper to convert Lat/Lon to Grid Index
const latLngToGrid = (lat: number, lon: number) => {
    const latSpan = BOUNDS[1][0] - BOUNDS[0][0];
    const lonSpan = BOUNDS[1][1] - BOUNDS[0][1];

    let r = Math.floor(((BOUNDS[1][0] - lat) / latSpan) * ROWS);
    let c = Math.floor(((lon - BOUNDS[0][1]) / lonSpan) * COLS);

    // Clamp
    r = Math.max(0, Math.min(ROWS - 1, r));
    c = Math.max(0, Math.min(COLS - 1, c));

    return [r, c];
};

const MapEvents = ({ onMapClick }: { onMapClick: (lat: number, lon: number) => void }) => {
    useMapEvents({
        click(e) {
            onMapClick(e.latlng.lat, e.latlng.lng);
        },
    });
    return null;
};

const MapSimulation: React.FC<SimulationProps> = ({ probGrid }) => {
    const [imageUrl, setImageUrl] = useState<string>("");
    const [path, setPath] = useState<[number, number][]>([]);

    // Canvas for generating the heatmap image
    const canvasRef = useRef<HTMLCanvasElement | null>(null);

    // Initialize canvas once
    useEffect(() => {
        const canvas = document.createElement('canvas');
        canvas.width = COLS;
        canvas.height = ROWS;
        canvasRef.current = canvas;
    }, []);

    // Update Image Overlay when probGrid changes
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
                    // Red color for fire, alpha depends on probability
                    data[idx] = 255;     // R
                    data[idx + 1] = 50;   // G
                    data[idx + 2] = 0;   // B
                    data[idx + 3] = Math.floor(prob * 200); // Alpha
                } else {
                    data[idx + 3] = 0; // Transparent
                }
            }
        }

        ctx.putImageData(imageData, 0, 0);
        setImageUrl(canvasRef.current.toDataURL());

    }, [probGrid]);

    const handleMapClick = async (lat: number, lon: number) => {
        const [r, c] = latLngToGrid(lat, lon);
        console.log(`Clicked at Lat: ${lat}, Lon: ${lon} -> Grid: [${r}, ${c}]`);

        // Define Goal (Center of the map for now, or could be another click)
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
                // Convert grid path back to lat/lon path
                const latLngPath = data.path.map(([pr, pc]: [number, number]) => gridToLatLng(pr, pc));
                setPath(latLngPath);
            }
        } catch (err) {
            console.error("Pathfinding error:", err);
        }
    };

    return (
        <div className="w-full h-[600px] rounded-xl overflow-hidden border border-orange-500 relative">
            <MapContainer
                center={[29.5700, 78.9200]}
                zoom={14}
                style={{ height: '100%', width: '100%' }}
                scrollWheelZoom={true}
            >
                <TileLayer
                    attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                    url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                />

                {/* Fire Grid Overlay */}
                {imageUrl && (
                    <ImageOverlay
                        url={imageUrl}
                        bounds={BOUNDS}
                        opacity={0.7}
                    />
                )}

                {/* Safe Path */}
                {path.length > 0 && (
                    <Polyline
                        positions={path}
                        color="lime"
                        weight={5}
                        opacity={0.9}
                    />
                )}

                <MapEvents onMapClick={handleMapClick} />
            </MapContainer>

            <div className="absolute top-4 right-4 bg-slate-900/90 text-white p-4 rounded-lg z-[1000] max-w-xs">
                <h3 className="font-bold mb-2">Uttarakhand Forest Safe Route</h3>
                <p className="text-xs text-gray-300">
                    Click anywhere on the map to find a safe path to the sector center avoiding fire zones.
                </p>
                <div className="mt-2 flex items-center gap-2">
                    <div className="w-4 h-4 bg-red-600 opacity-70 rounded"></div>
                    <span className="text-xs">Fire Risk Zone</span>
                </div>
                <div className="mt-1 flex items-center gap-2">
                    <div className="w-4 h-1 bg-lime-400 rounded"></div>
                    <span className="text-xs">Safe Route (D* Lite)</span>
                </div>
            </div>
        </div>
    );
};

export default MapSimulation;
