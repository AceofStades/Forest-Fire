"use client";
import React, { useRef, useEffect, useState } from "react";
import {
    MapContainer,
    TileLayer,
    ImageOverlay,
    useMapEvents,
    Rectangle,
    Polyline,
    CircleMarker
} from "react-leaflet";
import { Navigation } from "@/components/navigation";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Switch } from "@/components/ui/switch";
import { Flame, MapPin, Flag, Route, Trash2, ShieldAlert, Play, Pause, FastForward } from "lucide-react";

interface EventData {
    probGrid: number[][];
    groundTruth: [number, number][][];
    initialFire: [number, number][];
    bounds: [[number, number], [number, number]];
}

// Coordinate mapping constants
const MIN_LAT = 28.71806;
const MAX_LAT = 31.49096;
const MIN_LON = 77.50902;
const MAX_LON = 81.08195;

const rowColToLatLng = (row: number, col: number): [number, number] => {
    const lat = MAX_LAT - (row / 320) * (MAX_LAT - MIN_LAT);
    const lng = MIN_LON + (col / 400) * (MAX_LON - MIN_LON);
    return [lat, lng];
};

function MapClickHandler({
    onMapClick,
}: {
    onMapClick: (row: number, col: number, eType: 'click' | 'contextmenu') => void;
}) {
    useMapEvents({
        click(e) {
            const lat = e.latlng.lat;
            const lng = e.latlng.lng;
            const row = Math.floor((1 - (lat - MIN_LAT) / (MAX_LAT - MIN_LAT)) * 320);
            const col = Math.floor(((lng - MIN_LON) / (MAX_LON - MIN_LON)) * 400);

            if (row >= 0 && row < 320 && col >= 0 && col < 400) {
                onMapClick(row, col, 'click');
            }
        },
        contextmenu(e) {
            const lat = e.latlng.lat;
            const lng = e.latlng.lng;
            const row = Math.floor((1 - (lat - MIN_LAT) / (MAX_LAT - MIN_LAT)) * 320);
            const col = Math.floor(((lng - MIN_LON) / (MAX_LON - MIN_LON)) * 400);

            if (row >= 0 && row < 320 && col >= 0 && col < 400) {
                onMapClick(row, col, 'contextmenu');
            }
        },
    });
    return null;
}

const WindOverlay = ({ speed, direction, isSandbox }: { speed: number; direction: number; isSandbox: boolean; }) => {
    if (!isSandbox || speed === 0) return null;
    const duration = Math.max(1.5, 8 - speed / 15);
    return (
        <div className="absolute inset-0 pointer-events-none z-[400] overflow-hidden opacity-20 flex items-center justify-center mix-blend-screen">
            <div className="w-[200%] h-[200%] absolute" style={{ transform: `rotate(${direction}deg)` }}>
                <div className="w-full h-full" style={{
                        backgroundImage: `url("data:image/svg+xml,%3Csvg width='200' height='300' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M 100 300 Q 120 250, 100 200 T 100 100 M 40 150 Q 60 100, 40 50' stroke='rgba(255,255,255,0.4)' stroke-width='1.5' fill='none' stroke-linecap='round' /%3E%3C/svg%3E")`,
                        backgroundSize: "200px 300px",
                        animation: `windWave ${duration}s ease-in-out infinite`,
                    }}
                />
            </div>
            <style>{`
                @keyframes windWave {
                    0% { background-position: 0 300px; opacity: 0.3; }
                    50% { opacity: 0.8; transform: translateX(10px); }
                    100% { background-position: 0 0; opacity: 0.3; }
                }
            `}</style>
        </div>
    );
};

export default function MapSimulation() {
    const [events, setEvents] = useState<any[]>([]);
    const [selectedEventId, setSelectedEventId] = useState<number | null>(null);
    const [eventData, setEventData] = useState<EventData | null>(null);
    const [timeStep, setTimeStep] = useState<number>(0);
    const [isPlaying, setIsPlaying] = useState<boolean>(false);
    const [viewMode, setViewMode] = useState<"predicted" | "actual" | "compare">("predicted");

    // UI State
    const [interactionMode, setInteractionMode] = useState<"ignite" | "start" | "goal">("ignite");

    // D* Lite Pathfinding State
    const [evacStart, setEvacStart] = useState<[number, number] | null>(null);
    const [evacGoal, setEvacGoal] = useState<[number, number] | null>(null);
    const [safePath, setSafePath] = useState<[number, number][]>([]);
    const [isCalculatingPath, setIsCalculatingPath] = useState(false);
    const [routingStatus, setRoutingStatus] = useState<string>("");

    // Sandbox Settings
    const [windSpeed, setWindSpeed] = useState<number>(10);
    const [windDir, setWindDir] = useState<number>(90);
    const [isSandbox, setIsSandbox] = useState<boolean>(false);
    const [humidity, setHumidity] = useState<number>(25);
    const [ignitionThreshold, setIgnitionThreshold] = useState<number>(0.3);
    const [mapLayer, setMapLayer] = useState<"dark" | "satellite" | "terrain" | "streets">("dark");
    const [imageUrl, setImageUrl] = useState<string>("");

    const ROWS = 320;
    const COLS = 400;
    const stateGrid = useRef(new Uint8Array(ROWS * COLS));
    const stateHistory = useRef<Uint8Array[]>([]);
    const offscreenCanvasRef = useRef<HTMLCanvasElement | null>(null);

    const DATASET_BOUNDS: [[number, number], [number, number]] = [
        [MIN_LAT, MIN_LON],
        [MAX_LAT, MAX_LON],
    ];

    useEffect(() => {
        fetch("/api/historical-events")
            .then((res) => res.json())
            .then((data) => setEvents(data))
            .catch((err) => console.error(err));

        const cvs = document.createElement("canvas");
        cvs.width = COLS;
        cvs.height = ROWS;
        offscreenCanvasRef.current = cvs;
    }, []);

    const calculateSafePath = async (startTuple: [number, number], goalTuple: [number, number]) => {
        if (!eventData) return;
        setIsCalculatingPath(true);
        setRoutingStatus("Routing...");

        // Collect all active fires from the current stateGrid
        const activeFires: [number, number][] = [];
        for (let r = 0; r < ROWS; r++) {
            for (let c = 0; c < COLS; c++) {
                if (stateGrid.current[r * COLS + c] === 1) {
                    activeFires.push([r, c]);
                }
            }
        }

        try {
            const apiKey = process.env.NEXT_PUBLIC_ORS_API_KEY;

            if (apiKey && apiKey.length > 10) {
                try {
                    setRoutingStatus("ORS Vector Routing...");
                    // OPENROUTESERVICE VECTOR ROUTING
                    const startLatLng = rowColToLatLng(startTuple[0], startTuple[1]);
                    const goalLatLng = rowColToLatLng(goalTuple[0], goalTuple[1]);

                    // FILTER FIRES BY RELEVANCE (Bounding box around start/goal)
                    // We only care about fires that are somewhat near the path.
                    const margin = 40; // ~30km margin around the direct line
                    const minPathR = Math.min(startTuple[0], goalTuple[0]) - margin;
                    const maxPathR = Math.max(startTuple[0], goalTuple[0]) + margin;
                    const minPathC = Math.min(startTuple[1], goalTuple[1]) - margin;
                    const maxPathC = Math.max(startTuple[1], goalTuple[1]) + margin;

                    const relevantFires = activeFires.filter(([r, c]) => {
                        return r >= minPathR && r <= maxPathR && c >= minPathC && c <= maxPathC;
                    });

                    // Cluster relevant fires into blocks to prevent overloading ORS limits
                    let blockSize = 2;
                    let blocks = new Map<string, {minR: number, maxR: number, minC: number, maxC: number}>();
                    
                    while (blockSize <= 6) { // Strict cap on size to NEVER create massive area polygons
                        blocks.clear();
                        for(const [r, c] of relevantFires) {
                            const br = Math.floor(r / blockSize);
                            const bc = Math.floor(c / blockSize);
                            const key = `${br},${bc}`;
                            if (!blocks.has(key)) {
                                blocks.set(key, {minR: r, maxR: r, minC: c, maxC: c});
                            } else {
                                const b = blocks.get(key)!;
                                b.minR = Math.min(b.minR, r);
                                b.maxR = Math.max(b.maxR, r);
                                b.minC = Math.min(b.minC, c);
                                b.maxC = Math.max(b.maxC, c);
                            }
                        }
                        
                        if (blocks.size <= 20) break;
                        blockSize += 2;
                    }

                    // Convert blocks to an array
                    let sortedBlocks = Array.from(blocks.values());
                    
                    // If we still have > 20 blocks, prioritize those closest to the route's midpoint
                    if (sortedBlocks.length > 20) {
                        const midR = (startTuple[0] + goalTuple[0]) / 2;
                        const midC = (startTuple[1] + goalTuple[1]) / 2;
                        sortedBlocks.sort((a, b) => {
                            const centerA_R = (a.minR + a.maxR) / 2;
                            const centerA_C = (a.minC + a.maxC) / 2;
                            const distA = Math.pow(centerA_R - midR, 2) + Math.pow(centerA_C - midC, 2);
                            
                            const centerB_R = (b.minR + b.maxR) / 2;
                            const centerB_C = (b.minC + b.maxC) / 2;
                            const distB = Math.pow(centerB_R - midR, 2) + Math.pow(centerB_C - midC, 2);
                            
                            return distA - distB;
                        });
                        sortedBlocks = sortedBlocks.slice(0, 20);
                    }

                    const polygons = sortedBlocks.map(b => {
                        // Pad the bounding box by 1 pixel to ensure a safe distance
                        const p1 = rowColToLatLng(Math.max(0, b.minR - 1), Math.max(0, b.minC - 1)); // NW
                        const p2 = rowColToLatLng(Math.max(0, b.minR - 1), Math.min(COLS - 1, b.maxC + 1)); // NE
                        const p3 = rowColToLatLng(Math.min(ROWS - 1, b.maxR + 1), Math.min(COLS - 1, b.maxC + 1)); // SE
                        const p4 = rowColToLatLng(Math.min(ROWS - 1, b.maxR + 1), Math.max(0, b.minC - 1)); // SW
                        
                        // ORS expects [lon, lat] format for GeoJSON
                        // MUST BE COUNTER-CLOCKWISE! NW -> SW -> SE -> NE -> NW
                        return [[
                            [p1[1], p1[0]], // NW
                            [p4[1], p4[0]], // SW
                            [p3[1], p3[0]], // SE
                            [p2[1], p2[0]], // NE
                            [p1[1], p1[0]]  // NW (Close the loop)
                        ]];
                    });

                    const avoidPolygons = polygons.length > 0 ? polygons : undefined;

                    const requestBody: any = {
                        coordinates: [
                            [startLatLng[1], startLatLng[0]],
                            [goalLatLng[1], goalLatLng[0]]
                        ],
                        radiuses: [-1, -1]
                    };

                    if (avoidPolygons) {
                        requestBody.options = {
                            avoid_polygons: {
                                coordinates: avoidPolygons,
                                type: "MultiPolygon"
                            }
                        };
                    }

                    const res = await fetch("https://api.openrouteservice.org/v2/directions/driving-car/geojson", {
                        method: "POST",
                        headers: {
                            "Authorization": apiKey,
                            "Content-Type": "application/json"
                        },
                        body: JSON.stringify(requestBody)
                    });

                    const data = await res.json();
                    
                    if (data.features && data.features.length > 0) {
                        const coords = data.features[0].geometry.coordinates;
                        const latLngPath = coords.map((c: number[]) => [c[1], c[0]] as [number, number]);
                        setSafePath(latLngPath);
                        setRoutingStatus("ORS Route Active");
                        setIsCalculatingPath(false);
                        return; // Exit successfully
                    } else {
                        const errorMsg = data.error?.message || "No Route Found";
                        if (errorMsg.toLowerCase().includes("area of polygon")) {
                            console.warn("ORS area limit exceeded, silently falling back to D* Lite");
                            setRoutingStatus("Fire too large for ORS. Using D* Lite...");
                        } else {
                            console.warn("ORS routing failed, falling back to D* Lite...", data);
                            setRoutingStatus(`ORS Failed. Using D* Lite...`);
                        }
                    }
                } catch (orsError) {
                    console.error("ORS API Error:", orsError);
                    setRoutingStatus("ORS Error. Falling back to D* Lite...");
                }
            } else {
                setRoutingStatus("ORS Key Missing. Fallback D* Lite...");
            }

            // FALLBACK TO BACKEND D* LITE
            const response = await fetch("/api/get-safe-path", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ 
                    start: startTuple, 
                    goal: goalTuple,
                    active_fires: activeFires 
                }),
            });
            const data = await response.json();
            if (data.path) {
                // Convert row/col to lat/lng for Polyline
                const latLngPath = data.path.map((coord: [number, number]) => rowColToLatLng(coord[0], coord[1]));
                setSafePath(latLngPath);
                if (!routingStatus.includes("ORS Key Missing") && !routingStatus.includes("ORS Failed")) {
                     setRoutingStatus("D* Lite Route Active");
                }
            }
        } catch (error: any) {
            console.error("Pathfinding error:", error);
            setRoutingStatus("Routing Error");
        } finally {
            setIsCalculatingPath(false);
        }
    };

    // Auto trigger path calculation when both points are set
    useEffect(() => {
        if (evacStart && evacGoal) {
            calculateSafePath(evacStart, evacGoal);
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [evacStart, evacGoal, timeStep]); // Re-calc on step change too

    const loadEvent = async (id: number) => {
        setSelectedEventId(id);
        setIsPlaying(false);
        setTimeStep(0);
        setIsSandbox(false);
        clearPath();
        try {
            const res = await fetch(`/api/event-data/${id}?hours=30`);
            const data: EventData = await res.json();
            setEventData(data);

            stateGrid.current.fill(0);
            data.initialFire.forEach(([r, c]) => {
                stateGrid.current[r * COLS + c] = 1;
            });
            stateHistory.current = [new Uint8Array(stateGrid.current)];
            renderCanvas(data);
        } catch (err) {
            console.error(err);
        }
    };

    const enableSandbox = () => {
        setIsSandbox(true);
        setIsPlaying(false);
        setTimeStep(0);
        setEventData(null);
        clearPath();
        stateGrid.current.fill(0);
        stateHistory.current = [new Uint8Array(stateGrid.current)];

        fetch("/api/fire-grid")
            .then((res) => res.json())
            .then((data) => {
                const newData: EventData = {
                    probGrid: data.grid,
                    groundTruth: [],
                    initialFire: [],
                    bounds: DATASET_BOUNDS,
                };
                setEventData(newData);
                renderCanvas(newData);
            })
            .catch((err) => console.error("Error fetching fire grid", err));
    };

    const handleMapClick = (r: number, c: number, eType: 'click' | 'contextmenu') => {
        if (!eventData) return;

        // Right click always ignites fire (legacy support + quick action)
        if (eType === 'contextmenu') {
            if (!isSandbox) return;
            stateGrid.current[r * COLS + c] = 1;
            stateHistory.current = stateHistory.current.slice(0, timeStep + 1);
            stateHistory.current[timeStep] = new Uint8Array(stateGrid.current);
            renderCanvas(eventData);
            return;
        }

        // Left click depends on interactionMode
        if (interactionMode === 'ignite') {
            if (!isSandbox) return;
            stateGrid.current[r * COLS + c] = 1;
            stateHistory.current = stateHistory.current.slice(0, timeStep + 1);
            stateHistory.current[timeStep] = new Uint8Array(stateGrid.current);
            renderCanvas(eventData);
        } else if (interactionMode === 'start') {
            setEvacStart([r, c]);
        } else if (interactionMode === 'goal') {
            setEvacGoal([r, c]);
        }
    };

    const clearPath = () => {
        setEvacStart(null);
        setEvacGoal(null);
        setSafePath([]);
        setRoutingStatus("");
    };

    const runCAStep = () => {
        if (!eventData) return;
        const nextState = new Uint8Array(stateGrid.current);
        const { probGrid } = eventData;

        for (let r = 0; r < ROWS; r++) {
            for (let c = 0; c < COLS; c++) {
                const idx = r * COLS + c;
                if (stateGrid.current[idx] === 0) {
                    let hasNeighbor = false;
                    let neighborR = -1;
                    let neighborC = -1;

                    for (let dr = -1; dr <= 1; dr++) {
                        for (let dc = -1; dc <= 1; dc++) {
                            if (dr === 0 && dc === 0) continue;
                            const nr = r + dr;
                            const nc = c + dc;
                            if (nr >= 0 && nr < ROWS && nc >= 0 && nc < COLS) {
                                if (stateGrid.current[nr * COLS + nc] === 1) {
                                    hasNeighbor = true;
                                    neighborR = nr;
                                    neighborC = nc;
                                    break;
                                }
                            }
                        }
                        if (hasNeighbor) break;
                    }

                    if (hasNeighbor) {
                        let prob = (probGrid && probGrid[r] && probGrid[r][c]) !== undefined ? probGrid[r][c] : 0.05;

                        if (isSandbox) {
                            prob = prob * 1.5;
                            const hf = 1 - (humidity / 100) * 0.8; 
                            prob *= hf;

                            const spreadVecR = r - neighborR;
                            const spreadVecC = c - neighborC;
                            const angle = Math.atan2(spreadVecR, spreadVecC) * (180 / Math.PI);
                            let diff = Math.abs(((angle - windDir + 540) % 360) - 180);
                            const windBias = Math.cos((diff * Math.PI) / 180); 
                            
                            const wf = 1 + windBias * (windSpeed / 20); 
                            prob *= Math.max(0.1, wf);

                            if (prob < ignitionThreshold) {
                                prob *= 0.1;
                            }
                        } else {
                            prob = prob * 1.5 + 0.01;
                        }

                        if (Math.random() < prob) {
                            nextState[idx] = 1;
                        }
                    }
                } else if (stateGrid.current[idx] === 1) {
                    if (isSandbox) {
                        if (Math.random() < 0.1) nextState[idx] = 2;
                    } else {
                        if (Math.random() < 0.7) nextState[idx] = 2;
                    }
                }
            }
        }

        if (!isSandbox && eventData.groundTruth && timeStep + 1 < eventData.groundTruth.length) {
            const actualFires = eventData.groundTruth[timeStep + 1];
            if (actualFires) {
                actualFires.forEach(([r, c]) => {
                    nextState[r * COLS + c] = 1;
                });
            }
        }

        stateGrid.current = nextState;
        stateHistory.current = stateHistory.current.slice(0, timeStep + 1);
        stateHistory.current.push(new Uint8Array(nextState));

        setTimeStep((prev) => prev + 1);
        renderCanvas(eventData, timeStep + 1);
    };

    useEffect(() => {
        let interval: NodeJS.Timeout;
        if (isPlaying) {
            interval = setInterval(() => {
                if (!isSandbox && timeStep >= 30) {
                    setIsPlaying(false);
                } else {
                    runCAStep();
                }
            }, 500);
        }
        return () => clearInterval(interval);
    }, [isPlaying, isSandbox, timeStep, eventData]);

    const renderCanvas = (data: EventData, currentTime: number = timeStep) => {
        const cvs = offscreenCanvasRef.current;
        if (!cvs) return;
        const ctx = cvs.getContext("2d");
        if (!ctx) return;

        ctx.clearRect(0, 0, COLS, ROWS);

        for (let r = 0; r < ROWS; r++) {
            for (let c = 0; c < COLS; c++) {
                const state = stateGrid.current[r * COLS + c];

                if (viewMode === "predicted" || viewMode === "compare") {
                    if (state === 1) {
                        ctx.fillStyle = viewMode === "compare" ? "rgba(255, 0, 0, 0.6)" : "rgba(255, 69, 0, 0.8)";
                        ctx.fillRect(c, r, 1, 1);
                    } else if (state === 2) {
                        ctx.fillStyle = "rgba(80, 80, 80, 0.5)";
                        ctx.fillRect(c, r, 1, 1);
                    }
                }
            }
        }

        if (!isSandbox && (viewMode === "actual" || viewMode === "compare")) {
            const actualTime = Math.min(currentTime, data.groundTruth.length - 1);
            if (data.groundTruth[actualTime]) {
                data.groundTruth[actualTime].forEach(([r, c]) => {
                    if (viewMode === "compare") {
                        const isPredicted = stateGrid.current[r * COLS + c] === 1;
                        if (isPredicted) {
                            ctx.fillStyle = "rgba(128, 0, 128, 0.8)";
                        } else {
                            ctx.fillStyle = "rgba(0, 255, 0, 0.6)";
                        }
                    } else {
                        ctx.fillStyle = "rgba(255, 0, 0, 0.8)";
                    }
                    ctx.fillRect(c, r, 1, 1);
                });
            }
        }

        setImageUrl(cvs.toDataURL());
    };

    useEffect(() => {
        if (eventData) renderCanvas(eventData, timeStep);
    }, [viewMode, eventData]);

    return (
        <div className="flex flex-col min-h-screen bg-slate-950 text-white">
            <Navigation />
            <div className="flex flex-1 overflow-hidden">
                {/* Left Sidebar */}
                <div className="w-80 bg-slate-900 border-r border-slate-800 p-6 flex flex-col gap-6 overflow-y-auto relative z-20 shadow-2xl">
                    <h2 className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-orange-400 to-red-500 flex items-center gap-2">
                        <Flame className="w-5 h-5 text-orange-500" /> Fire Dynamics
                    </h2>

                    <Tabs defaultValue="historical" className="w-full" onValueChange={(v) => { if (v === "sandbox") enableSandbox(); }}>
                        <TabsList className="grid w-full grid-cols-2 bg-slate-800 text-slate-400 mb-4 p-1 rounded-lg">
                            <TabsTrigger value="historical" className="rounded-md data-[state=active]:bg-slate-700 data-[state=active]:text-white">Historical</TabsTrigger>
                            <TabsTrigger value="sandbox" className="rounded-md data-[state=active]:bg-blue-600 data-[state=active]:text-white">Sandbox</TabsTrigger>
                        </TabsList>

                        <TabsContent value="historical" className="space-y-3">
                            <div className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-2 flex items-center gap-2">
                                <Route className="w-4 h-4" /> Validated Events
                            </div>
                            <div className="space-y-2 max-h-[300px] overflow-y-auto pr-1">
                                {events.map((ev) => (
                                    <Button
                                        key={ev.id}
                                        variant={selectedEventId === ev.id && !isSandbox ? "default" : "outline"}
                                        className={`w-full justify-start ${selectedEventId === ev.id && !isSandbox ? 'bg-orange-600 hover:bg-orange-700 border-none' : 'border-slate-700 hover:bg-slate-800'}`}
                                        onClick={() => loadEvent(ev.id)}
                                    >
                                        {ev.name}
                                    </Button>
                                ))}
                            </div>
                        </TabsContent>

                        <TabsContent value="sandbox" className="space-y-4">
                            <div className="p-4 bg-slate-800/50 rounded-lg border border-slate-700 space-y-5 text-sm">
                                <div className="space-y-2">
                                    <div className="flex justify-between font-medium">
                                        <span className="text-slate-300">Wind Speed</span>
                                        <span className="text-orange-400">{windSpeed} km/h</span>
                                    </div>
                                    <input type="range" min="0" max="100" value={windSpeed} onChange={(e) => setWindSpeed(Number(e.target.value))} className="w-full accent-orange-500" />
                                </div>

                                <div className="space-y-2">
                                    <div className="flex justify-between font-medium">
                                        <span className="text-slate-300">Wind Direction</span>
                                        <span className="text-blue-400">{windDir}°</span>
                                    </div>
                                    <input type="range" min="0" max="360" value={windDir} onChange={(e) => setWindDir(Number(e.target.value))} className="w-full accent-blue-500" />
                                </div>

                                <div className="space-y-2">
                                    <div className="flex justify-between font-medium">
                                        <span className="text-slate-300">Humidity</span>
                                        <span className="text-teal-400">{humidity}%</span>
                                    </div>
                                    <input type="range" min="0" max="100" value={humidity} onChange={(e) => setHumidity(Number(e.target.value))} className="w-full accent-teal-500" />
                                </div>

                                <div className="space-y-2">
                                    <div className="flex justify-between font-medium">
                                        <span className="text-slate-300">Ignition Threshold</span>
                                        <span className="text-red-400">{ignitionThreshold.toFixed(2)}</span>
                                    </div>
                                    <input type="range" min="0" max="1" step="0.05" value={ignitionThreshold} onChange={(e) => setIgnitionThreshold(Number(e.target.value))} className="w-full accent-red-500" />
                                </div>
                            </div>
                        </TabsContent>
                    </Tabs>

                    <div className="mt-auto space-y-4">
                        <div className="pt-4 border-t border-slate-800">
                            <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">Map Topology</h3>
                            <div className="grid grid-cols-2 gap-2">
                                <Button variant={mapLayer === "dark" ? "secondary" : "outline"} onClick={() => setMapLayer("dark")} className="h-10 text-xs bg-slate-800 border-slate-700">Dark</Button>
                                <Button variant={mapLayer === "satellite" ? "secondary" : "outline"} onClick={() => setMapLayer("satellite")} className="h-10 text-xs bg-slate-800 border-slate-700">Sat</Button>
                                <Button variant={mapLayer === "terrain" ? "secondary" : "outline"} onClick={() => setMapLayer("terrain")} className="h-10 text-xs bg-slate-800 border-slate-700">Terr</Button>
                                <Button variant={mapLayer === "streets" ? "secondary" : "outline"} onClick={() => setMapLayer("streets")} className="h-10 text-xs bg-slate-800 border-slate-700">Streets</Button>
                            </div>
                        </div>

                        {!isSandbox && (
                            <div className="space-y-2 pt-2">
                                <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Analysis Mode</span>
                                <div className="flex bg-slate-800 rounded-md p-1">
                                    {(["predicted", "actual", "compare"] as const).map((mode) => (
                                        <button key={mode} onClick={() => setViewMode(mode)} className={`flex-1 text-xs py-1.5 rounded capitalize transition-all ${viewMode === mode ? "bg-slate-700 text-white shadow-sm" : "text-slate-400 hover:text-white"}`}>{mode}</button>
                                    ))}
                                </div>
                            </div>
                        )}
                    </div>
                </div>

                {/* Main Map Area */}
                <div className="flex-1 relative bg-slate-950 overflow-hidden flex flex-col">
                    
                    {/* Top Toolbar (Floating inside Map) */}
                    <div className="absolute top-4 right-4 z-[500] flex flex-col items-end gap-2">
                        <div className="flex gap-2">
                            {isSandbox && (
                                <div className="flex bg-slate-900/90 backdrop-blur-md border border-slate-700 rounded-lg p-1 shadow-xl">
                                    <Button 
                                        variant={interactionMode === 'ignite' ? 'default' : 'ghost'} 
                                        size="sm" 
                                        onClick={() => setInteractionMode('ignite')}
                                        className={interactionMode === 'ignite' ? 'bg-orange-600 hover:bg-orange-700' : 'text-slate-400 hover:text-orange-400'}
                                        title="Ignite Fire (Left Click)"
                                    >
                                        <Flame className="w-4 h-4 mr-1" /> Ignite
                                    </Button>
                                </div>
                            )}

                            <div className="flex bg-slate-900/90 backdrop-blur-md border border-slate-700 rounded-lg p-1 shadow-xl">
                                <Button 
                                    variant={interactionMode === 'start' ? 'secondary' : 'ghost'} 
                                    size="sm" 
                                    onClick={() => setInteractionMode('start')}
                                    className={interactionMode === 'start' ? 'bg-emerald-900/50 text-emerald-400 hover:bg-emerald-900/70' : 'text-slate-400 hover:text-emerald-400'}
                                >
                                    <MapPin className="w-4 h-4 mr-1" /> Set Evac Start
                                </Button>
                                <Button 
                                    variant={interactionMode === 'goal' ? 'secondary' : 'ghost'} 
                                    size="sm" 
                                    onClick={() => setInteractionMode('goal')}
                                    className={interactionMode === 'goal' ? 'bg-blue-900/50 text-blue-400 hover:bg-blue-900/70' : 'text-slate-400 hover:text-blue-400'}
                                >
                                    <Flag className="w-4 h-4 mr-1" /> Set Evac Goal
                                </Button>
                                
                                {(evacStart || evacGoal || safePath.length > 0) && (
                                    <div className="w-px bg-slate-700 mx-1 my-1"></div>
                                )}
                                {(evacStart || evacGoal || safePath.length > 0) && (
                                    <Button variant="ghost" size="sm" onClick={clearPath} className="text-red-400 hover:text-red-300 hover:bg-red-950/50" title="Clear Path">
                                        <Trash2 className="w-4 h-4" />
                                    </Button>
                                )}
                            </div>
                        </div>
                        
                        {routingStatus && (
                            <div className={`flex items-center px-3 py-1.5 bg-slate-900/90 backdrop-blur-md border rounded-lg text-sm font-medium shadow-xl ${
                                routingStatus.includes("ORS") && !routingStatus.includes("Failed") 
                                    ? "border-emerald-500/50 text-emerald-400" 
                                    : routingStatus.includes("Routing...") 
                                        ? "border-indigo-500/50 text-indigo-400 animate-pulse"
                                        : "border-orange-500/50 text-orange-400"
                            }`}>
                                <Route className="w-4 h-4 mr-2" /> {routingStatus}
                            </div>
                        )}
                    </div>

                    <div className="flex-1 relative z-0">
                        <MapContainer center={[30.1, 79.2]} zoom={8} style={{ height: "100%", width: "100%", background: "#0f172a" }}>
                            {mapLayer === "dark" && <TileLayer url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png" />}
                            {mapLayer === "satellite" && <TileLayer url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}" />}
                            {mapLayer === "terrain" && <TileLayer url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Terrain_Base/MapServer/tile/{z}/{y}/{x}" />}
                            {mapLayer === "streets" && <TileLayer url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}" />}

                            <Rectangle bounds={DATASET_BOUNDS} pathOptions={{ color: "#3b82f6", weight: 2, fill: false, dashArray: "5, 10" }} />

                            {eventData && imageUrl && <ImageOverlay url={imageUrl} bounds={eventData.bounds} opacity={1.0} zIndex={10} />}

                            {/* D* Lite Overlays */}
                            {safePath.length > 0 && (
                                <Polyline positions={safePath} pathOptions={{ color: '#10b981', weight: 4, dashArray: '10, 10', lineCap: 'round' }} />
                            )}
                            {evacStart && (
                                <CircleMarker center={rowColToLatLng(evacStart[0], evacStart[1])} radius={6} pathOptions={{ color: '#10b981', fillColor: '#10b981', fillOpacity: 1, weight: 2 }} />
                            )}
                            {evacGoal && (
                                <CircleMarker center={rowColToLatLng(evacGoal[0], evacGoal[1])} radius={6} pathOptions={{ color: '#3b82f6', fillColor: '#3b82f6', fillOpacity: 1, weight: 2 }} />
                            )}

                            <MapClickHandler onMapClick={handleMapClick} />
                        </MapContainer>
                        <WindOverlay speed={windSpeed} direction={windDir} isSandbox={isSandbox} />
                    </div>

                    {/* Bottom Playback Bar */}
                    {eventData && (
                        <div className="h-20 bg-slate-900 border-t border-slate-800 z-20 flex items-center px-6 gap-6 shadow-[0_-10px_40px_-10px_rgba(0,0,0,0.5)]">
                            <div className="flex gap-2">
                                <Button onClick={() => setIsPlaying(!isPlaying)} size="icon" className={`h-12 w-12 rounded-full ${isPlaying ? "bg-red-600 hover:bg-red-700 text-white" : "bg-emerald-600 hover:bg-emerald-700 text-white"}`}>
                                    {isPlaying ? <Pause className="w-5 h-5 fill-current" /> : <Play className="w-5 h-5 fill-current ml-1" />}
                                </Button>
                                <Button variant="outline" size="icon" onClick={runCAStep} disabled={isPlaying} className="h-12 w-12 rounded-full border-slate-700 bg-slate-800 hover:bg-slate-700">
                                    <FastForward className="w-5 h-5 text-slate-300" />
                                </Button>
                            </div>

                            <div className="flex-1 flex items-center gap-4">
                                <div className="text-slate-400 font-mono text-sm w-20">T + {timeStep}D</div>
                                <input
                                    type="range"
                                    min="0"
                                    max={stateHistory.current.length > 0 ? stateHistory.current.length - 1 : 0}
                                    value={timeStep}
                                    onChange={(e) => {
                                        const newStep = parseInt(e.target.value, 10);
                                        if (stateHistory.current[newStep]) {
                                            setIsPlaying(false);
                                            setTimeStep(newStep);
                                            stateGrid.current = new Uint8Array(stateHistory.current[newStep]);
                                            renderCanvas(eventData, newStep);
                                        }
                                    }}
                                    className="flex-1 h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-orange-500 hover:h-3 transition-all"
                                />
                            </div>

                            <div className="flex items-center gap-3 w-48 border-l border-slate-800 pl-6">
                                <ShieldAlert className={`w-5 h-5 ${isSandbox ? (humidity < 30 && windSpeed > 40 ? "text-red-500" : "text-orange-500") : "text-red-500"}`} />
                                <div>
                                    <div className="text-[10px] uppercase text-slate-500 font-bold">Fire Danger</div>
                                    <div className={`text-sm font-bold ${isSandbox ? (humidity < 30 && windSpeed > 40 ? "text-red-500" : "text-orange-500") : "text-red-500"}`}>
                                        {isSandbox ? (humidity < 30 && windSpeed > 40 ? "EXTREME" : "HIGH") : "HIGH"}
                                    </div>
                                </div>
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
