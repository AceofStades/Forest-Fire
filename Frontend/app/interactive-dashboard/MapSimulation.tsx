"use client";
import React, { useRef, useEffect, useState } from "react";
import {
    MapContainer,
    TileLayer,
    ImageOverlay,
    useMapEvents,
    Rectangle,
} from "react-leaflet";
import { Navigation } from "@/components/navigation";
import { Button } from "@/components/ui/button";

interface EventData {
    probGrid: number[][];
    groundTruth: [number, number][][];
    initialFire: [number, number][];
    bounds: [[number, number], [number, number]];
}

// Separate component to handle map right-clicks and interactions
function MapClickHandler({
    onMapClick,
}: {
    onMapClick: (row: number, col: number) => void;
}) {
    useMapEvents({
        contextmenu(e) {
            // Uses right-click to avoid interfering with map panning
            const lat = e.latlng.lat;
            const lng = e.latlng.lng;

            const minLat = 28.71806;
            const maxLat = 31.49096;
            const minLon = 77.50902;
            const maxLon = 81.08195;

            // Map standard: index [0,0] is top-left (maxLat, minLon)
            const row = Math.floor(
                (1 - (lat - minLat) / (maxLat - minLat)) * 320,
            );
            const col = Math.floor(((lng - minLon) / (maxLon - minLon)) * 400);

            if (row >= 0 && row < 320 && col >= 0 && col < 400) {
                onMapClick(row, col);
            }
        },
    });
    return null;
}

const WindOverlay = ({
    speed,
    direction,
    isSandbox,
}: {
    speed: number;
    direction: number;
    isSandbox: boolean;
}) => {
    if (!isSandbox || speed === 0) return null;

    // Slower, more gentle movement
    const duration = Math.max(1.5, 8 - speed / 15);

    return (
        <div className="absolute inset-0 pointer-events-none z-[400] overflow-hidden opacity-20 flex items-center justify-center mix-blend-screen">
            <div
                className="w-[200%] h-[200%] absolute"
                style={{ transform: `rotate(${direction}deg)` }}
            >
                <div
                    className="w-full h-full"
                    style={{
                        // Wider spacing, curved paths for wave effect
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
    const [viewMode, setViewMode] = useState<
        "predicted" | "actual" | "compare"
    >("predicted");

    // Environmental Controls for Sandbox
    const [windSpeed, setWindSpeed] = useState<number>(10);
    const [windDir, setWindDir] = useState<number>(90);
    const [isSandbox, setIsSandbox] = useState<boolean>(false);

    const [imageUrl, setImageUrl] = useState<string>("");

    // CA State Reference
    const ROWS = 320;
    const COLS = 400;
    const stateGrid = useRef(new Uint8Array(ROWS * COLS));
    const stateHistory = useRef<Uint8Array[]>([]);
    const offscreenCanvasRef = useRef<HTMLCanvasElement | null>(null);

    const DATASET_BOUNDS: [[number, number], [number, number]] = [
        [28.71806, 77.50902],
        [31.49096, 81.08195],
    ];

    useEffect(() => {
        fetch("http://127.0.0.1:8000/historical-events")
            .then((res) => res.json())
            .then((data) => setEvents(data))
            .catch((err) => console.error(err));

        const cvs = document.createElement("canvas");
        cvs.width = COLS;
        cvs.height = ROWS;
        offscreenCanvasRef.current = cvs;
    }, []);

    const loadEvent = async (id: number) => {
        setSelectedEventId(id);
        setIsPlaying(false);
        setTimeStep(0);
        setIsSandbox(false);
        try {
            const res = await fetch(
                `http://127.0.0.1:8000/event-data/${id}?hours=30`,
            );
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
        stateGrid.current.fill(0);
        stateHistory.current = [new Uint8Array(stateGrid.current)];

        fetch("http://127.0.0.1:8000/fire-grid")
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

    const handleMapClick = (r: number, c: number) => {
        if (!isSandbox || !eventData) return;
        stateGrid.current[r * COLS + c] = 1;
        // If we click the map, we truncate future history
        stateHistory.current = stateHistory.current.slice(0, timeStep + 1);
        stateHistory.current[timeStep] = new Uint8Array(stateGrid.current);
        renderCanvas(eventData);
    };

    const handleScrub = (e: React.ChangeEvent<HTMLInputElement>) => {
        const newStep = parseInt(e.target.value, 10);
        if (stateHistory.current[newStep]) {
            setIsPlaying(false);
            setTimeStep(newStep);
            stateGrid.current = new Uint8Array(stateHistory.current[newStep]);
            renderCanvas(eventData!, newStep);
        }
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

                    // Check 8-way neighbors
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
                        let prob =
                            (probGrid && probGrid[r] && probGrid[r][c]) !==
                            undefined
                                ? probGrid[r][c]
                                : 0.05;

                        if (isSandbox) {
                            // In sandbox mode, we want the fire to be highly responsive to user clicks and wind.
                            prob = prob * 1.5;

                            const windAngleRad =
                                (windDir - 90) * (Math.PI / 180);
                            const windVecR = Math.sin(windAngleRad);
                            const windVecC = Math.cos(windAngleRad);
                            const spreadVecR = r - neighborR;
                            const spreadVecC = c - neighborC;
                            const dotProduct =
                                spreadVecR * windVecR + spreadVecC * windVecC;

                            if (dotProduct > 0) {
                                prob += (windSpeed / 100) * 0.5 * dotProduct;
                            } else {
                                prob -=
                                    (windSpeed / 100) *
                                    0.2 *
                                    Math.abs(dotProduct);
                            }
                        } else {
                            // Historical mode: gentle base probability multiplier for realistic 24-hour spread
                            prob = prob * 1.5 + 0.01;
                        }

                        if (Math.random() < prob) {
                            nextState[idx] = 1;
                        }
                    }
                } else if (stateGrid.current[idx] === 1) {
                    // Burn out logic: Since each step is a full 24 hours, fire should naturally exhaust its local fuel quickly.
                    // 70% chance a pixel finishes burning and turns to ash (state 2) in a single day.
                    if (Math.random() < 0.7) nextState[idx] = 2; // Burn out
                }
            }
        }

        // If in historical mode, inject actual new satellite fires into the simulation so it can predict their subsequent spread
        if (
            !isSandbox &&
            eventData.groundTruth &&
            timeStep + 1 < eventData.groundTruth.length
        ) {
            const actualFires = eventData.groundTruth[timeStep + 1];
            if (actualFires) {
                actualFires.forEach(([r, c]) => {
                    // Inject spontaneous/new real-world ignitions into the simulation
                    nextState[r * COLS + c] = 1;
                });
            }
        }

        stateGrid.current = nextState;

        // Truncate any future history if we generated a new branch, then add new state
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
        // eslint-disable-next-line react-hooks/exhaustive-deps
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
                        ctx.fillStyle =
                            viewMode === "compare"
                                ? "rgba(255, 0, 0, 0.6)"
                                : "rgba(255, 69, 0, 0.8)";
                        ctx.fillRect(c, r, 1, 1);
                    } else if (state === 2) {
                        ctx.fillStyle = "rgba(80, 80, 80, 0.5)";
                        ctx.fillRect(c, r, 1, 1);
                    }
                }
            }
        }

        if (!isSandbox && (viewMode === "actual" || viewMode === "compare")) {
            const actualTime = Math.min(
                currentTime,
                data.groundTruth.length - 1,
            );
            if (data.groundTruth[actualTime]) {
                data.groundTruth[actualTime].forEach(([r, c]) => {
                    if (viewMode === "compare") {
                        const isPredicted =
                            stateGrid.current[r * COLS + c] === 1;
                        if (isPredicted) {
                            ctx.fillStyle = "rgba(128, 0, 128, 0.8)"; // Purple for overlap
                        } else {
                            ctx.fillStyle = "rgba(0, 255, 0, 0.6)"; // Green for actual only
                        }
                    } else {
                        ctx.fillStyle = "rgba(255, 0, 0, 0.8)"; // Red for actual mode
                    }
                    ctx.fillRect(c, r, 1, 1);
                });
            }
        }

        setImageUrl(cvs.toDataURL());
    };

    useEffect(() => {
        if (eventData) renderCanvas(eventData, timeStep);
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [viewMode, eventData]);

    return (
        <div className="flex flex-col min-h-screen bg-slate-950 text-white">
            <Navigation />
            <div className="flex flex-1 overflow-hidden">
                {/* Sidebar */}
                <div className="w-80 bg-slate-900 border-r border-slate-800 p-6 flex flex-col gap-6 overflow-y-auto">
                    <h2 className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-orange-400 to-red-500">
                        Fire Dynamics Engine
                    </h2>

                    {/* Historical Events */}
                    <div className="space-y-3">
                        <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wider">
                            Historical Scenarios
                        </h3>
                        {events.map((ev) => (
                            <Button
                                key={ev.id}
                                variant={
                                    selectedEventId === ev.id
                                        ? "default"
                                        : "outline"
                                }
                                className="w-full justify-start border-slate-700"
                                onClick={() => loadEvent(ev.id)}
                            >
                                {ev.name}
                            </Button>
                        ))}
                    </div>

                    <div className="h-px bg-slate-800 my-2"></div>

                    {/* Sandbox Mode */}
                    <div className="space-y-4">
                        <Button
                            variant={isSandbox ? "default" : "outline"}
                            className="w-full border-blue-600/50 hover:bg-blue-900/30"
                            onClick={enableSandbox}
                        >
                            Interactive Sandbox Mode
                        </Button>

                        {isSandbox && (
                            <div className="p-4 bg-slate-800/50 rounded-lg border border-slate-700 space-y-4 text-sm">
                                <p className="text-slate-300 mb-2 font-semibold text-blue-400">
                                    Right-Click on the map to ignite a fire.
                                </p>

                                <div className="space-y-2">
                                    <div className="flex justify-between">
                                        <span>Wind Speed</span>
                                        <span className="text-orange-400">
                                            {windSpeed} km/h
                                        </span>
                                    </div>
                                    <input
                                        type="range"
                                        min="0"
                                        max="100"
                                        value={windSpeed}
                                        onChange={(e) =>
                                            setWindSpeed(Number(e.target.value))
                                        }
                                        className="w-full accent-orange-500"
                                    />
                                </div>

                                <div className="space-y-2 pt-2">
                                    <div className="flex justify-between">
                                        <span>Wind Direction</span>
                                        <span className="text-blue-400">
                                            {windDir}°
                                        </span>
                                    </div>
                                    <input
                                        type="range"
                                        min="0"
                                        max="360"
                                        value={windDir}
                                        onChange={(e) =>
                                            setWindDir(Number(e.target.value))
                                        }
                                        className="w-full accent-blue-500"
                                    />
                                </div>
                            </div>
                        )}
                    </div>

                    {/* Controls */}
                    {eventData && (
                        <div className="mt-auto space-y-4 bg-slate-900 border-t border-slate-800 pt-6">
                            <div className="flex justify-between items-center text-sm">
                                <span className="text-slate-400">
                                    Simulation Time:
                                </span>
                                <span className="font-mono text-orange-400 font-bold">
                                    T + {timeStep} Days
                                </span>
                            </div>

                            <input
                                type="range"
                                min="0"
                                max={
                                    stateHistory.current.length > 0
                                        ? stateHistory.current.length - 1
                                        : 0
                                }
                                value={timeStep}
                                onChange={handleScrub}
                                className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-orange-500"
                            />

                            <div className="grid grid-cols-2 gap-2">
                                <Button
                                    onClick={() => setIsPlaying(!isPlaying)}
                                    className={
                                        isPlaying
                                            ? "bg-red-600 hover:bg-red-700 text-white"
                                            : "bg-emerald-600 hover:bg-emerald-700 text-white"
                                    }
                                >
                                    {isPlaying ? "Pause" : "Play Run"}
                                </Button>
                                <Button
                                    variant="secondary"
                                    onClick={runCAStep}
                                    disabled={isPlaying}
                                >
                                    Step +1 Day
                                </Button>
                            </div>

                            {!isSandbox && (
                                <div className="space-y-2 pt-4">
                                    <span className="text-xs font-semibold text-slate-500 uppercase">
                                        View Mode
                                    </span>
                                    <div className="flex bg-slate-800 rounded-md p-1">
                                        {(
                                            [
                                                "predicted",
                                                "actual",
                                                "compare",
                                            ] as const
                                        ).map((mode) => (
                                            <button
                                                key={mode}
                                                onClick={() =>
                                                    setViewMode(mode)
                                                }
                                                className={`flex-1 text-xs py-1.5 rounded capitalize transition-all ${viewMode === mode ? "bg-slate-700 text-white shadow-sm" : "text-slate-400 hover:text-white"}`}
                                            >
                                                {mode}
                                            </button>
                                        ))}
                                    </div>
                                    {viewMode === "compare" && (
                                        <div className="text-xs pt-2 flex justify-between px-2">
                                            <span className="flex items-center gap-1">
                                                <div className="w-2 h-2 bg-red-500 rounded-full"></div>{" "}
                                                Predicted
                                            </span>
                                            <span className="flex items-center gap-1">
                                                <div className="w-2 h-2 bg-green-500 rounded-full"></div>{" "}
                                                Actual
                                            </span>
                                        </div>
                                    )}
                                </div>
                            )}
                        </div>
                    )}
                </div>

                {/* Map Area */}
                <div className="flex-1 relative bg-slate-950 overflow-hidden">
                    {/* Compass / Wind Overlay */}
                    {isSandbox && windSpeed > 0 && (
                        <div className="absolute top-6 right-6 z-[400] w-28 h-28 bg-slate-900/80 rounded-full border border-slate-700 flex items-center justify-center pointer-events-none shadow-lg backdrop-blur-sm">
                            <div
                                className="absolute w-1 h-20 bg-gradient-to-t from-transparent via-blue-500/50 to-blue-400 rounded-full flex flex-col items-center"
                                style={{
                                    transform: `rotate(${windDir}deg)`,
                                    transition: "transform 0.3s ease-out",
                                }}
                            >
                                {/* Arrowhead */}
                                <div className="w-0 h-0 border-l-[6px] border-l-transparent border-r-[6px] border-r-transparent border-b-[10px] border-b-blue-400 -mt-2"></div>
                            </div>
                            <div className="absolute text-xs text-blue-300 font-bold bg-slate-900/60 px-2 py-0.5 rounded-full backdrop-blur-md">
                                {windSpeed} km/h
                            </div>
                        </div>
                    )}

                    <MapContainer
                        center={[30.1, 79.2]}
                        zoom={8}
                        style={{
                            height: "100%",
                            width: "100%",
                            background: "#0f172a",
                        }}
                    >
                        {/* Dark matter map tiles for good contrast */}
                        <TileLayer
                            url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
                            attribution='&copy; <a href="https://carto.com/">CARTO</a>'
                        />

                        {/* Model Bounds Box */}
                        <Rectangle
                            bounds={DATASET_BOUNDS}
                            pathOptions={{
                                color: "#3b82f6",
                                weight: 2,
                                fill: false,
                                dashArray: "5, 10",
                            }}
                        />

                        {eventData && imageUrl && (
                            <ImageOverlay
                                url={imageUrl}
                                bounds={eventData.bounds}
                                opacity={1.0}
                                zIndex={10}
                            />
                        )}

                        {/* Invisible layer to capture clicks */}
                        <MapClickHandler onMapClick={handleMapClick} />
                    </MapContainer>

                    {/* Wind particles overlay placed AFTER MapContainer to be on top */}
                    <WindOverlay
                        speed={windSpeed}
                        direction={windDir}
                        isSandbox={isSandbox}
                    />
                </div>
            </div>
        </div>
    );
}
