"use client";
import React, { useRef, useEffect, useState, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Play, Pause, RotateCcw, Flame, Wind, Mountain, ThermometerSun, MousePointerClick } from "lucide-react";

export default function ComparisonSandbox() {
    const SIZE = 120;
    
    // UI State
    const [isPlaying, setIsPlaying] = useState(false);
    const [step, setStep] = useState(0);
    
    // External Parameters
    const [windSpeed, setWindSpeed] = useState(30);
    const [windDir, setWindDir] = useState(90); // Degrees (0=North, 90=East)
    const [temperature, setTemperature] = useState(25); // Celsius
    const [steepness, setSteepness] = useState(50); // Multiplier for terrain effect

    const ndwsCanvasRef = useRef<HTMLCanvasElement>(null);
    const customCanvasRef = useRef<HTMLCanvasElement>(null);

    // Simulation Data Grids
    // State grids: 0=empty, 1=fire, 2=burned
    const ndwsGrid = useRef(new Uint8Array(SIZE * SIZE));
    const customGrid = useRef(new Uint8Array(SIZE * SIZE));
    
    // Topography Grid (0.0 to 1.0 elevation)
    const terrainGrid = useRef(new Float32Array(SIZE * SIZE));

    // Generate smoothed pseudo-random terrain
    const generateTerrain = () => {
        let terrain = new Float32Array(SIZE * SIZE);
        for (let i = 0; i < SIZE * SIZE; i++) terrain[i] = Math.random();
        
        // Smooth 6 times to create hills and valleys
        for (let passes = 0; passes < 6; passes++) {
            let next = new Float32Array(SIZE * SIZE);
            for (let r = 0; r < SIZE; r++) {
                for (let c = 0; c < SIZE; c++) {
                    let sum = 0, count = 0;
                    for (let dr = -2; dr <= 2; dr++) {
                        for (let dc = -2; dc <= 2; dc++) {
                            let nr = r + dr, nc = c + dc;
                            if (nr >= 0 && nr < SIZE && nc >= 0 && nc < SIZE) {
                                sum += terrain[nr * SIZE + nc];
                                count++;
                            }
                        }
                    }
                    next[r * SIZE + c] = sum / count;
                }
            }
            terrain = next;
        }
        
        // Normalize
        let min = 1, max = 0;
        for (let i = 0; i < SIZE * SIZE; i++) {
            if (terrain[i] < min) min = terrain[i];
            if (terrain[i] > max) max = terrain[i];
        }
        for (let i = 0; i < SIZE * SIZE; i++) {
            terrain[i] = (terrain[i] - min) / (max - min);
        }
        terrainGrid.current = terrain;
    };

    const resetSim = useCallback(() => {
        setIsPlaying(false);
        setStep(0);
        ndwsGrid.current.fill(0);
        customGrid.current.fill(0);
        generateTerrain();
        draw();
    }, []);

    useEffect(() => {
        resetSim();
    }, [resetSim]);

    // Handle interactive clicking to place fire
    const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
        const rect = e.currentTarget.getBoundingClientRect();
        // Calculate canvas scale since it's stretched via CSS
        const scaleX = SIZE / rect.width;
        const scaleY = SIZE / rect.height;
        
        const x = Math.floor((e.clientX - rect.left) * scaleX);
        const y = Math.floor((e.clientY - rect.top) * scaleY);

        if (x >= 0 && x < SIZE && y >= 0 && y < SIZE) {
            // Ignite a 3x3 block
            for (let i = -1; i <= 1; i++) {
                for (let j = -1; j <= 1; j++) {
                    const nr = y + i;
                    const nc = x + j;
                    if (nr >= 0 && nr < SIZE && nc >= 0 && nc < SIZE) {
                        const idx = nr * SIZE + nc;
                        ndwsGrid.current[idx] = 1;
                        customGrid.current[idx] = 1;
                    }
                }
            }
            draw();
        }
    };

    const stepSimulation = useCallback(async () => {
        try {
            const response = await fetch("http://127.0.0.1:8000/sandbox-step", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    size: SIZE,
                    ndws_grid: Array.from(ndwsGrid.current),
                    custom_grid: Array.from(customGrid.current),
                    terrain_grid: Array.from(terrainGrid.current),
                    wind_speed: windSpeed,
                    wind_dir: windDir,
                    temperature: temperature,
                    steepness: steepness,
                }),
            });

            if (!response.ok) {
                console.error("Failed to step simulation");
                setIsPlaying(false);
                return;
            }

            const data = await response.json();
            
            // Update grids
            ndwsGrid.current = new Uint8Array(data.next_ndws);
            customGrid.current = new Uint8Array(data.next_custom);
            
            setStep(s => s + 1);
            draw();
        } catch (err) {
            console.error("Error calling sandbox API:", err);
            setIsPlaying(false);
        }
    }, [temperature, windSpeed, windDir, steepness, isPlaying]);

    const draw = () => {
        const renderGrid = (grid: Uint8Array, canvasRef: React.RefObject<HTMLCanvasElement | null>) => {
            const cvs = canvasRef.current;
            if (!cvs) return;
            const ctx = cvs.getContext("2d");
            if (!ctx) return;

            const imgData = ctx.createImageData(SIZE, SIZE);
            
            for (let i = 0; i < SIZE * SIZE; i++) {
                const state = grid[i];
                const elev = terrainGrid.current[i];
                const pixelIdx = i * 4;
                
                // Base Topography Colors (Dark green to tan/brown)
                let r = Math.floor(30 + elev * 120);
                let g = Math.floor(70 + elev * 90);
                let b = Math.floor(30 + elev * 50);

                if (state === 1) { 
                    // Active Fire (Red/Orange overriding terrain)
                    r = 255; g = Math.floor(60 + Math.random() * 60); b = 0;
                } else if (state === 2) { 
                    // Burn Scar (Darken the terrain to represent char)
                    r = Math.floor(r * 0.3);
                    g = Math.floor(g * 0.3);
                    b = Math.floor(b * 0.3);
                }

                imgData.data[pixelIdx] = r;
                imgData.data[pixelIdx+1] = g;
                imgData.data[pixelIdx+2] = b;
                imgData.data[pixelIdx+3] = 255;
            }
            ctx.putImageData(imgData, 0, 0);
        };

        renderGrid(ndwsGrid.current, ndwsCanvasRef);
        renderGrid(customGrid.current, customCanvasRef);
    };

    useEffect(() => {
        let timeoutId: NodeJS.Timeout;
        let isActive = true;

        const loop = async () => {
            if (!isPlaying || !isActive) return;
            await stepSimulation();
            if (isActive) {
                timeoutId = setTimeout(loop, 200); // 5 FPS to not overwhelm backend
            }
        };

        if (isPlaying) {
            loop();
        }

        return () => {
            isActive = false;
            clearTimeout(timeoutId);
        };
    }, [isPlaying, stepSimulation]);

    return (
        <Card className="border-slate-800 bg-slate-900/50 shadow-2xl">
            <CardHeader className="border-b border-slate-800/50 bg-slate-900/30">
                <CardTitle className="flex items-center gap-2 text-xl">
                    <Flame className="w-5 h-5 text-orange-500" />
                    Interactive Physical Sandbox
                </CardTitle>
                <CardDescription className="text-slate-400">
                    Test the physical advection capabilities of both models on a procedurally generated topographical map. 
                    <strong className="text-emerald-400 ml-1">Click anywhere on the maps below to ignite a fire.</strong>
                </CardDescription>
            </CardHeader>
            <CardContent className="p-6">
                <div className="grid lg:grid-cols-12 gap-8">
                    
                    {/* Left Panel: The Maps (Spans 8 cols) */}
                    <div className="lg:col-span-8 flex flex-col md:flex-row gap-6 justify-center items-center">
                        {/* NDWS Map */}
                        <div className="flex flex-col items-center space-y-3 w-full flex-1">
                            <h3 className="font-semibold text-red-400 text-sm tracking-wider uppercase flex items-center gap-2">
                                Google NDWS
                            </h3>
                            <div className="relative group cursor-pointer" onClick={handleCanvasClick}>
                                <div className="absolute inset-0 bg-red-500/10 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center pointer-events-none rounded-lg z-10">
                                    <div className="bg-black/80 text-white text-xs px-3 py-1 rounded flex items-center gap-1 backdrop-blur-sm">
                                        <MousePointerClick className="w-3 h-3"/> Ignite Here
                                    </div>
                                </div>
                                <canvas 
                                    ref={ndwsCanvasRef} 
                                    width={SIZE} 
                                    height={SIZE} 
                                    className="w-full aspect-square rounded-lg border-2 border-slate-700 hover:border-red-500/50 transition-colors shadow-lg" 
                                    style={{ imageRendering: 'pixelated' }} 
                                />
                            </div>
                            <p className="text-xs text-slate-500 text-center leading-relaxed h-10">
                                Suffers from the "Identity Trap". Ignores terrain slope and wind physics, expanding radially at a constant rate.
                            </p>
                        </div>

                        {/* Custom Model Map */}
                        <div className="flex flex-col items-center space-y-3 w-full flex-1">
                            <h3 className="font-semibold text-emerald-400 text-sm tracking-wider uppercase flex items-center gap-2">
                                Custom Hybrid
                            </h3>
                            <div className="relative group cursor-pointer" onClick={handleCanvasClick}>
                                <div className="absolute inset-0 bg-emerald-500/10 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center pointer-events-none rounded-lg z-10">
                                    <div className="bg-black/80 text-white text-xs px-3 py-1 rounded flex items-center gap-1 backdrop-blur-sm">
                                        <MousePointerClick className="w-3 h-3"/> Ignite Here
                                    </div>
                                </div>
                                <canvas 
                                    ref={customCanvasRef} 
                                    width={SIZE} 
                                    height={SIZE} 
                                    className="w-full aspect-square rounded-lg border-2 border-slate-700 hover:border-emerald-500/50 transition-colors shadow-lg" 
                                    style={{ imageRendering: 'pixelated' }} 
                                />
                            </div>
                            <p className="text-xs text-slate-500 text-center leading-relaxed h-10">
                                Merges static fuel susceptibility with explicit Cellular Automaton physics (wind direction & uphill spread bias).
                            </p>
                        </div>
                    </div>

                    {/* Right Panel: Controls (Spans 4 cols) */}
                    <div className="lg:col-span-4 bg-slate-950 p-6 rounded-xl border border-slate-800 flex flex-col space-y-6 shadow-inner">
                        
                        <div className="flex gap-3 justify-center mb-2">
                            <Button 
                                onClick={() => setIsPlaying(!isPlaying)} 
                                className={`w-full font-semibold ${isPlaying ? "bg-red-600 hover:bg-red-700 text-white" : "bg-emerald-600 hover:bg-emerald-700 text-white"}`}
                            >
                                {isPlaying ? <Pause className="w-4 h-4 mr-2" /> : <Play className="w-4 h-4 mr-2" />}
                                {isPlaying ? "Pause Simulation" : "Run Simulation"}
                            </Button>
                            <Button variant="outline" className="border-slate-700 hover:bg-slate-800 px-3" onClick={resetSim} title="Regenerate Terrain & Clear Fire">
                                <RotateCcw className="w-4 h-4" />
                            </Button>
                        </div>

                        <div className="space-y-6 flex-1">
                            {/* Wind Speed */}
                            <div className="space-y-3">
                                <div className="flex justify-between text-sm">
                                    <span className="text-slate-300 flex items-center gap-2 font-medium">
                                        <Wind className="w-4 h-4 text-sky-400"/> Wind Speed
                                    </span>
                                    <span className="text-sky-400 font-mono bg-sky-950/50 px-2 py-0.5 rounded">{windSpeed} km/h</span>
                                </div>
                                <input type="range" min="0" max="80" value={windSpeed} onChange={(e) => setWindSpeed(Number(e.target.value))} className="w-full accent-sky-500" />
                            </div>

                            {/* Wind Direction */}
                            <div className="space-y-3">
                                <div className="flex justify-between text-sm">
                                    <span className="text-slate-300 flex items-center gap-2 font-medium">
                                        <Wind className="w-4 h-4 text-blue-400"/> Wind Direction
                                    </span>
                                    <span className="text-blue-400 font-mono bg-blue-950/50 px-2 py-0.5 rounded">{windDir}°</span>
                                </div>
                                <div className="relative pt-2 pb-6">
                                    <input type="range" min="0" max="360" value={windDir} onChange={(e) => setWindDir(Number(e.target.value))} className="w-full accent-blue-500" />
                                    <div className="absolute w-full flex justify-between text-[10px] text-slate-500 mt-1 font-mono">
                                        <span>N</span><span>E</span><span>S</span><span>W</span><span>N</span>
                                    </div>
                                </div>
                            </div>

                            {/* Temperature (Fuel Moisture proxy) */}
                            <div className="space-y-3">
                                <div className="flex justify-between text-sm">
                                    <span className="text-slate-300 flex items-center gap-2 font-medium">
                                        <ThermometerSun className="w-4 h-4 text-orange-400"/> Temperature
                                    </span>
                                    <span className="text-orange-400 font-mono bg-orange-950/50 px-2 py-0.5 rounded">{temperature}°C</span>
                                </div>
                                <input type="range" min="10" max="50" value={temperature} onChange={(e) => setTemperature(Number(e.target.value))} className="w-full accent-orange-500" />
                            </div>

                            {/* Topography Impact */}
                            <div className="space-y-3">
                                <div className="flex justify-between text-sm">
                                    <span className="text-slate-300 flex items-center gap-2 font-medium">
                                        <Mountain className="w-4 h-4 text-amber-600"/> Slope Impact
                                    </span>
                                    <span className="text-amber-600 font-mono bg-amber-950/50 px-2 py-0.5 rounded">{steepness}%</span>
                                </div>
                                <input type="range" min="0" max="100" value={steepness} onChange={(e) => setSteepness(Number(e.target.value))} className="w-full accent-amber-600" />
                            </div>
                        </div>

                        <div className="mt-auto pt-4 border-t border-slate-800 flex justify-between items-center text-xs">
                            <span className="text-slate-500">Simulation Time</span>
                            <span className="text-slate-300 font-mono bg-slate-900 px-2 py-1 rounded border border-slate-800">t + {step} hrs</span>
                        </div>
                    </div>
                </div>
            </CardContent>
        </Card>
    );
}
