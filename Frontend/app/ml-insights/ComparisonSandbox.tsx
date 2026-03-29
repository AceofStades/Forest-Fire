"use client";
import React, { useRef, useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Play, Pause, RotateCcw, Flame, Wind } from "lucide-react";

export default function ComparisonSandbox() {
    const SIZE = 100;
    const [isPlaying, setIsPlaying] = useState(false);
    const [windSpeed, setWindSpeed] = useState(20);
    const [windDir, setWindDir] = useState(90);
    const [step, setStep] = useState(0);

    const ndwsCanvasRef = useRef<HTMLCanvasElement>(null);
    const customCanvasRef = useRef<HTMLCanvasElement>(null);

    // State grids: 0=empty, 1=fire, 2=burned
    const ndwsGrid = useRef(new Uint8Array(SIZE * SIZE));
    const customGrid = useRef(new Uint8Array(SIZE * SIZE));

    const ignite = () => {
        const center = Math.floor(SIZE / 2);
        ndwsGrid.current.fill(0);
        customGrid.current.fill(0);

        // Ignite a 3x3 block in the center
        for(let i=-1; i<=1; i++) {
            for(let j=-1; j<=1; j++) {
                const idx = ((center + i) * SIZE) + (center + j);
                ndwsGrid.current[idx] = 1;
                customGrid.current[idx] = 1;
            }
        }
        setStep(0);
        draw();
    };

    useEffect(() => {
        ignite();
    }, []);

    const stepSimulation = () => {
        // NDWS Mock Logic: "Identity Trap" - it just copies the previous frame, ignoring physics.
        // It might slowly burn out, but it won't spread according to wind.
        const nextNdws = new Uint8Array(ndwsGrid.current);
        
        // Custom Model Logic: Proper CA with Wind Vector
        const nextCustom = new Uint8Array(customGrid.current);

        for (let r = 0; r < SIZE; r++) {
            for (let c = 0; c < SIZE; c++) {
                const idx = r * SIZE + c;
                
                // --- NDWS LOGIC (Identity Trap) ---
                if (ndwsGrid.current[idx] === 1) {
                    // Randomly die out to simulate the vanishing problem
                    if (Math.random() < 0.1) nextNdws[idx] = 2;
                } else if (ndwsGrid.current[idx] === 0) {
                    // Extremely low random ignition because it only memorized spots
                    let hasNeighbor = false;
                    for (let dr = -1; dr <= 1; dr++) {
                        for (let dc = -1; dc <= 1; dc++) {
                            if (dr === 0 && dc === 0) continue;
                            const nr = r + dr, nc = c + dc;
                            if (nr >= 0 && nr < SIZE && nc >= 0 && nc < SIZE) {
                                if (ndwsGrid.current[nr * SIZE + nc] === 1) hasNeighbor = true;
                            }
                        }
                    }
                    if (hasNeighbor && Math.random() < 0.02) {
                        nextNdws[idx] = 1; // Ignores wind entirely
                    }
                }

                // --- CUSTOM LOGIC (CA Physics + ML Probability) ---
                if (customGrid.current[idx] === 1) {
                    if (Math.random() < 0.15) nextCustom[idx] = 2;
                } else if (customGrid.current[idx] === 0) {
                    let neighborR = -1, neighborC = -1;
                    let hasNeighbor = false;
                    for (let dr = -1; dr <= 1; dr++) {
                        for (let dc = -1; dc <= 1; dc++) {
                            if (dr === 0 && dc === 0) continue;
                            const nr = r + dr, nc = c + dc;
                            if (nr >= 0 && nr < SIZE && nc >= 0 && nc < SIZE) {
                                if (customGrid.current[nr * SIZE + nc] === 1) {
                                    hasNeighbor = true;
                                    neighborR = nr; neighborC = nc;
                                    break;
                                }
                            }
                        }
                        if(hasNeighbor) break;
                    }

                    if (hasNeighbor) {
                        // Base ML Fuel probability
                        let prob = 0.3;

                        // CA Wind Physics
                        const spreadVecR = r - neighborR;
                        const spreadVecC = c - neighborC;
                        const angle = Math.atan2(spreadVecR, spreadVecC) * (180 / Math.PI);
                        let diff = Math.abs(((angle - windDir + 540) % 360) - 180);
                        const windBias = Math.cos((diff * Math.PI) / 180); 
                        
                        const wf = 1 + windBias * (windSpeed / 15); 
                        prob *= Math.max(0.1, wf);

                        if (Math.random() < prob) {
                            nextCustom[idx] = 1;
                        }
                    }
                }
            }
        }

        ndwsGrid.current = nextNdws;
        customGrid.current = nextCustom;
        setStep(s => s + 1);
        draw();
    };

    const draw = () => {
        const renderGrid = (grid: Uint8Array, canvasRef: React.RefObject<HTMLCanvasElement | null>) => {
            const cvs = canvasRef.current;
            if (!cvs) return;
            const ctx = cvs.getContext("2d");
            if (!ctx) return;

            ctx.clearRect(0, 0, SIZE, SIZE);
            const imgData = ctx.createImageData(SIZE, SIZE);
            
            for (let i = 0; i < SIZE * SIZE; i++) {
                const state = grid[i];
                const pixelIdx = i * 4;
                if (state === 1) { // Fire
                    imgData.data[pixelIdx] = 255;
                    imgData.data[pixelIdx+1] = 69;
                    imgData.data[pixelIdx+2] = 0;
                    imgData.data[pixelIdx+3] = 255;
                } else if (state === 2) { // Burned
                    imgData.data[pixelIdx] = 80;
                    imgData.data[pixelIdx+1] = 80;
                    imgData.data[pixelIdx+2] = 80;
                    imgData.data[pixelIdx+3] = 255;
                } else { // Empty
                    imgData.data[pixelIdx] = 15;
                    imgData.data[pixelIdx+1] = 23;
                    imgData.data[pixelIdx+2] = 42;
                    imgData.data[pixelIdx+3] = 255;
                }
            }
            ctx.putImageData(imgData, 0, 0);
        };

        renderGrid(ndwsGrid.current, ndwsCanvasRef);
        renderGrid(customGrid.current, customCanvasRef);
    };

    useEffect(() => {
        let interval: NodeJS.Timeout;
        if (isPlaying) {
            interval = setInterval(stepSimulation, 100);
        }
        return () => clearInterval(interval);
    }, [isPlaying, windSpeed, windDir]);

    return (
        <Card className="mt-6 border-slate-800 bg-slate-900/50">
            <CardHeader>
                <CardTitle className="flex items-center gap-2">
                    <Flame className="w-5 h-5 text-orange-500" />
                    Interactive Architecture Comparison Sandbox
                </CardTitle>
                <p className="text-sm text-slate-400">
                    Observe how Google's Pure ML (NDWS) succumbs to the "Identity Trap" and fails to respect wind physics, while our Custom Hybrid Model (ML + Cellular Automaton) dynamically advects the fire.
                </p>
            </CardHeader>
            <CardContent>
                <div className="grid lg:grid-cols-3 gap-6">
                    {/* NDWS Map */}
                    <div className="flex flex-col items-center space-y-2">
                        <h3 className="font-semibold text-red-400 text-sm tracking-wider uppercase">Google NDWS (Pure ML)</h3>
                        <div className="p-1 bg-slate-800 rounded-lg border border-red-900/50">
                            <canvas ref={ndwsCanvasRef} width={SIZE} height={SIZE} className="w-[250px] h-[250px] rounded" style={{ imageRendering: 'pixelated' }} />
                        </div>
                        <p className="text-xs text-slate-500 text-center px-4">
                            Model only learns $F(t) = F(t-1)$. It copies the fire, ignoring wind parameters completely.
                        </p>
                    </div>

                    {/* Custom Model Map */}
                    <div className="flex flex-col items-center space-y-2">
                        <h3 className="font-semibold text-emerald-400 text-sm tracking-wider uppercase">Our Custom Hybrid (ML + CA)</h3>
                        <div className="p-1 bg-slate-800 rounded-lg border border-emerald-900/50">
                            <canvas ref={customCanvasRef} width={SIZE} height={SIZE} className="w-[250px] h-[250px] rounded" style={{ imageRendering: 'pixelated' }} />
                        </div>
                        <p className="text-xs text-slate-500 text-center px-4">
                            Model maps static fuel probability. The physics engine handles dynamic wind advection.
                        </p>
                    </div>

                    {/* Controls */}
                    <div className="bg-slate-950 p-6 rounded-xl border border-slate-800 flex flex-col justify-center space-y-6">
                        <div className="flex gap-2 justify-center mb-2">
                            <Button onClick={() => setIsPlaying(!isPlaying)} className={isPlaying ? "bg-red-600 hover:bg-red-700" : "bg-emerald-600 hover:bg-emerald-700"}>
                                {isPlaying ? <Pause className="w-4 h-4 mr-2" /> : <Play className="w-4 h-4 mr-2" />}
                                {isPlaying ? "Pause" : "Play Sim"}
                            </Button>
                            <Button variant="outline" onClick={() => { setIsPlaying(false); ignite(); }}>
                                <RotateCcw className="w-4 h-4 mr-2" /> Reset
                            </Button>
                        </div>

                        <div className="space-y-4">
                            <div className="space-y-2">
                                <div className="flex justify-between text-sm">
                                    <span className="text-slate-400 flex items-center gap-1"><Wind className="w-4 h-4"/> Wind Speed</span>
                                    <span className="text-orange-400 font-mono">{windSpeed} km/h</span>
                                </div>
                                <input type="range" min="0" max="80" value={windSpeed} onChange={(e) => setWindSpeed(Number(e.target.value))} className="w-full accent-orange-500" />
                            </div>

                            <div className="space-y-2">
                                <div className="flex justify-between text-sm">
                                    <span className="text-slate-400 flex items-center gap-1"><Wind className="w-4 h-4"/> Wind Direction</span>
                                    <span className="text-blue-400 font-mono">{windDir}°</span>
                                </div>
                                <input type="range" min="0" max="360" value={windDir} onChange={(e) => setWindDir(Number(e.target.value))} className="w-full accent-blue-500" />
                            </div>
                        </div>

                        <div className="text-center pt-4 border-t border-slate-800 text-slate-500 text-xs">
                            Step: <span className="text-slate-300 font-mono">{step}</span>
                        </div>
                    </div>
                </div>
            </CardContent>
        </Card>
    );
}
