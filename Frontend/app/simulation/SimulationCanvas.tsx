"use client";
import React, { useRef, useEffect, useState } from 'react';

interface SimulationProps {
    probGrid: number[][]; 
}

const SimulationCanvas: React.FC<SimulationProps> = ({ probGrid }) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const containerRef = useRef<HTMLDivElement>(null);
    const [isRunning, setIsRunning] = useState(false);
    const pathRef = useRef<[number, number][]>([]);

    // Constants
    const ROWS = 320;
    const COLS = 400;
    const stateGrid = useRef(new Uint8Array(ROWS * COLS));
    
    // --- FIX 1: Frame Rate Control ---
    const lastTimeRef = useRef<number>(0);
    const FPS = 12; // Lower FPS = Slower, more realistic creeping fire
    const interval = 1000 / FPS;

    useEffect(() => {
        stateGrid.current[Math.floor(ROWS / 2) * COLS + Math.floor(COLS / 2)] = 1;
        
        // --- FIX 2: Dynamic Scaling Logic ---
        const handleResize = () => {
            if (canvasRef.current && containerRef.current) {
                const rect = containerRef.current.getBoundingClientRect();
                canvasRef.current.width = rect.width;
                canvasRef.current.height = rect.height;
            }
        };

        window.addEventListener('resize', handleResize);
        handleResize();
        return () => window.removeEventListener('resize', handleResize);
    }, []);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        let animationFrameId: number;

        const render = (timestamp: number) => {
            const deltaTime = timestamp - lastTimeRef.current;

            if (isRunning && deltaTime > interval) {
                updateSimulation();
                lastTimeRef.current = timestamp - (deltaTime % interval);
            }
            
            // Draw every frame for smooth path/UI, but simulation logic is throttled
            draw(ctx);
            animationFrameId = requestAnimationFrame(render);
        };

        const updateSimulation = () => {
             const nextState = new Uint8Array(stateGrid.current);
             for (let r = 0; r < ROWS; r++) {
                for (let c = 0; c < COLS; c++) {
                    const idx = r * COLS + c;
                    if (stateGrid.current[idx] === 0) {
                        if (hasBurningNeighbor(r, c) && Math.random() < probGrid[r][c]) {
                            nextState[idx] = 1;
                        }
                    } else if (stateGrid.current[idx] === 1) {
                        if (Math.random() < 0.05) nextState[idx] = 2;
                    }
                }
            }
            stateGrid.current = nextState;
        };

        const hasBurningNeighbor = (r: number, c: number) => {
             for (let dr = -1; dr <= 1; dr++) {
                for (let dc = -1; dc <= 1; dc++) {
                    const nr = r + dr;
                    const nc = c + dc;
                    if (nr >= 0 && nr < ROWS && nc >= 0 && nc < COLS) {
                        if (stateGrid.current[nr * COLS + nc] === 1) return true;
                    }
                }
            }
            return false;
        };

        const draw = (context: CanvasRenderingContext2D) => {
            const w = canvas.width;
            const h = canvas.height;
            const cellW = w / COLS;
            const cellH = h / ROWS;

            context.clearRect(0, 0, w, h);

            // Draw Fire Grid
            for (let r = 0; r < ROWS; r++) {
                for (let c = 0; c < COLS; c++) {
                    const state = stateGrid.current[r * COLS + c];
                    if (state === 0) continue; 
                    context.fillStyle = state === 1 ? '#ff4d00' : '#444444';
                    // Use calculated cellW/cellH for dynamic scaling
                    context.fillRect(c * cellW, r * cellH, cellW + 0.5, cellH + 0.5);
                }
            }

            // Draw Path Overlay
            const currentPath = pathRef.current;
            if (currentPath.length >= 2) {
                context.beginPath();
                context.strokeStyle = '#00ff00';
                context.lineWidth = 3;
                context.moveTo(currentPath[0][1] * cellW, currentPath[0][0] * cellH);
                for (let i = 1; i < currentPath.length; i++) {
                    context.lineTo(currentPath[i][1] * cellW, currentPath[i][0] * cellH);
                }
                context.stroke();
            }
        };

        animationFrameId = requestAnimationFrame(render);
        return () => cancelAnimationFrame(animationFrameId);
    }, [isRunning, probGrid, interval]);

    const handleCanvasClick = async (e: React.MouseEvent<HTMLCanvasElement>) => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const rect = canvas.getBoundingClientRect();
        
        // Translate click to grid coordinates based on dynamic scaling
        const x = Math.floor(((e.clientX - rect.left) / rect.width) * COLS);
        const y = Math.floor(((e.clientY - rect.top) / rect.height) * ROWS);

        const goal: [number, number] = [Math.floor(ROWS/2), Math.floor(COLS/2)];
        const start: [number, number] = [y, x];

        try {
            const response = await fetch('http://127.0.0.1:8000/get-safe-path', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ start, goal })
            });
            if(response.ok) {
                const data = await response.json();
                pathRef.current = data.path;
            }
        } catch (err) { console.error(err); }
    };

    return (
        <div className="flex flex-col items-center gap-4 bg-slate-900 p-6 rounded-xl w-full h-full">
            <div ref={containerRef} className="relative w-full h-[500px] border border-orange-500 overflow-hidden">
                <canvas
                    ref={canvasRef}
                    onClick={handleCanvasClick}
                    className="cursor-crosshair w-full h-full"
                />
            </div>
            <button
                onClick={() => setIsRunning(!isRunning)}
                className="px-6 py-2 bg-orange-600 text-white rounded-full hover:bg-orange-500"
            >
                {isRunning ? "Pause Simulation" : "Start Simulation"}
            </button>
        </div>
    );
};

export default SimulationCanvas;