"use client";
import React, { useRef, useEffect, useState } from 'react';

interface SimulationProps {
    probGrid: number[][]; 
}

const SimulationCanvas: React.FC<SimulationProps> = ({ probGrid }) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [isRunning, setIsRunning] = useState(false);
    
    // We use a Ref for path so the animation loop can access it without dependencies
    const pathRef = useRef<[number, number][]>([]);

    const ROWS = 320;
    const COLS = 400;
    const CELL_SIZE = 2; 

    // Simulation State
    const stateGrid = useRef(new Uint8Array(ROWS * COLS));

    // 1. Initialize logic
    useEffect(() => {
        // Ignite center
        stateGrid.current[Math.floor(ROWS / 2) * COLS + Math.floor(COLS / 2)] = 1;
    }, []);

    // 2. Animation Loop
    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        let animationFrameId: number;

        const render = () => {
            if (isRunning) updateSimulation();
            
            // Draw sequence
            drawGrid(ctx);      // Draw Fire
            drawPath(ctx);      // Draw Path (Overlay)
            
            animationFrameId = requestAnimationFrame(render);
        };

        const updateSimulation = () => {
             const nextState = new Uint8Array(stateGrid.current);
             // ... (Your existing CA logic is perfect, keeping it same) ...
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
             // ... (Your existing neighbor logic is correct) ...
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

        const drawGrid = (context: CanvasRenderingContext2D) => {
            context.clearRect(0, 0, canvas.width, canvas.height);
            for (let r = 0; r < ROWS; r++) {
                for (let c = 0; c < COLS; c++) {
                    const state = stateGrid.current[r * COLS + c];
                    if (state === 0) continue; 
                    context.fillStyle = state === 1 ? '#ff4d00' : '#444444';
                    context.fillRect(c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE);
                }
            }
        };

        const drawPath = (context: CanvasRenderingContext2D) => {
            const currentPath = pathRef.current;
            if (currentPath.length < 2) return;
            
            context.beginPath();
            context.strokeStyle = '#00ff00'; // Neon Green
            context.lineWidth = 3;
            // Move to first point
            context.moveTo(currentPath[0][1] * CELL_SIZE, currentPath[0][0] * CELL_SIZE);
            
            for (let i = 1; i < currentPath.length; i++) {
                // [row, col] -> x=col, y=row
                context.lineTo(currentPath[i][1] * CELL_SIZE, currentPath[i][0] * CELL_SIZE);
            }
            context.stroke();
        };

        render();
        return () => cancelAnimationFrame(animationFrameId);
    }, [isRunning, probGrid]); // Dependencies

    // 3. Interaction Handler
    const handleCanvasClick = async (e: React.MouseEvent<HTMLCanvasElement>) => {
        const rect = canvasRef.current?.getBoundingClientRect();
        if (!rect) return;

        const x = Math.floor((e.clientX - rect.left) / CELL_SIZE);
        const y = Math.floor((e.clientY - rect.top) / CELL_SIZE);

        console.log(`Click at Grid: [${y}, ${x}]`); // Debug log

        // Hardcoded goal for testing (middle of map)
        const goal: [number, number] = [160, 200];
        const start: [number, number] = [y, x];

        try {
            const response = await fetch('http://127.0.0.1:8000/get-safe-path', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ start, goal })
            });

            if(response.ok) {
                const data = await response.json();
                console.log("Path received:", data.path.length);
                pathRef.current = data.path; // Update the Ref, animation picks it up next frame
            }
        } catch (err) {
            console.error("Pathfinding error:", err);
        }
    };

    return (
        <div className="flex flex-col items-center gap-4 bg-slate-900 p-6 rounded-xl border border-slate-700">
            <h2 className="text-white font-bold text-xl">Fire Spread Simulation</h2>
            <div className="relative">
                <canvas
                    ref={canvasRef}
                    onClick={handleCanvasClick}  // <-- CRITICAL FIX: Event listener added
                    width={COLS * CELL_SIZE}
                    height={ROWS * CELL_SIZE}
                    className="rounded border border-orange-500 shadow-lg shadow-orange-500/20 cursor-crosshair"
                />
                <p className="text-xs text-slate-400 mt-2 text-center">
                    Click anywhere to calculate path to center
                </p>
            </div>
            
            <button
                onClick={() => setIsRunning(!isRunning)}
                className="px-6 py-2 bg-orange-600 text-white rounded-full hover:bg-orange-500 transition-colors"
            >
                {isRunning ? "Pause Simulation" : "Start Simulation"}
            </button>
        </div>
    );
};

export default SimulationCanvas;