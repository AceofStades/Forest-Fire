"use client";
import React, { useRef, useEffect, useState } from "react";

interface SimulationProps {
    probGrid: number[][]; // The 320x400 grid from FastAPI
}

const SimulationCanvas: React.FC<SimulationProps> = ({ probGrid }) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [isRunning, setIsRunning] = useState(false);
    const [path, setPath] = useState<[number, number][]>([]);

    // Simulation Constants
    const ROWS = 320;
    const COLS = 400;
    const CELL_SIZE = 2; // Pixels per grid cell

    // 0: Safe, 1: Burning, 2: Burned Out
    const stateGrid = useRef(new Uint8Array(ROWS * COLS));

    // Handle Canvas Clicks for Pathfinding
    const handleCanvasClick = async (
        e: React.MouseEvent<HTMLCanvasElement>,
    ) => {
        const rect = canvasRef.current?.getBoundingClientRect();
        if (!rect) return;

        // Convert click coordinates to Grid Coordinates (row, col)
        const x = Math.floor((e.clientX - rect.left) / CELL_SIZE);
        const y = Math.floor((e.clientY - rect.top) / CELL_SIZE);

        // For this demo, let's assume the center is always the goal
        const goal: [number, number] = [
            Math.floor(ROWS / 2),
            Math.floor(COLS / 2),
        ];
        const start: [number, number] = [y, x];

        try {
            const response = await fetch(
                "http://127.0.0.1:8000/get-safe-path",
                {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ start, goal }),
                },
            );

            if (response.ok) {
                const data = await response.json();
                setPath(data.path);
            }
        } catch (err) {
            console.error("Pathfinding error", err);
        }
    };

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext("2d");
        if (!ctx) return;

        // Initial ignition point (center for testing)
        stateGrid.current[Math.floor(ROWS / 2) * COLS + Math.floor(COLS / 2)] =
            1;

        let animationFrameId: number;

        const updateSimulation = () => {
            const nextState = new Uint8Array(stateGrid.current);

            for (let r = 0; r < ROWS; r++) {
                for (let c = 0; c < COLS; c++) {
                    const idx = r * COLS + c;
                    if (stateGrid.current[idx] === 0) {
                        // Check neighbors for spread
                        if (hasBurningNeighbor(r, c)) {
                            // Rule: Probability of ignition from model output
                            // We cap it or scale it if needed
                            if (Math.random() < probGrid[r][c]) {
                                nextState[idx] = 1;
                            }
                        }
                    } else if (stateGrid.current[idx] === 1) {
                        // Fire eventually burns out
                        if (Math.random() < 0.05) nextState[idx] = 2;
                    }
                }
            }
            stateGrid.current = nextState;
        };

        const hasBurningNeighbor = (r: number, c: number) => {
            for (let dr = -1; dr <= 1; dr++) {
                for (let dc = -1; dc <= 1; dc++) {
                    if (dr === 0 && dc === 0) continue;
                    const nr = r + dr;
                    const nc = c + dc;
                    if (nr >= 0 && nr < ROWS && nc >= 0 && nc < COLS) {
                        if (stateGrid.current[nr * COLS + nc] === 1)
                            return true;
                    }
                }
            }
            return false;
        };

        const drawPath = (context: CanvasRenderingContext2D) => {
            if (path.length < 2) return;
            context.beginPath();
            context.strokeStyle = "#00ff00"; // Neon Green for safety
            context.lineWidth = 2;
            context.moveTo(
                path[0][1] * CELL_SIZE + CELL_SIZE / 2,
                path[0][0] * CELL_SIZE + CELL_SIZE / 2,
            );
            for (let i = 1; i < path.length; i++) {
                context.lineTo(
                    path[i][1] * CELL_SIZE + CELL_SIZE / 2,
                    path[i][0] * CELL_SIZE + CELL_SIZE / 2,
                );
            }
            context.stroke();
        };

        const draw = (context: CanvasRenderingContext2D) => {
            context.clearRect(0, 0, canvas.width, canvas.height);
            for (let r = 0; r < ROWS; r++) {
                for (let c = 0; c < COLS; c++) {
                    const state = stateGrid.current[r * COLS + c];
                    if (state === 0) continue; // Don't draw safe cells (transparent)

                    context.fillStyle = state === 1 ? "#ff4d00" : "#444444";
                    context.fillRect(
                        c * CELL_SIZE,
                        r * CELL_SIZE,
                        CELL_SIZE,
                        CELL_SIZE,
                    );
                }
            }
            drawPath(context);
        };

        const render = () => {
            if (isRunning) {
                updateSimulation();
            }
            draw(ctx);
            animationFrameId = requestAnimationFrame(render);
        };

        render();
        return () => cancelAnimationFrame(animationFrameId);
    }, [isRunning, probGrid, path]);

    return (
        <div className="flex flex-col items-center gap-4 bg-slate-900 p-6 rounded-xl border border-slate-700">
            <h2 className="text-white font-bold text-xl">
                Fire Spread Simulation (ML Backend Connected)
            </h2>
            <canvas
                ref={canvasRef}
                width={COLS * CELL_SIZE}
                height={ROWS * CELL_SIZE}
                className="rounded border border-orange-500 shadow-lg shadow-orange-500/20 cursor-pointer"
                onClick={handleCanvasClick}
            />
            <p className="text-sm text-slate-400">
                Click on the canvas to find a safe path to the center.
            </p>
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
