"use client"; // Add this since we are using state in the page

import React, { useState, useEffect } from 'react';
import dynamic from 'next/dynamic'; // 1. Import dynamic
import SimulationCanvas from './SimulationCanvas';

// 2. Load the Map with SSR disabled
const RealTimeMap = dynamic(() => import('@/components/ui/RealTimeMap'), { 
  ssr: false, // 3. Crucial: Disables Server-Side Rendering for this component
  loading: () => <div className="w-full h-[600px] bg-slate-900 animate-pulse rounded-xl" />
});

export default function Page() {
  const [grid, setGrid] = useState<number[][] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // Fetch the grid from your FastAPI backend
    const getGrid = async () => {
      try {
        const response = await fetch('http://127.0.0.1:8000/fire-grid'); if (!response.ok) throw new Error("Failed to fetch fire grid");
        if (!response.ok) throw new Error("Failed to fetch fire grid");

        const data = await response.json();
        setGrid(data.grid);
      } catch (err) {
        console.error(err);
        setError("Could not connect to the backend server.");
      }
    };

    getGrid();
  }, []);

  return (
    <main className="p-8 bg-slate-950 min-h-screen flex flex-col items-center justify-center">
      <h1 className="text-white text-3xl font-bold mb-8">Forest Fire Simulation</h1>

      {error && <p className="text-red-500">{error}</p>}

      {/* Pass the 'grid' as 'probGrid' prop only when it exists. 
          This satisfies the TypeScript requirement.
      */}
      {grid ? (
        <SimulationCanvas probGrid={grid} />
      ) : (
        !error && <p className="text-slate-400">Loading Model Artifacts...</p>
      )}
    </main>
  );
}