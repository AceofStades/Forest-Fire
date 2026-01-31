"use client";
import React, { useState, useEffect } from 'react';
import dynamic from 'next/dynamic';
import SimulationCanvas from '../Simulation2/SimulationCanvas'; // Re-use existing canvas component
import { Slider } from "@/components/ui/slider";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Button } from "@/components/ui/button";

// Dynamic Map Import
const RealTimeMap = dynamic(() => import('@/components/ui/RealTimeMap'), { 
  ssr: false, 
  loading: () => <div className="w-full h-[500px] bg-slate-900 animate-pulse rounded-xl" />
});

export default function SimulationPage() {
  const [grid, setGrid] = useState<number[][] | null>(null);
  
  // Simulation Factors
  const [windSpeed, setWindSpeed] = useState([20]); // km/h
  const [humidity, setHumidity] = useState([40]);   // %
  const [windDir, setWindDir] = useState("N");

  const fetchGrid = async () => {
    try {
      const res = await fetch('http://127.0.0.1:8000/fire-grid');
      const data = await res.json();
      setGrid(data.grid);
    } catch (e) { console.error("Connection Error", e); }
  };

  // Initial Load
  useEffect(() => { fetchGrid(); }, []);

  // Update Simulation on Backend
  const applyDynamics = async () => {
    try {
      await fetch('http://127.0.0.1:8000/update-simulation', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          wind_speed: windSpeed[0],
          wind_direction: windDir,
          humidity: humidity[0]
        })
      });
      // Re-fetch grid to see changes visually
      fetchGrid();
    } catch (e) { console.error("Failed to update dynamics", e); }
  };

  return (
    <div className="min-h-screen bg-slate-950 p-6 text-white flex flex-col gap-6">
      
      {/* Header & Controls */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        <div className="bg-slate-900 p-6 rounded-xl border border-slate-800 lg:col-span-1 space-y-6">
          <h2 className="text-xl font-bold text-orange-500">Physics Engine</h2>
          
          <div className="space-y-3">
            <label className="text-sm text-slate-400">Wind Speed: {windSpeed} km/h</label>
            <Slider value={windSpeed} onValueChange={setWindSpeed} max={100} step={1} className="py-2" />
          </div>

          <div className="space-y-3">
            <label className="text-sm text-slate-400">Humidity: {humidity}%</label>
            <Slider value={humidity} onValueChange={setHumidity} max={100} step={1} className="py-2" />
          </div>

          <div className="space-y-3">
            <label className="text-sm text-slate-400">Wind Direction</label>
            <Select value={windDir} onValueChange={setWindDir}>
              <SelectTrigger className="bg-slate-800 border-slate-700">
                <SelectValue placeholder="Direction" />
              </SelectTrigger>
              <SelectContent className="bg-slate-800 border-slate-700 text-white">
                <SelectItem value="N">North (Blows South)</SelectItem>
                <SelectItem value="S">South (Blows North)</SelectItem>
                <SelectItem value="E">East (Blows West)</SelectItem>
                <SelectItem value="W">West (Blows East)</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <Button onClick={applyDynamics} className="w-full bg-orange-600 hover:bg-orange-500 font-bold">
            Apply Physics & Re-Simulate
          </Button>
        </div>

        {/* Dashboard Title */}
        <div className="lg:col-span-3 flex flex-col justify-end pb-2">
          <h1 className="text-4xl font-bold">Forest Fire Digital Twin</h1>
          <p className="text-slate-400">Real-time pathfinding adaptation based on live weather telemetry.</p>
        </div>
      </div>

      {/* Visualization Area */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 h-[600px]">
        {/* Left: 2D Simulation */}
        <div className="bg-black rounded-xl border border-slate-800 overflow-hidden flex items-center justify-center relative">
          <div className="absolute top-4 left-4 z-10 bg-black/50 px-3 py-1 rounded text-xs">
            Simulation View (Cellular Automata)
          </div>
          {grid ? <SimulationCanvas probGrid={grid} /> : "Loading Engine..."}
        </div>

        {/* Right: Map Navigation */}
        <div className="bg-slate-900 rounded-xl border border-slate-800 overflow-hidden relative">
           <div className="absolute top-4 left-14 z-[1000] bg-black/80 text-white px-3 py-1 rounded text-xs border border-slate-600 pointer-events-none">
            Tactical Map (Click to Navigate)
          </div>
          <RealTimeMap />
        </div>
      </div>
    </div>
  );
}