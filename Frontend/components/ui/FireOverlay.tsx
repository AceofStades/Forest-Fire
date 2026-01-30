"use client";
import React, { useEffect, useRef, useState } from 'react';
import { ImageOverlay } from 'react-leaflet';
import { getMapBounds } from '@/lib/geoUtils';

interface FireOverlayProps {
  probGrid: number[][]; // The UNet Probability Grid
}

const FireOverlay: React.FC<FireOverlayProps> = ({ probGrid }) => {
  const [imageUrl, setImageUrl] = useState<string>("");
  const [bounds] = useState(getMapBounds());
  
  // Hidden canvas for simulation logic
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  
  // State for CA (Burning/Safe/Burnt)
  const ROWS = 320;
  const COLS = 400;
  const stateGrid = useRef(new Uint8Array(ROWS * COLS));

  // 1. Initialize Canvas & Ignite
  useEffect(() => {
    // Create an off-screen canvas
    const canvas = document.createElement('canvas');
    canvas.width = COLS;
    canvas.height = ROWS;
    canvasRef.current = canvas;

    // Initial Ignition (Start Fire in Center)
    const centerIdx = Math.floor(ROWS / 2) * COLS + Math.floor(COLS / 2);
    stateGrid.current[centerIdx] = 1;
  }, []);

  // 2. The Simulation Loop
  useEffect(() => {
    let animationId: number;
    
    const loop = () => {
      if (!canvasRef.current) return;
      const ctx = canvasRef.current.getContext('2d');
      if (!ctx) return;

      updateFireState();
      drawToCanvas(ctx);
      
      // Export canvas to Image URL for Leaflet
      // We throttle this to save performance
      if (Math.random() < 0.3) { 
        setImageUrl(canvasRef.current.toDataURL());
      }

      animationId = requestAnimationFrame(loop);
    };

    loop();
    return () => cancelAnimationFrame(animationId);
  }, [probGrid]); 

  // Cellular Automata Logic
  const updateFireState = () => {
    const current = stateGrid.current;
    const next = new Uint8Array(current); 

    for (let r = 0; r < ROWS; r++) {
      for (let c = 0; c < COLS; c++) {
        const idx = r * COLS + c;
        if (current[idx] === 0) { // Safe -> Check neighbors
           if (hasBurningNeighbor(r, c) && Math.random() < probGrid[r][c]) {
             next[idx] = 1; // Ignite!
           }
        } else if (current[idx] === 1) { // Burning -> Burnt
           if (Math.random() < 0.1) next[idx] = 2; 
        } else {
           next[idx] = 2; // Stay Burnt
        }
      }
    }
    stateGrid.current = next;
  };

  const hasBurningNeighbor = (r: number, c: number) => {
    // Check 8 neighbors
    for (let dr = -1; dr <= 1; dr++) {
        for (let dc = -1; dc <= 1; dc++) {
            if (dr===0 && dc===0) continue;
            const nr = r + dr, nc = c + dc;
            if (nr>=0 && nr<ROWS && nc>=0 && nc<COLS) {
                if (stateGrid.current[nr*COLS+nc] === 1) return true;
            }
        }
    }
    return false;
  };

  const drawToCanvas = (ctx: CanvasRenderingContext2D) => {
    ctx.clearRect(0, 0, COLS, ROWS);
    const imgData = ctx.createImageData(COLS, ROWS);
    const data = imgData.data;

    for (let i = 0; i < stateGrid.current.length; i++) {
        const state = stateGrid.current[i];
        if (state === 1) { // Burning (Red/Orange)
            const ptr = i * 4;
            data[ptr] = 255;     // R
            data[ptr + 1] = 69;  // G
            data[ptr + 2] = 0;   // B
            data[ptr + 3] = 200; // Alpha
        } else if (state === 2) { // Burnt (Black Ash)
            const ptr = i * 4;
            data[ptr] = 20;
            data[ptr + 1] = 20;
            data[ptr + 2] = 20;
            data[ptr + 3] = 100;
        }
    }
    ctx.putImageData(imgData, 0, 0);
  };

  if (!imageUrl) return null;
  return <ImageOverlay url={imageUrl} bounds={bounds}  />;
};

export default FireOverlay;