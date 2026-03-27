"use client";

import React, { useState, useEffect } from "react";
import dynamic from "next/dynamic";

// Import the map component dynamically, disabling SSR because Leaflet uses window/document
const MapSimulation = dynamic(() => import("./MapSimulation"), {
    ssr: false,
    loading: () => (
        <div className="flex items-center justify-center min-h-screen bg-slate-950">
            <p className="text-orange-500 animate-pulse font-bold text-xl">
                Initializing Map Engine...
            </p>
        </div>
    ),
});

export default function Page() {
    return <MapSimulation />;
}
