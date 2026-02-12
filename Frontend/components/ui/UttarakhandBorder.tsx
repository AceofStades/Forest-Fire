"use client";
import React, { useEffect, useState } from 'react';
import { GeoJSON } from 'react-leaflet';

export default function UttarakhandBorder() {
  const [geoData, setGeoData] = useState<any>(null);

  useEffect(() => {
    // Fetch Uttarakhand GeoJSON from a public GitHub repository
    fetch('https://raw.githubusercontent.com/geohacker/india/master/state/uttarakhand.geojson')
      .then(res => res.json())
      .then(data => setGeoData(data))
      .catch(err => console.error("Failed to load state boundary:", err));
  }, []);

  if (!geoData) return null;

  return (
    <GeoJSON 
      data={geoData}
      style={{
        color: '#f97316',       // Orange outline (matches your theme)
        weight: 3,              // Thickness of the border
        fillColor: '#f97316',   // Slight orange fill
        fillOpacity: 0.05,      // Very transparent fill (just a tint)
        dashArray: '5, 5'       // Dotted line effect
      }} 
    />
  );
}