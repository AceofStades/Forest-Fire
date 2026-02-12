// Define the physical boundaries of your 320x400 grid
// This box covers a forest area near Pauri, Uttarakhand
export const FOREST_BOUNDS = {
  north: 30.1500, // Top Lat
  south: 30.0500, // Bottom Lat
  west: 79.0000,  // Left Lng
  east: 79.2000   // Right Lng
};

// Returns the LatLngBounds array for Leaflet: [[South, West], [North, East]]
export const getMapBounds = (): [[number, number], [number, number]] => {
  return [
    [FOREST_BOUNDS.south, FOREST_BOUNDS.west],
    [FOREST_BOUNDS.north, FOREST_BOUNDS.east]
  ];
};

export const gridToGPS = (row: number, col: number): [number, number] => {
  // Linear interpolation: Map 0..320 to North..South
  const lat = FOREST_BOUNDS.north - (row / 320) * (FOREST_BOUNDS.north - FOREST_BOUNDS.south);
  const lng = FOREST_BOUNDS.west + (col / 400) * (FOREST_BOUNDS.east - FOREST_BOUNDS.west);
  return [lat, lng];
};

export const gpsToGrid = (lat: number, lng: number): [number, number] => {
  const row = Math.floor(((FOREST_BOUNDS.north - lat) / (FOREST_BOUNDS.north - FOREST_BOUNDS.south)) * 320);
  const col = Math.floor(((lng - FOREST_BOUNDS.west) / (FOREST_BOUNDS.east - FOREST_BOUNDS.west)) * 400);
  // Clamp values to stay inside grid
  return [
    Math.max(0, Math.min(319, row)),
    Math.max(0, Math.min(399, col))
  ];
};