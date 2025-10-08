# 🔥 Forest Fire Spread Prediction and Simulation Portal

A full-stack **Next.js** web application that predicts forest fire probabilities and simulates fire spread over time using AI/ML techniques. Designed for vulnerable regions like **Uttarakhand**, it supports real-time 3D/2D visualization, geospatial raster uploads, and predictive insights.

---

## 🧠 Problem Statement

Forest fires threaten biodiversity, air quality, and public safety. Early detection and simulation are crucial for disaster preparedness. This solution provides:

* **AI/ML-based fire risk prediction**
* **Fire spread simulation (1–12 hours)**
* **Raster outputs (30m resolution)**
* **Support for user-uploaded map data**
* **Real-time 3D/2D visual rendering**

---

## 🌍 Features

| Feature                        | Description                                                  |
| ------------------------------ | ------------------------------------------------------------ |
| 🔮 **ML Fire Risk Prediction** | Predicts fire/no-fire zones using terrain, weather, and LULC |
| 🔥 **Fire Spread Simulation**  | Simulates spread based on slope, fuel, wind for 1–12 hours   |
| 🗺️ **GeoTIFF Raster Output**  | View and download predicted zones as `.tif` files            |
| 🧭 **3D Visualization (GLTF)** | Interactive terrain and fire spread rendered in 3D           |
| 📤 **Upload Raster Form**      | Upload your own GeoTIFF to run fire prediction               |
| 📊 **Dashboards & Analytics**  | Explore model insights and statistics                        |
| 📁 **Reports Export**          | Export raster maps, logs, and risk summaries                 |
| ⚡ **Next.js SSR/ISR**          | Optimized page loading and server rendering                  |
| 💡 **Responsive UI**           | Tailwind CSS + ShadCN + Framer Motion                        |

---

## 🧰 Tech Stack

| Layer               | Tools Used                                                                   |
| ------------------- | ---------------------------------------------------------------------------- |
| **Frontend**        | Next.js 14 (App Router) + TypeScript + Tailwind CSS + ShadCN + Framer Motion |
| **3D Rendering**    | Three.js + GLTFLoader                                                        |
| **Backend API**     | Python (FastAPI or Flask) for ML predictions and raster handling             |
| **Geospatial**      | Rasterio, GDAL, OpenLayers, Deck.gl                                          |
| **Deployment**      | Vercel (Frontend), Render/AWS/GCP (Backend)                                  |
| **Version Control** | GitHub                                                                       |

---

## 📁 Folder Structure (Frontend)

```
app/
├── 3d/                         # GLTF 3D Viewer
├── about/                      # About section
├── analytics/
│   ├── dashboard/              # Fire stats, overlays
│   └── ml-insights/            # Model performance
├── report/                     # Export section
├── upload/                     # Upload form + results
├── components/
│   └── ui/                     # Buttons, toasts, nav
├── hooks/                      # Custom hooks
├── public/                     # Static assets
├── styles/                     # Global styles
```

---

## 🧪 How It Works

### 🔢 Inputs

* Weather (Temp, Rainfall, Wind, Humidity)
* Topography (Slope, Aspect from DEM)
* LULC (Fuel model)
* Human Activity (Roads, Settlement from GHSL)
* Fire Labels (Historical VIIRS)

### 🔍 Prediction Flow

1. Raster stack built from inputs
2. U-NET/LSTM predicts fire/no fire raster
3. Output stored as `.tif`, visualized in web interface

### 🔄 Fire Spread Simulation

* Cellular Automata logic based on wind direction, slope, fuel
* Supports simulation for 1, 2, 3, 6, and 12-hour intervals
* Renders time-step spread in map/3D UI

---

## 🧪 Upload & Predict (Custom GeoTIFF)

Users can upload a **custom raster map** using the `Upload` tab. The backend returns a `.tif` fire prediction, which is visualized instantly.

---

## 🚀 Getting Started

```bash
# 1. Clone the repo
git clone https://github.com/your-org/forest-fire-prediction.git
cd forest-fire-prediction

# 2. Install dependencies
npm install --legacy-peer-deps

# 3. Start dev server
npm run dev

# 4. Backend (separate repo/folder)
cd backend
python app.py  # or uvicorn main:app
```

---

## 📌 Navigation Tabs

| Tab                    | Purpose                         |
| ---------------------- | ------------------------------- |
| `/` (Home)             | Overview, objectives            |
| `/upload`              | Upload your own map             |
| `/analytics/dashboard` | Real-time stats and ML insights |
| `/3d`                  | Interactive terrain view (GLTF) |
| `/report`              | Download maps and PDFs          |
| `/about`               | Team, goals, data sources       |

---

## 📚 Data Sources

* 🌍 [Bhuvan LULC](https://bhuvan.nrsc.gov.in)
* 🔥 [VIIRS Fire Dataset](https://firms.modaps.eosdis.nasa.gov)
* ⛰️ [Bhoonidhi DEM](https://bhoonidhi.nrsc.gov.in)
* 🌫️ [ERA5/MOSDAC Weather Data](https://cds.climate.copernicus.eu)
* 🏘️ [GHSL Human Settlement Data](https://ghsl.jrc.ec.europa.eu)

---

## Deployed Link
* https://forest-fire-predictor-nine.vercel.app

---

## 📌 Future Additions

* SMS/Email alerts for fire-prone areas
* Live satellite map overlay
* AI-based smoke detection
* Drone-based data ingestion

---

## 🧾 License

MIT License
