"use client"

import { useEffect, useRef, useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Navigation } from "@/components/navigation"
import { Button } from "@/components/ui/button"
import MapSimulation from "./MapSimulation"
import { DesignNotes } from "./DesignNotes"
import { SimulationControls } from "./SimulationControls"
import {
  Grid,
  loadImageGrid,
  simulateCA,
  drawFrameToCanvas,
  drawMapPreview,
  mulberry32,
  idx,
} from "./simulationUtils"

export default function Page() {
  // parameters
  const [windSpeed, setWindSpeed] = useState(6)
  const [windDir, setWindDir] = useState(90)
  const [humidity, setHumidity] = useState(25)
  const [ignitionThreshold, setIgnitionThreshold] = useState(0.6)
  const [nSteps, setNSteps] = useState(100)
  const [seed, setSeed] = useState(42)
  const [status, setStatus] = useState("Idle")

  // raster layers
  const [w, setW] = useState(0)
  const [h, setH] = useState(0)
  const [dem, setDem] = useState<Grid | null>(null)
  const [fuel, setFuel] = useState<Grid | null>(null)
  const [fireProb, setFireProb] = useState<Grid | null>(null)

  // frames & playback
  const [frames, setFrames] = useState<Uint8Array[]>([])
  const [cur, setCur] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [speedMs, setSpeedMs] = useState(400)
  const timerRef = useRef<number | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)

  // Compute effective playback interval (ms) that speeds up with higher wind and threshold
  const effectiveMs = () => {
    // base from slider; higher wind and threshold speed up playback
    const windFactor = 1 + windSpeed / 15 // 0.. ~3.6x for 40 m/s
    const ignFactor = 1 + ignitionThreshold * 1.5 // 1..2.5x
    const ms = speedMs / (windFactor * ignFactor)
    return Math.max(60, Math.min(1200, Math.floor(ms)))
  }

  async function loadDefaultMap() {
    setStatus("Loading map…")
    const { w, h, dem, fuel } = await loadImageGrid("/images/default-map.png", 200)
    setW(w)
    setH(h)
    setDem(dem)
    setFuel(fuel)

    // Create a simple "ML fire probability" layer:
    // combine fuel with a gentle eastward gradient to illustrate
    const fp = new Float32Array(w * h)
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const k = idx(x, y, w)
        const grad = x / w // eastward bias
        const base = Math.min(1, 0.15 + 0.7 * fuel[k] * (0.3 + 0.7 * grad))
        fp[k] = base
      }
    }
    setFireProb(fp)

    // Immediately show the provided map in the simulation canvas
    setFrames([])
    setCur(0)
    const cv = canvasRef.current
    if (cv) {
      await drawMapPreview(cv, "/images/default-map.png", w, h, 3)
    }
    setStatus("Map ready")
  }

  // Ensure frames exist (compute using current params); returns frames for immediate use
  function ensureFrames(): Uint8Array[] {
    if (!dem || !fuel || !fireProb) return []
    const f = simulateCA({
      dem,
      fuel,
      fireProb,
      w,
      h,
      nSteps,
      ignitionThreshold,
      windSpeed,
      windDir,
      humidity,
      seed,
    })
    setFrames(f)
    setCur(0)
    return f
  }

  // Randomize ignition propensity to mimic the reference "Random" control
  function randomizeIgnitions() {
    if (!w || !h || !fuel) return
    const rng = mulberry32(Math.floor(Math.random() * 1e9))
    const fp = new Float32Array(w * h)
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const k = idx(x, y, w)
        const grad = x / w
        const noise = (rng() - 0.5) * 0.2
        const base = 0.1 + 0.75 * fuel[k] * (0.25 + 0.75 * grad) + noise
        fp[k] = Math.min(1, Math.max(0, base))
      }
    }
    setFireProb(fp)
    setFrames([])
    setCur(0)
    setSeed(Math.floor(Math.random() * 1e9))
    const cv = canvasRef.current
    if (cv) {
      // show map again until user starts playback
      drawMapPreview(cv, "/images/default-map.png", w, h, 3)
    }
    setStatus("Randomized base ignition")
  }

  // Play: compute frames on-demand (no separate "Run in Browser" button), then start interval with effectiveMs()
  function play() {
    if (!dem || !fuel || !fireProb) return
    let f = frames
    if (!f.length) {
      f = ensureFrames()
      if (!f.length) return
    }
    setPlaying(true)
    // restart interval with current effectiveMs
    if (timerRef.current) window.clearInterval(timerRef.current)
    const len = f.length
    const t = window.setInterval(() => {
      setCur((c) => (c + 1) % len)
    }, effectiveMs())
    timerRef.current = t as unknown as number
  }

  // Pause: stop playback
  function pause() {
    setPlaying(false)
    if (timerRef.current) {
      window.clearInterval(timerRef.current)
      timerRef.current = null
    }
  }

  // Step: ensure frames exist, then advance one frame
  function step() {
    let f = frames
    if (!f.length) f = ensureFrames()
    if (f.length) setCur((c) => (c + 1) % f.length)
  }

  // Reset: stop playback, clear frames, and show the map preview
  function reset() {
    pause()
    setFrames([])
    setCur(0)
    const cv = canvasRef.current
    if (cv && w && h) {
      drawMapPreview(cv, "/images/default-map.png", w, h, 3)
    }
    setStatus("Cleared")
  }

  // Whenever speed slider, wind speed, or ignition threshold change, update the timer while playing
  useEffect(() => {
    if (!playing || !frames.length) return
    if (timerRef.current) {
      window.clearInterval(timerRef.current)
      timerRef.current = null
    }
    const len = frames.length
    const t = window.setInterval(() => {
      setCur((c) => (c + 1) % len)
    }, effectiveMs())
    timerRef.current = t as unknown as number
    return () => {
      if (timerRef.current) window.clearInterval(timerRef.current)
    }
  }, [playing, speedMs, windSpeed, ignitionThreshold, frames])

  // Autoload default map on first render
  useEffect(() => {
    loadDefaultMap().catch(() => setStatus("Failed to load map"))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const [grid, setGrid] = useState<number[][] | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const getGrid = async () => {
      try {
        const response = await fetch("http://127.0.0.1:8000/fire-grid")
        if (!response.ok) throw new Error("Failed to fetch fire grid")

        const data = await response.json()
        setGrid(data.grid)
      } catch (err) {
        console.error(err)
        setError("Could not connect to the backend server.")
      }
    }

    getGrid()
  }, [])
  const fetchGrid = async () => {
    try {
      const res = await fetch("http://127.0.0.1:8000/fire-grid")
      const data = await res.json()
      setGrid(data.grid)
    } catch (e) {
      console.error("Connection Error", e)
    }
  }

  // Initial Load
  useEffect(() => {
    fetchGrid()
  }, [])

  return (
    <main className="min-h-screen bg-[#060609] text-foreground">
      <Navigation />
      <div className="mx-auto max-w-7xl p-6 grid gap-6 md:grid-cols-[380px_1fr]">
        <div className="space-y-6">
          <Card className="glass-card-bright border-white/[0.08]">
            <CardHeader className="border-b border-white/[0.06]">
              <CardTitle className="text-pretty bg-gradient-to-r from-orange-400 to-red-400 bg-clip-text text-transparent">
                Wildfire CA Simulator
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <p className="text-sm text-muted-foreground">
                Uses the provided map as default to derive DEM and fuel. The CA consumes an ML fire probability layer
                and geodata to simulate spread over time.
              </p>

              <SimulationControls
                onLoadMap={loadDefaultMap}
                onStart={play}
                onPause={pause}
                onStep={step}
                onRandom={randomizeIgnitions}
                onReset={reset}
                playing={playing}
                windSpeed={windSpeed}
                setWindSpeed={setWindSpeed}
                windDir={windDir}
                setWindDir={setWindDir}
                humidity={humidity}
                setHumidity={setHumidity}
                ignitionThreshold={ignitionThreshold}
                setIgnitionThreshold={setIgnitionThreshold}
                nSteps={nSteps}
                setNSteps={setNSteps}
                seed={seed}
                setSeed={setSeed}
                speedMs={speedMs}
                setSpeedMs={setSpeedMs}
                effectiveMs={effectiveMs()}
              />

              <div className="text-xs text-muted-foreground">
                Status: {status} {w && h ? `• Grid ${w}×${h}` : ""}
              </div>
            </CardContent>
          </Card>

          <DesignNotes />
        </div>

        <div className="space-y-4">
          <Card className="glass-card-bright border-white/[0.08]">
            <CardHeader className="border-b border-white/[0.06]">
              <CardTitle className="text-pretty bg-gradient-to-r from-orange-400 to-amber-300 bg-clip-text text-transparent">
                Simulation Canvas
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex flex-col items-start gap-3">
                <div className="flex flex-col items-center w-full">
                  {grid ? (
                    <MapSimulation probGrid={grid} />
                  ) : (
                    <div className="flex flex-col items-center justify-center h-[500px] w-full bg-white/[0.02] border border-white/[0.06] rounded-xl animate-pulse">
                      <div className="w-10 h-10 border-2 border-orange-500/40 border-t-orange-500 rounded-full animate-spin mb-4" />
                      <p className="text-gray-500 font-medium">Loading Map & Fire Data...</p>
                    </div>
                  )}
                </div>
                <div className="text-xs text-muted-foreground">Green=Unburnt, Orange=Burning, Gray=Burnt</div>
              </div>
            </CardContent>
          </Card>

          <Card className="glass-card-bright border-white/[0.08]">
            <CardHeader className="border-b border-white/[0.06]">
              <CardTitle className="text-pretty text-gray-200">Demo API (optional)</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2 text-sm">
              <p>
                You can also fetch demo frames from the API at <code>/api/simulate</code> (returns small synthetic
                frames).
              </p>
              <Button
                variant="outline"
                onClick={async () => {
                  setStatus("Fetching demo frames…")
                  const res = await fetch("/api/simulate", { method: "POST" })
                  const data = await res.json()
                  const frames: Uint8Array[] = data.frames.map((f: number[][]) => Uint8Array.from(f.flat()))
                  const [hh, ww] = data.frames_shape
                  setW(ww)
                  setH(hh)
                  setFrames(frames)
                  setCur(0)
                  setStatus("Demo frames loaded")
                }}
              >
                Load Demo Frames
              </Button>
            </CardContent>
          </Card>
        </div>
      </div>
    </main>
  )
}
