"use client"

import { useEffect, useRef, useState } from "react"
import { Slider } from "@/components/ui/slider"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Navigation } from "@/components/navigation"
import { Label } from "@/components/ui/label"
import { Input } from "@/components/ui/input"
import SimulationCanvas from "./SimulationCanvas"
import dynamic from 'next/dynamic'

const MapSimulation = dynamic(() => import('./MapSimulation'), {
  ssr: false,
  loading: () => <p>Loading Map...</p>,
})

function mulberry32(a: number) {
  return () => {
    let t = (a += 0x6d2b79f5)
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

type Grid = Float32Array
type U8Grid = Uint8Array

const OFFS: Array<[number, number]> = [
  [-1, -1],
  [-1, 0],
  [-1, 1],
  [0, -1],
  [0, 1],
  [1, -1],
  [1, 0],
  [1, 1],
]
const OFFS_UNIT: Array<[number, number]> = OFFS.map(([dy, dx]) => {
  const n = Math.hypot(dy, dx)
  return [dy / n, dx / n]
})

function idx(x: number, y: number, w: number) {
  return y * w + x
}

async function loadImageGrid(
  src: string,
  targetW = 192,
): Promise<{
  w: number
  h: number
  dem: Grid
  fuel: Grid
}> {
  const img = new Image()
  img.crossOrigin = "anonymous"
  img.src = src
  await img.decode()

  // scale image proportionally to target width
  const scale = targetW / img.width
  const w = Math.max(16, Math.floor(img.width * scale))
  const h = Math.max(16, Math.floor(img.height * scale))

  const cv = document.createElement("canvas")
  cv.width = w
  cv.height = h
  const ctx = cv.getContext("2d")!
  ctx.drawImage(img, 0, 0, w, h)
  const { data } = ctx.getImageData(0, 0, w, h)

  const dem = new Float32Array(w * h)
  const fuel = new Float32Array(w * h)

  // simple conversions:
  // - dem from luminance
  // - fuel from greenness
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const p = (y * w + x) * 4
      const r = data[p] / 255
      const g = data[p + 1] / 255
      const b = data[p + 2] / 255
      const lum = 0.2126 * r + 0.7152 * g + 0.0722 * b // [0..1]
      dem[idx(x, y, w)] = lum // proxy elevation
      // Vegetation proxy: emphasize green, de-emphasize bright (rock/snow)
      fuel[idx(x, y, w)] = Math.min(1, Math.max(0, g * 0.8 + (g - r) * 0.2 - (lum - 0.6) * 0.3))
    }
  }

  // normalize dem to [0..1]
  let min = Number.POSITIVE_INFINITY,
    max = Number.NEGATIVE_INFINITY
  for (let i = 0; i < dem.length; i++) {
    if (dem[i] < min) min = dem[i]
    if (dem[i] > max) max = dem[i]
  }
  const rng = Math.max(1e-6, max - min)
  for (let i = 0; i < dem.length; i++) dem[i] = (dem[i] - min) / rng

  // clamp fuel
  for (let i = 0; i < fuel.length; i++) {
    fuel[i] = Math.min(1, Math.max(0, fuel[i]))
  }

  return { w, h, dem, fuel }
}

function computeSlope(dem: Grid, w: number, h: number): Grid {
  const out = new Float32Array(w * h)
  const get = (x: number, y: number) => dem[idx((x + w) % w, (y + h) % h, w)]
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const dzdx = (get(x + 1, y) - get(x - 1, y)) * 0.5
      const dzdy = (get(x, y + 1) - get(x, y - 1)) * 0.5
      const s = Math.sqrt(dzdx * dzdx + dzdy * dzdy)
      out[idx(x, y, w)] = s
    }
  }
  // scale slope to ~[0..1]
  let m = 0
  for (let i = 0; i < out.length; i++) if (out[i] > m) m = out[i]
  const scale = m > 1e-6 ? 1 / m : 1
  for (let i = 0; i < out.length; i++) out[i] *= scale
  return out
}

function fuelFactor(f: number) {
  return Math.min(2, Math.max(0.2, 0.2 + 1.8 * f))
}

function slopeFactor(s: number, k = 0.8) {
  // exp(k * slope) then clamp
  const v = Math.exp(k * s)
  return Math.min(3, Math.max(0.8, v))
}

function humidityFactor(humidityPct: number) {
  // 0..100 -> [0.1..1]
  const v = 1 - (humidityPct / 100) * 0.9
  return Math.min(1, Math.max(0.1, v))
}

function windAlignmentFactor(windSpeed: number, windDirDegTo: number, ndy: number, ndx: number) {
  // wind direction points TO (screen coords: +x right, +y down)
  const wdx = Math.cos((windDirDegTo * Math.PI) / 180)
  const wdy = -Math.sin((windDirDegTo * Math.PI) / 180)
  const dot = wdx * ndx + wdy * ndy // [-1..1]
  const mag = windSpeed / (windSpeed + 1)
  const v = 1 + dot * mag // [0..2] approx
  return Math.min(3, Math.max(0.2, v))
}
function simulateCA(params: {
  fireProb: Grid // 0..1
  dem: Grid // 0..1 proxy
  fuel: Grid // 0..1
  w: number
  h: number
  nSteps: number
  ignitionThreshold: number
  windSpeed: number
  windDir: number // degrees TO
  humidity: number // %
  seed: number
}) {
  const { fireProb, dem, fuel, w, h, nSteps, ignitionThreshold, windSpeed, windDir, humidity, seed } = params

  const rng = mulberry32(seed)
  const slope = computeSlope(dem, w, h)
  const state = new Uint8Array(w * h) // 0 unburnt, 1 burning, 2 burnt

  // 🔥 Fire starts at center
  const centerX = Math.floor(w / 2)
  const centerY = Math.floor(h / 2)
  state[idx(centerX, centerY, w)] = 1

  const frames: Array<Uint8Array> = [state.slice(0)]
  const hf = humidityFactor(humidity)

  for (let step = 0; step < nSteps; step++) {
    const next = state.slice(0)
    // burning -> burnt
    for (let i = 0; i < state.length; i++) if (state[i] === 1) next[i] = 2

    for (let n = 0; n < OFFS.length; n++) {
      const [dy, dx] = OFFS[n]
      const [ndy, ndx] = OFFS_UNIT[n]

      // 🌬️ Smooth wind bias (cosine-based curve instead of linear box)
      const angle = Math.atan2(-ndy, ndx) * (180 / Math.PI) // neighbor direction
      let diff = Math.abs(((angle - windDir + 540) % 360) - 180) // angular difference
      const windBias = Math.cos((diff * Math.PI) / 180) // -1..1 curve
      const wf = 1 + windBias * (windSpeed / 10) // smoother influence

      for (let y = 0; y < h; y++) {
        const ny = (y - dy + h) % h
        for (let x = 0; x < w; x++) {
          const nx = (x - dx + w) % w
          const center = idx(x, y, w)
          const neighbor = idx(nx, ny, w)

          if (state[center] !== 0) continue
          if (state[neighbor] !== 1) continue

          const base = fireProb[center]
          const ff = fuelFactor(fuel[center])
          const sf = slopeFactor(slope[neighbor])

          // 🎲 Add randomness for irregular/curved spread
          const noise = 0.85 + rng() * 0.3 // [0.85..1.15]

          let P = base * ff * sf * wf * hf * noise
          if (P > 1) P = 1
          if (rng() < P) next[center] = 1
        }
      }
    }

    for (let i = 0; i < state.length; i++) state[i] = next[i]
    frames.push(state.slice(0))
  }

  return frames
}


function stateColor(v: number): [number, number, number] {
  if (v === 0) return [34, 139, 34] // unburnt: green
  if (v === 1) return [255, 69, 0] // burning: orange/red
  if (v === 2) return [80, 80, 80] // burnt: gray
  return [0, 0, 0]
}

function drawFrameToCanvas(cv: HTMLCanvasElement, frame: U8Grid, w: number, h: number, scale = 3) {
  const ctx = cv.getContext("2d")!
  cv.width = w
  cv.height = h
  const img = ctx.createImageData(w, h)
  let j = 0
  for (let i = 0; i < frame.length; i++) {
    const [r, g, b] = stateColor(frame[i])
    img.data[j++] = r
    img.data[j++] = g
    img.data[j++] = b
    img.data[j++] = 255
  }
  ctx.putImageData(img, 0, 0)
  cv.style.width = `${w * scale}px`
  cv.style.height = `${h * scale}px`
}

// Helper to draw the map image into the simulation canvas as a preview
async function drawMapPreview(cv: HTMLCanvasElement, src: string, w: number, h: number, scale = 3) {
  const img = new Image()
  img.crossOrigin = "anonymous"
  img.src = src
  await img.decode()
  const ctx = cv.getContext("2d")!
  cv.width = w
  cv.height = h
  ctx.drawImage(img, 0, 0, w, h)
  cv.style.width = `${w * scale}px`
  cv.style.height = `${h * scale}px`
}

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
      // ignitePoints: [],
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

  useEffect(() => {
    const cv = canvasRef.current
    if (!cv || !frames.length) return
    drawFrameToCanvas(cv, frames[cur], w, h, 3)
  }, [frames, cur, w, h])

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

  const [grid, setGrid] = useState<number[][] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const getGrid = async () => {
      try {
        const response = await fetch('http://127.0.0.1:8000/fire-grid');
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
          wind_speed: windSpeed,
          wind_direction: windDir,
          humidity: humidity
        })
      });
      // Re-fetch grid to see changes visually
      fetchGrid();
    } catch (e) { console.error("Failed to update dynamics", e); }
  };



  return (
    <main className="min-h-screen bg-[#060609] text-foreground">
      <Navigation />
      <div className="mx-auto max-w-7xl p-6 grid gap-6 md:grid-cols-[380px_1fr]">
        <div className="space-y-6">
          <Card className="glass-card-bright border-white/[0.08]">
            <CardHeader className="border-b border-white/[0.06]">
              <CardTitle className="text-pretty bg-gradient-to-r from-orange-400 to-red-400 bg-clip-text text-transparent">Wildfire CA Simulator</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <p className="text-sm text-muted-foreground">
                Uses the provided map as default to derive DEM and fuel. The CA consumes an ML fire probability layer
                and geodata to simulate spread over time.
              </p>

              {/* Replace old buttons with reference-style controls */}
              <div className="flex flex-wrap items-center gap-2">
                <Button onClick={loadDefaultMap} variant="secondary">
                  Load Default Map
                </Button>
                {!playing ? (
                  <Button onClick={play}>Start</Button>
                ) : (
                  <Button variant="secondary" onClick={pause}>
                    Pause
                  </Button>
                )}
                <Button variant="outline" onClick={step}>
                  Step
                </Button>
                <Button variant="outline" onClick={randomizeIgnitions}>
                  Random
                </Button>
                <Button variant="ghost" onClick={reset}>
                  Clear
                </Button>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <Label htmlFor="windSpeed">Wind speed (km/h)</Label>
                  <Input
                    id="windSpeed"
                    type="number"
                    min={0}
                    max={100}
                    step={1}
                    value={windSpeed}
                    onChange={(e) => setWindSpeed(Number(e.target.value))}
                  />
                </div>
                <div>
                  <Label htmlFor="windDir">Wind dir (deg TO)</Label>
                  <Input
                    id="windDir"
                    type="number"
                    min={0}
                    max={360}
                    step={1}
                    value={windDir}
                    onChange={(e) => setWindDir(Number(e.target.value))}
                  />
                </div>
                <div>
                  <Label htmlFor="humidity">Humidity (%)</Label>
                  <Input
                    id="humidity"
                    type="number"
                    min={0}
                    max={100}
                    step={1}
                    value={humidity}
                    onChange={(e) => setHumidity(Number(e.target.value))}
                  />
                </div>
                <div>
                  <Label htmlFor="threshold">Ignition threshold</Label>
                  <Input
                    id="threshold"
                    type="number"
                    min={0}
                    max={1}
                    step={0.05}
                    value={ignitionThreshold}
                    onChange={(e) => setIgnitionThreshold(Number(e.target.value))}
                  />
                </div>
                <div>
                  <Label htmlFor="steps">Steps</Label>
                  <Input
                    id="steps"
                    type="number"
                    min={1}
                    max={200}
                    step={1}
                    value={nSteps}
                    onChange={(e) => setNSteps(Number(e.target.value))}
                  />
                </div>
                <div>
                  <Label htmlFor="seed">Random seed</Label>
                  <Input
                    id="seed"
                    type="number"
                    min={1}
                    step={1}
                    value={seed}
                    onChange={(e) => setSeed(Number(e.target.value))}
                  />
                </div>
              </div>

              {/* Playback speed still available, used as base for effective speed */}
              <div className="space-y-2">
                <Label>Speed (base ms)</Label>
                <Slider value={[speedMs]} min={100} max={1500} step={50} onValueChange={(v) => setSpeedMs(v[0])} />
                <div className="text-xs text-muted-foreground">
                  Effective interval: {effectiveMs()} ms (faster with higher wind & threshold)
                </div>
              </div>

              <div className="text-xs text-muted-foreground">
                Status: {status} {w && h ? `• Grid ${w}×${h}` : ""}
              </div>
            </CardContent>
          </Card>

          <Card className="glass-card-bright border-white/[0.08]">
            <CardHeader className="border-b border-white/[0.06]">
              <CardTitle className="text-pretty text-gray-200">Design Notes</CardTitle>
            </CardHeader>
            <CardContent className="text-sm space-y-2 text-gray-400">
              <ul className="list-disc pl-5 space-y-1">
                <li>States: 0=Unburnt, 1=Burning, 2=Burnt. Moore 8-neighborhood.</li>
                <li>Base probability from ML per-pixel map.</li>
                <li>Fuel scales ignition; denser fuel → higher chance.</li>
                <li>Slope factor f_slope = exp(k·slope). Uphill spreads faster.</li>
                <li>Wind factor from dot-product with neighbor direction.</li>
                <li>Humidity reduces spread probability.</li>
                <li>Distance: immediate neighbors; ember spotting optional.</li>
                <li>Stochastic CA: RNG decides ignition per neighbor per step.</li>
              </ul>
            </CardContent>
          </Card>
        </div>

        <div className="space-y-4">
          <Card className="glass-card-bright border-white/[0.08]">
            <CardHeader className="border-b border-white/[0.06]">
              <CardTitle className="text-pretty bg-gradient-to-r from-orange-400 to-amber-300 bg-clip-text text-transparent">Simulation Canvas</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex flex-col items-start gap-3">
                {/* <div className="flex flex-col items-center">
                  {grid ? (
                    <SimulationCanvas probGrid={grid} />
                  ) : (
                    <div className="text-slate-400 animate-pulse">Loading Model Grid...</div>
                  )}
                </div> */}
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
