
export const OFFS: Array<[number, number]> = [
    [-1, -1],
    [-1, 0],
    [-1, 1],
    [0, -1],
    [0, 1],
    [1, -1],
    [1, 0],
    [1, 1],
]
export const OFFS_UNIT: Array<[number, number]> = OFFS.map(([dy, dx]) => {
    const n = Math.hypot(dy, dx)
    return [dy / n, dx / n]
})

export type Grid = Float32Array
export type U8Grid = Uint8Array

export function idx(x: number, y: number, w: number) {
    return y * w + x
}

export function mulberry32(a: number) {
    return () => {
        let t = (a += 0x6d2b79f5)
        t = Math.imul(t ^ (t >>> 15), t | 1)
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296
    }
}

export async function loadImageGrid(
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

export function computeSlope(dem: Grid, w: number, h: number): Grid {
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

export function fuelFactor(f: number) {
    return Math.min(2, Math.max(0.2, 0.2 + 1.8 * f))
}

export function slopeFactor(s: number, k = 0.8) {
    // exp(k * slope) then clamp
    const v = Math.exp(k * s)
    return Math.min(3, Math.max(0.8, v))
}

export function humidityFactor(humidityPct: number) {
    // 0..100 -> [0.1..1]
    const v = 1 - (humidityPct / 100) * 0.9
    return Math.min(1, Math.max(0.1, v))
}

export function windAlignmentFactor(windSpeed: number, windDirDegTo: number, ndy: number, ndx: number) {
    // wind direction points TO (screen coords: +x right, +y down)
    const wdx = Math.cos((windDirDegTo * Math.PI) / 180)
    const wdy = -Math.sin((windDirDegTo * Math.PI) / 180)
    const dot = wdx * ndx + wdy * ndy // [-1..1]
    const mag = windSpeed / (windSpeed + 1)
    const v = 1 + dot * mag // [0..2] approx
    return Math.min(3, Math.max(0.2, v))
}

export function simulateCA(params: {
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

export function stateColor(v: number): [number, number, number] {
    if (v === 0) return [34, 139, 34] // unburnt: green
    if (v === 1) return [255, 69, 0] // burning: orange/red
    if (v === 2) return [80, 80, 80] // burnt: gray
    return [0, 0, 0]
}

export function drawFrameToCanvas(cv: HTMLCanvasElement, frame: U8Grid, w: number, h: number, scale = 3) {
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

export async function drawMapPreview(cv: HTMLCanvasElement, src: string, w: number, h: number, scale = 3) {
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
