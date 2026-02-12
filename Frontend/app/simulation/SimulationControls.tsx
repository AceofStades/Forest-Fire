
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Slider } from "@/components/ui/slider"

interface SimulationControlsProps {
    // Actions
    onLoadMap: () => void
    onStart: () => void
    onPause: () => void
    onStep: () => void
    onRandom: () => void
    onReset: () => void

    // State
    playing: boolean
    windSpeed: number
    setWindSpeed: (v: number) => void
    windDir: number
    setWindDir: (v: number) => void
    humidity: number
    setHumidity: (v: number) => void
    ignitionThreshold: number
    setIgnitionThreshold: (v: number) => void
    nSteps: number
    setNSteps: (v: number) => void
    seed: number
    setSeed: (v: number) => void
    speedMs: number
    setSpeedMs: (v: number) => void
    effectiveMs: number
}

export function SimulationControls({
    onLoadMap,
    onStart,
    onPause,
    onStep,
    onRandom,
    onReset,
    playing,
    windSpeed,
    setWindSpeed,
    windDir,
    setWindDir,
    humidity,
    setHumidity,
    ignitionThreshold,
    setIgnitionThreshold,
    nSteps,
    setNSteps,
    seed,
    setSeed,
    speedMs,
    setSpeedMs,
    effectiveMs,
}: SimulationControlsProps) {
    return (
        <div className="space-y-4">
            {/* Buttons */}
            <div className="flex flex-wrap items-center gap-2">
                <Button onClick={onLoadMap} variant="secondary">
                    Load Default Map
                </Button>
                {!playing ? (
                    <Button onClick={onStart}>Start</Button>
                ) : (
                    <Button variant="secondary" onClick={onPause}>
                        Pause
                    </Button>
                )}
                <Button variant="outline" onClick={onStep}>
                    Step
                </Button>
                <Button variant="outline" onClick={onRandom}>
                    Random
                </Button>
                <Button variant="ghost" onClick={onReset}>
                    Clear
                </Button>
            </div>

            {/* Inputs Grid */}
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

            {/* Speed Slider */}
            <div className="space-y-2">
                <Label>Speed (base ms)</Label>
                <Slider
                    value={[speedMs]}
                    min={100}
                    max={1500}
                    step={50}
                    onValueChange={(v) => setSpeedMs(v[0])}
                />
                <div className="text-xs text-muted-foreground">
                    Effective interval: {effectiveMs} ms (faster with higher wind & threshold)
                </div>
            </div>
        </div>
    )
}
