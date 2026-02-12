
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

export function DesignNotes() {
    return (
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
    )
}
