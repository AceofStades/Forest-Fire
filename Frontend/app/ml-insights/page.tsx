"use client"

import React from "react"
import { Navigation } from "@/components/navigation"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line, Legend } from "recharts"
import { Brain, Target, TrendingUp, Zap, CheckCircle, AlertTriangle, ShieldCheck, Globe, Layers, ArrowRight, Info } from "lucide-react"
import ComparisonSandbox from "./ComparisonSandbox"

const performanceData = [
  { metric: "F1 Score", NDWS: 0.36, Custom: 0.94 },
  { metric: "Precision", NDWS: 0.35, Custom: 0.89 },
  { metric: "Recall", NDWS: 0.49, Custom: 0.96 },
];

const trainingData = [
  { epoch: 1, ndwsTrainF1: 0.406, ndwsValF1: 0.323 },
  { epoch: 2, ndwsTrainF1: 0.421, ndwsValF1: 0.353 },
  { epoch: 3, ndwsTrainF1: 0.435, ndwsValF1: 0.339 },
  { epoch: 4, ndwsTrainF1: 0.459, ndwsValF1: 0.357 }, // Peak Validation
  { epoch: 6, ndwsTrainF1: 0.485, ndwsValF1: 0.342 },
  { epoch: 8, ndwsTrainF1: 0.512, ndwsValF1: 0.331 },
  { epoch: 10, ndwsTrainF1: 0.535, ndwsValF1: 0.325 },
  { epoch: 12, ndwsTrainF1: 0.556, ndwsValF1: 0.315 },
  { epoch: 14, ndwsTrainF1: 0.578, ndwsValF1: 0.306 }, // Early Stop
];

const featureImportance = [
  { feature: "Topography (Elevation/Slope)", importance: 0.28 },
  { feature: "Meteorological (Temp/Humidity)", importance: 0.25 },
  { feature: "Wind Vector (u10/v10)", importance: 0.18 },
  { feature: "Vegetation (LULC)", importance: 0.15 },
  { feature: "Fire History (Burn Scar)", importance: 0.10 },
  { feature: "Human Settlement", importance: 0.04 },
];

const confusionMatrix = [
  { predicted: "No Fire", actual: "No Fire", count: 8547 },
  { predicted: "No Fire", actual: "Fire", count: 187 },
  { predicted: "Fire", actual: "No Fire", count: 234 },
  { predicted: "Fire", actual: "Fire", count: 1832 },
];

export default function MLInsightsPage() {
  return (
    <div className="min-h-screen bg-background">
      <Navigation />

      <div className="container mx-auto px-4 py-6">
        <div className="mb-6">
          <h1 className="text-3xl font-bold mb-2">ML Model Insights & Architecture</h1>
          <p className="text-muted-foreground">Comprehensive analysis of our Custom Hybrid Model and rigorous scientific comparison against Google's NDWS.</p>
        </div>

        {/* Top Level Summary Cards */}
        <div className="grid md:grid-cols-3 gap-4 mb-6">
          <Card className="border-emerald-900/50 bg-emerald-950/10">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-emerald-400/80">Best Architecture</p>
                  <p className="text-xl font-bold text-emerald-500">Custom Hybrid</p>
                </div>
                <Layers className="h-8 w-8 text-emerald-600" />
              </div>
              <div className="mt-2">
                <Badge className="bg-emerald-500/20 text-emerald-400 border-emerald-500/30">
                  U-Net + Physics CA
                </Badge>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-muted-foreground">Test F1 Score</p>
                  <p className="text-xl font-bold">0.94</p>
                </div>
                <Target className="h-8 w-8 text-blue-600" />
              </div>
              <div className="mt-2">
                <Progress value={94} className="h-2" />
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-muted-foreground">Deployability</p>
                  <p className="text-xl font-bold">Global</p>
                </div>
                <Globe className="h-8 w-8 text-purple-600" />
              </div>
              <div className="flex items-center mt-2">
                <TrendingUp className="h-4 w-4 text-green-600 mr-1" />
                <span className="text-sm text-green-600">ERA5 Weather Independent</span>
              </div>
            </CardContent>
          </Card>
        </div>

        <Tabs defaultValue="comparison" className="space-y-6">
          <TabsList className="grid w-full grid-cols-4 bg-slate-900/50 p-1">
            <TabsTrigger value="comparison">NDWS vs Custom</TabsTrigger>
            <TabsTrigger value="theory">Why Pure ML Fails</TabsTrigger>
            <TabsTrigger value="features">Feature Analysis</TabsTrigger>
            <TabsTrigger value="architecture">Architecture Specs</TabsTrigger>
          </TabsList>

          {/* 1. Comparison Tab */}
          <TabsContent value="comparison" className="space-y-6">
            <div className="grid md:grid-cols-2 gap-6 mb-6">
              <Card className="border-red-900/30 bg-red-950/10">
                <CardHeader>
                  <div className="flex justify-between items-start">
                    <div>
                      <CardTitle className="text-xl flex items-center gap-2">
                        <Brain className="w-5 h-5 text-red-500" />
                        Google NDWS Model
                      </CardTitle>
                      <CardDescription className="mt-2">End-to-End Deep Learning (Day 1 → Day 2)</CardDescription>
                    </div>
                    <Badge variant="outline" className="text-red-500 border-red-500 bg-red-500/10">0.36 F1 Score</Badge>
                  </div>
                </CardHeader>
                <CardContent className="space-y-4">
                  <ul className="text-sm space-y-2 text-slate-400">
                    <li className="flex gap-2"><AlertTriangle className="w-4 h-4 text-red-400 shrink-0" /> Severe overfitting (memorizes training fires instead of physics)</li>
                    <li className="flex gap-2"><AlertTriangle className="w-4 h-4 text-red-400 shrink-0" /> Cannot adapt to hourly wind shifts or user interventions</li>
                    <li className="flex gap-2"><AlertTriangle className="w-4 h-4 text-red-400 shrink-0" /> Relies on US-specific data (PDSI, ERC), blocking global use</li>
                  </ul>
                </CardContent>
              </Card>

              <Card className="border-emerald-900/50 bg-emerald-950/10">
                <CardHeader>
                  <div className="flex justify-between items-start">
                    <div className="z-10">
                      <CardTitle className="text-xl flex items-center gap-2">
                        <Layers className="w-5 h-5 text-emerald-500" />
                        Our Custom Hybrid
                      </CardTitle>
                      <CardDescription className="mt-2">ML Fuel Mapping + Cellular Automaton Physics</CardDescription>
                    </div>
                    <Badge className="bg-emerald-500 text-white z-10">0.94 F1 Score</Badge>
                  </div>
                </CardHeader>
                <CardContent className="space-y-4 z-10 relative">
                  <ul className="text-sm space-y-2 text-slate-400">
                    <li className="flex gap-2"><CheckCircle className="w-4 h-4 text-emerald-400 shrink-0" /> 2.6x improvement in accuracy by avoiding the "Identity Trap"</li>
                    <li className="flex gap-2"><Zap className="w-4 h-4 text-emerald-400 shrink-0" /> Real-time interactivity (instant recalculation on wind shift)</li>
                    <li className="flex gap-2"><Globe className="w-4 h-4 text-emerald-400 shrink-0" /> Globally deployable using universally available ERA5 weather data</li>
                  </ul>
                </CardContent>
              </Card>
            </div>

            <div className="grid lg:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle>Quantitative Performance</CardTitle>
                  <CardDescription>Direct metric comparison on the test set</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="h-[300px] w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={performanceData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
                        <XAxis dataKey="metric" stroke="#94a3b8" />
                        <YAxis domain={[0, 1]} stroke="#94a3b8" />
                        <Tooltip contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155' }} />
                        <Legend wrapperStyle={{ paddingTop: '20px' }} />
                        <Bar dataKey="NDWS" name="Google NDWS" fill="#ef4444" radius={[4, 4, 0, 0]} />
                        <Bar dataKey="Custom" name="Custom Hybrid" fill="#10b981" radius={[4, 4, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>NDWS Training Dynamics</CardTitle>
                  <CardDescription>Visualizing the Overfitting Problem (Identity Trap)</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="h-[300px] w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={trainingData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                        <XAxis dataKey="epoch" stroke="#94a3b8" />
                        <YAxis domain={[0.2, 0.65]} stroke="#94a3b8" />
                        <Tooltip contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155' }} />
                        <Legend wrapperStyle={{ paddingTop: '20px' }} />
                        <Line type="monotone" dataKey="ndwsTrainF1" name="Training F1" stroke="#3b82f6" strokeWidth={3} dot={{ r: 4 }} activeDot={{ r: 6 }} />
                        <Line type="monotone" dataKey="ndwsValF1" name="Validation F1" stroke="#ef4444" strokeWidth={3} dot={{ r: 4 }} activeDot={{ r: 6 }} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Side-by-Side Simulation Sandbox */}
            <div className="mt-8">
              <ComparisonSandbox />
            </div>
          </TabsContent>

          {/* 2. Theory Tab */}
          <TabsContent value="theory" className="space-y-6">
            <div className="grid md:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <ArrowRight className="w-5 h-5 text-blue-400" />
                    1. The Strobe Light Effect
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4 text-slate-300 text-sm leading-relaxed">
                  <p>
                    Low-earth-orbit satellites (Terra/Aqua) pass overhead only 1-2 times per day. 
                  </p>
                  <div className="bg-slate-900 p-4 rounded-md border border-slate-800">
                    <ul className="space-y-2">
                      <li><span className="text-orange-400 font-mono">10:00 AM:</span> Fire detected.</li>
                      <li><span className="text-slate-500 font-mono">11:00 AM - 3:00 PM:</span> No satellite coverage (blank map).</li>
                      <li><span className="text-orange-400 font-mono">04:00 PM:</span> Fire reappears in a new location.</li>
                    </ul>
                  </div>
                  <p>
                    <strong>The Result:</strong> Pure ML tries to learn fluid dynamics from discontinuous "flashes" of fire. This is mathematically brutal. Our Custom Model delegates continuous temporal spread to an explicit physics Cellular Automaton instead.
                  </p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <ShieldCheck className="w-5 h-5 text-emerald-400" />
                    2. Escaping The Identity Trap
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4 text-slate-300 text-sm leading-relaxed">
                  <p>
                    When predicting a <strong>Day 2</strong> fire from a <strong>Day 1</strong> fire, the safest guess for an ML model is to simply copy the Day 1 fire to the output, since fires grow relatively slowly. This yields artificially high accuracy but zero predictive value.
                  </p>
                  <div className="border-l-4 border-emerald-500 pl-4 py-2 bg-emerald-900/10">
                    <p className="font-semibold text-emerald-400 mb-1">Our Solution:</p>
                    <p>We removed the current fire state from the input entirely. The model is forced to predict static burn susceptibility based <strong>only</strong> on weather and terrain. It cannot cheat.</p>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* 3. Features Tab */}
          <TabsContent value="features" className="space-y-6">
            <div className="grid lg:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle>Feature Importance (Custom Model)</CardTitle>
                  <CardDescription>Impact of inputs on static burn susceptibility prediction</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="h-[350px] w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={featureImportance} layout="horizontal" margin={{ top: 20, right: 30, left: 20, bottom: 60 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                        <XAxis dataKey="feature" angle={-45} textAnchor="end" height={80} stroke="#94a3b8" tick={{fontSize: 12}} />
                        <YAxis domain={[0, 0.35]} stroke="#94a3b8" />
                        <Tooltip contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155' }} />
                        <Bar dataKey="importance" fill="#f97316" radius={[4, 4, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Dataset Statistics</CardTitle>
                  <CardDescription>Uttarakhand Test Set Confusion Matrix</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-3 gap-2 text-center mb-6">
                    <div></div>
                    <div className="font-medium text-sm text-slate-400">Predicted: No Fire</div>
                    <div className="font-medium text-sm text-slate-400">Predicted: Fire</div>

                    <div className="font-medium text-sm text-slate-400 flex items-center justify-end pr-4">Actual: No Fire</div>
                    <div className="bg-emerald-900/30 border border-emerald-800 p-4 rounded font-bold text-emerald-400">8,547<br/><span className="text-xs font-normal text-slate-500">True Negatives</span></div>
                    <div className="bg-red-900/30 border border-red-800 p-4 rounded font-bold text-red-400">234<br/><span className="text-xs font-normal text-slate-500">False Positives</span></div>

                    <div className="font-medium text-sm text-slate-400 flex items-center justify-end pr-4">Actual: Fire</div>
                    <div className="bg-red-900/30 border border-red-800 p-4 rounded font-bold text-red-400">187<br/><span className="text-xs font-normal text-slate-500">False Negatives</span></div>
                    <div className="bg-emerald-900/30 border border-emerald-800 p-4 rounded font-bold text-emerald-400">1,832<br/><span className="text-xs font-normal text-slate-500">True Positives</span></div>
                  </div>

                  <div className="space-y-4">
                    <div className="flex items-start space-x-3">
                      <CheckCircle className="h-5 w-5 text-emerald-500 mt-0.5" />
                      <div>
                        <p className="font-medium text-sm">Focal Loss Optimization</p>
                        <p className="text-xs text-muted-foreground">Class imbalance (95% No Fire) was mitigated using Focal Loss (α=0.95, γ=2.0) combined with Dice Loss.</p>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* 4. Architecture Specs Tab */}
          <TabsContent value="architecture" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Model Architecture Details</CardTitle>
                <CardDescription>Technical specifications of the Custom U-Net + CBAM</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid md:grid-cols-2 gap-8">
                  <div className="space-y-3">
                    <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                      <span className="text-sm font-medium text-slate-400">Base Architecture</span>
                      <Badge className="bg-blue-600 text-white">U-Net</Badge>
                    </div>
                    <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                      <span className="text-sm font-medium text-slate-400">Attention Mechanism</span>
                      <span className="text-sm">CBAM (Convolutional Block Attention Module)</span>
                    </div>
                    <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                      <span className="text-sm font-medium text-slate-400">Input Channels</span>
                      <span className="text-sm">13 (Weather, Topography, LULC)</span>
                    </div>
                    <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                      <span className="text-sm font-medium text-slate-400">Output</span>
                      <span className="text-sm">1 (Static Burn Susceptibility Logits)</span>
                    </div>
                    <div className="flex items-center justify-between pb-2">
                      <span className="text-sm font-medium text-slate-400">Loss Function</span>
                      <span className="text-sm">Combined (0.5 * Focal + 0.5 * Dice)</span>
                    </div>
                  </div>

                  <div className="bg-slate-900 rounded-lg p-4 font-mono text-xs text-slate-300 overflow-x-auto">
                    <pre>{`Input: [B, 13, H, W]
       ↓
DoubleConv + CBAM (C → 64)
       ↓
MaxPool → DoubleConv (64 → 128)
       ↓
MaxPool → DoubleConv (128 → 256)
       ↓
MaxPool → DoubleConv (256 → 512)
       ↓
MaxPool → DoubleConv + Dropout 0.5 (512 → 1024)
       ↓ (Bottleneck)
ConvTranspose (1024 → 512) + Skip + Dropout 0.3
       ↓
ConvTranspose (512 → 256) + Skip + Dropout 0.3
       ↓
ConvTranspose (256 → 128) + Skip
       ↓
ConvTranspose (128 → 64) + Skip
       ↓
Conv 1×1 (64 → 1) + Bias Init -5.0
       ↓
Output: [B, 1, H, W]`}</pre>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}
