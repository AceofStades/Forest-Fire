"use client"

import React from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, LineChart, Line } from "recharts"
import { Brain, CheckCircle, AlertTriangle, ShieldCheck, Zap, Globe, Layers, ArrowRight } from "lucide-react"

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

export function ModelComparisonContent() {
  return (
    <div className="container mx-auto px-4 py-8 flex-1">
      <div className="mb-8">
        <h1 className="text-3xl font-bold mb-3">Model Architecture Comparison</h1>
        <p className="text-muted-foreground text-lg">
          A rigorous scientific comparison between Google's Next Day Wildfire Spread (NDWS) and our Custom Hybrid Architecture.
        </p>
      </div>

      <div className="grid md:grid-cols-2 gap-6 mb-8">
        {/* NDWS Model Card */}
        <Card className="border-red-900/30 bg-red-950/10">
          <CardHeader>
            <div className="flex justify-between items-start">
              <div>
                <CardTitle className="text-xl flex items-center gap-2">
                  <Brain className="w-5 h-5 text-red-500" />
                  NDWS (End-to-End Deep Learning)
                </CardTitle>
                <CardDescription className="mt-2">Pure ML approach predicting Day 2 spread from Day 1 state</CardDescription>
              </div>
              <Badge variant="outline" className="text-red-500 border-red-500 bg-red-500/10">0.36 F1 Score</Badge>
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <h4 className="font-semibold text-sm text-slate-300 mb-2 uppercase tracking-wide">The Approach</h4>
              <p className="text-sm text-slate-400">Attempts to learn complex fluid dynamics (fire spread) directly from sparse, 24-hour satellite snapshots using a pure neural network.</p>
            </div>
            <div>
              <h4 className="font-semibold text-sm text-slate-300 mb-2 uppercase tracking-wide">Key Weaknesses</h4>
              <ul className="text-sm space-y-2 text-slate-400">
                <li className="flex gap-2"><AlertTriangle className="w-4 h-4 text-red-400 shrink-0" /> Severe overfitting (memorizes training fires instead of physics)</li>
                <li className="flex gap-2"><AlertTriangle className="w-4 h-4 text-red-400 shrink-0" /> Cannot adapt to hourly wind shifts or user interventions</li>
                <li className="flex gap-2"><AlertTriangle className="w-4 h-4 text-red-400 shrink-0" /> Relies on US-specific data (PDSI, ERC), blocking global use</li>
              </ul>
            </div>
          </CardContent>
        </Card>

        {/* Custom Hybrid Card */}
        <Card className="border-emerald-900/50 bg-emerald-950/10 relative overflow-hidden">
          <div className="absolute top-0 right-0 p-4 opacity-10">
            <Zap className="w-32 h-32 text-emerald-500" />
          </div>
          <CardHeader>
            <div className="flex justify-between items-start">
              <div className="z-10">
                <CardTitle className="text-xl flex items-center gap-2">
                  <Layers className="w-5 h-5 text-emerald-500" />
                  Custom Hybrid (ML + Physics CA)
                </CardTitle>
                <CardDescription className="mt-2">Separates static fuel mapping from temporal spread dynamics</CardDescription>
              </div>
              <Badge className="bg-emerald-500 text-white z-10">0.94 F1 Score</Badge>
            </div>
          </CardHeader>
          <CardContent className="space-y-4 z-10 relative">
            <div>
              <h4 className="font-semibold text-sm text-slate-300 mb-2 uppercase tracking-wide">The Approach</h4>
              <p className="text-sm text-slate-400">Uses Deep Learning strictly for static topography/fuel mapping, delegating temporal fluid dynamics to a Cellular Automaton physics engine.</p>
            </div>
            <div>
              <h4 className="font-semibold text-sm text-slate-300 mb-2 uppercase tracking-wide">Key Strengths</h4>
              <ul className="text-sm space-y-2 text-slate-400">
                <li className="flex gap-2"><CheckCircle className="w-4 h-4 text-emerald-400 shrink-0" /> 2.6x improvement in accuracy by avoiding the "Identity Trap"</li>
                <li className="flex gap-2"><Zap className="w-4 h-4 text-emerald-400 shrink-0" /> Real-time interactivity (instant recalculation on wind shift)</li>
                <li className="flex gap-2"><Globe className="w-4 h-4 text-emerald-400 shrink-0" /> Globally deployable using universally available ERA5 weather data</li>
              </ul>
            </div>
          </CardContent>
        </Card>
      </div>

      <Tabs defaultValue="quantitative" className="w-full">
        <TabsList className="grid w-full grid-cols-3 mb-6 bg-slate-900/50 p-1">
          <TabsTrigger value="quantitative">Quantitative Performance</TabsTrigger>
          <TabsTrigger value="overfitting">The Overfitting Problem</TabsTrigger>
          <TabsTrigger value="theory">Why NDWS Fails (Theory)</TabsTrigger>
        </TabsList>

        <TabsContent value="quantitative" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Metric Comparison</CardTitle>
              <CardDescription>Direct performance comparison between the two models on the test set.</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-[400px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={performanceData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
                    <XAxis dataKey="metric" stroke="#94a3b8" />
                    <YAxis domain={[0, 1]} stroke="#94a3b8" />
                    <Tooltip 
                      contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155' }}
                      cursor={{ fill: 'rgba(255, 255, 255, 0.05)' }}
                    />
                    <Legend wrapperStyle={{ paddingTop: '20px' }} />
                    <Bar dataKey="NDWS" name="Google NDWS" fill="#ef4444" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="Custom" name="Custom Hybrid Model" fill="#10b981" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="overfitting" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>NDWS Training Dynamics (The Identity Trap)</CardTitle>
              <CardDescription>
                Notice how training accuracy keeps climbing while validation accuracy peaks at Epoch 4 and declines. 
                The model is memorizing specific fires rather than learning generalizable fluid dynamics.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-[400px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={trainingData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                    <XAxis dataKey="epoch" stroke="#94a3b8" label={{ value: 'Epoch', position: 'insideBottom', offset: -5, fill: '#94a3b8' }} />
                    <YAxis domain={[0.2, 0.65]} stroke="#94a3b8" label={{ value: 'F1 Score', angle: -90, position: 'insideLeft', fill: '#94a3b8' }} />
                    <Tooltip 
                      contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155' }}
                    />
                    <Legend wrapperStyle={{ paddingTop: '20px' }} />
                    <Line type="monotone" dataKey="ndwsTrainF1" name="Training F1" stroke="#3b82f6" strokeWidth={3} dot={{ r: 4 }} activeDot={{ r: 6 }} />
                    <Line type="monotone" dataKey="ndwsValF1" name="Validation F1" stroke="#ef4444" strokeWidth={3} dot={{ r: 4 }} activeDot={{ r: 6 }} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

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
                  <strong>The Result:</strong> NDWS tries to learn fluid dynamics from discontinuous "flashes" of fire. This is mathematically brutal. Our Custom Model delegates continuous temporal spread to an explicit physics Cellular Automaton instead.
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
      </Tabs>
    </div>
  )
}
