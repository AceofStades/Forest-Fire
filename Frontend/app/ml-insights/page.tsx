"use client"

import { Navigation } from "@/components/navigation"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line } from "recharts"
import { Brain, Target, TrendingUp, Zap, CheckCircle, AlertCircle, Info } from "lucide-react"

const modelPerformance = [
  { model: "U-Net", accuracy: 87.3, precision: 85.2, recall: 89.1, f1Score: 87.1 },
  { model: "LSTM", accuracy: 82.7, precision: 80.5, recall: 84.8, f1Score: 82.6 },
  { model: "Cellular Automata", accuracy: 79.4, precision: 77.8, recall: 81.2, f1Score: 79.5 },
  { model: "Hybrid (U-Net + LSTM)", accuracy: 91.2, precision: 89.7, recall: 92.5, f1Score: 91.1 },
]

const trainingHistory = [
  { epoch: 1, loss: 0.85, accuracy: 0.65, valLoss: 0.92, valAccuracy: 0.61 },
  { epoch: 5, loss: 0.42, accuracy: 0.78, valLoss: 0.48, valAccuracy: 0.75 },
  { epoch: 10, loss: 0.28, accuracy: 0.84, valLoss: 0.31, valAccuracy: 0.82 },
  { epoch: 15, loss: 0.19, accuracy: 0.89, valLoss: 0.22, valAccuracy: 0.87 },
  { epoch: 20, loss: 0.14, accuracy: 0.91, valLoss: 0.18, valAccuracy: 0.89 },
  { epoch: 25, loss: 0.11, accuracy: 0.93, valLoss: 0.15, valAccuracy: 0.91 },
]

const featureImportance = [
  { feature: "Temperature", importance: 0.28 },
  { feature: "Humidity", importance: 0.22 },
  { feature: "Wind Speed", importance: 0.18 },
  { feature: "Vegetation Density", importance: 0.15 },
  { feature: "Slope", importance: 0.08 },
  { feature: "Elevation", importance: 0.06 },
  { feature: "Distance to Roads", importance: 0.03 },
]

const confusionMatrix = [
  { predicted: "No Fire", actual: "No Fire", count: 8547 },
  { predicted: "No Fire", actual: "Fire", count: 234 },
  { predicted: "Fire", actual: "No Fire", count: 187 },
  { predicted: "Fire", actual: "Fire", count: 1832 },
]

export default function MLInsightsPage() {
  return (
    <div className="min-h-screen bg-background">
      <Navigation />

      <div className="container mx-auto px-4 py-6">
        <div className="mb-6">
          <h1 className="text-3xl font-bold mb-2">ML Model Insights</h1>
          <p className="text-muted-foreground">Detailed analysis of AI/ML model performance and explainability</p>
        </div>

        {/* Model Overview Cards */}
        <div className="grid md:grid-cols-3 gap-4 mb-6">
          <Card>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-muted-foreground">Best Model</p>
                  <p className="text-xl font-bold">U-Net + LSTM</p>
                </div>
                <Brain className="h-8 w-8 text-purple-600" />
              </div>
              <div className="mt-2">
                <Badge variant="outline" className="bg-purple-50 text-purple-700 border-purple-200">
                  Hybrid Architecture
                </Badge>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-muted-foreground">Overall Accuracy</p>
                  <p className="text-xl font-bold">91.2%</p>
                </div>
                <Target className="h-8 w-8 text-green-600" />
              </div>
              <div className="mt-2">
                <Progress value={91.2} className="h-2" />
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-muted-foreground">Training Time</p>
                  <p className="text-xl font-bold">4.2 hrs</p>
                </div>
                <Zap className="h-8 w-8 text-yellow-600" />
              </div>
              <div className="flex items-center mt-2">
                <TrendingUp className="h-4 w-4 text-green-600 mr-1" />
                <span className="text-sm text-green-600">Optimized</span>
              </div>
            </CardContent>
          </Card>
        </div>

        <Tabs defaultValue="performance" className="space-y-6">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="performance">Model Performance</TabsTrigger>
            <TabsTrigger value="training">Training History</TabsTrigger>
            <TabsTrigger value="features">Feature Importance</TabsTrigger>
            <TabsTrigger value="confusion">Confusion Matrix</TabsTrigger>
          </TabsList>

          <TabsContent value="performance" className="space-y-6">
            <div className="grid lg:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle>Model Comparison</CardTitle>
                  <CardDescription>Performance metrics across different AI/ML models</CardDescription>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={modelPerformance}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="model" />
                      <YAxis domain={[70, 95]} />
                      <Tooltip />
                      <Bar dataKey="accuracy" fill="#3b82f6" name="Accuracy" />
                      <Bar dataKey="f1Score" fill="#10b981" name="F1-Score" />
                    </BarChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Model Architecture Details</CardTitle>
                  <CardDescription>Technical specifications of the best performing model</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium">Architecture</span>
                      <Badge>U-Net + LSTM Hybrid</Badge>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium">Input Resolution</span>
                      <span className="text-sm">256x256 pixels</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium">Parameters</span>
                      <span className="text-sm">23.5M</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium">Training Data</span>
                      <span className="text-sm">50,000 samples</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium">Validation Split</span>
                      <span className="text-sm">20%</span>
                    </div>
                  </div>

                  <div className="pt-4 border-t">
                    <h4 className="font-medium mb-2">Key Features:</h4>
                    <ul className="text-sm space-y-1 text-muted-foreground">
                      <li>• Spatial feature extraction via U-Net</li>
                      <li>• Temporal sequence modeling with LSTM</li>
                      <li>• Multi-scale attention mechanism</li>
                      <li>• Data augmentation techniques</li>
                    </ul>
                  </div>
                </CardContent>
              </Card>
            </div>

            <Card>
              <CardHeader>
                <CardTitle>Performance Metrics</CardTitle>
                <CardDescription>Detailed breakdown of model evaluation metrics</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid md:grid-cols-4 gap-4">
                  {modelPerformance[3] && (
                    <>
                      <div className="text-center">
                        <div className="text-2xl font-bold text-blue-600">{modelPerformance[3].accuracy}%</div>
                        <div className="text-sm text-muted-foreground">Accuracy</div>
                        <Progress value={modelPerformance[3].accuracy} className="mt-2" />
                      </div>
                      <div className="text-center">
                        <div className="text-2xl font-bold text-green-600">{modelPerformance[3].precision}%</div>
                        <div className="text-sm text-muted-foreground">Precision</div>
                        <Progress value={modelPerformance[3].precision} className="mt-2" />
                      </div>
                      <div className="text-center">
                        <div className="text-2xl font-bold text-orange-600">{modelPerformance[3].recall}%</div>
                        <div className="text-sm text-muted-foreground">Recall</div>
                        <Progress value={modelPerformance[3].recall} className="mt-2" />
                      </div>
                      <div className="text-center">
                        <div className="text-2xl font-bold text-purple-600">{modelPerformance[3].f1Score}%</div>
                        <div className="text-sm text-muted-foreground">F1-Score</div>
                        <Progress value={modelPerformance[3].f1Score} className="mt-2" />
                      </div>
                    </>
                  )}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="training" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Training History</CardTitle>
                <CardDescription>Model performance evolution during training</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={400}>
                  <LineChart data={trainingHistory}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="epoch" />
                    <YAxis />
                    <Tooltip />
                    <Line
                      type="monotone"
                      dataKey="accuracy"
                      stroke="#3b82f6"
                      name="Training Accuracy"
                      strokeWidth={2}
                    />
                    <Line
                      type="monotone"
                      dataKey="valAccuracy"
                      stroke="#10b981"
                      name="Validation Accuracy"
                      strokeWidth={2}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="features" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Feature Importance Analysis</CardTitle>
                <CardDescription>Impact of different input features on model predictions</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={400}>
                  <BarChart data={featureImportance} layout="horizontal">
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis type="number" domain={[0, 0.3]} />
                    <YAxis dataKey="feature" type="category" width={120} />
                    <Tooltip />
                    <Bar dataKey="importance" fill="#f97316" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="confusion" className="space-y-6">
            <div className="grid lg:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle>Confusion Matrix</CardTitle>
                  <CardDescription>Model prediction accuracy breakdown</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-3 gap-2 text-center">
                    <div></div>
                    <div className="font-medium text-sm">Predicted: No Fire</div>
                    <div className="font-medium text-sm">Predicted: Fire</div>

                    <div className="font-medium text-sm">Actual: No Fire</div>
                    <div className="bg-green-100 p-4 rounded font-bold text-green-800">8,547</div>
                    <div className="bg-red-100 p-4 rounded font-bold text-red-800">187</div>

                    <div className="font-medium text-sm">Actual: Fire</div>
                    <div className="bg-red-100 p-4 rounded font-bold text-red-800">234</div>
                    <div className="bg-green-100 p-4 rounded font-bold text-green-800">1,832</div>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Model Insights</CardTitle>
                  <CardDescription>Key observations and recommendations</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex items-start space-x-3">
                    <CheckCircle className="h-5 w-5 text-green-600 mt-0.5" />
                    <div>
                      <p className="font-medium">High True Positive Rate</p>
                      <p className="text-sm text-muted-foreground">
                        Model correctly identifies 88.7% of actual fire cases
                      </p>
                    </div>
                  </div>

                  <div className="flex items-start space-x-3">
                    <CheckCircle className="h-5 w-5 text-green-600 mt-0.5" />
                    <div>
                      <p className="font-medium">Low False Positive Rate</p>
                      <p className="text-sm text-muted-foreground">
                        Only 2.1% of non-fire areas incorrectly classified as fire
                      </p>
                    </div>
                  </div>

                  <div className="flex items-start space-x-3">
                    <AlertCircle className="h-5 w-5 text-orange-600 mt-0.5" />
                    <div>
                      <p className="font-medium">Room for Improvement</p>
                      <p className="text-sm text-muted-foreground">
                        Consider ensemble methods to reduce false negatives
                      </p>
                    </div>
                  </div>

                  <div className="flex items-start space-x-3">
                    <Info className="h-5 w-5 text-blue-600 mt-0.5" />
                    <div>
                      <p className="font-medium">Balanced Performance</p>
                      <p className="text-sm text-muted-foreground">Good balance between precision and recall metrics</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}
