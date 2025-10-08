"use client"

import { Navigation } from "@/components/navigation"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  LineChart,
  Line,
  Area,
  AreaChart,
} from "recharts"
import { TrendingUp, TrendingDown, AlertTriangle, MapPin, Calendar, Download, Share } from "lucide-react"

const riskDistributionData = [
  { name: "High Risk", value: 18.5, color: "#ef4444" },
  { name: "Medium Risk", value: 34.2, color: "#f97316" },
  { name: "Low Risk", value: 47.3, color: "#22c55e" },
]

const districtData = [
  { district: "North Valley", high: 25, medium: 35, low: 40 },
  { district: "East Hills", high: 15, medium: 45, low: 40 },
  { district: "South Plains", high: 30, medium: 25, low: 45 },
  { district: "West Coast", high: 12, medium: 38, low: 50 },
  { district: "Central", high: 22, medium: 33, low: 45 },
]

const timeSeriesData = [
  { time: "00:00", risk: 15 },
  { time: "04:00", risk: 12 },
  { time: "08:00", risk: 18 },
  { time: "12:00", risk: 35 },
  { time: "16:00", risk: 42 },
  { time: "20:00", risk: 28 },
  { time: "24:00", risk: 20 },
]

const spreadData = [
  { hour: 0, area: 0 },
  { hour: 1, area: 2.5 },
  { hour: 2, area: 8.2 },
  { hour: 3, area: 18.7 },
  { hour: 6, area: 45.3 },
  { hour: 12, area: 89.6 },
  { hour: 24, area: 156.8 },
]

export default function AnalyticsPage() {
  return (
    <div className="min-h-screen bg-background">
      <Navigation />

      <div className="container mx-auto px-4 py-6">
        <div className="mb-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold mb-2">Analytics Dashboard</h1>
              <p className="text-muted-foreground">Comprehensive fire risk analysis and statistical insights</p>
            </div>
            <div className="flex space-x-2">
              <Button variant="outline">
                <Share className="mr-2 h-4 w-4" />
                Share
              </Button>
              <Button>
                <Download className="mr-2 h-4 w-4" />
                Export Report
              </Button>
            </div>
          </div>
        </div>

        {/* Key Metrics */}
        <div className="grid md:grid-cols-4 gap-4 mb-6">
          <Card>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-muted-foreground">Total Area Analyzed</p>
                  <p className="text-2xl font-bold">12,450 km²</p>
                </div>
                <MapPin className="h-8 w-8 text-blue-600" />
              </div>
              <div className="flex items-center mt-2">
                <TrendingUp className="h-4 w-4 text-green-600 mr-1" />
                <span className="text-sm text-green-600">+5.2% from last month</span>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-muted-foreground">High Risk Areas</p>
                  <p className="text-2xl font-bold">2,303 km²</p>
                </div>
                <AlertTriangle className="h-8 w-8 text-red-600" />
              </div>
              <div className="flex items-center mt-2">
                <TrendingDown className="h-4 w-4 text-red-600 mr-1" />
                <span className="text-sm text-red-600">-2.1% from last week</span>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-muted-foreground">Predictions Made</p>
                  <p className="text-2xl font-bold">1,247</p>
                </div>
                <Calendar className="h-8 w-8 text-green-600" />
              </div>
              <div className="flex items-center mt-2">
                <TrendingUp className="h-4 w-4 text-green-600 mr-1" />
                <span className="text-sm text-green-600">+12.5% this week</span>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-muted-foreground">Model Accuracy</p>
                  <p className="text-2xl font-bold">87.3%</p>
                </div>
                <Badge variant="outline" className="bg-green-50 text-green-700 border-green-200">
                  Excellent
                </Badge>
              </div>
              <div className="flex items-center mt-2">
                <TrendingUp className="h-4 w-4 text-green-600 mr-1" />
                <span className="text-sm text-green-600">+1.2% improvement</span>
              </div>
            </CardContent>
          </Card>
        </div>

        <Tabs defaultValue="overview" className="space-y-6">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="districts">By District</TabsTrigger>
            <TabsTrigger value="temporal">Temporal Analysis</TabsTrigger>
            <TabsTrigger value="spread">Fire Spread</TabsTrigger>
          </TabsList>

          <TabsContent value="overview" className="space-y-6">
            <div className="grid lg:grid-cols-2 gap-6">
              {/* Risk Distribution Pie Chart */}
              <Card>
                <CardHeader>
                  <CardTitle>Risk Distribution</CardTitle>
                  <CardDescription>Area breakdown by risk level</CardDescription>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <PieChart>
                      <Pie
                        data={riskDistributionData}
                        cx="50%"
                        cy="50%"
                        labelLine={false}
                        label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(1)}%`}
                        outerRadius={80}
                        fill="#8884d8"
                        dataKey="value"
                      >
                        {riskDistributionData.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.color} />
                        ))}
                      </Pie>
                      <Tooltip />
                    </PieChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              {/* Daily Risk Trend */}
              <Card>
                <CardHeader>
                  <CardTitle>Daily Risk Trend</CardTitle>
                  <CardDescription>24-hour fire risk variation</CardDescription>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <AreaChart data={timeSeriesData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="time" />
                      <YAxis />
                      <Tooltip />
                      <Area type="monotone" dataKey="risk" stroke="#f97316" fill="#fed7aa" />
                    </AreaChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="districts" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Fire Risk by District</CardTitle>
                <CardDescription>Comparative analysis across different regions</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={400}>
                  <BarChart data={districtData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="district" />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="high" stackId="a" fill="#ef4444" name="High Risk" />
                    <Bar dataKey="medium" stackId="a" fill="#f97316" name="Medium Risk" />
                    <Bar dataKey="low" stackId="a" fill="#22c55e" name="Low Risk" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="temporal" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Temporal Risk Analysis</CardTitle>
                <CardDescription>Fire risk patterns over time</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={400}>
                  <LineChart data={timeSeriesData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="time" />
                    <YAxis />
                    <Tooltip />
                    <Line
                      type="monotone"
                      dataKey="risk"
                      stroke="#f97316"
                      strokeWidth={3}
                      dot={{ fill: "#f97316", strokeWidth: 2, r: 6 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="spread" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Fire Spread Simulation</CardTitle>
                <CardDescription>Projected fire spread area over time</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={400}>
                  <AreaChart data={spreadData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="hour" label={{ value: "Hours", position: "insideBottom", offset: -5 }} />
                    <YAxis label={{ value: "Area (km²)", angle: -90, position: "insideLeft" }} />
                    <Tooltip />
                    <Area type="monotone" dataKey="area" stroke="#dc2626" fill="#fecaca" />
                  </AreaChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}
