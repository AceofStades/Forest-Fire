"use client"

import { useState } from "react"
import { Navigation } from "@/components/navigation"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { Checkbox } from "@/components/ui/checkbox"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Badge } from "@/components/ui/badge"
import { Download, FileText, ImageIcon, Video, Share2, Calendar, MapPin, BarChart3, Settings } from "lucide-react"
import { useToast } from "@/hooks/use-toast"

export default function ReportPage() {
  const [reportConfig, setReportConfig] = useState({
    title: "Forest Fire Risk Assessment Report",
    region: "Northern California",
    dateRange: "2024-01-01 to 2024-01-31",
    includeMap: true,
    includeCharts: true,
    include3D: false,
    includeMLInsights: true,
    format: "pdf",
  })
  const [isGenerating, setIsGenerating] = useState(false)
  const { toast } = useToast()

  const handleGenerateReport = async () => {
    setIsGenerating(true)

    // Simulate report generation
    setTimeout(() => {
      setIsGenerating(false)
      toast({
        title: "Report generated successfully",
        description: "Your fire risk assessment report is ready for download",
      })
    }, 3000)
  }

  return (
    <div className="min-h-screen bg-background">
      <Navigation />

      <div className="container mx-auto px-4 py-6">
        <div className="mb-6">
          <h1 className="text-3xl font-bold mb-2">Generate Report</h1>
          <p className="text-muted-foreground">
            Create comprehensive fire risk assessment reports with maps, charts, and analysis
          </p>
        </div>

        <div className="grid lg:grid-cols-3 gap-6">
          {/* Report Configuration */}
          <div className="lg:col-span-2 space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center">
                  <Settings className="mr-2 h-5 w-5" />
                  Report Configuration
                </CardTitle>
                <CardDescription>Customize your report content and format</CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="report-title">Report Title</Label>
                    <Input
                      id="report-title"
                      value={reportConfig.title}
                      onChange={(e) => setReportConfig((prev) => ({ ...prev, title: e.target.value }))}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="region">Region</Label>
                    <Input
                      id="region"
                      value={reportConfig.region}
                      onChange={(e) => setReportConfig((prev) => ({ ...prev, region: e.target.value }))}
                    />
                  </div>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="date-range">Date Range</Label>
                  <Input
                    id="date-range"
                    value={reportConfig.dateRange}
                    onChange={(e) => setReportConfig((prev) => ({ ...prev, dateRange: e.target.value }))}
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="description">Description (Optional)</Label>
                  <Textarea
                    id="description"
                    placeholder="Add any additional context or notes for this report..."
                    rows={3}
                  />
                </div>

                <div className="space-y-4">
                  <Label className="text-base font-medium">Include in Report</Label>
                  <div className="space-y-3">
                    <div className="flex items-center space-x-2">
                      <Checkbox
                        id="include-map"
                        checked={reportConfig.includeMap}
                        onCheckedChange={(checked) =>
                          setReportConfig((prev) => ({ ...prev, includeMap: checked as boolean }))
                        }
                      />
                      <Label htmlFor="include-map" className="flex items-center">
                        <MapPin className="mr-2 h-4 w-4" />
                        Fire Risk Maps
                      </Label>
                    </div>

                    <div className="flex items-center space-x-2">
                      <Checkbox
                        id="include-charts"
                        checked={reportConfig.includeCharts}
                        onCheckedChange={(checked) =>
                          setReportConfig((prev) => ({ ...prev, includeCharts: checked as boolean }))
                        }
                      />
                      <Label htmlFor="include-charts" className="flex items-center">
                        <BarChart3 className="mr-2 h-4 w-4" />
                        Analytics Charts
                      </Label>
                    </div>

                    <div className="flex items-center space-x-2">
                      <Checkbox
                        id="include-3d"
                        checked={reportConfig.include3D}
                        onCheckedChange={(checked) =>
                          setReportConfig((prev) => ({ ...prev, include3D: checked as boolean }))
                        }
                      />
                      <Label htmlFor="include-3d" className="flex items-center">
                        <Video className="mr-2 h-4 w-4" />
                        3D Visualization Screenshots
                      </Label>
                    </div>

                    <div className="flex items-center space-x-2">
                      <Checkbox
                        id="include-ml"
                        checked={reportConfig.includeMLInsights}
                        onCheckedChange={(checked) =>
                          setReportConfig((prev) => ({ ...prev, includeMLInsights: checked as boolean }))
                        }
                      />
                      <Label htmlFor="include-ml" className="flex items-center">
                        <FileText className="mr-2 h-4 w-4" />
                        ML Model Insights
                      </Label>
                    </div>
                  </div>
                </div>

                <div className="space-y-2">
                  <Label>Output Format</Label>
                  <Select
                    value={reportConfig.format}
                    onValueChange={(value) => setReportConfig((prev) => ({ ...prev, format: value }))}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="pdf">PDF Document</SelectItem>
                      <SelectItem value="html">HTML Report</SelectItem>
                      <SelectItem value="docx">Word Document</SelectItem>
                      <SelectItem value="pptx">PowerPoint Presentation</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Report Preview & Actions */}
          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Report Preview</CardTitle>
                <CardDescription>Summary of your report configuration</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <h4 className="font-medium text-sm text-muted-foreground">Title</h4>
                  <p className="font-medium">{reportConfig.title}</p>
                </div>

                <div>
                  <h4 className="font-medium text-sm text-muted-foreground">Region</h4>
                  <p>{reportConfig.region}</p>
                </div>

                <div>
                  <h4 className="font-medium text-sm text-muted-foreground">Date Range</h4>
                  <p>{reportConfig.dateRange}</p>
                </div>

                <div>
                  <h4 className="font-medium text-sm text-muted-foreground">Format</h4>
                  <Badge variant="outline">{reportConfig.format.toUpperCase()}</Badge>
                </div>

                <div>
                  <h4 className="font-medium text-sm text-muted-foreground mb-2">Included Sections</h4>
                  <div className="space-y-1">
                    {reportConfig.includeMap && (
                      <div className="flex items-center text-sm">
                        <MapPin className="mr-2 h-3 w-3 text-green-600" />
                        Fire Risk Maps
                      </div>
                    )}
                    {reportConfig.includeCharts && (
                      <div className="flex items-center text-sm">
                        <BarChart3 className="mr-2 h-3 w-3 text-green-600" />
                        Analytics Charts
                      </div>
                    )}
                    {reportConfig.include3D && (
                      <div className="flex items-center text-sm">
                        <Video className="mr-2 h-3 w-3 text-green-600" />
                        3D Visualizations
                      </div>
                    )}
                    {reportConfig.includeMLInsights && (
                      <div className="flex items-center text-sm">
                        <FileText className="mr-2 h-3 w-3 text-green-600" />
                        ML Insights
                      </div>
                    )}
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Generate Report</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <Button onClick={handleGenerateReport} disabled={isGenerating} className="w-full">
                  {isGenerating ? (
                    <>Generating Report...</>
                  ) : (
                    <>
                      <Download className="mr-2 h-4 w-4" />
                      Generate & Download
                    </>
                  )}
                </Button>

                <Button variant="outline" className="w-full bg-transparent">
                  <Share2 className="mr-2 h-4 w-4" />
                  Share Report Link
                </Button>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Quick Downloads</CardTitle>
                <CardDescription>Pre-generated reports and data exports</CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                <Button variant="outline" size="sm" className="w-full justify-start bg-transparent">
                  <ImageIcon className="mr-2 h-4 w-4" />
                  Current Risk Map (PNG)
                </Button>
                <Button variant="outline" size="sm" className="w-full justify-start bg-transparent">
                  <FileText className="mr-2 h-4 w-4" />
                  Risk Data (GeoTIFF)
                </Button>
                <Button variant="outline" size="sm" className="w-full justify-start bg-transparent">
                  <BarChart3 className="mr-2 h-4 w-4" />
                  Analytics Data (CSV)
                </Button>
                <Button variant="outline" size="sm" className="w-full justify-start bg-transparent">
                  <Video className="mr-2 h-4 w-4" />
                  Simulation Video (MP4)
                </Button>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center">
                  <Calendar className="mr-2 h-5 w-5" />
                  Recent Reports
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="text-sm">
                  <div className="flex items-center justify-between">
                    <span>Weekly Assessment</span>
                    <Badge variant="outline" className="text-xs">
                      PDF
                    </Badge>
                  </div>
                  <p className="text-xs text-muted-foreground">Generated 2 days ago</p>
                </div>
                <div className="text-sm">
                  <div className="flex items-center justify-between">
                    <span>Monthly Summary</span>
                    <Badge variant="outline" className="text-xs">
                      HTML
                    </Badge>
                  </div>
                  <p className="text-xs text-muted-foreground">Generated 1 week ago</p>
                </div>
                <div className="text-sm">
                  <div className="flex items-center justify-between">
                    <span>Emergency Response</span>
                    <Badge variant="outline" className="text-xs">
                      DOCX
                    </Badge>
                  </div>
                  <p className="text-xs text-muted-foreground">Generated 2 weeks ago</p>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  )
}
