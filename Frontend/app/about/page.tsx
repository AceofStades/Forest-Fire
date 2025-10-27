import { Navigation } from "@/components/navigation"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Users, Database, Award, ExternalLink, Github, Mail, Globe } from "lucide-react"

export default function AboutPage() {
  return (
    <div className="min-h-screen bg-background">
      <Navigation />

      <div className="container mx-auto px-4 py-6">
        <div className="mb-6">
          <h1 className="text-3xl font-bold mb-2">About Forest Fire Prediction System</h1>
          <p className="text-muted-foreground">
            Learn about our project, team, and the technology behind fire prediction
          </p>
        </div>

        <div className="space-y-8">
          {/* Project Overview */}
          <Card>
            <CardHeader>
              <CardTitle>Project Overview</CardTitle>
              <CardDescription>Advanced AI/ML-powered forest fire prediction and simulation platform</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <p className="text-muted-foreground leading-relaxed">
                Our Forest Fire Prediction System represents a cutting-edge approach to wildfire risk assessment and
                spread simulation. By leveraging advanced machine learning techniques including U-Net neural networks,
                LSTM time-series models, and Cellular Automata, we provide accurate next-day fire probability maps and
                real-time spread animations.
              </p>
              <p className="text-muted-foreground leading-relaxed">
                The platform integrates multiple data sources including satellite imagery, weather data, terrain
                information, and vegetation indices to create comprehensive risk assessments. Our 3D visualization
                capabilities, 2D to 3D conversion tools, and interactive mapping make complex fire behavior data
                accessible to researchers, emergency responders, and land managers.
              </p>
            </CardContent>
          </Card>

          {/* Technology Stack */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center">
                <Database className="mr-2 h-5 w-5" />
                Technology Stack
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid md:grid-cols-3 gap-6">
                <div>
                  <h4 className="font-medium mb-3">Frontend</h4>
                  <div className="space-y-2">
                    <Badge variant="outline">React + Next.js</Badge>
                    <Badge variant="outline">TypeScript</Badge>
                    <Badge variant="outline">Tailwind CSS</Badge>
                    <Badge variant="outline">shadcn/ui</Badge>
                    <Badge variant="outline">React-Leaflet</Badge>
                    <Badge variant="outline">Three.js</Badge>
                    <Badge variant="outline">Recharts</Badge>
                  </div>
                </div>
                <div>
                  <h4 className="font-medium mb-3">AI/ML Models</h4>
                  <div className="space-y-2">
                    <Badge variant="outline">U-Net (PyTorch)</Badge>
                    <Badge variant="outline">LSTM Networks</Badge>
                    <Badge variant="outline">Cellular Automata</Badge>
                    <Badge variant="outline">TensorFlow</Badge>
                    <Badge variant="outline">Scikit-learn</Badge>
                    <Badge variant="outline">OpenCV</Badge>
                  </div>
                </div>
                <div>
                  <h4 className="font-medium mb-3">Data & Infrastructure</h4>
                  <div className="space-y-2">
                    <Badge variant="outline">VIIRS Satellite Data</Badge>
                    <Badge variant="outline">SRTM DEM</Badge>
                    <Badge variant="outline">Weather APIs</Badge>
                    <Badge variant="outline">PostGIS</Badge>
                    <Badge variant="outline">Docker</Badge>
                    <Badge variant="outline">Vercel</Badge>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Team */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center">
                <Users className="mr-2 h-5 w-5" />
                Project Team
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
                <div className="text-center">
                  <div className="w-20 h-20 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full mx-auto mb-3 flex items-center justify-center text-white font-bold text-xl">
                    VB
                  </div>
                  <h4 className="font-medium">Vishal Barai</h4>
                  <p className="text-sm text-muted-foreground">Project Lead</p>
                  <p className="text-xs text-muted-foreground mt-1">Backend Developer</p>
                </div>
                <div className="text-center">
                  <div className="w-20 h-20 bg-gradient-to-br from-green-500 to-teal-600 rounded-full mx-auto mb-3 flex items-center justify-center text-white font-bold text-xl">
                    SB
                  </div>
                  <h4 className="font-medium">Shravani Bhogle</h4>
                  <p className="text-sm text-muted-foreground">Frontend Developer</p>
                  <p className="text-xs text-muted-foreground mt-1">UI/UX and 3D Visualization</p>
                </div>
                <div className="text-center">
                  <div className="w-20 h-20 bg-gradient-to-br from-orange-500 to-red-600 rounded-full mx-auto mb-3 flex items-center justify-center text-white font-bold text-xl">
                    RB
                  </div>
                  <h4 className="font-medium">Rahul Bokade</h4>
                  <p className="text-sm text-muted-foreground">AIML Engineer</p>
                  <p className="text-xs text-muted-foreground mt-1">Geospatial Technology Expert</p>
                </div>
                <div className="text-center">
                  <div className="w-20 h-20 bg-gradient-to-br from-purple-500 to-pink-600 rounded-full mx-auto mb-3 flex items-center justify-center text-white font-bold text-xl">
                    YC
                  </div>
                  <h4 className="font-medium">Yug Chapherkar</h4>
                  <p className="text-sm text-muted-foreground">Data Analyst </p>
                  <p className="text-xs text-muted-foreground mt-1">System Architecture & APIs</p>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Data Sources */}
          <Card>
            <CardHeader>
              <CardTitle>Data Sources & Acknowledgments</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid md:grid-cols-2 gap-6">
                <div>
                  <h4 className="font-medium mb-3">Satellite Data</h4>
                  <ul className="text-sm space-y-2 text-muted-foreground">
                    <li>• VIIRS Active Fire Detection (NASA)</li>
                    <li>• MODIS Land Cover Type (USGS)</li>
                    <li>• Landsat 8/9 Surface Reflectance</li>
                    <li>• Sentinel-2 Multispectral Imagery</li>
                  </ul>
                </div>
                <div>
                  <h4 className="font-medium mb-3">Environmental Data</h4>
                  <ul className="text-sm space-y-2 text-muted-foreground">
                    <li>• SRTM Digital Elevation Model</li>
                    <li>• NOAA Weather Station Data</li>
                    <li>• Global Human Settlement Layer</li>
                    <li>• OpenStreetMap Road Networks</li>
                  </ul>
                </div>
              </div>

              <div className="pt-4 border-t">
                <p className="text-sm text-muted-foreground">
                  We acknowledge the use of data from NASA, NOAA, USGS, and ESA. Special thanks to the open-source
                  community for providing essential tools and libraries that made this project possible. This project
                  focuses on forest fire prediction and simulation for Uttarakhand, India region.
                </p>
              </div>
            </CardContent>
          </Card>

          {/* Contact & Links */}
          <div className="grid md:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center">
                  <Award className="mr-2 h-5 w-5" />
                  Recognition
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-sm">Best AI Project 2024</span>
                  <Badge>University Award</Badge>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm">Innovation in GIS</span>
                  <Badge variant="outline">Conference Paper</Badge>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm">Open Source Contribution</span>
                  <Badge variant="outline">GitHub</Badge>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm">2D to 3D Innovation</span>
                  <Badge variant="outline">Tech Innovation</Badge>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Contact & Resources</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <Button variant="outline" className="w-full justify-start bg-transparent">
                  <Github className="mr-2 h-4 w-4" />
                  View Source Code
                  <ExternalLink className="ml-auto h-4 w-4" />
                </Button>
                <Button variant="outline" className="w-full justify-start bg-transparent">
                  <Globe className="mr-2 h-4 w-4" />
                  Research Paper
                  <ExternalLink className="ml-auto h-4 w-4" />
                </Button>
                <Button variant="outline" className="w-full justify-start bg-transparent">
                  <Mail className="mr-2 h-4 w-4" />
                  Contact Team
                  <ExternalLink className="ml-auto h-4 w-4" />
                </Button>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  )
}
