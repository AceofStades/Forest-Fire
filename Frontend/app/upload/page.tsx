"use client";

import { useState } from "react";
import { useRouter } from "next/navigation"; // ✅ import router for redirect
import { Navigation } from "@/components/navigation";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Progress } from "@/components/ui/progress";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import {
  Upload,
  FileText,
  MapPin,
  CheckCircle,
  AlertCircle,
  Download,
  Eye,
  AlertTriangle,
} from "lucide-react";
import { useToast } from "@/hooks/use-toast";

export default function UploadPage() {
  const router = useRouter(); // ✅ initialize router
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isUploading, setIsUploading] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [results, setResults] = useState<any>(null);
  const [status, setStatus] = useState<{ type: "success" | "error"; message: string } | null>(null);

  const [areaName, setAreaName] = useState("");
  const [location, setLocation] = useState("");
  const [model, setModel] = useState("unet-lstm");
  const [description, setDescription] = useState("");

  const { toast } = useToast();

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      if (file.size > 50 * 1024 * 1024) {
        toast({
          title: "File too large",
          description: "Please upload a file smaller than 50MB",
          variant: "destructive",
        });
        return;
      }
      setSelectedFile(file);
      setStatus(null);
      toast({
        title: "File selected",
        description: `${file.name} is ready for upload`,
      });
    }
  };

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    if (!selectedFile) {
      toast({
        title: "No file selected",
        description: "Please select a file to upload",
        variant: "destructive",
      });
      return;
    }

    setIsUploading(true);
    setUploadProgress(0);
    setStatus(null);

    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("area_name", areaName);
    formData.append("location", location);
    formData.append("model", model);
    formData.append("description", description);

    const progressInterval = setInterval(() => {
      setUploadProgress((prev) => {
        if (prev >= 90) {
          clearInterval(progressInterval);
          return 90;
        }
        return prev + 10;
      });
    }, 200);

    try {
      const response = await fetch("http://127.0.0.1:8000/upload-data", {
        method: "POST",
        body: formData,
      });

      const result = await response.json();

      if (!response.ok) {
        throw new Error(result.detail || "Failed to upload data.");
      }

      setUploadProgress(100);
      setStatus({ type: "success", message: result.message || "File uploaded successfully!" });

      setResults({
        areaName: areaName || "Uploaded Region",
        totalArea: "2,450 hectares",
        riskDistribution: {
          high: 18.5,
          medium: 34.2,
          low: 47.3,
        },
        modelUsed: model.toUpperCase(),
        confidence: 87.3,
        processingTime: "2.4 seconds",
      });

      toast({
        title: "Upload successful",
        description: "Redirecting to Chat...",
      });

      // ✅ Redirect to /chat after a short delay
      setTimeout(() => {
        router.push("/chat");
      }, 1500);

    } catch (err: any) {
      setStatus({
        type: "error",
        message: err.message || "An unexpected error occurred.",
      });
      toast({
        title: "Upload failed",
        description: err.message || "An unexpected error occurred.",
        variant: "destructive",
      });
    } finally {
      clearInterval(progressInterval);
      setIsUploading(false);
    }
  };

  return (
    <div className="min-h-screen bg-background">
      <Navigation />

      <div className="container mx-auto px-4 py-6">
        <div className="mb-6">
          <h1 className="text-3xl font-bold mb-2">Upload & Predict</h1>
          <p className="text-muted-foreground">
            Upload your ARGO or Map Data to get personalized AI/ML-based analysis
          </p>
        </div>

        <div className="grid lg:grid-cols-2 gap-6">
          {/* Upload Form */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center">
                <Upload className="mr-2 h-5 w-5" />
                Upload Dataset
              </CardTitle>
              <CardDescription>
                Supported formats: .nc, GeoTIFF, GeoJSON (Max: 50MB)
              </CardDescription>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleSubmit} className="space-y-6">
                <div className="space-y-2">
                  <Label htmlFor="file-upload">Select File</Label>
                  <div className="border-2 border-dashed border-muted-foreground/25 rounded-lg p-6 text-center hover:border-muted-foreground/50 transition-colors">
                    <Input
                      id="file-upload"
                      type="file"
                      accept=".nc,.tiff,.tif,.geojson,.json"
                      onChange={handleFileUpload}
                      className="hidden"
                    />
                    <Label htmlFor="file-upload" className="cursor-pointer">
                      <div className="space-y-2">
                        <FileText className="h-10 w-10 mx-auto text-muted-foreground" />
                        <div className="text-sm">
                          {selectedFile ? (
                            <div className="space-y-1">
                              <p className="font-medium">{selectedFile.name}</p>
                              <p className="text-muted-foreground">
                                {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                              </p>
                            </div>
                          ) : (
                            <>
                              <p className="font-medium">Click to upload or drag and drop</p>
                              <p className="text-muted-foreground">
                                NetCDF, GeoTIFF, or GeoJSON
                              </p>
                            </>
                          )}
                        </div>
                      </div>
                    </Label>
                  </div>
                </div>

                {/* Area Name and Location */}
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="area-name">Area Name</Label>
                    <Input
                      id="area-name"
                      placeholder="e.g., Northern Bay"
                      value={areaName}
                      onChange={(e) => setAreaName(e.target.value)}
                      required
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="location">Location</Label>
                    <Input
                      id="location"
                      placeholder="e.g., 37.7749, -122.4194"
                      value={location}
                      onChange={(e) => setLocation(e.target.value)}
                    />
                  </div>
                </div>

                {/* Model Selection */}
                <div className="space-y-2">
                  <Label htmlFor="model-select">AI/ML Model</Label>
                  <Select
                    value={model}
                    onValueChange={(value) => setModel(value)}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="unet">U-Net (Segmentation)</SelectItem>
                      <SelectItem value="lstm">LSTM (Time Series)</SelectItem>
                      <SelectItem value="unet-lstm">U-Net + LSTM (Hybrid)</SelectItem>
                      <SelectItem value="cellular-automata">
                        Cellular Automata
                      </SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                {/* Description */}
                <div className="space-y-2">
                  <Label htmlFor="description">Description (Optional)</Label>
                  <Textarea
                    id="description"
                    placeholder="Additional notes about your dataset..."
                    rows={3}
                    value={description}
                    onChange={(e) => setDescription(e.target.value)}
                  />
                </div>

                {/* Upload Progress */}
                {isUploading && (
                  <div className="space-y-2">
                    <div className="flex items-center justify-between text-sm">
                      <span>Uploading...</span>
                      <span>{uploadProgress}%</span>
                    </div>
                    <Progress value={uploadProgress} />
                  </div>
                )}

                {/* Status Messages */}
                {status && (
                  <Alert
                    variant={status.type === "error" ? "destructive" : "default"}
                  >
                    {status.type === "success" ? (
                      <CheckCircle className="h-4 w-4" />
                    ) : (
                      <AlertTriangle className="h-4 w-4" />
                    )}
                    <AlertDescription>{status.message}</AlertDescription>
                  </Alert>
                )}

                <Button
                  type="submit"
                  className="w-full"
                  disabled={!selectedFile || isUploading}
                >
                  {isUploading ? "Uploading..." : "Upload & Analyze"}
                </Button>
              </form>
            </CardContent>
          </Card>

          {/* Right Panel (instructions or results) */}
          <div className="space-y-6">
            {!results ? (
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center">
                    <MapPin className="mr-2 h-5 w-5" />
                    Upload Instructions
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <Alert>
                    <AlertCircle className="h-4 w-4" />
                    <AlertDescription>
                      Upload your dataset and fill in details to start analysis.
                    </AlertDescription>
                  </Alert>

                  <ul className="text-sm space-y-1 text-muted-foreground">
                    <li>• Supported: .nc, .tiff, .geojson</li>
                    <li>• Max size: 50MB</li>
                    <li>• Choose model & optional metadata</li>
                  </ul>
                </CardContent>
              </Card>
            ) : (
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center">
                    <CheckCircle className="mr-2 h-5 w-5 text-green-600" />
                    Prediction Results
                  </CardTitle>
                  <CardDescription>
                    Analysis completed for {results.areaName}
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  {/* Results Summary */}
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <p className="text-sm text-muted-foreground">Total Area</p>
                      <p className="font-medium">{results.totalArea}</p>
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground">Model Used</p>
                      <p className="font-medium">{results.modelUsed}</p>
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground">Confidence</p>
                      <p className="font-medium">{results.confidence}%</p>
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground">Processing Time</p>
                      <p className="font-medium">{results.processingTime}</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
