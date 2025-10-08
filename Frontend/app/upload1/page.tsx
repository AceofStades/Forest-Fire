"use client";

import { useState } from "react";
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
	Brain,
	CheckCircle,
	AlertCircle,
	Download,
	Eye,
	AlertTriangle,
} from "lucide-react";
import { useToast } from "@/hooks/use-toast";

export default function UploadPage() {
	const [uploadProgress, setUploadProgress] = useState(0);
	const [isUploading, setIsUploading] = useState(false);
	const [selectedFile, setSelectedFile] = useState<File | null>(null);
	const [results, setResults] = useState<any>(null);
	const [status, setStatus] = useState<{ type: "success" | "error"; message: string } | null>(null);
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

		// Simulate progress bar while uploading
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
				throw new Error(result.detail || "Failed to upload file.");
			}

			setUploadProgress(100);
			setStatus({ type: "success", message: result.message || "Upload successful!" });
			setResults({
				areaName: "Uploaded ARGO Data",
				totalArea: "N/A",
				modelUsed: "Backend Processing",
				confidence: "N/A",
				processingTime: "Few seconds",
				riskDistribution: {
					high: 0,
					medium: 0,
					low: 100,
				},
			});

			toast({
				title: "Upload successful",
				description: result.message || "File uploaded successfully.",
			});
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
					<h1 className="text-3xl font-bold mb-2">Upload & Analyze ARGO Data</h1>
					<p className="text-muted-foreground">
						Upload your NetCDF (.nc) ARGO dataset for backend processing and database storage
					</p>
				</div>

				<div className="grid lg:grid-cols-2 gap-6">
					{/* Upload Form */}
					<Card>
						<CardHeader>
							<CardTitle className="flex items-center">
								<Upload className="mr-2 h-5 w-5" />
								Upload ARGO Data
							</CardTitle>
							<CardDescription>Supported format: .nc (NetCDF)</CardDescription>
						</CardHeader>
						<CardContent>
							<form onSubmit={handleSubmit} className="space-y-6">
								<div className="space-y-2">
									<Label htmlFor="file-upload">NetCDF File</Label>
									<div className="border-2 border-dashed border-muted-foreground/25 rounded-lg p-6 text-center hover:border-muted-foreground/50 transition-colors">
										<Input
											id="file-upload"
											type="file"
											accept=".nc"
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
															<p className="text-muted-foreground">NetCDF (.nc) file only</p>
														</>
													)}
												</div>
											</div>
										</Label>
									</div>
								</div>

								<div className="space-y-2">
									<Label htmlFor="description">Description (Optional)</Label>
									<Textarea
										id="description"
										placeholder="Add notes about this dataset..."
										rows={3}
									/>
								</div>

								{isUploading && (
									<div className="space-y-2">
										<div className="flex items-center justify-between text-sm">
											<span>Uploading...</span>
											<span>{uploadProgress}%</span>
										</div>
										<Progress value={uploadProgress} />
									</div>
								)}

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
									{isUploading ? "Uploading..." : "Upload File"}
								</Button>
							</form>
						</CardContent>
					</Card>

					{/* Results Panel */}
					<div className="space-y-6">
						{results ? (
							<Card>
								<CardHeader>
									<CardTitle className="flex items-center">
										<CheckCircle className="mr-2 h-5 w-5 text-green-600" />
										Processing Results
									</CardTitle>
									<CardDescription>
										Analysis completed for {results.areaName}
									</CardDescription>
								</CardHeader>
								<CardContent className="space-y-4">
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
											<p className="text-sm text-muted-foreground">Processing Time</p>
											<p className="font-medium">{results.processingTime}</p>
										</div>
									</div>

									<div className="space-y-3">
										<h4 className="font-medium">Risk Distribution</h4>
										<div className="space-y-2">
											<div className="flex items-center justify-between">
												<div className="flex items-center space-x-2">
													<div className="w-3 h-3 bg-red-500 rounded"></div>
													<span className="text-sm">High Risk</span>
												</div>
												<Badge variant="destructive">{results.riskDistribution.high}%</Badge>
											</div>
											<div className="flex items-center justify-between">
												<div className="flex items-center space-x-2">
													<div className="w-3 h-3 bg-orange-500 rounded"></div>
													<span className="text-sm">Medium Risk</span>
												</div>
												<Badge variant="secondary">{results.riskDistribution.medium}%</Badge>
											</div>
											<div className="flex items-center justify-between">
												<div className="flex items-center space-x-2">
													<div className="w-3 h-3 bg-green-500 rounded"></div>
													<span className="text-sm">Low Risk</span>
												</div>
												<Badge variant="outline">{results.riskDistribution.low}%</Badge>
											</div>
										</div>
									</div>

									<div className="flex space-x-2">
										<Button className="flex-1">
											<Eye className="mr-2 h-4 w-4" />
											View on Map
										</Button>
										<Button variant="outline" className="flex-1 bg-transparent">
											<Download className="mr-2 h-4 w-4" />
											Download
										</Button>
									</div>
								</CardContent>
							</Card>
						) : (
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
											Upload your NetCDF file to get started with backend analysis
										</AlertDescription>
									</Alert>

									<ul className="text-sm space-y-1 text-muted-foreground">
										<li>• Supported file type: .nc (NetCDF)</li>
										<li>• Maximum file size: 50MB</li>
										<li>• Data will be processed and stored in the backend</li>
									</ul>
								</CardContent>
							</Card>
						)}
					</div>
				</div>
			</div>
		</div>
	);
}
