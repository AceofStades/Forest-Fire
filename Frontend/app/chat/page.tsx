"use client";

import { useState, useRef, useEffect } from "react";
import { ChatHistorySidebar } from "@/components/chat-history-sidebar";
import { QuickExamplesSidebar } from "@/components/quick-examples-sidebar";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Menu, Lightbulb, Sparkles, TrendingUp, BarChart3, Send, Loader2 } from "lucide-react";
import { Navigation } from "@/components/navigation";

interface Message {
	id: string;
	content: string;
	isUser: boolean;
	timestamp: string;
	isHtml?: boolean;
}

export default function ChatPage() {
	const [messages, setMessages] = useState<Message[]>([
		{
			id: "1",
			content:
				"Hello! I'm your AI assistant for exploring ocean data. Please upload a NetCDF file first, then ask me a question about it.",
			isUser: false,
			timestamp: new Date().toLocaleTimeString(),
		},
	]);
	const [isLoading, setIsLoading] = useState(false);
	const [showChatHistory, setShowChatHistory] = useState(false);
	const [showQuickExamples, setShowQuickExamples] = useState(false);
	const [inputValue, setInputValue] = useState("");
	const messagesEndRef = useRef<HTMLDivElement>(null);

	// Auto-scroll to bottom when new messages are added
	useEffect(() => {
		messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
	}, [messages]);

	const handleSendMessage = async (content: string) => {
		if (!content.trim()) return;

		const userMessage: Message = {
			id: Date.now().toString(),
			content,
			isUser: true,
			timestamp: new Date().toLocaleTimeString(),
		};

		const newMessages = [...messages, userMessage];
		setMessages(newMessages);
		setIsLoading(true);
		setInputValue("");

		try {
			const formData = new FormData();
			formData.append("query", content);

			const response = await fetch(
				"http://127.0.0.1:8000/chatbot-response",
				{
					method: "POST",
					body: formData,
				},
			);

			if (!response.ok) {
				const errorData = await response.json();
				throw new Error(
					errorData.detail || "Failed to get AI response",
				);
			}

			const contentType = response.headers.get("content-type");
			let aiMessage: Message;

			if (contentType && contentType.includes("text/html")) {
				const htmlContent = await response.text();
				aiMessage = {
					id: (Date.now() + 1).toString(),
					content: htmlContent,
					isUser: false,
					timestamp: new Date().toLocaleTimeString(),
					isHtml: true,
				};
			} else {
				const data = await response.json();
				const messageContent =
					data.message || "No data found for this query.";
				const preview = data.preview
					? `\n\n**Data Preview:**\n\`\`\`json\n${JSON.stringify(
						data.preview,
						null,
						2,
					)}\n\`\`\``
					: "";

				aiMessage = {
					id: (Date.now() + 1).toString(),
					content: messageContent + preview,
					isUser: false,
					timestamp: new Date().toLocaleTimeString(),
				};
			}

			setMessages((prevMessages) => [...prevMessages, aiMessage]);
		} catch (error: any) {
			console.error("Chat error:", error);

			const errorMessage: Message = {
				id: (Date.now() + 1).toString(),
				content: `I apologize, but an error occurred. Please ensure the backend is running and data has been uploaded.\n\n**Error:** ${error.message}`,
				isUser: false,
				timestamp: new Date().toLocaleTimeString(),
			};

			setMessages((prevMessages) => [...prevMessages, errorMessage]);
		} finally {
			setIsLoading(false);
		}
	};

	return (
		<div className="h-screen flex flex-col bg-[#001222]">
			{/* Professional Navbar */}
			{/* <Navbar /> */}
			<Navigation />


			{/* Main Container */}
			<div className="flex flex-1 relative overflow-hidden">
				{/* Sidebars */}
				<ChatHistorySidebar
					isOpen={showChatHistory}
					onClose={() => setShowChatHistory(false)}
					onNewChat={() => {
						setMessages([
							{
								id: "1",
								content:
									"Hello! I'm your AI assistant for exploring ocean data. Please upload a NetCDF file first, then ask me a question about it.",
								isUser: false,
								timestamp: new Date().toLocaleTimeString(),
							},
						]);
					}}
					onSelectChat={(chatId: string) => {
						console.log("Selected chat:", chatId);
					}}
					activeChat={""}
					chatHistory={[]}
				/>
				<QuickExamplesSidebar
					isOpen={showQuickExamples}
					onClose={() => setShowQuickExamples(false)}
					onSelectExample={(example: string) => {
						setInputValue(example);
						setShowQuickExamples(false);
					}}
				/>

				{/* Main Chat Container */}
				<div className="flex-1 flex flex-col max-w-5xl mx-auto w-full px-4 py-6 overflow-hidden">
					{/* Header Buttons */}
					<div className="flex items-center justify-between mb-4 flex-shrink-0">
						<Button
							variant="outline"
							size="sm"
							onClick={() => setShowChatHistory(true)}
							className="flex items-center gap-2"
							style={{
								backgroundColor: '#775253',
								color: '#e5cdc8',
								borderColor: '#775253'
							}}
						>
							<Menu className="w-4 h-4" />
							Chat History
						</Button>

						<Button
							variant="outline"
							size="sm"
							onClick={() => setShowQuickExamples(true)}
							className="flex items-center gap-2"
							style={{
								backgroundColor: '#775253',
								color: '#e5cdc8',
								borderColor: '#775253'
							}}
						>
							<Lightbulb className="w-4 h-4" />
							Quick Examples
						</Button>
					</div>

					{/* Messages Area */}
					<div
						className="flex-1 rounded-2xl shadow-lg overflow-hidden mb-4"
						style={{ backgroundColor: '#1a1a1a' }}
					>
						<ScrollArea className="h-full">
							<div className="p-6 space-y-6">
								{messages.map((message) => (
									<div
										key={message.id}
										className={`flex ${message.isUser
											? "justify-end"
											: "justify-start"
											}`}
									>
										<div className="flex gap-3 max-w-[80%]">
											{!message.isUser && (
												<div
													className="w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 mt-1 shadow-sm"
													style={{
														backgroundColor: '#0a8754',
														color: 'white'
													}}
												>
													<span className="text-sm font-semibold">AI</span>
												</div>
											)}
											<div className="flex flex-col gap-1">
												<div
													className="rounded-2xl px-5 py-3 shadow-sm"
													style={{
														backgroundColor: message.isUser
															? '#004f2d'
															: '#775253',
														color: message.isUser
															? '#e5cdc8'
															: 'white'
													}}
												>
													{message.isHtml ? (
														<div dangerouslySetInnerHTML={{ __html: message.content }} />
													) : (
														<div className="whitespace-pre-wrap leading-relaxed">
															{message.content}
														</div>
													)}
												</div>
												<span
													className="text-xs px-2"
													style={{ color: '#e5cdc8', opacity: 0.6 }}
												>
													{message.timestamp}
												</span>
											</div>
											{message.isUser && (
												<div
													className="w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 mt-1 shadow-sm"
													style={{
														backgroundColor: '#0a8754',
														color: 'white'
													}}
												>
													<span className="text-sm font-semibold">U</span>
												</div>
											)}
										</div>
									</div>
								))}
								{isLoading && (
									<div className="flex justify-start">
										<div className="flex gap-3 max-w-[80%]">
											<div
												className="w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 mt-1 shadow-sm"
												style={{
													backgroundColor: '#0a8754',
													color: 'white'
												}}
											>
												<span className="text-sm font-semibold">AI</span>
											</div>
											<div
												className="rounded-2xl px-5 py-3 shadow-sm flex items-center gap-2"
												style={{
													backgroundColor: '#775253',
													color: 'white'
												}}
											>
												<Loader2 className="w-4 h-4 animate-spin" />
												<span>Analyzing your query...</span>
											</div>
										</div>
									</div>
								)}
								<div ref={messagesEndRef} />
							</div>
						</ScrollArea>
					</div>

					{/* Input Area */}
					<div
						className="rounded-2xl shadow-lg p-4"
						style={{ backgroundColor: '#1a1a1a' }}
					>
						<div className="flex items-center gap-3">
							<input
								type="text"
								value={inputValue}
								onChange={(e) => setInputValue(e.target.value)}
								onKeyPress={(e) => {
									if (e.key === 'Enter' && !isLoading) {
										handleSendMessage(inputValue);
									}
								}}
								placeholder="Ask about the uploaded ocean data..."
								disabled={isLoading}
								className="flex-1 px-4 py-3 rounded-xl border-2 focus:outline-none focus:ring-2 transition-all"
								style={{
									borderColor: '#351431',
									backgroundColor: '#2a2a2a',
									color: '#e5cdc8'
								}}
							/>
							<button
								onClick={() => handleSendMessage(inputValue)}
								disabled={isLoading || !inputValue.trim()}
								className="p-3 rounded-xl shadow-md hover:shadow-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed"
								style={{
									backgroundColor: '#0a8754',
									color: 'white'
								}}
							>
								<Send className="w-5 h-5" />
							</button>
						</div>
					</div>
				</div>
			</div>
		</div>
	);
}