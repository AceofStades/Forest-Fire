"use client";

import type React from "react";

import { Avatar, AvatarFallback } from "@/src/components/ui/avatar";
import { Card, CardContent } from "@/src/components/ui/card";
import { Bot, User } from "lucide-react";

interface ChatMessageProps {
	message: string;
	isUser: boolean;
	timestamp?: string;
	children?: React.ReactNode;
	isHtml?: boolean;
}

export function ChatMessage({
	message,
	isUser,
	timestamp,
	children,
	isHtml,
}: ChatMessageProps) {
	return (
		<div
			className={`flex gap-3 ${isUser ? "flex-row-reverse" : "flex-row"}`}
		>
			<Avatar className="w-8 h-8 flex-shrink-0">
				<AvatarFallback
					className={
						isUser
							? "bg-primary text-primary-foreground"
							: "bg-blue-100"
					}
				>
					{isUser ? (
						<User className="w-4 h-4" />
					) : (
						<Bot className="w-4 h-4 text-blue-600" />
					)}
				</AvatarFallback>
			</Avatar>

			<div
				className={`flex-1 max-w-[80%] ${isUser ? "text-right" : "text-left"}`}
			>
				<Card
					className={`${isUser ? "bg-primary text-primary-foreground" : "bg-muted"}`}
				>
					<CardContent className="p-3">
						{isHtml ? (
							<div
								dangerouslySetInnerHTML={{ __html: message }}
							/>
						) : (
							<div className="whitespace-pre-wrap text-sm leading-relaxed">
								{message.split("\n").map((line, index) => {
									// Handle bullet points
									if (
										line.trim().startsWith("•") ||
										line.trim().startsWith("-")
									) {
										return (
											<div
												key={index}
												className="flex items-start gap-2 my-1"
											>
												<span className="text-blue-500 font-bold">
													•
												</span>
												<span>
													{line.replace(
														/^[•-]\s*/,
														"",
													)}
												</span>
											</div>
										);
									}
									// Handle bold text (markdown-style)
									if (line.includes("**")) {
										const parts = line.split("**");
										return (
											<div key={index} className="my-1">
												{parts.map((part, partIndex) =>
													partIndex % 2 === 1 ? (
														<strong key={partIndex}>
															{part}
														</strong>
													) : (
														part
													),
												)}
											</div>
										);
									}
									return (
										<div key={index} className="my-1">
											{line}
										</div>
									);
								})}
							</div>
						)}
						{children}
					</CardContent>
				</Card>

				{timestamp && (
					<p className="text-xs text-muted-foreground mt-1 px-1">
						{timestamp}
					</p>
				)}
			</div>
		</div>
	);
}
