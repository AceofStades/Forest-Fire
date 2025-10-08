"use client"

import type React from "react"

import { useState } from "react"
import { Button } from "@/src/components/ui/button"
import { Textarea } from "@/src/components/ui/textarea"
import { Send } from "lucide-react"

interface ChatInputProps {
  onSendMessage: (message: string) => void
  disabled?: boolean
  placeholder?: string
}

export function ChatInput({
  onSendMessage,
  disabled = false,
  placeholder = "Ask about ARGO floats and ocean data...",
}: ChatInputProps) {
  const [message, setMessage] = useState("")

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (message.trim() && !disabled) {
      onSendMessage(message.trim())
      setMessage("")
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e)
    }
  }

  return (
    <form onSubmit={handleSubmit} className="p-4 border-t border-border">
      <div className="flex gap-2">
        <Textarea
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          disabled={disabled}
          className="min-h-[60px] resize-none"
          rows={2}
        />
        <Button
          type="submit"
          disabled={disabled || !message.trim()}
          size="icon"
          className="h-[60px] w-[60px] flex-shrink-0"
        >
          <Send className="w-4 h-4" />
        </Button>
      </div>
    </form>
  )
}
