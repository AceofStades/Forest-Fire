"use client"
import { Button } from "@/src/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/src/components/ui/card"
import { ScrollArea } from "@/src/components/ui/scroll-area"
import { X, Plus, MessageCircle } from "lucide-react"

interface ChatHistoryItem {
  id: string
  title: string
  date: string
  messages: any[]
}

interface ChatHistorySidebarProps {
  isOpen: boolean
  onClose: () => void
  onNewChat: () => void
  onSelectChat: (chatId: string) => void
  activeChat: string
  chatHistory: ChatHistoryItem[]
}

export function ChatHistorySidebar({
  isOpen,
  onClose,
  onNewChat,
  onSelectChat,
  activeChat,
  chatHistory,
}: ChatHistorySidebarProps) {
  if (!isOpen) return null

  return (
    <div className="z-50 w-96 mr-10">
      {/* Overlay for mobile */}
      <div className="absolute inset-0 bg-background/10 backdrop-blur-sm lg:hidden" onClick={onClose} />

      {/* Sidebar */}
      <Card className=" h-[100vh] w-90 lg:w-full shadow-lg">
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-lg font-semibold">Chat History</CardTitle>
          <Button variant="ghost" size="icon" onClick={onClose}>
            <X className="w-4 h-4" />
          </Button>
        </CardHeader>

        <CardContent className="p-0">
          <div className="p-4 border-b border-border">
            <Button onClick={onNewChat} className="w-full justify-start bg-transparent" variant="outline">
              <Plus className="w-4 h-4 mr-2" />
              New Chat
            </Button>
          </div>

          <ScrollArea className="h-[calc(100vh-12rem)]">
            <div className="p-4 space-y-2">
              {chatHistory.length === 0 && (
                <p className="text-sm text-muted-foreground">No chats yet. Start a new one!</p>
              )}

              {chatHistory.map((chat) => (
                <Button
                  key={chat.id}
                  variant={activeChat === chat.id ? "secondary" : "ghost"}
                  className="w-full justify-start h-auto p-3 text-left"
                  onClick={() => onSelectChat(chat.id)}
                >
                  <div className="flex items-start space-x-3">
                    <MessageCircle className="w-4 h-4 mt-0.5 flex-shrink-0" />
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium truncate">{chat.title}</p>
                      <p className="text-xs text-muted-foreground">{chat.date}</p>
                    </div>
                  </div>
                </Button>
              ))}
            </div>
          </ScrollArea>
        </CardContent>
      </Card>
    </div>
  )
}
