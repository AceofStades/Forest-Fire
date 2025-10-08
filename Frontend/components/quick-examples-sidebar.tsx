"use client"

import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { ScrollArea } from "@/components/ui/scroll-area"
import { X, Lightbulb } from "lucide-react"

interface QuickExamplesSidebarProps {
  isOpen: boolean
  onClose: () => void
  onSelectExample: (example: string) => void
}

export function QuickExamplesSidebar({ isOpen, onClose, onSelectExample }: QuickExamplesSidebarProps) {
  const examples = [
    {
      category: "Temperature Analysis",
      queries: [
        "Show me temperature profiles in the North Atlantic",
        "What's the average temperature at 1000m depth?",
        "Compare temperature trends over the last year",
      ],
    },
    {
      category: "Salinity Data",
      queries: [
        "What are the salinity levels near the equator?",
        "Show salinity variations in the Pacific Ocean",
        "Find areas with highest salinity concentrations",
      ],
    },
    {
      category: "BGC Floats",
      queries: [
        "Find BGC data from the last 6 months",
        "Show oxygen levels in the Southern Ocean",
        "What's the chlorophyll distribution globally?",
      ],
    },
    {
      category: "Float Operations",
      queries: [
        "How many floats are currently active?",
        "Show float deployment locations",
        "Which floats need maintenance?",
      ],
    },
  ]

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 mr-10 z-50 h-[100vh] lg:relative lg:inset-auto">
      {/* Overlay for mobile */}
      <div className="absolute inset-0 bg-background/80 backdrop-blur-sm lg:hidden" onClick={onClose} />

      {/* Sidebar */}
      <Card className="absolute right-0 top-0  w-80 lg:relative lg:w-full shadow-lg">
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-lg font-semibold flex items-center gap-2">
            <Lightbulb className="w-5 h-5" />
            Quick Examples
          </CardTitle>
          <Button variant="ghost" size="icon" onClick={onClose}>
            <X className="w-4 h-4" />
          </Button>
        </CardHeader>

        <CardContent className="p-0">
          <ScrollArea className="h-[calc(100vh-8rem)]">
            <div className="p-4 space-y-6">
              {examples.map((category, categoryIndex) => (
                <div key={categoryIndex} className="space-y-3">
                  <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wide">
                    {category.category}
                  </h3>
                  <div className="space-y-2">
                    {category.queries.map((query, queryIndex) => (
                      <Button
                        key={queryIndex}
                        variant="outline"
                        className="w-full justify-start h-auto p-3 text-left text-wrap bg-transparent"
                        onClick={() => onSelectExample(query)}
                      >
                        <div className="text-sm leading-relaxed">{query}</div>
                      </Button>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </ScrollArea>
        </CardContent>
      </Card>
    </div>
  )
}
