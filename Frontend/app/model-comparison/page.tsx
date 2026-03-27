import { Navigation } from "@/components/navigation"
import { ModelComparisonContent } from "./content"

export default function ModelComparisonPage() {
  return (
    <div className="min-h-screen bg-background text-foreground flex flex-col">
      <Navigation />
      <ModelComparisonContent />
    </div>
  )
}
