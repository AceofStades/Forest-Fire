import { Link } from 'lucide-react'
import React from 'react'
import { Button } from './ui/button'

const CTA = () => {
    return (
        <div>
            <section className="container mx-auto px-4 py-16 text-center">
                <h2 className="text-3xl font-bold mb-6">Ready to Explore Fire Prediction?</h2>
                <p className="text-xl text-muted-foreground mb-8 max-w-2xl mx-auto">
                    Start by exploring our interactive dashboard, convert your 2D maps to 3D, or upload your own data for custom
                    predictions
                </p>
                <div className="flex flex-col sm:flex-row gap-4 justify-center">
                    <Button
                        asChild
                        size="lg"
                        className="bg-gradient-to-r from-orange-500 to-red-600 hover:from-orange-600 hover:to-red-700"
                    >
                        <Link href="/dashboard">Start Exploring</Link>
                    </Button>
                    <Button asChild variant="outline" size="lg">
                        <Link href="/2d-to-3d"className='text-2xl' >Try 2D to 3D</Link>
                    </Button>
                </div>
            </section>
        </div>
    )
}

export default CTA
