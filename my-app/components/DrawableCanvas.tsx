"use client"

import { useRef, useEffect, useState, forwardRef, useImperativeHandle } from "react"
import { Pencil, Eraser } from "lucide-react"

export interface DrawableCanvasRef {
  downloadCanvas: () => string | null
}

const DrawableCanvas = forwardRef<DrawableCanvasRef, { setDownload: (fn: () => string | null) => void }>(
  ({ setDownload }, ref) => {
    const canvasRef = useRef<HTMLCanvasElement>(null)
    const [isDrawing, setIsDrawing] = useState(false)
    const [tool, setTool] = useState<"pen" | "eraser">("pen")
    const lastPositionRef = useRef<{ x: number; y: number } | null>(null)

    useEffect(() => {
      const canvas = canvasRef.current
      if (canvas) {
        const context = canvas.getContext("2d")
        if (context) {
          context.fillStyle = "#FFFFFF"
          context.fillRect(0, 0, canvas.width, canvas.height)
        }
      }
    }, [])

    const downloadCanvas = () => {
      const canvas = canvasRef.current
      if (canvas) {
        return canvas.toDataURL("image/png")
      }
      return null;
    }

    useImperativeHandle(ref, () => ({
      downloadCanvas
    }))

    useEffect(() => {
      setDownload(downloadCanvas)
    }, [setDownload])

    const getPos = (e: React.MouseEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current
      if (!canvas) return { x: 0, y: 0 }
      const rect = canvas.getBoundingClientRect()
      return { x: e.clientX - rect.left, y: e.clientY - rect.top }
    }

    const startDrawing = (e: React.MouseEvent<HTMLCanvasElement>) => {
      setIsDrawing(true)
      lastPositionRef.current = getPos(e)
    }

    const stopDrawing = () => {
      setIsDrawing(false)
      lastPositionRef.current = null
    }

    const draw = (e: React.MouseEvent<HTMLCanvasElement>) => {
      if (!isDrawing) return
      const canvas = canvasRef.current
      const context = canvas?.getContext("2d")
      if (context && canvas) {
        const currentPosition = getPos(e)

        // ✏️ Tool settings
        context.lineWidth = tool === "eraser" ? 30 : 4   // change eraser size here
        context.lineCap = "round"
        context.strokeStyle = tool === "eraser" ? "#FFFFFF" : "#000000"

        context.beginPath()
        if (lastPositionRef.current) {
          context.moveTo(lastPositionRef.current.x, lastPositionRef.current.y)
        } else {
          context.moveTo(currentPosition.x, currentPosition.y)
        }
        context.lineTo(currentPosition.x, currentPosition.y)
        context.stroke()

        lastPositionRef.current = currentPosition
      }
    }

    return (
      <div className="relative inline-block">
        {/* Tool buttons (top-right) */}
        <div className="absolute right-2 top-2 z-10 flex gap-2">
          <button
            type="button"
            aria-label="Pen"
            onClick={() => setTool("pen")}
            className={`p-1 rounded-md border shadow ${
              tool === "pen" ? "bg-yellow-200 border-black" : "bg-white border-gray-300"
            }`}
          >
            <Pencil className="h-4 w-4" />
          </button>
          <button
            type="button"
            aria-label="Eraser"
            onClick={() => setTool("eraser")}
            className={`p-1 rounded-md border shadow ${
              tool === "eraser" ? "bg-yellow-200 border-black" : "bg-white border-gray-300"
            }`}
          >
            <Eraser className="h-4 w-4" />
          </button>
        </div>

        <canvas
          ref={canvasRef}
          width={800}
          height={400}
          className="border-4 border-yellow-400 rounded-lg shadow-lg cursor-crosshair"
          onMouseDown={startDrawing}
          onMouseUp={stopDrawing}
          onMouseMove={draw}
          onMouseLeave={stopDrawing}
        />
      </div>
    )
  }
)

DrawableCanvas.displayName = "DrawableCanvas"

export default DrawableCanvas
