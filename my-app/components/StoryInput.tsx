"use client"

import { useState } from "react"
import { ArrowRight } from 'lucide-react'
import { useTranslation } from "react-i18next"
import { DotLottieReact } from '@lottiefiles/dotlottie-react' // Import the Lottie component

interface StoryInputProps {
  onSubmit: (storyData: any) => void
  triggerCanvasDownload: () => string | null | undefined
}

export default function StoryInput({ onSubmit, triggerCanvasDownload }: StoryInputProps) {
  const [input, setInput] = useState("")
  const [isLoading, setIsLoading] = useState(false) // Added loading state
  const { t } = useTranslation()

  const handleSubmit = async () => {
    // Set loading state to true when submit is clicked
    setIsLoading(true)
    
    // Trigger the canvas download
    const drawingBase64 = triggerCanvasDownload();

    // Submit the story
    try {
      const user = JSON.parse(sessionStorage.getItem("user") || "{}");

// Get the age from the stored user object
        const age = user?.age || 10; // fallback to 10 if not found

        // Then send it in the request
        const response = await fetch("http://localhost:5000/StoryTeller", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ text: input, age: age, image: drawingBase64 }),
        });
      const data = await response.json()
      console.log("Response from server:", data)
      if (data.error) {
        alert("Server Error (It might be an API Quota Issue): " + data.error)
        return
      }
      onSubmit(data)
    } catch (error) {
      console.error("Error:", error)
    } finally {
      // Set loading state to false once the request is finished
      setIsLoading(false)
    }
  }

  return (
    <div className="flex flex-col items-center mt-4 w-full max-w-2xl relative">
      <div className="flex items-center mt-4 w-full">
        <textarea
          className="flex-grow p-4 border-2 border-blue-400 rounded-l-lg focus:outline-none focus:border-blue-600 font-serif text-lg"
          placeholder={t("Onceuponatime")}
          rows={3}
          value={input}
          onChange={(e) => setInput(e.target.value)}
        />
        <button
          onClick={handleSubmit}
          className="bg-blue-500 text-white p-4 rounded-r-lg hover:bg-blue-600 focus:outline-none transition duration-300 ease-in-out transform hover:scale-105"
          disabled={isLoading} // Disable the button while loading
        >
          {isLoading ? (
            <span className="loader">Loading...</span> // Placeholder text for loading
          ) : (
            <ArrowRight size={24} />
          )}
        </button>
      </div>

      {isLoading && (
  <div className="fixed inset-0 flex items-center justify-center bg-white bg-opacity-90 z-50">
    <div className="text-center">
      <div className="animate-spin rounded-full h-16 w-16 border-t-4 border-blue-500 border-solid mx-auto mb-4"></div>
      <p className="text-lg font-semibold text-blue-700">Loading your story...</p>
    </div>
  </div>
)}



    </div>
  )
}