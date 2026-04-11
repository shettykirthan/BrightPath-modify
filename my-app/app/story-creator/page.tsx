"use client"

import { useRef, useState, useCallback } from "react"
import { useRouter } from "next/navigation"
import Background from "../../components/Background"
import Sidebar from "../../components/Sidebar"
import DrawableCanvas, { DrawableCanvasRef } from "../../components/DrawableCanvas"
import StoryInput from "../../components/StoryInput"
import Book from "../../components/Book"
import { Button } from "@/components/ui/button"
import { useTranslation } from "react-i18next"
import { motion, AnimatePresence } from "framer-motion"

export default function StoryCreator() {
  const router = useRouter()
  const bookRef = useRef<HTMLDivElement>(null)
  const [showBook, setShowBook] = useState(false)
  const [story, setStory] = useState<any>("")
  const { t } = useTranslation()

  const canvasRef = useRef<DrawableCanvasRef>(null)
  const [downloadCanvas, setDownloadCanvas] = useState<(() => string | null) | null>(null)

  const handleSubmit = (storyText: any) => {
    console.log("Pages: ", storyText)
    setStory(storyText)
    setShowBook(true)

    // Trigger canvas download when story is submitted
    downloadCanvas?.()

    setTimeout(() => {
      bookRef.current?.scrollIntoView({ behavior: "smooth" })
    }, 100)
  }

  const navigateToQuiz = (storyText: any) => {
    router.push(`/quiz?story=${encodeURIComponent(storyText.response)}`)
  }

  const setDownload = useCallback((fn: () => string | null) => {
    setDownloadCanvas(() => fn)
  }, [])

  return (
    <div className="relative min-h-screen overflow-hidden bg-blue-200">
      <Background />
      <Sidebar />
      <main className="relative z-10 p-8 ml-10">
        <h1 className="text-5xl font-bold text-blue-800 mb-8 font-serif">{t("StoryCreator")}</h1>
        <div className="flex flex-col items-center space-y-8">
          <div className="bg-white p-6 rounded-xl shadow-xl">
            <h2 className="text-2xl font-bold text-blue-600 mb-4">{t("DrawYourStory")}</h2>
            <DrawableCanvas ref={canvasRef} setDownload={setDownload} />
          </div>
          <div className="w-full max-w-2xl">
            <h2 className="text-2xl font-bold text-blue-600 mb-4">{t("TellYourStory")}</h2>
            <StoryInput onSubmit={handleSubmit} triggerCanvasDownload={() => downloadCanvas?.()} />
          </div>
        </div>
        <AnimatePresence>
          {showBook && (
            <motion.div
              ref={bookRef}
              initial={{ opacity: 0, y: 50 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 50 }}
              transition={{ duration: 0.5 }}
              className="mt-16"
            >
              <h2 className="text-3xl font-bold text-blue-800 mb-8 font-serif text-center">{t("YourStorybook")}</h2>
              {story && story.drawing && story.drawing !== "scribbles" && (
                <p className="text-2xl font-bold text-center text-purple-600 mb-4 font-serif">
                  AI noticed you drew: {story.drawing}!
                </p>
              )}
              <Book storyData={story} />
              <div className="mt-8 flex justify-center">
                <Button
                  onClick={() => navigateToQuiz(story)}
                  className="bg-purple-500 hover:bg-purple-600 text-white font-bold py-2 px-4 rounded-full text-xl"
                >
                  {t("TakeaQuiz")}
                </Button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </main>
    </div>
  )
}