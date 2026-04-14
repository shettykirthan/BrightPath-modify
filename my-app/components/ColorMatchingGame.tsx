import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import confetti from 'canvas-confetti'
import { useTranslation } from "react-i18next"
import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import { CheckCircle, XCircle } from 'lucide-react'

const colors = [
  { name: 'red', bg: 'bg-red-400', text: 'text-red-600' },
  { name: 'blue', bg: 'bg-blue-400', text: 'text-blue-600' },
  { name: 'green', bg: 'bg-green-400', text: 'text-green-600' },
  { name: 'yellow', bg: 'bg-yellow-400', text: 'text-yellow-600' },
  { name: 'purple', bg: 'bg-purple-400', text: 'text-purple-600' },
  { name: 'pink', bg: 'bg-pink-400', text: 'text-pink-600' }
]

interface Shape {
  id: number
  color: typeof colors[0]
}

export default function ColorMatchingGame() {
  const [shapes, setShapes] = useState<Shape[]>([])
  const [slots, setSlots] = useState<typeof colors[0][]>([])
  const [correctAnswers, setCorrectAnswers] = useState(0)
  const [incorrectAnswers, setIncorrectAnswers] = useState(0)
  const [score, setScore] = useState(0)
  const [gameOver, setGameOver] = useState(false)
  const [feedback, setFeedback] = useState<{ message: string, isCorrect: boolean } | null>(null)
  const [selectedShape, setSelectedShape] = useState<Shape | null>(null)
  const { t, i18n } = useTranslation()
  const changeLanguage = (lang: string) => {
    i18n.changeLanguage(lang)
  }

  useEffect(() => {
    startNewGame()
  }, [])

  useEffect(() => {
    if (gameOver) {
      saveScoreToLocalStorage()
    }
  }, [gameOver])

  const getTodayKey = () => {
    const today = new Date()
    const year = today.getFullYear()
    const month = String(today.getMonth() + 1).padStart(2, '0')
    const day = String(today.getDate()).padStart(2, '0')
    return `${year}-${month}-${day}`
  }
const saveScoreToLocalStorage = async () => {
  const todayKey = getTodayKey()
  let currentData = JSON.parse(localStorage.getItem("colorMatchingGameScores") || "[]")

  const matchScore = correctAnswers - incorrectAnswers
  const averageScore =
    correctAnswers + incorrectAnswers > 0 ? matchScore / correctAnswers : 0

  const newMatch = {
    match:
      currentData.length > 0
        ? currentData[currentData.length - 1].matches.length + 1
        : 1,
    score: matchScore,
    correct: correctAnswers,
    incorrect: incorrectAnswers,
    totalQuestions: 5,
    averageScore: averageScore.toFixed(2),
  }

  // ✅ update today's entry
  if (
    currentData.length === 0 ||
    currentData[currentData.length - 1].date !== todayKey
  ) {
    currentData.push({
      date: todayKey,
      TotalMatches: 1,
      TotalAverageScore: newMatch.averageScore,
      matches: [newMatch],
    })
  } else {
    const todayHistory = currentData[currentData.length - 1]
    todayHistory.matches.push(newMatch)
    todayHistory.TotalMatches = todayHistory.matches.length
    todayHistory.TotalAverageScore = (
      todayHistory.matches.reduce(
        (sum, match) => sum + parseFloat(match.averageScore),
        0
      ) / todayHistory.matches.length
    ).toFixed(2)
  }

  // ✅ Save locally
  localStorage.setItem("colorMatchingGameScores", JSON.stringify(currentData))

  // ✅ Push to backend
  try {
    const user = JSON.parse(sessionStorage.getItem("user")|| "{}");
    if (!user || !user.id) {
      console.error("User not found in sessionStorage");
      return;
    }

    const userId = user.id;

    const response = await fetch("http://localhost:5001/api/game/save", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        userId,
        gameName: "ColorMatchingGame",
        gameHistory: currentData,
      }),
    })

    const result = await response.json()
    if (!result.success) {
      console.error("Backend save failed:", result.error)
    } else {
      console.log("✅ Game history synced with backend", result.data)
    }
  } catch (err) {
    console.error("Error saving to backend:", err)
  }
}


  const startNewGame = () => {
    const gameColors = colors.slice(0, 4)
    const newShapes = gameColors.map((color, index) => ({ id: index, color }))
    const newSlots = [...gameColors].sort(() => Math.random() - 0.5)
    setShapes(newShapes)
    setSlots(newSlots)
    setCorrectAnswers(0)
    setIncorrectAnswers(0)
    setScore(0)
    setGameOver(false)
    setFeedback(null)
    setSelectedShape(null)
  }

  const handleShapeClick = (shape: Shape) => {
    setSelectedShape(shape)
  }

  const handleSlotClick = (targetColor: typeof colors[0]) => {
    if (!selectedShape) return

    if (selectedShape.color.name === targetColor.name) {
      setShapes(shapes.filter(s => s.id !== selectedShape.id))
      setCorrectAnswers(correctAnswers + 1)
      setFeedback({ message: t("CorrectGreatjob"), isCorrect: true })
      if (shapes.length === 1) {
        setGameOver(true)
        triggerConfetti()
      }
    } else {
      setFeedback({ message: t("OopsTryagain"), isCorrect: false })
      setIncorrectAnswers(incorrectAnswers + 1)
      shakeAnimation()
    }
    setSelectedShape(null)
  }

  const triggerConfetti = () => {
    confetti({
      particleCount: 100,
      spread: 70,
      origin: { y: 0.6 }
    })
  }

  const shakeAnimation = () => {
    const gameArea = document.getElementById('game-area')
    if (gameArea) {
      gameArea.classList.add('shake')
      setTimeout(() => gameArea.classList.remove('shake'), 500)
    }
  }

  return (
    <div className="text-center">
      <h2 className="text-3xl font-bold mb-4 text-blue-600">{t("ColorMatchingGame")}</h2>
      <div className="flex justify-between items-center mb-4">
        <p className="text-xl">{t("Score")} {correctAnswers - incorrectAnswers} | {t("Correct")} {correctAnswers} | {t("Incorrect")} {incorrectAnswers}</p>
        <Dialog>
          <DialogTrigger asChild>
            <Button variant="outline">{t("HowtoPlay")}</Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>{t("HowtoPlayColorMatchingGame")}</DialogTitle>
              <DialogDescription>
                {t("Clickonacoloredcircletoselectit")}<br />
                {t("Thenclickonthematchingcoloredsquare")}<br />
                {t("Matchallcolorscorrectlytowinthegame")}<br />
                {t("Ifyoumakeamistakeyoucantryagain")}<br />
                {t("Havefunandlearnyourcolors")}
              </DialogDescription>
            </DialogHeader>
          </DialogContent>
        </Dialog>
      </div>
      <div id="game-area" className="mb-8">
        <div className="flex justify-center flex-wrap gap-4 mb-8">
          {slots.map((color, index) => (
            <motion.div
              key={index}
              className={`w-32 h-32 rounded-lg border-4 border-white/50 shadow-lg ${color.bg} cursor-pointer flex items-center justify-center`}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => handleSlotClick(color)}
            />
          ))}
        </div>
        <div className="flex justify-center flex-wrap gap-4">
          <AnimatePresence>
            {shapes.map((shape) => (
              <motion.div
                key={shape.id}
                className={`w-32 h-32 rounded-full shadow-lg ${shape.color.bg} cursor-pointer ${selectedShape?.id === shape.id ? 'border-4 border-blue-500' : ''} flex items-center justify-center`}
                onClick={() => handleShapeClick(shape)}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              />
            ))}
          </AnimatePresence>
        </div>
      </div>
      <AnimatePresence>
        {feedback && (
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className={`text-xl font-bold mb-4 ${feedback.isCorrect ? 'text-green-500' : 'text-red-500'}`}
          >
            {feedback.message}
            {feedback.isCorrect ? <CheckCircle className="inline ml-2" /> : <XCircle className="inline ml-2" />}
          </motion.div>
        )}
      </AnimatePresence>
      {gameOver && (
        <motion.div
          className="mt-8 text-2xl font-bold text-green-500"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          🎉{t("CongratulationsYoumatchedallthecolors")}🎉
        </motion.div>
      )}
      <motion.button
        onClick={startNewGame}
        className="mt-8 bg-blue-500 hover:bg-blue-600 text-white font-bold py-3 px-6 rounded-full text-lg shadow-lg hover:shadow-xl transition-all duration-300"
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
      >
        {t("NewGame")}
      </motion.button>
    </div>
  )
}