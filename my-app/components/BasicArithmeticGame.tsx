'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import confetti from 'canvas-confetti'
import { Button } from "@/components/ui/button"
import { useTranslation } from "react-i18next"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import { CheckCircle, XCircle } from 'lucide-react'
import { useRouter } from 'next/navigation'

export default function BasicArithmeticGame() {
  const router = useRouter()
  const [currentLevel, setCurrentLevel] = useState(0)
  const { t, i18n } = useTranslation()
  const changeLanguage = (lang: string) => {
    i18n.changeLanguage(lang)
  }

  const levels = [
    { name: t("Level1"), operations: ['+', '-'], maxNumber: 10, questions: 5, time: 60 },
    { name: t("Level2"), operations: ['+', '-', '*'], maxNumber: 15, questions: 7, time: 90 },
    { name: t("Level3"), operations: ['+', '-', '*'], maxNumber: 20, questions: 10, time: 120 },
  ]

  const [question, setQuestion] = useState('')
  const [answer, setAnswer] = useState('')
  const [score, setScore] = useState(0)
  const [timeLeft, setTimeLeft] = useState(levels[currentLevel].time)
  const [gameOver, setGameOver] = useState(false)
  const [feedback, setFeedback] = useState<string | null>(null)
  const [correctAnswers, setCorrectAnswers] = useState(0)
  const [incorrectAnswers, setIncorrectAnswers] = useState(0)

  useEffect(() => {
    if (timeLeft > 0 && !gameOver) {
      const timer = setTimeout(() => setTimeLeft(timeLeft - 1), 1000)
      return () => clearTimeout(timer)
    } else if (timeLeft === 0 && !gameOver) {
      endGame()
    }
  }, [timeLeft, gameOver])

  useEffect(() => {
    generateQuestion()
  }, [currentLevel])

  const generateQuestion = () => {
    const { operations, maxNumber } = levels[currentLevel]
    const operation = operations[Math.floor(Math.random() * operations.length)]
    let num1 = Math.floor(Math.random() * maxNumber) + 1
    let num2 = Math.floor(Math.random() * maxNumber) + 1

    if (operation === '*') {
      num1 = Math.floor(Math.random() * 5) + 1
      num2 = Math.floor(Math.random() * 5) + 1
    }

    setQuestion(`${num1} ${operation} ${num2} = ?`)
    setAnswer('')
    setFeedback(null)
  }

  const handleSubmit = async () => {
    if (answer === '') return
    const correctAnswer = eval(question.replace('=', '').replace('?', ''))
    const isCorrect = parseInt(answer) === correctAnswer

    if (isCorrect) {
      setFeedback('✅ Correct!')
      setCorrectAnswers(prev => prev + 1)
      setScore(score + 1)
    } else {
      setFeedback('❌ Incorrect. Try again!')
      setIncorrectAnswers(prev => prev + 1)
    }

    await saveGameDataToLocalStorage(
      correctAnswers + (isCorrect ? 1 : 0),
      incorrectAnswers + (!isCorrect ? 1 : 0)
    )

    setTimeout(generateQuestion, 1000)
    setAnswer('')
  }

  const endGame = async () => {
    setGameOver(true)
    await saveGameDataToLocalStorage(correctAnswers, incorrectAnswers)
    if (score === levels[currentLevel].questions && currentLevel === levels.length - 1) {
      triggerConfetti()
    }
  }

  const triggerConfetti = () => {
    confetti({
      particleCount: 100,
      spread: 70,
      origin: { y: 0.6 }
    })
  }

  const getTodayKey = () => {
    const today = new Date()
    const year = today.getFullYear()
    const month = String(today.getMonth() + 1).padStart(2, '0')
    const day = String(today.getDate()).padStart(2, '0')
    return `${year}-${month}-${day}`
  }

  // ✅ FIXED SAVE FUNCTION
  const saveGameDataToLocalStorage = async (correct: number, incorrect: number) => {
    const todayKey = getTodayKey()
    let currentData = JSON.parse(localStorage.getItem("basicArithmeticGame") || "[]")


        // ✅ FIXED
      const totalQuestions = correct + incorrect
      const matchScore = correct // total correct answers are your score

      // ✅ Prevent division by zero
      let averageScore = 0
      if (totalQuestions > 0) {
        averageScore = correct / totalQuestions
      } else {
        averageScore = 0
      }


    const newMatch = {
      match:
        currentData.length > 0
          ? currentData[currentData.length - 1].matches.length + 1
          : 1,
      score: matchScore,
      correct: correct,
      incorrect: incorrect,
      totalQuestions: correct + incorrect,
      averageScore: averageScore.toFixed(2),
    }

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

    localStorage.setItem("basicArithmeticGame", JSON.stringify(currentData))

    // ✅ Sync with MongoDB
    try {
      const user = JSON.parse(sessionStorage.getItem("user") || "{}")
      if (!user || !user.id) {
        console.error("User not found in sessionStorage")
        return
      }

      const userId = user.id
      const response = await fetch("http://localhost:5001/api/game/save", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          userId,
          gameName: "BasicArithmeticGame",
          gameHistory: currentData,
        }),
      })

      const result = await response.json()
      if (!result.success) {
        console.error("❌ Backend save failed:", result.error)
      } else {
        console.log("✅ BasicArithmeticGame synced with backend", result.data)
      }
    } catch (err) {
      console.error("Error saving BasicArithmeticGame to backend:", err)
    }
  }

  // ✅ Back to Games → Save current results then navigate
  const handleBackToGames = async () => {
    await saveGameDataToLocalStorage(correctAnswers, incorrectAnswers)
    router.push("/games")
  }

  return (
    <div className="text-center">
      <h2 className="text-3xl font-bold mb-4 text-blue-600">{t("BasicArithmeticAdventure")}</h2>
      <div className="flex justify-between items-center mb-4 flex-wrap gap-2">
        <p className="text-xl">{levels[currentLevel].name}</p>
        <p className="text-xl">{t("Score")} {score}/{levels[currentLevel].questions}</p>
        <p className="text-xl">{t("Time")} {timeLeft}s</p>
        <div className="flex gap-2">
          <Button onClick={handleBackToGames} variant="outline" className="bg-gray-200 hover:bg-gray-300">
            ⬅️ {t("BackToGames")}
          </Button>
          <Dialog>
            <DialogTrigger asChild>
              <Button variant="outline">{t("HowtoPlay")}</Button>
            </DialogTrigger>
            <DialogContent>
              <DialogHeader>
                <DialogTitle>{t("HowtoPlayBasicArithmeticAdventure")}</DialogTitle>
                <DialogDescription>
                  {t("SolvethearithmeticproblemstoprogressthroughlevelsAnswerallquestionscorrectlybeforetimerunsouttowin")}
                </DialogDescription>
              </DialogHeader>
            </DialogContent>
          </Dialog>
        </div>
      </div>

      {!gameOver ? (
        <div className="mb-8">
          <div className="text-3xl mb-4">{question}</div>
          <input
            type="number"
            value={answer}
            onChange={(e) => setAnswer(e.target.value)}
            className="border-2 border-gray-300 rounded px-4 py-2 mb-4"
          />
          <Button
            onClick={handleSubmit}
            className="bg-green-500 hover:bg-green-600 text-white font-bold py-2 px-4 rounded ml-4"
          >
            {t("Submit")}
          </Button>
          {feedback && (
            <motion.div
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              className={`mt-4 text-xl ${feedback.includes('Correct') ? 'text-green-500' : 'text-red-500'}`}
            >
              {feedback}
              {feedback.includes('Correct')
                ? <CheckCircle className="inline ml-2" />
                : <XCircle className="inline ml-2" />}
            </motion.div>
          )}
        </div>
      ) : (
        <div>
          <p className="text-xl mb-4">
            {t("GameOverYourfinalscore:")} {score}/{levels[currentLevel].questions}
          </p>
          <p className="text-xl mb-2">{t("CorrectAnswers")} {correctAnswers}</p>
          <p className="text-xl mb-4">{t("IncorrectAnswers")} {incorrectAnswers}</p>
          <Button
            onClick={() => {
              setCurrentLevel(0)
              setScore(0)
              setCorrectAnswers(0)
              setIncorrectAnswers(0)
              setTimeLeft(levels[0].time)
              setGameOver(false)
              generateQuestion()
            }}
            className="bg-blue-500 hover:bg-blue-600 text-white font-bold py-2 px-4 rounded"
          >
            {t("PlayAgain")}
          </Button>
        </div>
      )}
    </div>
  )
}
