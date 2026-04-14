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

const levels = [
  { 
    name: 'Level 1', 
    sentences: [
      { text: "The cat is sleeping on the bed.", correct: true },
      { text: "She don't like ice cream.", correct: false },
      { text: "They are going to the park.", correct: true },
    ],
    time: 30
  },
  { 
    name: 'Level 2', 
    sentences: [
      { text: "Neither of the students have finished their homework.", correct: false },
      { text: "The team is practicing for the big game.", correct: true },
      { text: "Every one of the apples are ripe.", correct: false },
    ],
    time: 30
  },
  { 
    name: 'Level 3', 
    sentences: [
      { text: "If I were you, I would study harder.", correct: true },
      { text: "The data show that the experiment was successful.", correct: true },
      { text: "She is one of the only people who understands the problem.", correct: false },
    ],
    time: 30
  },
]

export default function GrammarDetectiveGame() {
  const [currentLevel, setCurrentLevel] = useState(0)
  const { t, i18n } = useTranslation()
  const [currentSentence, setCurrentSentence] = useState(0)
  const [score, setScore] = useState(0)
  const [correctAnswers, setCorrectAnswers] = useState(0)
  const [incorrectAnswers, setIncorrectAnswers] = useState(0)
  const [timeLeft, setTimeLeft] = useState(levels[currentLevel].time)
  const [gameOver, setGameOver] = useState(false)
  const [feedback, setFeedback] = useState<string | null>(null)

  useEffect(() => {
    if (timeLeft > 0 && !gameOver) {
      const timer = setTimeout(() => setTimeLeft(timeLeft - 1), 1000)
      return () => clearTimeout(timer)
    } else if (timeLeft === 0 && !gameOver) {
      saveGameDataToLocalStorage()
      endGame()
    }
  }, [timeLeft, gameOver])

  useEffect(() => {
    startNewLevel()
  }, [currentLevel])

  useEffect(() => {
    if (gameOver) {
      saveGameDataToLocalStorage()
    }
  }, [gameOver])

  const startNewLevel = () => {
    setCurrentSentence(0)
    setTimeLeft(levels[currentLevel].time)
    setGameOver(false)
    setFeedback(null)
  }

  const handleAnswer = (isCorrect: boolean) => {
    if (isCorrect === levels[currentLevel].sentences[currentSentence].correct) {
      setScore(score + 1)
      setCorrectAnswers(correctAnswers + 1)
      setFeedback('✅ Correct!')
    } else {
      setIncorrectAnswers(incorrectAnswers + 1)
      setFeedback('❌ Incorrect. Try again!')
    }

    if (currentSentence + 1 < levels[currentLevel].sentences.length) {
      setCurrentSentence(currentSentence + 1)
    } else {
      if (currentLevel < levels.length - 1) {
        setCurrentLevel(currentLevel + 1)
      } else {
        endGame()
      }
    }
  }

  const endGame = () => {
    setGameOver(true)
    if (score === levels[currentLevel].sentences.length && currentLevel === levels.length - 1) {
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

  // ✅ FIXED: Correct average calculation and variable usage
  const saveGameDataToLocalStorage = async () => {
    const todayKey = getTodayKey()
    let currentData = JSON.parse(localStorage.getItem("grammarDetectiveGame") || "[]")

    const totalQuestions = correctAnswers + incorrectAnswers
    const matchScore = correctAnswers

    // ✅ Prevent division by zero
    let averageScore = 0
    if (totalQuestions > 0) {
      averageScore = correctAnswers / totalQuestions
    }

    const newMatch = {
      match:
        currentData.length > 0
          ? currentData[currentData.length - 1].matches.length + 1
          : 1,
      score: matchScore,
      correct: correctAnswers,
      incorrect: incorrectAnswers,
      totalQuestions: totalQuestions,
      averageScore: averageScore.toFixed(2),
    }

    // ✅ Update or create today's record
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

    localStorage.setItem("grammarDetectiveGame", JSON.stringify(currentData))

    // ✅ Push to MongoDB backend
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
          gameName: "GrammarDetectiveGame",
          gameHistory: currentData,
        }),
      })

      const result = await response.json()
      if (!result.success) {
        console.error("❌ Backend save failed:", result.error)
      } else {
        console.log("✅ GrammarDetectiveGame history synced with backend", result.data)
      }
    } catch (err) {
      console.error("Error saving GrammarDetectiveGame to backend:", err)
    }
  }

  return (
    <div className="text-center">
      <h2 className="text-3xl font-bold mb-4 text-blue-600">{t("GrammarDetectiveGame")}</h2>
      <div className="flex justify-between items-center mb-4 flex-wrap gap-2">
        <p className="text-xl">{levels[currentLevel].name}</p>
        <p className="text-xl">
          {t("Score")} {score}/{levels.reduce((acc, level) => acc + level.sentences.length, 0)}
        </p>
        <p className="text-xl">{t("Time")} {timeLeft}s</p>
        <Dialog>
          <DialogTrigger asChild>
            <Button variant="outline">{t("HowtoPlay")}</Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>{t("HowtoPlayGrammarDetectiveGame")}</DialogTitle>
              <DialogDescription>
                {t("Readthesentencecarefully")}<br/>
                {t("Decideifthesentenceisgrammaticallycorrectorincorrect")}<br/>
                {t("Click'Correct'or'Incorrect'basedonyourdecision")}<br/>
                {t("Completeallsentencesineachleveltoadvance")}<br/>
                {t("Thereare3levelswithincreasingdifficulty")}<br/>
                {t("Winthegamebycompletingalllevelsbeforetimerunsout")}<br/>
                {t("Havefunandimproveyourgrammarskills")}
              </DialogDescription>
            </DialogHeader>
          </DialogContent>
        </Dialog>
      </div>

      {!gameOver ? (
        <div className="mb-8">
          <div className="text-2xl mb-4">{levels[currentLevel].sentences[currentSentence].text}</div>
          <div className="flex justify-center gap-4">
            <Button 
              onClick={() => handleAnswer(true)}
              className="bg-green-500 hover:bg-green-600 text-white font-bold py-2 px-4 rounded"
            >
              {t("Correct")}
            </Button>
            <Button 
              onClick={() => handleAnswer(false)}
              className="bg-red-500 hover:bg-red-600 text-white font-bold py-2 px-4 rounded"
            >
              {t("Incorrect")}
            </Button>
          </div>
          {feedback && (
            <div className={`mt-4 text-xl ${feedback.includes('Correct') ? 'text-green-500' : 'text-red-500'}`}>
              {feedback}
            </div>
          )}
        </div>
      ) : (
        <div>
          <p className="text-xl mb-4">
            {t("GameOverYourfinalscore:")} {score}/{levels.reduce((acc, level) => acc + level.sentences.length, 0)}
          </p>
          {score === levels.reduce((acc, level) => acc + level.sentences.length, 0) ? (
            <motion.div
              className="text-2xl font-bold text-green-500"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
            >
              🎉 {t("CongratulationsYouvecompletedalllevels")} 🎉
            </motion.div>
          ) : (
            <Button 
              onClick={() => {
                setCurrentLevel(0)
                setScore(0)
                setCorrectAnswers(0)
                setIncorrectAnswers(0)
                startNewLevel()
              }}
              className="bg-blue-500 hover:bg-blue-600 text-white font-bold py-2 px-4 rounded"
            >
              {t("PlayAgain")}
            </Button>
          )}
        </div>
      )}
    </div>
  )
}
