'use client'

import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
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

const instruments = ['🎹', '🎸', '🥁', '🎺']

interface GameMatch {
  match: number
  score: number
  correct: number
  incorrect: number
  totalQuestions: number
  averageScore: string
}

interface DayRecord {
  date: string
  TotalMatches: number
  TotalAverageScore: string
  matches: GameMatch[]
}

export default function MusicalPatternsGame() {
  const { t, i18n } = useTranslation()
  const changeLanguage = (lang: string) => { i18n.changeLanguage(lang) }

  const levels = [
    { name: t("Level1"), patternLength: 3, speed: 1000 },
    { name: t("Level2"), patternLength: 5, speed: 800 },
    { name: t("Level3"), patternLength: 7, speed: 600 },
  ]

  const [currentLevel, setCurrentLevel] = useState(0)
  const [pattern, setPattern] = useState<string[]>([])
  const [playerPattern, setPlayerPattern] = useState<string[]>([])
  const [isPlaying, setIsPlaying] = useState(false)
  const [gameOver, setGameOver] = useState(false)
  const [feedback, setFeedback] = useState<string | null>(null)
  const [isCorrect, setIsCorrect] = useState<boolean | null>(null)
  const [timeRemaining, setTimeRemaining] = useState<number>(30)
  const [timerActive, setTimerActive] = useState<boolean>(false)

  const [gameData, setGameData] = useState(() => ({
    score: 0,
    correct: 0,
    incorrect: 0,
    match: 1,
    totalQuestions: 0
  }))

  // ---------- Timer ----------
  useEffect(() => {
    if (timerActive && timeRemaining > 0) {
      const timer = setInterval(() => setTimeRemaining(prev => prev - 1), 1000)
      return () => clearInterval(timer)
    } else if (timeRemaining === 0) {
      handleGameOver()
    }
  }, [timerActive, timeRemaining])

  // ---------- Generate pattern ----------
  useEffect(() => { generatePattern() }, [currentLevel])

  const generatePattern = () => {
    setTimeRemaining(30)
    const newPattern = Array.from({ length: levels[currentLevel].patternLength }, () =>
      instruments[Math.floor(Math.random() * instruments.length)]
    )
    setPattern(newPattern)
    setPlayerPattern([])
    setIsPlaying(true)
    playPattern(newPattern)
    setTimerActive(true)
  }

  const playPattern = async (patternToPlay: string[]) => {
    for (let instrument of patternToPlay) {
      setFeedback(instrument)
      await new Promise(r => setTimeout(r, levels[currentLevel].speed))
      setFeedback(null)
      await new Promise(r => setTimeout(r, 200))
    }
    setIsPlaying(false)
  }

  // ---------- Update Game Scores & save to MongoDB ----------
  const updateGameScores = async (isAnswerCorrect: boolean) => {
    const todayKey = new Date().toISOString().split('T')[0]
    const storedData: DayRecord[] = JSON.parse(localStorage.getItem('musicalGameScore') || '[]')
    let todayRecord = storedData.find(r => r.date === todayKey)

    if (!todayRecord) {
      todayRecord = { date: todayKey, TotalMatches: 0, TotalAverageScore: "0.00", matches: [] }
      storedData.push(todayRecord)
    }

    // Update game stats
    const updatedGameData = {
      ...gameData,
      correct: isAnswerCorrect ? gameData.correct + 1 : gameData.correct,
      incorrect: isAnswerCorrect ? gameData.incorrect : gameData.incorrect + 1,
      totalQuestions: gameData.totalQuestions + 1
    }

    const totalQuestions = updatedGameData.correct + updatedGameData.incorrect
    const matchScore = updatedGameData.correct
    const averageScore = totalQuestions > 0 ? (updatedGameData.correct / totalQuestions).toFixed(2) : "0.00"

    const matchRecord: GameMatch = {
      match: gameData.match,
      score: matchScore,
      correct: updatedGameData.correct,
      incorrect: updatedGameData.incorrect,
      totalQuestions,
      averageScore
    }

    // Add or update match
    const matchIndex = todayRecord.matches.findIndex(m => m.match === gameData.match)
    if (matchIndex >= 0) todayRecord.matches[matchIndex] = matchRecord
    else todayRecord.matches.push(matchRecord)

    // Update daily totals
    todayRecord.TotalMatches = todayRecord.matches.length
    todayRecord.TotalAverageScore = (
      todayRecord.matches.reduce((sum, m) => sum + parseFloat(m.averageScore), 0) /
      todayRecord.TotalMatches
    ).toFixed(2)

    localStorage.setItem('musicalGameScore', JSON.stringify(storedData))
    setGameData(updatedGameData)

    // ---------- Save to MongoDB ----------
    try {
      const user = JSON.parse(sessionStorage.getItem("user") || "{}")
      if (!user || !user.id) return
      const res = await fetch("http://localhost:5001/api/game/save", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          userId: user.id,
          gameName: "MusicalPatternsGame",
          gameHistory: storedData
        })
      })
      const result = await res.json()
      if (!result.success) console.error(result.error)
      else console.log("✅ Synced with backend")
    } catch (err) {
      console.error("Error saving to backend:", err)
    }
  }

  // ---------- Handle clicks ----------
  const handleInstrumentClick = (instrument: string) => {
    if (isPlaying || timeRemaining === 0) return

    const newPlayerPattern = [...playerPattern, instrument]
    setPlayerPattern(newPlayerPattern)

    if (newPlayerPattern.length === pattern.length) {
      const correct = newPlayerPattern.every((i, idx) => i === pattern[idx])
      setIsCorrect(correct)
      updateGameScores(correct)

      if (correct) {
        setFeedback(t("CorrectGreatjob"))
        if (gameData.score + 1 === 3 && currentLevel < levels.length - 1) {
          setCurrentLevel(prev => prev + 1)
          setGameData(prev => ({ ...prev, score: 0, match: prev.match + 1 }))
        } else if (gameData.score + 1 === 3 && currentLevel === levels.length - 1) {
          setGameOver(true)
          triggerConfetti()
        } else setTimeout(generatePattern, 1500)
      } else {
        setFeedback(t("OopsTryagain"))
        setTimeout(() => { setIsCorrect(null); generatePattern() }, 1500)
      }
    }
  }

  const handleGameOver = () => { setGameOver(true); setTimerActive(false); triggerConfetti() }
  const triggerConfetti = () => { confetti({ particleCount: 100, spread: 70, origin: { y: 0.6 } }) }

  const getTodayKey = () => new Date().toISOString().split('T')[0]

  // ---------- JSX ----------
  return (
    <div className="text-center">
      <h2 className="text-3xl font-bold mb-4 text-blue-600">{t("MusicalPatternsGame")}</h2>

      <div className="flex justify-between items-center mb-4">
        <p className="text-xl">{levels[currentLevel].name}</p>
        <p className="text-xl">{t("Score")} {gameData.score}/3</p>
        <p className="text-xl">{t("Correct")}{gameData.correct}</p>
        <p className="text-xl">{t("Incorrect")} {gameData.incorrect}</p>
        <p className="text-xl">{t("Match")} {gameData.match}</p>
        <p className="text-xl">{t("Time")}{timeRemaining}s</p>

        <Dialog>
          <DialogTrigger asChild><Button variant="outline">{t("HowtoPlay")}</Button></DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>{t("HowtoPlayMusicalPatternsGame")}</DialogTitle>
              <DialogDescription>
                {t("Watchthepatternofinstruments")}<br/>
                {t("Clicktheinstrumentsinthesameordertorepeatthepattern")}<br/>
                {t("Completepatternscorrectlytoadvancetothenextlevel")}<br/>
                {t("Thereare3levelswithincreasingdifficulty")}<br/>
                {t("Winthegamebycompletingalllevels")}<br/>
                {t("Havefunandimproveyourmemoryandpatternrecognitionskills")}
              </DialogDescription>
            </DialogHeader>
          </DialogContent>
        </Dialog>
      </div>

      <div className="mb-8 flex justify-center gap-4">
        {instruments.map((instrument, index) => (
          <motion.button
            key={index}
            className="w-40 h-40 text-6xl bg-purple-200 rounded-lg shadow-lg flex items-center justify-center"
            onClick={() => handleInstrumentClick(instrument)}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            disabled={isPlaying || timeRemaining === 0}
          >
            {instrument}
          </motion.button>
        ))}
      </div>

      <AnimatePresence>
        {feedback && (
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="text-2xl font-bold mb-4"
          >
            {feedback} {isCorrect !== null && (isCorrect ? <CheckCircle className="inline text-green-500 ml-2"/> : <XCircle className="inline text-red-500 ml-2"/>)}
          </motion.div>
        )}
      </AnimatePresence>

      {gameOver ?
        <motion.div className="text-2xl font-bold text-green-500">🎉 {t("CongratulationsYouvecompletedalllevels")} 🎉</motion.div> :
        <Button onClick={generatePattern} disabled={isPlaying || timeRemaining === 0} className="bg-blue-500 hover:bg-blue-600 text-white font-bold py-2 px-4 rounded">
          {gameData.score === 0 ? t("StartGame") : t("NextPattern")}
        </Button>
      }
    </div>
  )
}
