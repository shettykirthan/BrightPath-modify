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

const shapes = [
    { name: 'circle', icon: '●', bg: 'bg-red-400' },
    { name: 'square', icon: '■', bg: 'bg-blue-400' },
    { name: 'triangle', icon: '▲', bg: 'bg-green-400' },
    { name: 'star', icon: '★', bg: 'bg-yellow-400' },
]

interface ShapeItem {
    id: number
    shape: typeof shapes[0]
}

// ⚠️ IMPORTANT: Define constants for API and Game Name
const API_BASE_URL = "http://localhost:5001/api/game";
const GAME_NAME = "ShapeSortingGame";

export default function ShapeSortingGame() {
    const [items, setItems] = useState<ShapeItem[]>([])
    const [slots, setSlots] = useState<typeof shapes[]>([])
    const [correctAnswers, setCorrectAnswers] = useState(0)
    const [incorrectAnswers, setIncorrectAnswers] = useState(0)
    const [score, setScore] = useState(0)
    const [gameOver, setGameOver] = useState(false)
    const [feedback, setFeedback] = useState<{ message: string, isCorrect: boolean } | null>(null)
    const [selectedItem, setSelectedItem] = useState<ShapeItem | null>(null)
    const { t, i18n } = useTranslation()
    const changeLanguage = (lang: string) => {
        i18n.changeLanguage(lang)
    }

    useEffect(() => {
        startNewGame()
    }, [])

    useEffect(() => {
        if (gameOver) {
            // Pass the final score values directly since state updates might be batched
            saveMatchData(correctAnswers, incorrectAnswers)
        }
    }, [gameOver])

    // REMOVED loadGameState - It relied on local storage which is now removed.

    const getTodayKey = () => {
        const today = new Date()
        const year = today.getFullYear()
        const month = String(today.getMonth() + 1).padStart(2, '0')
        const day = String(today.getDate()).padStart(2, '0')
        return `${year}-${month}-${day}`
    }

    // UPDATED: This function is now ASYNCHRONOUS and handles all backend communication
    const saveMatchData = async (finalCorrect: number, finalIncorrect: number) => {
        const todayKey = getTodayKey()
        const TOTAL_QUESTIONS = shapes.length // 4 in this case

        const user = JSON.parse(sessionStorage.getItem("user") || "{}");
        if (!user || !user.id) {
            console.error("User not found in sessionStorage. Cannot save score to backend.");
            return;
        }
        const userId = user.id;

        // --- STEP 1: Calculate the new match data ---
        const matchScore = finalCorrect - finalIncorrect
        const averageScore = finalCorrect > 0 ? matchScore / finalCorrect : 0

        const newMatch = {
            // match number will be determined after fetching the current data
            score: matchScore,
            correct: finalCorrect,
            incorrect: finalIncorrect,
            totalQuestions: TOTAL_QUESTIONS,
            averageScore: averageScore.toFixed(2),
            timestamp: new Date().toISOString(),
        }

        try {
            // --- STEP 2: Fetch the current game history from the backend ---
            // ⚠️ Assuming your backend has a GET endpoint for history: /api/game/history/:userId/:gameName
            const fetchResponse = await fetch(`${API_BASE_URL}/history/${userId}/${GAME_NAME}`)

            let currentData: any[] = []
            if (fetchResponse.ok) {
                const historyResult = await fetchResponse.json()
                // Assuming history is in a 'data' field or is the response root
                currentData = historyResult.data || []
            } else if (fetchResponse.status !== 404) {
                console.error("Failed to fetch game history:", fetchResponse.statusText);
            }

            // --- STEP 3: Update the history array with the new match ---
            // Determine match number based on fetched data
            newMatch.match = currentData.length > 0 && currentData[currentData.length - 1].date === todayKey
                ? currentData[currentData.length - 1].matches.length + 1
                : 1;

            if (currentData.length === 0 || currentData[currentData.length - 1].date !== todayKey) {
                currentData.push({
                    date: todayKey,
                    TotalMatches: 1,
                    TotalAverageScore: newMatch.averageScore,
                    matches: [newMatch]
                })
            } else {
                currentData[currentData.length - 1].matches.push(newMatch)
                currentData[currentData.length - 1].TotalMatches = currentData[currentData.length - 1].matches.length
                
                const todayHistory = currentData[currentData.length - 1];
                const totalAverageScore = (
                    todayHistory.matches.reduce((sum, match) => sum + parseFloat(match.averageScore), 0) /
                    todayHistory.matches.length
                ).toFixed(2)
                todayHistory.TotalAverageScore = totalAverageScore
            }

            // --- STEP 4: Push the complete updated history back to the backend ---
            const saveResponse = await fetch(`${API_BASE_URL}/save`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    userId,
                    gameName: GAME_NAME,
                    gameHistory: currentData, // Send the entire updated array
                }),
            })

            const result = await saveResponse.json()
            if (!saveResponse.ok || !result.success) {
                console.error("Backend save failed:", result.error || 'Unknown error')
            } else {
                console.log(`✅ Game history synced with backend for ${GAME_NAME}`, result.data)
            }
        } catch (err) {
            console.error("Error communicating with backend:", err)
        }
    }
    // END of saveMatchData

    const startNewGame = () => {
        const gameShapes = shapes.slice(0, 4)
        const newItems = gameShapes.map((shape, index) => ({ id: index, shape }))

        // Randomize slots
        const newSlots = [...gameShapes].sort(() => Math.random() - 0.5)
        setItems(newItems)
        setSlots(newSlots)
        setScore(0)
        setCorrectAnswers(0)
        setIncorrectAnswers(0)
        setGameOver(false)
        setFeedback(null)
        setSelectedItem(null)
    }

    const handleItemClick = (item: ShapeItem) => {
        setSelectedItem(item)
    }

    const handleSlotClick = (targetShape: typeof shapes[0]) => {
        if (!selectedItem) return;

        let newCorrectAnswers = correctAnswers;
        let newIncorrectAnswers = incorrectAnswers;

        if (selectedItem.shape.name === targetShape.name) {
            // Correct match
            setItems(items.filter(i => i.id !== selectedItem.id)); // Remove matched shape
            newCorrectAnswers = correctAnswers + 1; // Prepare new value
            
            // If all items are matched correctly, game is over
            if (items.length === 1) {
                setGameOver(true);
                triggerConfetti();
            }

            setFeedback({ message: t("CorrectGreatjob"), isCorrect: true });

        } else {
            // Incorrect match
            newIncorrectAnswers = incorrectAnswers + 1; // Prepare new value
            setFeedback({ message: t("OopsTryagain"), isCorrect: false });
            shakeAnimation();
        }

        // Calculate score as correct answers minus incorrect answers
        const newScore = newCorrectAnswers - newIncorrectAnswers;

        // Update state with new values
        setScore(newScore);
        setCorrectAnswers(newCorrectAnswers);
        setIncorrectAnswers(newIncorrectAnswers);

        // REMOVED localStorage save from here. Saving only happens on gameOver.
        
        // Reset selected item
        setSelectedItem(null);
    };

    const triggerConfetti = () => {
        confetti({
            particleCount: 100,
            spread: 70,
            origin: { y: 0.6 }
        })
    }

    const shakeAnimation = () => {
        const gameArea = document.getElementById('shape-game-area')
        if (gameArea) {
            gameArea.classList.add('shake')
            setTimeout(() => gameArea.classList.remove('shake'), 500)
        }
    }

    return (
        <div className="text-center">
            <h2 className="text-3xl font-bold mb-4 text-blue-600">{t("ShapeSortingAdventure")}</h2>
            <div className="flex justify-between items-center mb-4">
                <p className="text-xl">{t("Score")} {score}</p>
                <Dialog>
                    <DialogTrigger asChild>
                        <Button variant="outline">{t("HowtoPlay")}</Button>
                    </DialogTrigger>
                    <DialogContent>
                        <DialogHeader>
                            <DialogTitle>{t("HowtoPlayShapeSortingAdventure")}</DialogTitle>
                            <DialogDescription>
                                {t("Clickonacoloredshapetoselectit")}<br />
                                {t("Thenclickonthematchingshapeoutline")}<br />
                                {t("Matchallshapescorrectlytowinthegame")}<br />
                                {t("Ifyoumakeamistake,youcantryagain")}<br />
                                {t("Havefunandlearnyourshapes")}
                            </DialogDescription>
                        </DialogHeader>
                    </DialogContent>
                </Dialog>
            </div>
            <div id="shape-game-area" className="mb-8">
                <div className="flex justify-center flex-wrap gap-4 mb-8">
                    {slots.map((shape, index) => (
                        <motion.div
                            key={index}
                            className={`w-24 h-24 rounded-lg border-4 border-dashed ${shape.bg} flex items-center justify-center text-4xl text-white/50 cursor-pointer`}
                            whileHover={{ scale: 1.05 }}
                            whileTap={{ scale: 0.95 }}
                            onClick={() => handleSlotClick(shape)}
                        >
                            {shape.icon}
                        </motion.div>
                    ))}
                </div>
                <div className="flex justify-center flex-wrap gap-4">
                    <AnimatePresence>
                        {items.map((item) => (
                            <motion.div
                                key={item.id}
                                className={`w-24 h-24 rounded-full shadow-lg ${item.shape.bg} flex items-center justify-center text-3xl text-white cursor-pointer ${selectedItem?.id === item.id ? 'border-4 border-blue-500' : ''}`}
                                onClick={() => handleItemClick(item)}
                                initial={{ opacity: 0, y: 20 }}
                                animate={{ opacity: 1, y: 0 }}
                                exit={{ opacity: 0, y: -20 }}
                                whileHover={{ scale: 1.05 }}
                                whileTap={{ scale: 0.95 }}
                            >
                                {item.shape.icon}
                            </motion.div>
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
                    🎉 {t("CongratulationsYoumatchedalltheshapes")} 🎉
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