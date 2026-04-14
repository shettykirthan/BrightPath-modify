'use client'

import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import confetti from 'canvas-confetti'
import { Button } from "@/components/ui/button"
import { useTranslation } from 'react-i18next'
import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogHeader,
    DialogTitle,
    DialogTrigger,
} from "@/components/ui/dialog"
import { CheckCircle, XCircle } from 'lucide-react'


interface EmotionItem {
    id: number
    emotion: typeof emotions[0]
}

// ⚠️ IMPORTANT: Define constants for API and Game Name
const API_BASE_URL = "http://localhost:5001/api/game";
const GAME_NAME = "EmotionRecognitionGame";
const TOTAL_QUESTIONS = 4; // Based on the emotions array slice in startNewGame

export default function EmotionRecognitionGame() {
    const [items, setItems] = useState<EmotionItem[]>([])
    const [correctAnswers, setCorrectAnswers] = useState(0)
    const [incorrectAnswers, setIncorrectAnswers] = useState(0)
    const [score, setScore] = useState(0)
    const [gameOver, setGameOver] = useState(false)
    const [feedback, setFeedback] = useState<{ message: string, isCorrect: boolean } | null>(null)
    const [selectedItem, setSelectedItem] = useState<EmotionItem | null>(null)
    // Removed [match, setMatch] as it was unused
    const { t, i18n } = useTranslation()
    const changeLanguage = (lang: string) => {
        i18n.changeLanguage(lang)
    }
    // Defined emotions array outside or before the interface
    const emotions = [
        { name: t("happy"), emoji: '😊', bg: 'bg-yellow-400' },
        { name: t("sad"), emoji: '😢', bg: 'bg-blue-400' },
        { name: t("angry"), emoji: '😠', bg: 'bg-red-400' },
        { name: t("surprised"), emoji: '😲', bg: 'bg-purple-400' },
    ];

    useEffect(() => {
        startNewGame()
    }, [])

    // The save function is triggered when gameOver becomes true
    useEffect(() => {
        if (gameOver) {
            // Pass the final score values directly since state updates might be batched
            saveMatchData(correctAnswers, incorrectAnswers);
        }
    }, [gameOver])


    const startNewGame = () => {
        const gameEmotions = emotions.slice(0, 4)
        const newItems = gameEmotions.map((emotion, index) => ({ id: index, emotion }))
        setItems(newItems)
        setCorrectAnswers(0)
        setIncorrectAnswers(0)
        setScore(0)
        setGameOver(false)
        setFeedback(null)
        setSelectedItem(null)
    }

    const handleItemClick = (item: EmotionItem) => {
        setSelectedItem(item)
    }

    const handleSlotClick = (targetEmotion: typeof emotions[0]) => {
        if (!selectedItem) return

        if (selectedItem.emotion.name === targetEmotion.name) {
            setItems(items.filter(i => i.id !== selectedItem.id))
            setCorrectAnswers(prev => prev + 1) // Use functional update
            setScore(prev => prev + 1)
            setFeedback({ message: t("CorrectGreatjob"), isCorrect: true })
            if (items.length === 1) {
                setGameOver(true)
                triggerConfetti()
            }
        } else {
            setIncorrectAnswers(prev => prev + 1) // Use functional update
            setScore(prev => prev - 1)
            setFeedback({ message: t("OopsTryagain"), isCorrect: false })
            shakeAnimation()
        }
        setSelectedItem(null)
    }

    const getTodayKey = () => {
        const today = new Date()
        const year = today.getFullYear()
        const month = String(today.getMonth() + 1).padStart(2, '0')
        const day = String(today.getDate()).padStart(2, '0')
        return `${year}-${month}-${day}`
    }

    // ⭐ This function now correctly fetches, updates the match index, and saves to the backend. ⭐
    const saveMatchData = async (finalCorrect: number, finalIncorrect: number) => {
        const todayKey = getTodayKey();

        const user = JSON.parse(sessionStorage.getItem("user") || "{}");
        if (!user || !user.id) {
            console.error("User not found in sessionStorage. Cannot save score to backend.");
            return;
        }
        const userId = user.id;

        // --- STEP 1: Fetch the current game history from the backend ---
        try {
            const fetchResponse = await fetch(`${API_BASE_URL}/history/${userId}/${GAME_NAME}`);

            let currentData: any[] = [];
            if (fetchResponse.ok) {
                const historyResult = await fetchResponse.json();
                currentData = historyResult.data || [];
            } else if (fetchResponse.status !== 404) {
                console.warn("Failed to fetch game history. Starting new history array.");
            }

            // --- STEP 2: Calculate the new match data and correct match index ---
            const matchScore = finalCorrect - finalIncorrect;
            // NOTE: Using finalCorrect as divisor as per your original logic, consider using (finalCorrect + finalIncorrect) for average accuracy.
            const averageScore = finalCorrect > 0 ? matchScore / finalCorrect : 0;

            const newMatch = {
                // Correctly calculates the incremented match index based on fetched data
                match:
                    currentData.length > 0 && currentData[currentData.length - 1].date === todayKey
                        ? currentData[currentData.length - 1].matches.length + 1
                        : 1,
                score: matchScore,
                correct: finalCorrect,
                incorrect: finalIncorrect,
                totalQuestions: TOTAL_QUESTIONS, // Uses defined constant
                averageScore: averageScore.toFixed(2),
                timestamp: new Date().toISOString(),
            };


            // --- STEP 3: Update the history array (Logic from your original file) ---
            if (currentData.length === 0 || currentData[currentData.length - 1].date !== todayKey) {
                currentData.push({
                    date: todayKey,
                    TotalMatches: 1,
                    TotalAverageScore: newMatch.averageScore,
                    matches: [newMatch]
                });
            } else {
                currentData[currentData.length - 1].matches.push(newMatch);
                currentData[currentData.length - 1].TotalMatches = currentData[currentData.length - 1].matches.length;

                const todayHistory = currentData[currentData.length - 1];
                const totalAverageScore = (
                    todayHistory.matches.reduce((sum, match) => sum + parseFloat(match.averageScore), 0) /
                    todayHistory.matches.length
                ).toFixed(2);
                todayHistory.TotalAverageScore = totalAverageScore;
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
            });

            const result = await saveResponse.json();
            if (!saveResponse.ok || !result.success) {
                console.error("Backend save failed:", result.error || 'Unknown error');
            } else {
                console.log(`✅ Game history synced with backend for ${GAME_NAME}`, result.data);
            }
        } catch (err) {
            console.error("Error communicating with backend:", err);
        }
    }

    // ... (rest of the component remains the same)

    const triggerConfetti = () => {
        confetti({
            particleCount: 100,
            spread: 70,
            origin: { y: 0.6 }
        })
    }

    const shakeAnimation = () => {
        const gameArea = document.getElementById('emotion-game-area')
        if (gameArea) {
            gameArea.classList.add('shake')
            setTimeout(() => gameArea.classList.remove('shake'), 500)
        }
    }

    return (
        <div className="text-center">
            <h2 className="text-3xl font-bold mb-4 text-blue-600">{t("EmotionRecognitionGame")}</h2>
            <div className="flex justify-between items-center mb-4">
                <Dialog>
                    <DialogTrigger asChild>
                        <Button variant="outline">{t("HowtoPlay")}</Button>
                    </DialogTrigger>
                    <DialogContent>
                        <DialogHeader>
                            <DialogTitle>{t("HowtoPlayEmotionRecognitionGame")}</DialogTitle>
                            <DialogDescription>
                                {t("Clickonanemojitoselectit")}<br />
                                {t("Thenclickonthematchingemotionword")}<br />
                                {t("Matchallemotionscorrectlytowinthegame")}<br />
                                {t("Ifyoumakeamistakeyoucantryagain")}<br />
                                {t("Havefunandlearntorecognizeemotions")}
                            </DialogDescription>
                        </DialogHeader>
                    </DialogContent>
                </Dialog>
            </div>
            <div id="emotion-game-area" className="mb-8">
                <div className="flex justify-center flex-wrap gap-4 mb-8">
                    {emotions.map((emotion, index) => (
                        <motion.div
                            key={index}
                            className={`w-32 h-32 rounded-lg border-4 border-white/50 shadow-lg ${emotion.bg} flex items-center justify-center text-xl font-bold text-white cursor-pointer`}
                            whileHover={{ scale: 1.05 }}
                            whileTap={{ scale: 0.95 }}
                            onClick={() => handleSlotClick(emotion)}
                        >
                            {emotion.name}
                        </motion.div>
                    ))}
                </div>
                <div className="flex justify-center flex-wrap gap-4">
                    <AnimatePresence>
                        {items.map((item) => (
                            <motion.div
                                key={item.id}
                                className={`w-32 h-32 rounded-full shadow-lg ${item.emotion.bg} flex items-center justify-center text-4xl cursor-pointer ${selectedItem?.id === item.id ? 'border-4 border-blue-500' : ''}`}
                                onClick={() => handleItemClick(item)}
                                initial={{ opacity: 0, y: 20 }}
                                animate={{ opacity: 1, y: 0 }}
                                exit={{ opacity: 0, y: -20 }}
                                whileHover={{ scale: 1.05 }}
                                whileTap={{ scale: 0.95 }}
                            >
                                {item.emotion.emoji}
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
                    🎉 {t("CongratulationsYoumatchedalltheemotions")} 🎉
                </motion.div>
            )}
            <motion.button
                onClick={startNewGame}
                className="mt-8 bg-blue-500 hover:bg-blue-600 text-white font-bold py-3 px-6 rounded-full text-lg shadow-lg hover:shadow-xl transition-all duration-300"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
            >
                {t("RestartGame")}
            </motion.button>
        </div>
    )
}