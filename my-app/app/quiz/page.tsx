'use client'

import React, { useState, useEffect } from 'react'
import { useSearchParams } from 'next/navigation' // Import to access query params
import { motion } from 'framer-motion'
import Background from '../../components/Background'
import Sidebar from '../../components/Sidebar'
import QuizBox from '../../components/QuizBox'
import Bot from '../../components/Bot'
import { Button } from '@/components/ui/button'

interface Question {
  question: string;
  options: string[];
  correctAnswer: string;
}

const defaultQuestions: Question[] = [
  {
    question: "What color is the sky on a clear day?",
    options: ["Blue", "Green", "Red", "Yellow"],
    correctAnswer: "Blue"
  },
  {
    question: "How many legs does a cat have?",
    options: ["Two", "Four", "Six", "Eight"],
    correctAnswer: "Four"
  },
  {
    question: "Which animal says 'moo'?",
    options: ["Dog", "Cat", "Cow", "Sheep"],
    correctAnswer: "Cow"
  },
  {
    question: "What shape is a ball?",
    options: ["Square", "Triangle", "Circle", "Rectangle"],
    correctAnswer: "Circle"
  },
  {
    question: "Which fruit is red and grows on a tree?",
    options: ["Banana", "Orange", "Apple", "Grapes"],
    correctAnswer: "Apple"
  }
]

export default function QuizPage() {
  const [currentQuestion, setCurrentQuestion] = useState(0)
  const [quizStarted, setQuizStarted] = useState(false)
  const [score, setScore] = useState(0)
  const [lastAnswerCorrect, setLastAnswerCorrect] = useState<boolean | null>(null)
  const [loading, setLoading] = useState(false);
  const [questions, setQuestions] = useState<Question[]>([]);
  const [countdown, setCountdown] = useState(20); // Countdown state for the timer
  const [isButtonDisabled, setIsButtonDisabled] = useState(true); // State to disable the start button initially
  const [questionAttempts, setQuestionAttempts] = useState<any[]>([]);

  const searchParams = useSearchParams()
  const story = searchParams.get('story') // Retrieve 'story' from query params

  useEffect(() => {
    if (story) {
      console.log('Received story:', story);
      fetchQuizQuestions(story);
    }
  }, [story]);

  useEffect(() => {
    // Countdown Timer logic
    if (countdown > 0) {
      const timer = setInterval(() => {
        setCountdown(prev => prev - 1); // Decrease countdown every second
      }, 1000);
      
      return () => clearInterval(timer); // Cleanup on component unmount or countdown reaching zero
    } else {
      setIsButtonDisabled(false); // Enable the start button when countdown is 0
    }
  }, [countdown]);

  const fetchQuizQuestions = async (storyText: string) => {
    setLoading(true);
    try {
      const user = JSON.parse(sessionStorage.getItem("user") || "{}");

// Get the age from the stored user object
        const age = user?.age || 10; // fallback to 10 if not found
      const response = await fetch('http://localhost:5000/QuizBot', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: storyText , age: age}),
      });

      const data = await response.json();
      console.log('Fetched Questions1 :', data);
      const fetchedQuestions = data.response.map((item: any) => ({
        question: item.question,
        options: item.options,
        correctAnswer: item.correctAnswer,
      }))
      console.log('Fetched Questions2 :', data.response);
      console.log('Fetched Questions3 :', fetchedQuestions);

      setQuestions(fetchedQuestions);
    } catch (error) {
      console.error('Error fetching questions:', error);
    } finally {
      setLoading(false);
    }
  };

  const startQuiz = () => {
    setQuizStarted(true)
    setScore(0)
    setCurrentQuestion(0)
    setLastAnswerCorrect(null)
    setQuestionAttempts([])
  }

  const handleAnswer = (selectedOption: string) => {
    const correctAnswerRaw = questions[currentQuestion].correctAnswer.toLowerCase().trim()
    const optionRaw = selectedOption.toLowerCase().trim()
    const optionIndex = questions[currentQuestion].options.indexOf(selectedOption)
    const selectedLetter = String.fromCharCode(97 + optionIndex) // Convert index to letter (a, b, c, d)
    
    const isCorrect = 
      correctAnswerRaw === selectedLetter ||
      correctAnswerRaw.startsWith(selectedLetter + ')') ||
      correctAnswerRaw.startsWith(selectedLetter + '.') ||
      correctAnswerRaw.startsWith(selectedLetter + ' ') ||
      correctAnswerRaw === optionRaw ||
      (optionRaw.length > 2 && correctAnswerRaw.includes(optionRaw)) ||
      (correctAnswerRaw.length > 2 && optionRaw.includes(correctAnswerRaw));
    
    // Debug logging
    console.log('Selected Option:', selectedOption)
    console.log('Option Index:', optionIndex)
    console.log('Selected Letter:', selectedLetter)
    console.log('Correct Answer Raw:', correctAnswerRaw)
    console.log('Is Correct:', isCorrect)
    
    const attempt = {
      questionText: questions[currentQuestion].question,
      selectedOption: selectedOption,
      correctAnswer: questions[currentQuestion].correctAnswer,
      isCorrect: isCorrect
    }
    const updatedAttempts = [...questionAttempts, attempt]
    setQuestionAttempts(updatedAttempts)
    
    setLastAnswerCorrect(isCorrect)
    const newScore = isCorrect ? score + 1 : score
    if (isCorrect) {
      setScore(newScore)
    }
    setTimeout(async () => {
      if (currentQuestion < questions.length - 1) {
        setCurrentQuestion(currentQuestion + 1)
        setLastAnswerCorrect(null)
      } else {
        setQuizStarted(false)
        try {
          const user = JSON.parse(sessionStorage.getItem("user") || "{}");
          if (user._id || user.id) {
            await fetch("http://localhost:5001/api/quiz/save", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                userId: user._id || user.id,
                storyTopic: story || "General Quiz",
                score: newScore,
                totalQuestions: questions.length,
                questionsData: updatedAttempts
              })
            });
          }
        } catch (err) {
          console.error("Failed to save quiz progress:", err);
        }
      }
    }, 1500)
  }

  return (
    <div className="relative min-h-screen overflow-hidden bg-gradient-to-b from-purple-300 to-pink-200">
      <Background />
      <Sidebar />
      <main className="relative z-10 p-8 ml-10">
        <h1 className="text-5xl font-bold text-purple-800 mb-8 font-serif text-center">Fun Quiz Time!</h1>
        
        {!quizStarted ? (
          <motion.div 
            initial={{ opacity: 0, y: 50 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex flex-col items-center space-y-4"
          >
            <h2 className="text-2xl font-bold text-purple-600">
              Press start in {countdown} seconds
            </h2>

            {currentQuestion > 0 && (
              <h2 className="text-3xl font-bold text-purple-600">Your Score: {score}/{questions.length}</h2>
            )}
            <Button 
              onClick={startQuiz} 
              className="bg-purple-500 hover:bg-purple-600 text-white font-bold py-2 px-4 rounded-full text-xl"
              disabled={isButtonDisabled} // Disable button until countdown reaches 0
            >
              {currentQuestion > 0 ? 'Play Again' : 'Start Quiz'}
            </Button>
          </motion.div>
        ) : (
          <div className="flex flex-col items-center justify-center h-[calc(100vh-200px)]">
            <div className="mb-4">
              <Bot isCorrect={lastAnswerCorrect} />
            </div>
            <div className="w-full max-w-2xl">
              {questions.length > 0 && questions[currentQuestion] ? (
                <QuizBox 
                  question={questions[currentQuestion].question}
                  options={questions[currentQuestion].options}
                  onAnswer={handleAnswer}
                  isCorrect={lastAnswerCorrect}
                />
              ) : (
                <div className="text-center text-lg text-gray-600">Loading questions...</div>
              )}
            </div>
          </div>
        )}
      </main>
    </div>
  )
}