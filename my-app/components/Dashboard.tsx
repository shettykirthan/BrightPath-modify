"use client";

import { motion } from 'framer-motion';
import { Brain, BookOpen, MessageSquare, Clock, Lightbulb, Mic, PenTool } from 'lucide-react';
import { MetricCard } from './MetricCard';
import { ProgressChart } from './ProgressChart';
import { MilestoneCard } from './MilestoneCard';
import { EmotionTracker } from './EmotionTracker';
import { CommunicationCard } from './CommunicationCard';
import { AISuggestionsCard } from './AISuggestionsCard';
import Background from './Background_copy';
import Sidebar from './Sidebar';
import axios from 'axios';
import { useState, useEffect, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

// --- CONSTANTS ---
const API_BASE_URL = "http://localhost:5001/api/game";
// Keys matching the exact 'gameName' strings saved to the backend
const GAME_SCORE_KEYS = [
  { key: 'ColorMatchingGame', name: 'ColorMatchingGame' },
  { key: 'EmotionRecognitionGame', name: 'EmotionRecognitionGame' },
  { key: 'ShapeSortingGame', name: 'ShapeSortingGame' },
  
  { key: 'GrammarDetectiveGame', name: 'GrammarDetectiveGame' },
  { key: 'BasicArithmeticGame', name: 'BasicArithmeticGame' }
];
// -----------------

export default function Dashboard() {
  const [userId, setUserId] = useState(null);
  const [aiSuggestions, setAiSuggestions] = useState([]);
  const [allGameTotalAverageScore, setAllGameTotalAverageScore] = useState(0);
  const [dayStreak, setDayStreak] = useState(0);
  const { t, i18n } = useTranslation();

  const [analyticalData, setAnalyticalData] = useState([]);
  const [consistencyData, setConsistencyData] = useState([]);

  // --- Initial Setup: Get User ID ---
  useEffect(() => {
    const user = JSON.parse(sessionStorage.getItem("user") || "{}");
    if (user && user.id) {
      setUserId(user.id);
    } else {
      console.warn("User ID not found in sessionStorage. Cannot load dashboard data.");
    }
  }, []);

  // --- Helper to get week dates and initialize consistency data ---
  const getCurrentWeekDates = useCallback(() => {
    const today = new Date();
    const dayOfWeek = today.getDay(); // Sunday - Saturday: 0 - 6
    const startOfWeek = new Date(today);
    // Adjust for Sunday (0) to be the end of the previous week, setting the start to Monday (1)
    startOfWeek.setDate(today.getDate() - dayOfWeek + (dayOfWeek === 0 ? -6 : 1));

    const weekDays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
    const dates = [];

    for (let i = 0; i < 7; i++) {
      const date = new Date(startOfWeek);
      date.setDate(startOfWeek.getDate() + i);
      dates.push({
        day: weekDays[i],
        date: date.toISOString().split("T")[0], // YYYY-MM-DD
      });
    }

    // Initialize consistency data structure
    return dates.map(d => ({
        day: d.day, 
        date: d.date, 
        value: 0
    }));

  }, []);

  // --- Helper to shorten game names for display on the bar chart ---
  const getShortGameName = (longName) => {
      return longName
          .replace(/Game/g, '')
          .replace(/Matching/g, 'Match')
          .replace(/Recognition/g, 'Rec')
          .replace(/Sorting/g, 'Sort')
          .replace(/Detective/g, 'Detect')
          .replace(/Arithmetic/g, 'Math')
          .trim();
  };

  // --- Combined Fetch Game Data & Calculate Metrics (Core Logic) ---
  const fetchGameDataAndCalculateMetrics = useCallback(async () => {
    if (!userId) return;

    const initialConsistencyData = getCurrentWeekDates();
    setConsistencyData(initialConsistencyData); 
    
    // Fetch AI suggestions
    try {
      const response = await axios.get("http://localhost:5000/AiSuggestionBot");
      const rawSuggestions = response.data.response;
      const suggestions = rawSuggestions
        .split('\n\n')
        .map((suggestion, index) => {
          const [categoryText, ...suggestionTextParts] = suggestion.split(':');
          const suggestionText = suggestionTextParts.join(':').trim();

          return {
            id: String(index + 1),
            suggestion: suggestionText.replace(/\*\*/g, '').trim(),
            category: categoryText.replace(/\*/g, '').trim()
          };
        });
      setAiSuggestions(suggestions);
    } catch (error) {
      console.error("Error fetching AI suggestions:", error);
    }


    // --- Fetch ALL Game History and Process Metrics ---
    const allDatesPlayed = new Set();
    const weeklyScores = new Map(initialConsistencyData.map(entry => [entry.date, 0]));

    let totalAverageScoreSum = 0;
    let totalEntriesCount = 0;
    const updatedAnalyticalData = [];

    const fetchPromises = GAME_SCORE_KEYS.map(game =>
      // Use the GET endpoint: /api/game/:userId/:gameName
      axios.get(`${API_BASE_URL}/${userId}/${game.name}`)
        .catch(err => {
          console.warn(`Could not fetch history for ${game.name}:`, err.message);
          return { data: { data: [] } }; // Return empty array on failure
        })
    );
    const results = await Promise.all(fetchPromises);


    results.forEach((response, index) => {
      const { name } = GAME_SCORE_KEYS[index];
      const history = response.data.data || []; 

      let categoryTotal = 0;
      let categoryCount = 0;
      
      if (Array.isArray(history)) {
        history.forEach((dayEntry) => {
          const date = dayEntry.date;
          const score = parseFloat(dayEntry.TotalAverageScore) || 0; 

          categoryTotal += score;
          categoryCount += 1;
          allDatesPlayed.add(date);

          if (weeklyScores.has(date)) {
            weeklyScores.set(date, weeklyScores.get(date) + score);
          }
        });
      }

      const categoryAverage = categoryCount > 0 ? (categoryTotal / categoryCount) : 0;
      
      // Shorten the name for the chart X-axis label
      const shortName = getShortGameName(name);

      updatedAnalyticalData.push({
        name: shortName, 
        value: parseFloat(categoryAverage.toFixed(2)) * 100,
        color: index % 2 === 0 ? '#4ADE80' : '#FCD34D',
      });

      totalAverageScoreSum += categoryTotal;
      totalEntriesCount += categoryCount;
    });

    // --- 4. Final Data State Updates ---

    setAnalyticalData(updatedAnalyticalData);
    const calculatedOverallAverage = totalEntriesCount > 0 ? (totalAverageScoreSum / totalEntriesCount).toFixed(4) : 0;
    setAllGameTotalAverageScore(parseFloat(calculatedOverallAverage) * 100);

    // Day Streak Calculation (using allDatesPlayed set)
    let streak = 0;
    const todayStr = new Date().toISOString().split('T')[0];

    if (allDatesPlayed.has(todayStr)) {
      streak = 1;
      let previousDate = new Date(todayStr);

      while (true) {
        previousDate.setDate(previousDate.getDate() - 1);
        const prevDateStr = previousDate.toISOString().split('T')[0];

        if (allDatesPlayed.has(prevDateStr)) {
          streak += 1;
        } else {
          break;
        }
      }
    }
    setDayStreak(streak);

    // Weekly Consistency Data
    setConsistencyData(prev => prev.map(entry => {
      const score = weeklyScores.get(entry.date) || 0;
      return {
        ...entry,
        value: score
      };
    }));

  }, [userId, getCurrentWeekDates]);


  // --- Primary useEffect to trigger data fetching ---
  useEffect(() => {
    fetchGameDataAndCalculateMetrics();
  }, [fetchGameDataAndCalculateMetrics]);


  // --- Static/Random Data for other cards (No changes needed here) ---
  const [understandingData, setUnderstandingData] = useState([
    { month: 'Jan', value: 30 },
    { month: 'Feb', value: 45 },
    { month: 'Mar', value: 60 },
    { month: 'Apr', value: 10 },
  ]);

  useEffect(() => {
    const getRandomNumber = () => {
      return Math.floor(Math.random() * (80 - 30 + 1)) + 30;
    };
    const updatedUnderstandingData = understandingData.map(data => ({
      ...data,
      value: getRandomNumber(),
    }));
    setUnderstandingData(updatedUnderstandingData);
  }, []);

  const daysStreak = [
    { name: t('Games'), value: 40, color: '#FF4B91' },
    { name: t('Learning'), value: 35, color: '#65B741' },
    { name: t('Communication'), value: 25, color: '#4477CE' },
  ];

  const milestones = [
    {
      id: '1',
      title: t("FirstQuizCompleted"),
      achieved: true,
      icon: 'trophy' as const,
      date: t("daysago")
    },
    {
      id: '2',
      title: t("GamesPlayed"),
      achieved: true,
      icon: 'star' as const,
      date: t("weekago")
    },
    {
      id: '3',
      title: t("VocabularyMaster"),
      achieved: false,
      icon: 'award' as const
    },
  ];

  const emotions = [
    { type: 'happy' as const, count: 15, percentage: 75 },
    { type: 'neutral' as const, count: 4, percentage: 20 },
    { type: 'sad' as const, count: 1, percentage: 5 },
  ];

  const [communicationSkills, setCommunicationSkills] = useState([
    { name: t("VerbalSkills"), value: 85, icon: Mic },
    { name: t("WrittenSkills"), value: 78, icon: PenTool },
    { name: t("Comprehension"), value: 90, icon: BookOpen },
  ]);

  useEffect(() => {
    const getRandomNumber = () => {
      return Math.floor(Math.random() * (85 - 50 + 1)) + 50;
    };

    const updatedCommunicationSkills = communicationSkills.map(skill => ({
      ...skill,
      value: getRandomNumber(),
    }));
    setCommunicationSkills(updatedCommunicationSkills);
  }, []);



  return (
    <div className="min-h-screen relative">
      <Background />
      <Sidebar />
      <div className="relative z-10 p-8">
        <div className="max-w-7xl mx-auto space-y-6">
          {/* Title */}
          <motion.h1
            className="text-4xl font-bold text-gray-800 text-center mb-8"
            initial={{ y: -50, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ duration: 0.5 }}
          >
            {t("ChildDevelopmentDashboard")}
          </motion.h1>

          {/* Top Metrics */}
          <motion.div
            className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
          >
            <MetricCard
              title={t("AnalyticalSkills")}
              value={allGameTotalAverageScore}
              icon={Brain}
              color="#FF4B91"
              subtitle={t("AverageScore")}
              unit="%"
            />
            <MetricCard
              title={t("Understanding")}
              value={78}
              icon={Lightbulb}
              color="#65B741"
              subtitle={t("TopicsLearned")}
              unit="%"
            />
            <MetricCard
              title={t("Communication")}
              value={92}
              icon={MessageSquare}
              color="#4477CE"
              subtitle={t("GradeLevel")}
              unit="%"
            />
            <MetricCard
              title={t("Streak")}
              value={dayStreak}
              icon={Clock}
              color="#FFB534"
              subtitle={`${dayStreak} ${t("daysofstreak")}`}
            />

          </motion.div>

          {/* Charts Row 1 */}
          <motion.div
            className="grid grid-cols-1 md:grid-cols-2 gap-6"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.4 }}
          >
            <ProgressChart
              title={t("AnalyticalPerformance")}
              data={analyticalData}
              color="#FF4B91"
              type="bar"
              dataKey="value"
              xAxisKey="name"
              height={200}
              lastUpdated={t("updatednow")}
              tooltipFormatter={(value) => `${value}%`}
            />
            <ProgressChart
              title={t("UnderstandingProgress")}
              data={understandingData}
              color="#65B741"
              type="line"
              dataKey="value"
              xAxisKey="month"
              height={200}
              lastUpdated={t("updatednow")}
              tooltipFormatter={(value) => `${value}%`}
            />
          </motion.div>

          {/* Communication and Weekly Consistency */}
          <motion.div
            className="grid grid-cols-1 md:grid-cols-2 gap-6"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.6 }}
          >
            <CommunicationCard skills={communicationSkills} />
            <ProgressChart
              title={t("WeeklyConsistency")}
              data={consistencyData}
              color="#FFB534"
              type="bar"
              dataKey="value"
              xAxisKey="day"
              height={200}
              lastUpdated={t("updatednow")}
              tooltipFormatter={(value) => `${value.toFixed(2)} score`}
            />
          </motion.div>

          {/* Charts Row 2 */}
          <motion.div
            className="grid grid-cols-1 md:grid-cols-2 gap-6"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.8 }}
          >
            <ProgressChart
              title={t("TimeDistribution")}
              data={daysStreak}
              color="#4477CE"
              type="pie"
              dataKey="value"
              height={200}
              lastUpdated={t("updatednow")}
              tooltipFormatter={(value) => `${value}%`}
            />
            <AISuggestionsCard suggestions={aiSuggestions} />
          </motion.div>

          {/* Bottom Row */}
          <motion.div
            className="grid grid-cols-1 md:grid-cols-2 gap-6"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 1.0 }}
          >
            <EmotionTracker
              emotions={emotions}
              trend="positive"
              focusTime={45}
              responseTime={2.5}
            />
            <MilestoneCard milestones={milestones} />
          </motion.div>
        </div>
      </div>
    </div>
  );
}