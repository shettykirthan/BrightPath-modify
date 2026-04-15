const express = require("express");
const QuizProgress = require("../models/QuizProgress");

const router = express.Router();

// Save quiz attempt
router.post("/save", async (req, res) => {
  try {
    const { userId, storyTopic, score, totalQuestions, questionsData } = req.body;

    let progress = await QuizProgress.findOne({ userId });
    
    if (!progress) {
      progress = new QuizProgress({ userId, attempts: [] });
    }

    // Add new attempt
    progress.attempts.push({
      storyTopic,
      score,
      totalQuestions,
      questionsData
    });

    // Update total average score
    const totalScore = progress.attempts.reduce((acc, curr) => acc + curr.score, 0);
    const totalAttemptedQuestions = progress.attempts.reduce((acc, curr) => acc + curr.totalQuestions, 0);
    
    if (totalAttemptedQuestions > 0) {
      progress.totalAverageScore = (totalScore / totalAttemptedQuestions) * 100;
    }

    await progress.save();

    res.json({ success: true, data: progress });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
});

// Get quiz history
router.get("/:userId", async (req, res) => {
  try {
    const { userId } = req.params;
    const progress = await QuizProgress.findOne({ userId });

    if (!progress) {
      return res.json({ success: true, data: [] });
    }

    res.json({ success: true, data: progress.attempts });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
});

module.exports = router;
