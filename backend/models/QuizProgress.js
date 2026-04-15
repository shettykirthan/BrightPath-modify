const mongoose = require("mongoose");
const { Schema } = mongoose;

const questionAttemptSchema = new Schema({
  questionText: String,
  selectedOption: String,
  correctAnswer: String,
  isCorrect: Boolean,
});

const quizAttemptSchema = new Schema({
  date: { type: Date, default: Date.now },
  storyTopic: String,
  score: Number,
  totalQuestions: Number,
  questionsData: [questionAttemptSchema],
});

const quizProgressSchema = new Schema({
  userId: { type: Schema.Types.ObjectId, ref: "User", required: true },
  attempts: [quizAttemptSchema],
  totalAverageScore: { type: Number, default: 0 },
});

module.exports =
  mongoose.models.QuizProgress ||
  mongoose.model("QuizProgress", quizProgressSchema);
