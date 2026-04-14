"use client"

import { useState } from "react"
import { useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/card"
import Link from "next/link"

export default function SignUpPage() {
  const router = useRouter()
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState("")
  const [formData, setFormData] = useState({
    childName: "",
    // 1. Add childAge to the state
    childAge: "",
    parentEmail: "",
    password: "",
    confirmPassword: "",
  })

  const handleInputChange = (field: string, value: string) => {
    setFormData((prev) => ({ ...prev, [field]: value }))
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError("")

    if (formData.password !== formData.confirmPassword) {
      setError("Passwords do not match ❌")
      return
    }

    // Optional: Basic validation to ensure age is a number
    if (isNaN(Number(formData.childAge)) || Number(formData.childAge) <= 0) {
        setError("Please enter a valid age for your child 🎂");
        return;
    }


    setIsLoading(true)

    try {
      const res = await fetch("http://localhost:5001/api/auth/signup", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        // 3. Include childAge in the JSON payload sent to the backend
        body: JSON.stringify({
          username: formData.childName,
          age: Number(formData.childAge), // Convert to number for backend
          email: formData.parentEmail,
          password: formData.password,
        }),
      })

      if (!res.ok) {
        const data = await res.json()
        throw new Error(data.message || "Signup failed")
      }

      const data = await res.json()

      // ✅ Save user + token
      sessionStorage.setItem("user", JSON.stringify(data.user))
      sessionStorage.setItem("token", data.token)

      // ✅ Redirect
      router.push("/")
    } catch (err: any) {
      setError(err.message)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-400 via-blue-500 to-purple-600 relative overflow-hidden">
      {/* Animated Background */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute top-10 left-10 w-32 h-20 bg-white rounded-full opacity-80 animate-float"></div>
        <div className="absolute top-32 right-20 w-24 h-16 bg-white rounded-full opacity-70 animate-float-delayed"></div>
        <div className="absolute bottom-40 left-1/4 w-28 h-18 bg-white rounded-full opacity-60 animate-float"></div>
      </div>

      <div className="flex items-center justify-center min-h-screen p-4">
        <Card className="w-full max-w-md bg-white/95 backdrop-blur-sm shadow-2xl border-0 animate-slide-up">
          <CardHeader className="text-center space-y-4">
            <div className="mx-auto w-16 h-16 bg-gradient-to-r from-yellow-400 to-orange-400 rounded-full flex items-center justify-center text-2xl animate-pulse">
              ⭐
            </div>
            <CardTitle className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
              Join BrightPath!
            </CardTitle>
            <CardDescription className="text-lg text-gray-600">
              Let&apos;s start your amazing learning adventure! 🚀
            </CardDescription>
          </CardHeader>

          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-6">
              <div className="space-y-2">
                <label className="text-sm font-semibold text-gray-700 flex items-center gap-2">
                  <span className="text-lg">👶</span>Child&apos;s Name
                </label>
                <Input
                  type="text"
                  placeholder="What should we call your little star?"
                  value={formData.childName}
                  onChange={(e) => handleInputChange("childName", e.target.value)}
                  className="h-12 text-lg border-2 border-blue-200 focus:border-blue-400 rounded-xl transition-all duration-300 focus:scale-105"
                  required
                />
              </div>

              {/* 2. New Input Field for Child's Age */}
              <div className="space-y-2">
                <label className="text-sm font-semibold text-gray-700 flex items-center gap-2">
                  <span className="text-lg">🎂</span>Child&apos;s Age
                </label>
                <Input
                  type="number"
                  placeholder="How old are they? (e.g., 5)"
                  value={formData.childAge}
                  // Using value.replace for better control over number input
                  onChange={(e) => handleInputChange("childAge", e.target.value.replace(/[^0-9]/g, ""))}
                  className="h-12 text-lg border-2 border-orange-200 focus:border-orange-400 rounded-xl transition-all duration-300 focus:scale-105"
                  required
                  min="1"
                  max="100" // Assuming a typical age range for a learning app
                />
              </div>
              {/* --- */}

              <div className="space-y-2">
                <label className="text-sm font-semibold text-gray-700 flex items-center gap-2">
                  <span className="text-lg">👨‍👩‍👧‍👦</span>Parent&apos;s Email
                </label>
                <Input
                  type="email"
                  placeholder="your.email@example.com"
                  value={formData.parentEmail}
                  onChange={(e) =>
                    handleInputChange("parentEmail", e.target.value)
                  }
                  className="h-12 text-lg border-2 border-green-200 focus:border-green-400 rounded-xl transition-all duration-300 focus:scale-105"
                  required
                />
              </div>

              <div className="space-y-2">
                <label className="text-sm font-semibold text-gray-700 flex items-center gap-2">
                  <span className="text-lg">🔐</span>Password
                </label>
                <Input
                  type="password"
                  placeholder="Create a super secret password"
                  value={formData.password}
                  onChange={(e) =>
                    handleInputChange("password", e.target.value)
                  }
                  className="h-12 text-lg border-2 border-pink-200 focus:border-pink-400 rounded-xl transition-all duration-300 focus:scale-105"
                  required
                />
              </div>

              <div className="space-y-2">
                <label className="text-sm font-semibold text-gray-700 flex items-center gap-2">
                  <span className="text-lg">✅</span>Confirm Password
                </label>
                <Input
                  type="password"
                  placeholder="Type your password again"
                  value={formData.confirmPassword}
                  onChange={(e) =>
                    handleInputChange("confirmPassword", e.target.value)
                  }
                  className="h-12 text-lg border-2 border-purple-200 focus:border-purple-400 rounded-xl transition-all duration-300 focus:scale-105"
                  required
                />
              </div>

              {error && (
                <p className="text-red-600 text-sm font-semibold">{error}</p>
              )}

              <Button
                type="submit"
                disabled={isLoading}
                className="w-full h-14 text-lg font-bold bg-gradient-to-r from-yellow-400 to-orange-400 hover:from-yellow-500 hover:to-orange-500 text-white rounded-xl shadow-lg hover:shadow-xl transition-all duration-300 hover:scale-105 disabled:opacity-50"
              >
                {isLoading ? (
                  <div className="flex items-center gap-2">
                    <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                    Creating your account...
                  </div>
                ) : (
                  <span className="flex items-center gap-2">
                    Start Learning! <span className="text-xl">🎉</span>
                  </span>
                )}
              </Button>
            </form>

            <div className="mt-6 text-center">
              <p className="text-gray-600">
                Already have an account?{" "}
                <Link
                  href="/Auth/login"
                  className="font-semibold text-blue-600 hover:text-blue-800 transition-colors duration-200 hover:underline"
                >
                  Sign in here! 👋
                </Link>
              </p>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}