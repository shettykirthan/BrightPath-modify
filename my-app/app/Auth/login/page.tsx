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

export default function LoginPage() {
  const router = useRouter()
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState("")
  const [formData, setFormData] = useState({
    email: "",
    password: "",
  })

  const handleInputChange = (field: string, value: string) => {
    setFormData((prev) => ({ ...prev, [field]: value }))
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setIsLoading(true)
    setError("")

    try {
      const res = await fetch("http://localhost:5001/api/auth/signin", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData),
      })

      if (!res.ok) {
        const data = await res.json()
        throw new Error(data.message || "Failed to login")
      }

      const data = await res.json()
      // ✅ Store user + token
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
    <div className="min-h-screen bg-gradient-to-br from-green-400 via-blue-500 to-purple-600 relative overflow-hidden">
      {/* Background animation */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute top-16 right-16 w-36 h-24 bg-white rounded-full opacity-80 animate-float"></div>
        <div className="absolute top-1/3 left-12 w-28 h-18 bg-white rounded-full opacity-70 animate-float-delayed"></div>
        <div className="absolute bottom-32 right-1/4 w-32 h-20 bg-white rounded-full opacity-60 animate-float"></div>
      </div>

      {/* Main Card */}
      <div className="flex items-center justify-center min-h-screen p-4">
        <Card className="w-full max-w-md bg-white/95 backdrop-blur-sm shadow-2xl border-0 animate-slide-up">
          <CardHeader className="text-center space-y-4">
            <div className="mx-auto w-16 h-16 bg-gradient-to-r from-green-400 to-blue-400 rounded-full flex items-center justify-center text-2xl animate-pulse">
              🌟
            </div>
            <CardTitle className="text-3xl font-bold bg-gradient-to-r from-green-600 to-blue-600 bg-clip-text text-transparent">
              Welcome Back!
            </CardTitle>
            <CardDescription className="text-lg text-gray-600">
              Ready to continue your learning journey? 🎓
            </CardDescription>
          </CardHeader>

          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-6">
              <div className="space-y-2">
                <label className="text-sm font-semibold text-gray-700 flex items-center gap-2">
                  <span className="text-lg">📧</span>Email Address
                </label>
                <Input
                  type="email"
                  placeholder="Enter your email"
                  value={formData.email}
                  onChange={(e) => handleInputChange("email", e.target.value)}
                  className="h-12 text-lg border-2 border-green-200 focus:border-green-400 rounded-xl transition-all duration-300 focus:scale-105"
                  required
                />
              </div>

              <div className="space-y-2">
                <label className="text-sm font-semibold text-gray-700 flex items-center gap-2">
                  <span className="text-lg">🔑</span>Password
                </label>
                <Input
                  type="password"
                  placeholder="Enter your password"
                  value={formData.password}
                  onChange={(e) => handleInputChange("password", e.target.value)}
                  className="h-12 text-lg border-2 border-blue-200 focus:border-blue-400 rounded-xl transition-all duration-300 focus:scale-105"
                  required
                />
              </div>

              {error && (
                <p className="text-red-600 text-sm font-semibold">{error}</p>
              )}

              <div className="text-right">
                <Link
                  href="/forgot-password"
                  className="text-sm text-blue-600 hover:text-blue-800 transition-colors duration-200 hover:underline"
                >
                  Forgot your password? 🤔
                </Link>
              </div>

              <Button
                type="submit"
                disabled={isLoading}
                className="w-full h-14 text-lg font-bold bg-gradient-to-r from-green-400 to-blue-400 hover:from-green-500 hover:to-blue-500 text-white rounded-xl shadow-lg hover:shadow-xl transition-all duration-300 hover:scale-105 disabled:opacity-50"
              >
                {isLoading ? (
                  <div className="flex items-center gap-2">
                    <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                    Signing you in...
                  </div>
                ) : (
                  <span className="flex items-center gap-2">
                    Let&apos;s Learn! <span className="text-xl">🚀</span>
                  </span>
                )}
              </Button>
            </form>

            <div className="mt-6 text-center">
              <p className="text-gray-600">
                New to BrightPath?{" "}
                <Link
                  href="/Auth"
                  className="font-semibold text-green-600 hover:text-green-800 transition-colors duration-200 hover:underline"
                >
                  Create an account! ✨
                </Link>
              </p>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
