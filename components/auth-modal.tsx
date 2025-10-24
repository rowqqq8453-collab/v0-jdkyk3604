"use client"

import { useState } from "react"
import { motion } from "framer-motion"
import { GlassCard } from "@/components/glass-card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { X, Mail, Lock, User, Loader2, AlertCircle } from "lucide-react"
import { useAuth } from "@/lib/auth-context"

type AuthMode = "login" | "signup"

export function AuthModal({ onClose, onGuestContinue }: { onClose: () => void; onGuestContinue?: () => void }) {
  const [mode, setMode] = useState<AuthMode>("login")
  const [name, setName] = useState("")
  const [studentId, setStudentId] = useState("")
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [rememberMe, setRememberMe] = useState(false)
  const [agreedToTerms, setAgreedToTerms] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState("")

  const { login, signup } = useAuth()

  const handleSubmit = async () => {
    setError("")

    if (mode === "signup") {
      if (!name || !studentId || !email || !password) {
        setError("모든 필드를 입력해주세요.")
        return
      }
      if (!agreedToTerms) {
        setError("약관에 동의해주세요.")
        return
      }
    } else {
      if (!email || !password) {
        setError("아이디와 비밀번호를 입력해주세요.")
        return
      }
    }

    setIsLoading(true)

    try {
      if (mode === "signup") {
        await signup(name, studentId, email, password)
      } else {
        await login(email, password, rememberMe)
      }
      onClose()
    } catch (err: any) {
      setError(err.message || "오류가 발생했습니다.")
    } finally {
      setIsLoading(false)
    }
  }

  const handleGuestContinue = () => {
    if (onGuestContinue) {
      onGuestContinue()
    }
    onClose()
  }

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/60 backdrop-blur-md p-4">
      <motion.div
        initial={{ opacity: 0, scale: 0.95, y: 20 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        exit={{ opacity: 0, scale: 0.95, y: 20 }}
        className="w-full max-w-md"
      >
        <GlassCard className="p-6 space-y-4 rounded-2xl">
          <div className="flex items-center justify-between">
            <h2 className="text-xl font-bold text-gray-900">{mode === "login" ? "로그인" : "회원가입"}</h2>
            <button onClick={onClose} className="p-1.5 hover:bg-gray-100 rounded-full transition-colors">
              <X className="w-5 h-5 text-gray-500" />
            </button>
          </div>

          {error && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              className="p-3 bg-red-50 border border-red-200 rounded-lg flex items-start gap-2"
            >
              <AlertCircle className="w-4 h-4 text-red-600 flex-shrink-0 mt-0.5" />
              <p className="text-xs text-red-700">{error}</p>
            </motion.div>
          )}

          <div className="space-y-3">
            {mode === "signup" && (
              <motion.div
                key="signup-fields"
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                exit={{ opacity: 0, height: 0 }}
                className="space-y-3"
              >
                <div>
                  <label className="text-xs font-medium text-gray-700 mb-1.5 block">아이디</label>
                  <div className="relative">
                    <User className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
                    <Input
                      value={name}
                      onChange={(e) => setName(e.target.value)}
                      placeholder="아이디"
                      className="pl-10 h-11 text-sm"
                    />
                  </div>
                </div>

                <div>
                  <label className="text-xs font-medium text-gray-700 mb-1.5 block">비밀번호</label>
                  <div className="relative">
                    <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
                    <Input
                      type="password"
                      value={password}
                      onChange={(e) => setPassword(e.target.value)}
                      placeholder="비밀번호"
                      className="pl-10 h-11 text-sm"
                    />
                  </div>
                </div>

                <div>
                  <label className="text-xs font-medium text-gray-700 mb-1.5 block">이메일 (선택)</label>
                  <div className="relative">
                    <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
                    <Input
                      type="email"
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      placeholder="example@email.com"
                      className="pl-10 h-11 text-sm"
                    />
                  </div>
                </div>
              </motion.div>
            )}

            {mode === "login" && (
              <>
                <div>
                  <label className="text-xs font-medium text-gray-700 mb-1.5 block">아이디</label>
                  <div className="relative">
                    <User className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
                    <Input
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      placeholder="아이디"
                      className="pl-10 h-11 text-sm placeholder:text-gray-400"
                    />
                  </div>
                </div>

                <div>
                  <label className="text-xs font-medium text-gray-700 mb-1.5 block">비밀번호</label>
                  <div className="relative">
                    <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
                    <Input
                      type="password"
                      value={password}
                      onChange={(e) => setPassword(e.target.value)}
                      placeholder="비밀번호"
                      className="pl-10 h-11 text-sm placeholder:text-gray-400"
                    />
                  </div>
                </div>

                <div className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    id="rememberMe"
                    checked={rememberMe}
                    onChange={(e) => setRememberMe(e.target.checked)}
                    className="w-4 h-4 rounded border-gray-300 text-indigo-600 focus:ring-indigo-500 cursor-pointer"
                  />
                  <label htmlFor="rememberMe" className="text-sm text-gray-700 cursor-pointer">
                    로그인 정보 저장
                  </label>
                </div>
              </>
            )}

            {mode === "signup" && (
              <div className="p-3 bg-amber-50 border border-amber-200 rounded-lg">
                <div className="flex items-start gap-2">
                  <input
                    type="checkbox"
                    checked={agreedToTerms}
                    onChange={(e) => setAgreedToTerms(e.target.checked)}
                    className="mt-0.5 w-4 h-4 rounded border-gray-300 text-indigo-600 focus:ring-indigo-500 cursor-pointer"
                  />
                  <div className="flex-1">
                    <p className="text-xs text-gray-800 leading-relaxed">
                      <span className="font-semibold">서비스 이용약관</span> 및{" "}
                      <span className="font-semibold">개인정보 처리방침</span>에 동의합니다.
                    </p>
                    <p className="text-[10px] text-gray-600 mt-1">민감한 정보는 암호화되어 안전하게 보관됩니다.</p>
                  </div>
                </div>
              </div>
            )}
          </div>

          <div className="space-y-2.5">
            <Button
              onClick={handleSubmit}
              disabled={isLoading}
              className="w-full rounded-xl bg-gradient-to-r from-indigo-500 to-purple-500 hover:from-indigo-600 hover:to-purple-600 text-white h-11 text-sm font-medium shadow-sm transition-all"
            >
              {isLoading ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  처리중...
                </>
              ) : mode === "login" ? (
                "로그인"
              ) : (
                "회원가입"
              )}
            </Button>

            <Button
              onClick={() => {
                setMode(mode === "login" ? "signup" : "login")
                setError("")
              }}
              variant="outline"
              className="w-full rounded-xl border-2 border-gray-200 hover:bg-gray-50 h-10 text-sm font-medium"
            >
              {mode === "login" ? "회원가입하기" : "로그인하기"}
            </Button>

            <div className="relative">
              <div className="absolute inset-0 flex items-center">
                <div className="w-full border-t border-gray-200"></div>
              </div>
              <div className="relative flex justify-center text-xs">
                <span className="bg-white px-3 py-1 text-gray-500 rounded-full border border-gray-200">또는</span>
              </div>
            </div>

            <Button
              onClick={handleGuestContinue}
              variant="outline"
              className="w-full rounded-xl border-2 border-gray-300 hover:bg-gray-50 h-10 text-sm font-medium text-gray-600 bg-transparent"
            >
              비회원으로 돌아가기
            </Button>

            <p className="text-[10px] text-gray-500 text-center leading-relaxed">
              멘토링 매칭 기능은 회원만 사용할 수 있습니다.
            </p>
          </div>
        </GlassCard>
      </motion.div>
    </div>
  )
}
