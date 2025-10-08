"use client"

import { createContext, useContext, useState, useEffect, type ReactNode } from "react"

interface User {
  id: string
  email: string
  name: string
  type: "guest" | "normal" | "premium"
}

interface AuthContextType {
  user: User | null
  isAuthenticated: boolean
  login: (email: string, password: string) => Promise<boolean>
  register: (email: string, password: string, name: string) => Promise<boolean>
  logout: () => void
  continueAsGuest: () => void
}

const AuthContext = createContext<AuthContextType | undefined>(undefined)

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null)
  const [isAuthenticated, setIsAuthenticated] = useState(false)

  useEffect(() => {
    // Check for existing session
    const savedUser = localStorage.getItem("ocean-platform-user")
    if (savedUser) {
      const userData = JSON.parse(savedUser)
      setUser(userData)
      setIsAuthenticated(true)
    }
  }, [])

  const login = async (email: string, password: string): Promise<boolean> => {
    // Simulate API call
    await new Promise((resolve) => setTimeout(resolve, 1000))

    // Mock authentication - in real app, this would be an API call
    if (email && password) {
      const userData: User = {
        id: "user-" + Date.now(),
        email,
        name: email.split("@")[0],
        type: email.includes("premium") ? "premium" : "normal",
      }

      setUser(userData)
      setIsAuthenticated(true)
      localStorage.setItem("ocean-platform-user", JSON.stringify(userData))
      return true
    }

    return false
  }

  const register = async (email: string, password: string, name: string): Promise<boolean> => {
    // Simulate API call
    await new Promise((resolve) => setTimeout(resolve, 1000))

    if (email && password && name) {
      const userData: User = {
        id: "user-" + Date.now(),
        email,
        name,
        type: "normal",
      }

      setUser(userData)
      setIsAuthenticated(true)
      localStorage.setItem("ocean-platform-user", JSON.stringify(userData))
      return true
    }

    return false
  }

  const logout = () => {
    setUser(null)
    setIsAuthenticated(false)
    localStorage.removeItem("ocean-platform-user")
    localStorage.removeItem("ocean-platform-chat-history")
  }

  const continueAsGuest = () => {
    const guestUser: User = {
      id: "guest-" + Date.now(),
      email: "guest@ocean-platform.com",
      name: "Guest User",
      type: "guest",
    }

    setUser(guestUser)
    setIsAuthenticated(true)
    // Don't save guest session to localStorage
  }

  return (
    <AuthContext.Provider
      value={{
        user,
        isAuthenticated,
        login,
        register,
        logout,
        continueAsGuest,
      }}
    >
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const context = useContext(AuthContext)
  if (context === undefined) {
    throw new Error("useAuth must be used within an AuthProvider")
  }
  return context
}
