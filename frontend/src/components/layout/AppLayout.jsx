import React from "react";
import { ThemeProvider } from "./ThemeProvider";
import { Button } from "../ui/button";
import { Moon, Sun, Home, BarChart3, LogOut } from "lucide-react";
import { useTheme } from "next-themes";
import { useNavigate, useLocation } from "react-router-dom";
import { useGmailConnection } from "../../hooks/useGmailConnection";

export function AppLayout({ children }) {
  return (
    <ThemeProvider>
      <Header />
      <main className="mx-auto w-full max-w-6xl px-4 py-6">{children}</main>
    </ThemeProvider>
  );
}

function Header() {
  const { theme, setTheme } = useTheme();
  const isDark = theme === "dark";
  const navigate = useNavigate();
  const location = useLocation();
  const { isConnected, email, disconnect } = useGmailConnection();

  const isActive = (path) => location.pathname === path;
  const handleLogout = async () => {
    try {
      await AuthService.logout();
    } catch (error) {
      console.warn("Logout API call failed:", error);
    }
    disconnect();
    navigate("/");
  };

  return (
    <header className="sticky top-0 z-30 w-full border-b border-gray-200 dark:border-gray-800 bg-white/80 dark:bg-gray-900/80 backdrop-blur-md">
      <div className="mx-auto flex w-full max-w-6xl items-center justify-between px-4 py-3">
        {/* Logo */}
        <button
          onClick={() => navigate("/")}
          className="text-xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent hover:from-blue-700 hover:to-purple-700 transition-all"
        >
          JASH
        </button>

        {/* Navigation */}
        <div className="flex items-center gap-2">
          {/* Home Button */}
          <Button
            variant={isActive("/") ? "default" : "ghost"}
            onClick={() => navigate("/")}
            className="flex items-center gap-2"
          >
            <Home size={16} />
            <span className="hidden sm:inline">Home</span>
          </Button>

          {/* Classifications Button */}
          <Button
            variant={isActive("/classifications") ? "default" : "ghost"}
            onClick={() => navigate("/classifications")}
            disabled={!isConnected}
            className="flex items-center gap-2"
          >
            <BarChart3 size={16} />
            <span className="hidden sm:inline">Classifications</span>
          </Button>

          {/* Connection Status & Logout */}
          {isConnected && (
            <div className="flex items-center gap-2">
              {email && (
                <div className="hidden md:flex items-center gap-2 px-3 py-1 bg-emerald-50 dark:bg-emerald-900/20 rounded-full text-emerald-700 dark:text-emerald-300 text-sm">
                  <div className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse"></div>
                  Connected: {email.split("@")[0]}
                </div>
              )}
              <Button
                variant="outline"
                onClick={handleLogout}
                className="flex items-center gap-2 text-sm"
              >
                <LogOut size={14} />
                <span className="hidden sm:inline">Logout</span>
              </Button>
            </div>
          )}

          {/* Theme Toggle */}
          <Button
            variant="ghost"
            onClick={() => setTheme(isDark ? "light" : "dark")}
            className="h-9 w-9 p-0"
            title="Toggle theme"
          >
            {isDark ? <Sun size={18} /> : <Moon size={18} />}
          </Button>
        </div>
      </div>
    </header>
  );
}

export default AppLayout;
