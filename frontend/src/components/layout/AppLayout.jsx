import React, { useState } from "react";
import { ThemeProvider } from "./ThemeProvider";
import { Button } from "../ui/button";
import { Moon, Sun, User } from "lucide-react";
import { useTheme } from "next-themes";
import { useNavigate } from "react-router-dom";
import { AuthService } from "../../services/api";
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
  const { isConnected, email, disconnect } = useGmailConnection();
  const [dropdownOpen, setDropdownOpen] = useState(false);

  return (
    <header className="sticky top-0 z-30 w-full border-b border-[hsl(var(--border))] bg-[hsl(var(--background))]">
      <div className="mx-auto flex w-full max-w-6xl items-center justify-between px-4 py-3">
        <div className="text-sm font-semibold">
          JASH – Email Classification Dashboard
        </div>
        <div className="flex items-center gap-2 relative">
          <Button variant="outline" onClick={() => navigate("/settings")}>
            Settings
          </Button>
          <Button
            variant="outline"
            onClick={async () => {
              if (isConnected) {
                navigate("/classifications");
              } else {
                try {
                  const data = await AuthService.login();
                  window.location.href = data.authorization_url;
                } catch (e) {
                  console.error(e);
                }
              }
            }}
          >
            {isConnected ? "View Classifications" : "Connect Gmail"}
          </Button>

          {/* Profile dropdown */}
          <div className="relative">
            <button
              aria-label="Profile"
              title="Profile"
              className="h-9 w-9 rounded-md border border-[hsl(var(--border))] flex items-center justify-center"
              onClick={() => setDropdownOpen((prev) => !prev)}
            >
              <User size={16} />
            </button>
            {dropdownOpen && (
              <div className="absolute right-0 mt-2 w-48 rounded-md border bg-white shadow-lg p-2 text-sm">
                {isConnected ? (
                  <>
                    <div className="px-2 py-1 text-gray-600">
                      {email || "Fetching email..."}
                    </div>
                    <Button
                      variant="ghost"
                      className="w-full justify-start"
                      onClick={() => {
                        disconnect();
                        navigate("/");
                        setDropdownOpen(false);
                      }}
                    >
                      Sign out
                    </Button>
                  </>
                ) : (
                  <div className="px-2 py-1 text-gray-500">Not connected</div>
                )}
              </div>
            )}
          </div>

          {/* Theme toggle */}
          <Button
            variant="ghost"
            aria-label="Toggle theme"
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
