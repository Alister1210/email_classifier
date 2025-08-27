import React, { useState, useEffect } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { motion } from "framer-motion";
import { Mail, LogIn, LogOut, ArrowRight } from "lucide-react";
import api from "../services/api";
import { Button } from "./ui/button";
import { Alert } from "./ui/alert";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "./ui/card";
import { Spinner } from "./ui/spinner";

const Login = () => {
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [isConnected, setIsConnected] = useState(
    localStorage.getItem("gmailConnected") === "true"
  );
  const navigate = useNavigate();
  const location = useLocation();

  useEffect(() => {
    // Check URL query parameters for auth status
    const params = new URLSearchParams(location.search);
    const authStatus = params.get("auth");
    const authMessage = params.get("message");
    if (authStatus === "success") {
      setIsConnected(true);
      localStorage.setItem("gmailConnected", "true");
      navigate("/classifications", { replace: true });
    } else if (authStatus === "error") {
      setError(`Authorization failed: ${authMessage || "Unknown error"}`);
      setIsConnected(false);
      localStorage.removeItem("gmailConnected");
    }
  }, [location, navigate]);

  const handleLogin = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await api.get("/api/auth/login", { timeout: 5000 });
      const authUrl = response.data.authorization_url;
      window.location.href = authUrl; // Redirect to Google OAuth
    } catch (err) {
      console.error("Login error:", err);
      if (err.code === "ERR_NETWORK") {
        setError(
          "Cannot connect to backend. Ensure the server is running at https://localhost:8000."
        );
      } else {
        setError(`Error initiating login: ${err.message}. Please try again.`);
      }
    } finally {
      setLoading(false);
    }
  };

  const handleLogout = () => {
    setIsConnected(false);
    localStorage.removeItem("gmailConnected");
    setError(null);
    // Optionally, call a backend endpoint to clear credentials
    navigate("/");
  };

  const handleViewClassifications = () => {
    navigate("/classifications");
  };

  const MotionDiv = motion.div;

  return (
    <div className="flex items-center justify-center py-16">
      <MotionDiv
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
        className="w-full max-w-xl"
      >
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-2xl">
              <Mail className="text-[hsl(var(--primary))]" size={22} />
              JASH – Email Classification
            </CardTitle>
            <CardDescription>
              Connect your Gmail and view live spam/ham classifications.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {error && <Alert variant="error">{error}</Alert>}
            {isConnected && <Alert variant="success">Gmail Connected</Alert>}
            <div className="flex flex-wrap items-center gap-3">
              {!isConnected ? (
                <Button
                  onClick={handleLogin}
                  disabled={loading}
                  className="inline-flex items-center gap-2"
                >
                  {loading ? (
                    <>
                      <Spinner size={16} /> Connecting...
                    </>
                  ) : (
                    <>
                      <LogIn size={16} /> Connect to Gmail
                    </>
                  )}
                </Button>
              ) : (
                <Button
                  variant="secondary"
                  onClick={handleLogout}
                  className="inline-flex items-center gap-2"
                >
                  <LogOut size={16} /> Disconnect Gmail
                </Button>
              )}
              <Button
                variant="outline"
                onClick={handleViewClassifications}
                className="inline-flex items-center gap-2"
              >
                View Classifications <ArrowRight size={16} />
              </Button>
            </div>
          </CardContent>
        </Card>
      </MotionDiv>
    </div>
  );
};

export default Login;
