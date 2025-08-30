import React, { useState, useEffect } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { motion } from "framer-motion";
import { Mail, LogIn, ArrowRight, AlertCircle } from "lucide-react";
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
import { useGmailConnection } from "../hooks/useGmailConnection";

const Login = () => {
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();
  const location = useLocation();
  const { isConnected, connect, disconnect } = useGmailConnection();

  useEffect(() => {
    // Check for auth status in URL
    const params = new URLSearchParams(location.search);
    const authStatus = params.get("auth");
    const authMessage = params.get("message");

    if (authStatus === "success") {
      const email = params.get("email");
      if (email) {
        connect(email);
      }
      navigate("/classifications", { replace: true });
    } else if (authStatus === "error") {
      setError(authMessage || "Authentication failed");
      disconnect();
    }
  }, [location, navigate, connect, disconnect]);

  // Redirect if already connected
  useEffect(() => {
    if (isConnected) {
      navigate("/classifications");
    }
  }, [isConnected, navigate]);

  const handleLogin = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await api.get("/api/auth/login", { timeout: 8000 });
      const authUrl = response.data.authorization_url;

      if (!authUrl) {
        throw new Error("No authorization URL received");
      }

      // Redirect to Google OAuth
      window.location.href = authUrl;
    } catch (err) {
      console.error("Login error:", err);

      if (err.code === "ERR_NETWORK") {
        setError(
          "Cannot connect to backend server. Please ensure it's running on localhost:8000."
        );
      } else if (err.code === "ECONNREFUSED") {
        setError(
          "Connection refused. Make sure the backend server is running."
        );
      } else {
        setError(err.message || "Failed to initiate login. Please try again.");
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex items-center justify-center py-16">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="w-full max-w-md"
      >
        <Card className="shadow-xl">
          <CardHeader className="text-center pb-4">
            <div className="mx-auto w-16 h-16 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full flex items-center justify-center mb-4">
              <Mail className="text-white" size={32} />
            </div>
            <CardTitle className="text-2xl font-bold">
              Connect Your Gmail
            </CardTitle>
            <CardDescription className="text-base">
              Securely connect your Gmail account to start classifying emails
              with AI
            </CardDescription>
          </CardHeader>

          <CardContent className="space-y-6">
            {error && (
              <Alert variant="destructive" className="flex items-start gap-2">
                <AlertCircle size={16} className="mt-0.5 flex-shrink-0" />
                <span>{error}</span>
              </Alert>
            )}

            <div className="space-y-4">
              <Button
                onClick={handleLogin}
                disabled={loading}
                className="w-full bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white py-3 text-lg font-semibold shadow-lg hover:shadow-xl transition-all duration-300"
                size="lg"
              >
                {loading ? (
                  <>
                    <Spinner className="w-5 h-5 mr-2" />
                    Connecting...
                  </>
                ) : (
                  <>
                    <LogIn size={20} className="mr-2" />
                    Connect with Google
                    <ArrowRight size={20} className="ml-2" />
                  </>
                )}
              </Button>

              <div className="text-center">
                <Button
                  variant="ghost"
                  onClick={() => navigate("/")}
                  className="text-sm"
                >
                  ← Back to Homepage
                </Button>
              </div>
            </div>

            <div className="border-t pt-4">
              <div className="text-xs text-gray-500 dark:text-gray-400 space-y-2">
                <div className="flex items-center justify-center gap-1">
                  <span>🔒</span>
                  <span>Secured with OAuth2</span>
                </div>
                <p className="text-center">
                  We never store your Gmail password. Your data is processed
                  securely and privately.
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </motion.div>
    </div>
  );
};

export default Login;
