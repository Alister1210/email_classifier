import React, { useEffect, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { useGmailConnection } from "../hooks/useGmailConnection";
import { motion } from "framer-motion";
import { CheckCircle, XCircle, Loader2 } from "lucide-react";

export default function AuthCallback() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const { connect } = useGmailConnection();
  const [status, setStatus] = useState("processing");
  const [error, setError] = useState(null);

  useEffect(() => {
    const processAuth = async () => {
      try {
        // Get parameters from URL
        const accessToken = searchParams.get("access_token");
        const refreshToken = searchParams.get("refresh_token");
        const email = searchParams.get("email");
        const authError = searchParams.get("auth");

        console.log("Processing auth callback:", {
          accessToken: !!accessToken,
          email,
          authError,
        });

        // Check for auth error
        if (authError === "error") {
          const message =
            searchParams.get("message") || "Authentication failed";
          console.error("OAuth error:", message);
          setStatus("error");
          setError(message);
          return;
        }

        // Validate required data
        if (!accessToken || !email) {
          console.error("Missing authentication data");
          setStatus("error");
          setError("Missing authentication information");
          return;
        }

        // Store tokens in localStorage for API requests
        localStorage.setItem("access_token", accessToken);
        if (refreshToken) {
          localStorage.setItem("refresh_token", refreshToken);
        }

        // Update connection state
        await connect(email);

        console.log("Authentication successful for:", email);
        setStatus("success");

        // Clear URL parameters for security
        window.history.replaceState(
          {},
          document.title,
          window.location.pathname
        );

        // Redirect after success animation
        setTimeout(() => {
          navigate("/classifications", { replace: true });
        }, 2000);
      } catch (err) {
        console.error("Auth processing failed:", err);
        setStatus("error");
        setError(err.message || "Failed to complete authentication");

        // Redirect to home after error display
        setTimeout(() => {
          navigate("/", { replace: true });
        }, 3000);
      }
    };

    processAuth();
  }, [searchParams, connect, navigate]);

  const renderContent = () => {
    switch (status) {
      case "processing":
        return (
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="text-center space-y-6"
          >
            <div className="w-16 h-16 mx-auto bg-blue-100 dark:bg-blue-900/30 rounded-full flex items-center justify-center">
              <Loader2 className="w-8 h-8 text-blue-600 animate-spin" />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-gray-900 dark:text-gray-100 mb-2">
                Completing Authentication
              </h2>
              <p className="text-gray-600 dark:text-gray-400">
                Finalizing your Gmail connection...
              </p>
            </div>
          </motion.div>
        );

      case "success":
        return (
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="text-center space-y-6"
          >
            <motion.div
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ delay: 0.2, type: "spring", stiffness: 200 }}
              className="w-16 h-16 mx-auto bg-emerald-100 dark:bg-emerald-900/30 rounded-full flex items-center justify-center"
            >
              <CheckCircle className="w-8 h-8 text-emerald-600" />
            </motion.div>
            <div>
              <h2 className="text-2xl font-bold text-gray-900 dark:text-gray-100 mb-2">
                Successfully Connected!
              </h2>
              <p className="text-gray-600 dark:text-gray-400 mb-4">
                Your Gmail account is now connected and ready for
                classification.
              </p>
              <div className="flex items-center justify-center gap-2 text-sm text-emerald-600 dark:text-emerald-400">
                <div className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse"></div>
                Redirecting to dashboard...
              </div>
            </div>
          </motion.div>
        );

      case "error":
        return (
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="text-center space-y-6"
          >
            <div className="w-16 h-16 mx-auto bg-red-100 dark:bg-red-900/30 rounded-full flex items-center justify-center">
              <XCircle className="w-8 h-8 text-red-600" />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-gray-900 dark:text-gray-100 mb-2">
                Connection Failed
              </h2>
              <p className="text-gray-600 dark:text-gray-400 mb-6">{error}</p>
              <Button
                onClick={() => navigate("/", { replace: true })}
                className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-2"
              >
                Try Again
              </Button>
            </div>
          </motion.div>
        );

      default:
        return null;
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50 dark:bg-gray-900">
      <div className="max-w-md w-full mx-4">
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-2xl p-8 border border-gray-200 dark:border-gray-700">
          {renderContent()}
        </div>
      </div>
    </div>
  );
}
