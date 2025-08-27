import React, { useState, useEffect } from "react";
import { Button, Typography, Box, Alert } from "@mui/material";
import { useNavigate, useLocation } from "react-router-dom";
import api from "../services/api";

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

  return (
    <Box textAlign="center" mt={5}>
      <Typography variant="h4" gutterBottom>
        Email Classifier App
      </Typography>
      {error && <Alert severity="error">{error}</Alert>}
      {isConnected && (
        <Alert severity="success" sx={{ mb: 2 }}>
          Gmail Connected
        </Alert>
      )}
      {!isConnected ? (
        <Button
          variant="contained"
          color="primary"
          onClick={handleLogin}
          disabled={loading}
          sx={{ mr: 2 }}
        >
          {loading ? "Connecting..." : "Connect to Gmail"}
        </Button>
      ) : (
        <Button
          variant="contained"
          color="secondary"
          onClick={handleLogout}
          sx={{ mr: 2 }}
        >
          Disconnect Gmail
        </Button>
      )}
      <Button
        variant="outlined"
        color="secondary"
        onClick={handleViewClassifications}
      >
        View Classifications
      </Button>
    </Box>
  );
};

export default Login;
