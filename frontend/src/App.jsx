import React, { useEffect } from "react";
import {
  BrowserRouter as Router,
  Routes,
  Route,
  useLocation,
  useNavigate,
} from "react-router-dom";
import Login from "./components/Login";
import Classifications from "./components/Classifications";
import AppLayout from "./components/layout/AppLayout";
import Hero from "./components/Hero";
import Settings from "./components/Settings";
import NotFound from "./components/NotFound";
import ProtectedRoute from "./components/ProtectedRoute";
import ErrorBoundary from "./components/ErrorBoundary";
import AppToaster from "./components/ui/toaster";

function App() {
  return (
    <Router>
      <AppLayout>
        <AppToaster />
        <AuthWatcher />
        <ErrorBoundary>
          <Routes>
            <Route path="/" element={<Hero />} />
            <Route path="/login" element={<Login />} />
            <Route
              path="/classifications"
              element={
                <ProtectedRoute>
                  <Classifications />
                </ProtectedRoute>
              }
            />
            <Route path="/settings" element={<Settings />} />
            <Route path="*" element={<NotFound />} />
          </Routes>
        </ErrorBoundary>
      </AppLayout>
    </Router>
  );
}

function AuthWatcher() {
  const location = useLocation();
  const navigate = useNavigate();
  useEffect(() => {
    const params = new URLSearchParams(location.search);
    const authStatus = params.get("auth");
    const authMessage = params.get("message");
    if (authStatus === "success") {
      localStorage.setItem("gmailConnected", "true");
      navigate("/classifications", { replace: true });
    } else if (authStatus === "error") {
      localStorage.removeItem("gmailConnected");
      console.error("Authorization failed:", authMessage);
    }
  }, [location, navigate]);
  return null;
}

export default App;
