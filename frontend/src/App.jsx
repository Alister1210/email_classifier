import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Hero from "./components/Hero";
import Classifications from "./components/Classifications";
import Login from "./components/Login";
import AuthCallback from "./components/AuthCallback";
import AppLayout from "./components/layout/AppLayout";
import NotFound from "./components/NotFound";
import ErrorBoundary from "./components/ErrorBoundary";

function App() {
  return (
    <Router>
      <AppLayout>
        <ErrorBoundary>
          <Routes>
            <Route path="/" element={<Hero />} />
            <Route path="/login" element={<Login />} />
            <Route path="/classifications" element={<Classifications />} />
            <Route path="/auth/callback" element={<AuthCallback />} />
            <Route path="*" element={<NotFound />} />
          </Routes>
        </ErrorBoundary>
      </AppLayout>
    </Router>
  );
}

export default App;
