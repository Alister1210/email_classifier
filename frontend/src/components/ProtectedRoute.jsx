import React from "react";
import { Navigate } from "react-router-dom";
import { useGmailConnection } from "../hooks/useGmailConnection";

export default function ProtectedRoute({ children }) {
  const { isConnected } = useGmailConnection();
  if (!isConnected) return <Navigate to="/" replace />;
  return children;
}
