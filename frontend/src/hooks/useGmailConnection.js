import { useCallback, useEffect, useState } from "react";

export function useGmailConnection() {
  const [isConnected, setIsConnected] = useState(
    localStorage.getItem("gmailConnected") === "true"
  );
  const [email, setEmail] = useState(localStorage.getItem("gmailEmail") || "");

  useEffect(() => {
    const handler = () => {
      setIsConnected(localStorage.getItem("gmailConnected") === "true");
      setEmail(localStorage.getItem("gmailEmail") || "");
    };
    window.addEventListener("storage", handler);
    return () => window.removeEventListener("storage", handler);
  }, []);

  const connect = useCallback((email) => {
    setIsConnected(true);
    setEmail(email);
    localStorage.setItem("gmailConnected", "true");
    localStorage.setItem("gmailEmail", email);
  }, []);

  const disconnect = useCallback(() => {
    setIsConnected(false);
    setEmail("");
    localStorage.removeItem("gmailConnected");
    localStorage.removeItem("gmailEmail");
  }, []);

  return { isConnected, email, connect, disconnect };
}

export default useGmailConnection;
