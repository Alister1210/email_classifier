import { useCallback, useEffect, useState } from "react";

export function useGmailConnection() {
  const [isConnected, setIsConnected] = useState(
    localStorage.getItem("gmailConnected") === "true"
  );
  const [email, setEmail] = useState(localStorage.getItem("gmailEmail") || "");

  // Listen for localStorage changes (useful for multiple tabs)
  useEffect(() => {
    const handleStorageChange = () => {
      setIsConnected(localStorage.getItem("gmailConnected") === "true");
      setEmail(localStorage.getItem("gmailEmail") || "");
    };

    window.addEventListener("storage", handleStorageChange);
    return () => window.removeEventListener("storage", handleStorageChange);
  }, []);

  const connect = useCallback((userEmail) => {
    console.log("Connecting Gmail for:", userEmail);
    setIsConnected(true);
    setEmail(userEmail);
    localStorage.setItem("gmailConnected", "true");
    localStorage.setItem("gmailEmail", userEmail);
  }, []);

  const disconnect = useCallback(() => {
    console.log("Disconnecting Gmail");
    setIsConnected(false);
    setEmail("");
    localStorage.removeItem("gmailConnected");
    localStorage.removeItem("gmailEmail");
    localStorage.removeItem("access_token");
    localStorage.removeItem("refresh_token");
  }, []);

  return {
    isConnected,
    email,
    connect,
    disconnect,
  };
}

export default useGmailConnection;
