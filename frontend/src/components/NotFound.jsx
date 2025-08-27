import React from "react";
import { Button } from "./ui/button";
import { useNavigate } from "react-router-dom";

export default function NotFound() {
  const navigate = useNavigate();
  return (
    <div className="flex min-h-[50vh] flex-col items-center justify-center gap-3 text-center">
      <div className="text-4xl font-bold">404</div>
      <div className="text-[hsl(var(--muted-foreground))]">Page not found</div>
      <Button onClick={() => navigate("/")}>Go Home</Button>
    </div>
  );
}
