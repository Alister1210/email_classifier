import React from "react";
import { cn } from "../../lib/utils";

export function Skeleton({ className }) {
  return (
    <div
      className={cn(
        "animate-pulse rounded-md bg-[hsl(var(--muted))] dark:bg-white/10",
        className
      )}
    />
  );
}
