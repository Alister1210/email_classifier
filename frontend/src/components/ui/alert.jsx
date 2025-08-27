import React from "react";
import { cn } from "../../lib/utils";

export function Alert({ variant = "default", className, children }) {
  const styles = {
    default: "border-[hsl(var(--border))] bg-[hsl(var(--accent))]",
    error:
      "border-red-400 bg-red-50 text-red-900 dark:bg-red-950/40 dark:text-red-100",
    success:
      "border-emerald-400 bg-emerald-50 text-emerald-900 dark:bg-emerald-950/40 dark:text-emerald-100",
    warning:
      "border-yellow-400 bg-yellow-50 text-yellow-900 dark:bg-yellow-950/40 dark:text-yellow-100",
    info: "border-sky-400 bg-sky-50 text-sky-900 dark:bg-sky-950/40 dark:text-sky-100",
  };
  return (
    <div
      className={cn(
        "rounded-md p-3 text-sm border",
        styles[variant],
        className
      )}
    >
      {children}
    </div>
  );
}
