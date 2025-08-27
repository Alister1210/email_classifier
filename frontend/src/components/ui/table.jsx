import React from "react";
import { cn } from "../../lib/utils";

export function Table({ className, ...props }) {
  return (
    <div className={cn("w-full overflow-auto", className)}>
      <table className="w-full caption-bottom text-sm" {...props} />
    </div>
  );
}

export function THead({ className, ...props }) {
  return <thead className={cn("[&_tr]:border-b", className)} {...props} />;
}

export function TBody({ className, ...props }) {
  return (
    <tbody className={cn("[&_tr:last-child]:border-0", className)} {...props} />
  );
}

export function TR({ className, ...props }) {
  return (
    <tr
      className={cn(
        "border-b transition-colors hover:bg-[hsl(var(--accent))]",
        className
      )}
      {...props}
    />
  );
}

export function TH({ className, ...props }) {
  return (
    <th
      className={cn(
        "h-10 px-2 text-left align-middle font-medium text-[hsl(var(--muted-foreground))]",
        className
      )}
      {...props}
    />
  );
}

export function TD({ className, ...props }) {
  return <td className={cn("p-2 align-middle", className)} {...props} />;
}
