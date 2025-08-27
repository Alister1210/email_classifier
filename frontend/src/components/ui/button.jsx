import React from "react";
import { Slot } from "@radix-ui/react-slot";
import { cn } from "../../lib/utils";

const buttonVariants = {
  default:
    "inline-flex items-center justify-center whitespace-nowrap rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:pointer-events-none disabled:opacity-50 bg-[hsl(var(--primary))] text-[hsl(var(--primary-foreground))] hover:opacity-90",
  secondary:
    "bg-[hsl(var(--secondary))] text-[hsl(var(--secondary-foreground))] hover:bg-[hsl(var(--accent))]",
  outline:
    "border border-[hsl(var(--border))] bg-transparent hover:bg-[hsl(var(--accent))]",
  ghost: "hover:bg-[hsl(var(--accent))]",
  link: "underline-offset-4 hover:underline",
};

export const Button = React.forwardRef(
  ({ className, variant = "default", asChild = false, ...props }, ref) => {
    const Comp = asChild ? Slot : "button";
    return (
      <Comp
        ref={ref}
        className={cn(
          "h-9 px-4 py-2",
          buttonVariants[variant] || buttonVariants.default,
          className
        )}
        {...props}
      />
    );
  }
);
Button.displayName = "Button";

export default Button;
