import React from "react";
import { motion } from "framer-motion";
import { Button } from "./ui/button";
import { AuthService } from "../services/api";
import { toast } from "sonner";
import { track } from "../lib/analytics";
import { ArrowRight, ShieldCheck, Sparkles, MailSearch } from "lucide-react";

export default function Hero() {
  return (
    <section className="relative overflow-hidden rounded-2xl border border-[hsl(var(--border))] p-8">
      <BackgroundFX />
      <div className="relative z-10 grid gap-8 md:grid-cols-2">
        <div className="space-y-4">
          <motion.h1
            className="text-3xl font-bold md:text-4xl"
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
          >
            JASH – Smarter Email Classification
          </motion.h1>
          <motion.p
            className="text-[hsl(var(--muted-foreground))]"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.1 }}
          >
            Connect Gmail, visualize trends, and keep your inbox safe with a
            modern, animated dashboard.
          </motion.p>
          <div className="flex flex-wrap items-center gap-3">
            <Button
              aria-label="Connect Gmail"
              onClick={async () => {
                try {
                  const data = await AuthService.login();
                  window.location.href = data.authorization_url;
                } catch (e) {
                  toast.error("Failed to start Google OAuth");
                  track("auth_login_error", {
                    message: String(e?.message || e),
                  });
                }
              }}
              className="inline-flex items-center gap-2"
            >
              Connect Gmail <ArrowRight size={16} />
            </Button>
            <div className="flex items-center gap-2 text-sm text-[hsl(var(--muted-foreground))]">
              <ShieldCheck size={16} /> OAuth2 Secure
            </div>
          </div>
          <ul className="mt-4 grid grid-cols-1 gap-2 text-sm md:grid-cols-2">
            <li className="flex items-center gap-2">
              <Sparkles size={14} /> Live classifications
            </li>
            <li className="flex items-center gap-2">
              <MailSearch size={14} /> Search, sort, filter
            </li>
            <li className="flex items-center gap-2">
              <Sparkles size={14} /> Glass UI + dark mode
            </li>
            <li className="flex items-center gap-2">
              <Sparkles size={14} /> Charts & analytics
            </li>
          </ul>
        </div>
        <motion.div
          className="relative"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.15 }}
        >
          <div className="grid grid-cols-3 gap-2">
            <img
              src="/hero-1.jpg"
              alt="preview1"
              className="glass h-28 w-full rounded-md object-cover"
            />
            <img
              src="/hero-2.jpg"
              alt="preview2"
              className="glass h-28 w-full rounded-md object-cover"
            />
            <img
              src="/hero-3.jpg"
              alt="preview3"
              className="glass h-28 w-full rounded-md object-cover"
            />
            <img
              src="/hero-4.jpg"
              alt="preview4"
              className="glass h-28 w-full rounded-md object-cover"
            />
            <img
              src="/hero-5.jpg"
              alt="preview5"
              className="glass h-28 w-full rounded-md object-cover"
            />
            <img
              src="/hero-6.jpg"
              alt="preview6"
              className="glass h-28 w-full rounded-md object-cover"
            />
          </div>
        </motion.div>
      </div>
    </section>
  );
}

function BackgroundFX() {
  return (
    <div className="pointer-events-none absolute inset-0">
      <div className="absolute -top-24 left-20 h-56 w-56 rounded-full bg-purple-500/20 blur-3xl" />
      <div className="absolute bottom-0 right-20 h-56 w-56 rounded-full bg-cyan-500/20 blur-3xl" />
    </div>
  );
}
