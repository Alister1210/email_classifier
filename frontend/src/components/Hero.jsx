import React from "react";
import { motion } from "framer-motion";
import { Button } from "./ui/button";
import { useNavigate } from "react-router-dom";
import { useGmailConnection } from "../hooks/useGmailConnection";
import api from "../services/api";
import {
  ArrowRight,
  ShieldCheck,
  BarChart3,
  Mail,
  Zap,
  Eye,
} from "lucide-react";

export default function Hero() {
  const navigate = useNavigate();
  const { isConnected } = useGmailConnection();

  const handleConnect = async () => {
    if (isConnected) {
      navigate("/classifications");
      return;
    }

    try {
      const response = await api.get("/api/auth/login");
      window.location.href = response.data.authorization_url;
    } catch (err) {
      console.error("Login error:", err);
    }
  };

  return (
    <div className="relative overflow-hidden">
      {/* Background Effects */}
      <div className="absolute inset-0 -z-10">
        <div className="absolute top-1/4 left-1/4 h-96 w-96 rounded-full bg-gradient-to-r from-blue-500/20 to-purple-500/20 blur-3xl" />
        <div className="absolute bottom-1/4 right-1/4 h-96 w-96 rounded-full bg-gradient-to-r from-emerald-500/20 to-cyan-500/20 blur-3xl" />
      </div>

      <div className="container mx-auto px-4 py-16">
        <div className="text-center space-y-8">
          {/* Main Heading */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="space-y-4"
          >
            <h1 className="text-5xl md:text-6xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
              JASH
            </h1>
            <h2 className="text-2xl md:text-3xl font-semibold text-gray-800 dark:text-gray-200">
              Smart Email Classification
            </h2>
            <p className="text-lg text-gray-600 dark:text-gray-400 max-w-2xl mx-auto">
              Automatically classify your emails as spam or legitimate with
              AI-powered analysis. Get real-time insights and protect your inbox
              effortlessly.
            </p>
          </motion.div>

          {/* Feature Cards */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="grid md:grid-cols-3 gap-6 max-w-4xl mx-auto"
          >
            <FeatureCard
              icon={<Zap className="text-yellow-500" size={24} />}
              title="Real-time Analysis"
              description="Instant email classification as messages arrive"
            />
            <FeatureCard
              icon={<BarChart3 className="text-blue-500" size={24} />}
              title="Visual Analytics"
              description="Charts and statistics to track your email patterns"
            />
            <FeatureCard
              icon={<ShieldCheck className="text-emerald-500" size={24} />}
              title="Secure & Private"
              description="OAuth2 authentication with Google's secure protocols"
            />
          </motion.div>

          {/* Call to Action */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.4 }}
            className="space-y-6"
          >
            <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
              <Button
                onClick={handleConnect}
                size="lg"
                className="bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white px-8 py-3 text-lg font-semibold shadow-lg hover:shadow-xl transition-all duration-300"
              >
                <Mail size={20} className="mr-2" />
                {isConnected ? "View Classifications" : "Connect Gmail"}
                <ArrowRight size={20} className="ml-2" />
              </Button>

              {isConnected && (
                <Button
                  onClick={() => navigate("/classifications")}
                  variant="outline"
                  size="lg"
                  className="px-8 py-3 text-lg"
                >
                  <Eye size={20} className="mr-2" />
                  Dashboard
                </Button>
              )}
            </div>

            {isConnected && (
              <div className="inline-flex items-center gap-2 px-4 py-2 bg-emerald-50 dark:bg-emerald-900/20 rounded-full text-emerald-700 dark:text-emerald-300 text-sm">
                <ShieldCheck size={16} />
                Gmail Connected Successfully
              </div>
            )}
          </motion.div>

          {/* Status Section */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.6, delay: 0.6 }}
            className="pt-8"
          >
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 max-w-2xl mx-auto">
              <StatCard label="Secure" value="100%" />
              <StatCard label="Fast" value="< 1s" />
              <StatCard label="Accurate" value="95%+" />
              <StatCard label="Real-time" value="Live" />
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  );
}

function FeatureCard({ icon, title, description }) {
  return (
    <motion.div
      whileHover={{ scale: 1.05 }}
      className="p-6 rounded-xl bg-white/50 dark:bg-gray-800/50 backdrop-blur-sm border border-gray-200/50 dark:border-gray-700/50 shadow-lg"
    >
      <div className="flex flex-col items-center text-center space-y-3">
        <div className="p-3 rounded-full bg-gray-100/50 dark:bg-gray-700/50">
          {icon}
        </div>
        <h3 className="font-semibold text-gray-900 dark:text-gray-100">
          {title}
        </h3>
        <p className="text-sm text-gray-600 dark:text-gray-400">
          {description}
        </p>
      </div>
    </motion.div>
  );
}

function StatCard({ label, value }) {
  return (
    <div className="text-center space-y-1">
      <div className="text-2xl font-bold text-gray-900 dark:text-gray-100">
        {value}
      </div>
      <div className="text-xs text-gray-500 dark:text-gray-400">{label}</div>
    </div>
  );
}
