import axios from "axios";

const api = axios.create({
  baseURL:
    import.meta.env.VITE_REACT_APP_API_BASE_URL || "https://localhost:8000",
  timeout: 15000,
  headers: { "Content-Type": "application/json" },
});

// Allow status codes 200-299
api.defaults.validateStatus = (status) => status >= 200 && status < 300;

export const ResultsService = {
  async list() {
    const res = await api.get("/api/results", { timeout: 5000 });
    return res.data;
  },
  async cleanupDuplicates() {
    const res = await api.post("/api/cleanup/duplicates");
    return res.data;
  },
};

export const HealthService = {
  async health() {
    const res = await api.get("/api/health", { timeout: 4000 });
    return res.data;
  },
};

export const EmailsService = {
  async full(limit = 20) {
    const res = await api.post(
      "/api/emails/full",
      { limit },
      { timeout: 30000 }
    );
    return res.data;
  },
};
// services/api.js

export const AuthService = {
  login: async () => {
    const res = await fetch("/api/auth/login");
    return res.json();
  },
  getProfile: async () => {
    const res = await fetch("/api/auth/profile");
    if (!res.ok) throw new Error("Failed to fetch profile");
    return res.json();
  },
};

export default api;
