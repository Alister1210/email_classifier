import axios from "axios";

// Create axios instance with base configuration
const api = axios.create({
  baseURL:
    import.meta.env.VITE_REACT_APP_API_BASE_URL || "https://localhost:8000",
  timeout: 15000,
  headers: { "Content-Type": "application/json" },
});

// Add token to requests if available
api.interceptors.request.use((config) => {
  const token = localStorage.getItem("access_token");
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Handle token refresh on 401 errors
api.interceptors.response.use(
  (response) => response,
  async (error) => {
    if (error.response?.status === 401) {
      // Clear tokens and redirect to login
      localStorage.removeItem("access_token");
      localStorage.removeItem("refresh_token");
      localStorage.removeItem("gmailConnected");
      localStorage.removeItem("gmailEmail");
      window.location.href = "/";
    }
    return Promise.reject(error);
  }
);

// Validate successful status codes
api.defaults.validateStatus = (status) => status >= 200 && status < 300;

// Service modules
export const AuthService = {
  async login() {
    const response = await api.get("/api/auth/login");
    return response.data;
  },

  async getProfile() {
    const response = await api.get("/api/auth/profile");
    return response.data;
  },

  async logout() {
    try {
      await api.post("/api/auth/logout");
    } catch (error) {
      console.warn("Logout API call failed:", error);
    } finally {
      // Clear local storage regardless
      localStorage.removeItem("access_token");
      localStorage.removeItem("refresh_token");
      localStorage.removeItem("gmailConnected");
      localStorage.removeItem("gmailEmail");
    }
  },
};

export const ClassificationsService = {
  async getResults() {
    const response = await api.get("/api/results");
    return response.data;
  },

  async processEmails(limit = 20) {
    const response = await api.post(
      "/api/emails/full",
      { limit },
      { timeout: 30000 }
    );
    return response.data;
  },
};

export const HealthService = {
  async check() {
    const response = await api.get("/api/health", { timeout: 5000 });
    return response.data;
  },
};

// Token management utilities
export const TokenManager = {
  setTokens(accessToken, refreshToken = null) {
    localStorage.setItem("access_token", accessToken);
    if (refreshToken) {
      localStorage.setItem("refresh_token", refreshToken);
    }
  },

  getTokens() {
    return {
      accessToken: localStorage.getItem("access_token"),
      refreshToken: localStorage.getItem("refresh_token"),
    };
  },

  clearTokens() {
    localStorage.removeItem("access_token");
    localStorage.removeItem("refresh_token");
  },

  hasValidToken() {
    return !!localStorage.getItem("access_token");
  },
};

export default api;
