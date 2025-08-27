import axios from "axios";

const api = axios.create({
  baseURL:
    import.meta.env.VITE_REACT_APP_API_BASE_URL || "https://localhost:8000",
  timeout: 10000,
  headers: { "Content-Type": "application/json" },
});

// Allow status codes 200-299
api.defaults.validateStatus = (status) => status >= 200 && status < 300;

export default api;
