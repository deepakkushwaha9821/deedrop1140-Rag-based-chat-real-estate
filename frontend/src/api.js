import axios from "axios";

// Resolve API base: VITE_API_URL env var → local dev proxy fallback
const envUrl = import.meta.env.VITE_API_URL?.trim().replace(/\/+$/, "");
const apiBaseUrl = envUrl
  ? (envUrl.endsWith("/api") ? envUrl : `${envUrl}/api`)
  : "/api";

const api = axios.create({
  baseURL: apiBaseUrl,
});

api.interceptors.request.use((config) => {
  const token = localStorage.getItem("token");
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

export default api;
