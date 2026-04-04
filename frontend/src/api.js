import axios from "axios";

// Priority: VITE_API_URL env var > local dev proxy > fallback HF Spaces URL
const isLocalHost = ["localhost", "127.0.0.1"].includes(window.location.hostname);
const envApiUrl = import.meta.env.VITE_API_URL?.trim();
const normalizedEnvApiUrl = envApiUrl ? envApiUrl.replace(/\/+$/, "") : "";
const envBaseUrl = normalizedEnvApiUrl
  ? (normalizedEnvApiUrl.endsWith("/api") ? normalizedEnvApiUrl : `${normalizedEnvApiUrl}/api`)
  : "";

// Set VITE_API_URL in Vercel to your HF Spaces backend URL, e.g.:
// https://your-username-propai-backend.hf.space
const apiBaseUrl = envBaseUrl || (isLocalHost ? "/api" : "/api");

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
