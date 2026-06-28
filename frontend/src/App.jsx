import { Link, Navigate, Route, Routes, useNavigate } from "react-router-dom";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import api from "./api";


function TypewriterText({ text, animate, onDone }) {
  const [visible, setVisible] = useState(animate ? "" : text);

  useEffect(() => {
    if (!animate) { setVisible(text); return; }
    setVisible("");
    let i = 0;
    const id = window.setInterval(() => {
      i += 1;
      setVisible(text.slice(0, i));
      if (i >= text.length) { window.clearInterval(id); onDone?.(); }
    }, 14);
    return () => window.clearInterval(id);
  }, [text, animate, onDone]);

  return <>{visible}</>;
}

const METRIC_LABELS = [
  ["context_precision", "Context Precision"],
  ["context_recall", "Context Recall"],
  ["precision_at_k", "Precision@K"],
  ["recall_at_k", "Recall@K"],
  ["mrr", "MRR"],
  ["map", "MAP"],
  ["ndcg", "NDCG"],
  ["faithfulness", "Faithfulness"],
  ["answer_relevancy", "Answer Relevancy"],
  ["correctness", "Correctness"],
  ["hallucination_rate", "Hallucination Rate"],
  ["semantic_similarity", "Semantic Similarity"],
];

function MetricChips({ metrics }) {
  if (!metrics) return null;

  return (
    <div className="message-metrics">
      {METRIC_LABELS.map(([key, label]) => {
        const value = metrics[key];
        if (value === undefined || value === null) return null;
        return (
          <span key={key} className="metric-chip" title={`${label}: ${value.toFixed(3)}`}>
            <strong>{label}</strong>
            <span>{`${(value * 100).toFixed(1)}%`}</span>
          </span>
        );
      })}
    </div>
  );
}


function AuthPage({ mode }) {
  const navigate = useNavigate();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const submit = async (e) => {
    e.preventDefault();
    setError(""); setLoading(true);
    try {
      if (mode === "register") await api.post("/auth/register", { username, password });
      const res = await api.post("/auth/login", { username, password });
      localStorage.setItem("token", res.data.access_token);
      navigate("/");
    } catch (err) {
      setError(err.response?.data?.detail || "Request failed");
    } finally { setLoading(false); }
  };

  return (
    <div className="auth-shell">
      <div>
        <div className="auth-brand">
          <span className="auth-brand-icon">🏠</span>
          <h1>PropAI</h1>
          <p>Real Estate Intelligence Assistant</p>
        </div>
        <form className="auth-card" onSubmit={submit}>
          <h2>{mode === "login" ? "Welcome Back" : "Create Account"}</h2>
          <input id="username" value={username} onChange={e => setUsername(e.target.value)}
            placeholder="Username" required autoComplete="username" />
          <input id="password" type="password" value={password} onChange={e => setPassword(e.target.value)}
            placeholder="Password" required autoComplete={mode === "login" ? "current-password" : "new-password"} />
          <button type="submit" className="btn-primary" disabled={loading}>
            {loading ? "Please wait…" : mode === "login" ? "Sign In" : "Create Account"}
          </button>
          {error && <p className="error-banner" style={{ margin: "12px 0 0" }}>{error}</p>}
          <p className="auth-footer">
            {mode === "login" ? "No account?" : "Already registered?"}{" "}
            <Link to={mode === "login" ? "/register" : "/login"}>
              {mode === "login" ? "Register" : "Sign In"}
            </Link>
          </p>
        </form>
      </div>
    </div>
  );
}


function AboutPage() {
  const [info, setInfo] = useState(null);
  useEffect(() => { api.get("/about").then(r => setInfo(r.data)); }, []);

  return (
    <div className="page-card">
      <h1>🏠 About PropAI</h1>
      <p style={{ color: "var(--text-secondary)", marginBottom: "8px" }}>
        {info?.app_name || "PropAI — Real Estate Intelligence Assistant"}
      </p>
      <p style={{ color: "var(--text-muted)", fontSize: "0.85rem", lineHeight: 1.6 }}>
        Advanced RAG system with hybrid retrieval (BM25 + ChromaDB), cross-encoder re-ranking,
        and multi-turn conversational memory. Improved retrieval accuracy ~35% over naive vector search.
      </p>
      <div className="stack-pills">
        {(info?.stack || []).map(item => (
          <span key={item} className="stack-pill">{item}</span>
        ))}
      </div>
      <div style={{ marginTop: "24px" }}>
        <Link to="/">← Back to Chat</Link>
      </div>
    </div>
  );
}


function WelcomeScreen({ onNew }) {
  const features = [
    { icon: "🔍", title: "Property Search", desc: "Find properties by intent, budget & location" },
    { icon: "📄", title: "Document Q&A", desc: "Upload legal docs and get instant answers" },
    { icon: "📊", title: "Market Insights", desc: "Investment analysis and trend queries" },
  ];
  return (
    <div className="welcome-screen">
      <div style={{ textAlign: "center" }}>
        <span className="welcome-icon">🏠</span>
        <h2 className="welcome-title">PropAI Assistant</h2>
        <p className="welcome-sub">Your AI-powered real estate intelligence — search, analyze, decide.</p>
      </div>
      <div className="welcome-cards">
        {features.map(f => (
          <div key={f.title} className="welcome-card" onClick={onNew}>
            <span className="wc-icon">{f.icon}</span>
            <h3>{f.title}</h3>
            <p>{f.desc}</p>
          </div>
        ))}
      </div>
    </div>
  );
}


function ChatPage() {
  const navigate = useNavigate();
  const messagesEndRef = useRef(null);

  const [me, setMe] = useState(null);
  const [chats, setChats] = useState([]);
  const [activeChatId, setActiveChatId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [typingMessageId, setTypingMessageId] = useState(null);
  const [files, setFiles] = useState([]);
  const [text, setText] = useState("");
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [uploadLoading, setUploadLoading] = useState(false);

  const activeChat = useMemo(() => chats.find(c => c.id === activeChatId) || null, [chats, activeChatId]);

  const refreshChats = useCallback(async () => {
    const r = await api.get("/chats");
    setChats(r.data);
    return r.data;
  }, []);

  const loadDetail = useCallback(async (chatId) => {
    const r = await api.get(`/chats/${chatId}`);
    setMessages(r.data.messages);
    setFiles(r.data.files);
  }, []);

  // Bootstrap
  useEffect(() => {
    (async () => {
      try {
        const u = await api.get("/auth/me");
        setMe(u.data);
        const data = await refreshChats();
        if (data.length > 0) setActiveChatId(data[0].id);
      } catch { localStorage.removeItem("token"); navigate("/login"); }
    })();
  }, [refreshChats, navigate]);

  useEffect(() => {
    if (activeChatId) loadDetail(activeChatId);
    else { setMessages([]); setFiles([]); setTypingMessageId(null); }
  }, [activeChatId, loadDetail]);

  // Scroll on new messages / loading
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, isLoading]);

  useEffect(() => {
    if (!typingMessageId) return;
    const id = window.setInterval(() => {
      messagesEndRef.current?.scrollIntoView({ behavior: "auto", block: "end" });
    }, 50);
    return () => window.clearInterval(id);
  }, [typingMessageId]);

  const createChat = async () => {
    const r = await api.post("/chats");
    await refreshChats();
    setActiveChatId(r.data.id);
  };

  const sendMessage = async (e) => {
    e.preventDefault();
    if (!activeChatId || !text.trim() || isLoading) return;
    setError("");
    const msg = text;
    setText("");
    setIsLoading(true);

    // Optimistic user bubble
    const tempId = `temp-${Date.now()}`;
    setMessages(prev => [...prev, { id: tempId, role: "user", content: msg, timestamp: new Date().toISOString() }]);

    try {
      const r = await api.post(`/chats/${activeChatId}/messages`, { message: msg });
      setTypingMessageId(r.data.id);
      await loadDetail(activeChatId);
      await refreshChats();
    } catch (err) {
      setError(err.response?.data?.detail || "Failed to send message");
      setMessages(prev => prev.filter(m => m.id !== tempId));
    } finally { setIsLoading(false); }
  };

  const upload = async (e) => {
    if (!activeChatId) return;
    const file = e.target.files?.[0];
    if (!file) return;
    const form = new FormData();
    form.append("file", file);
    setError(""); setUploadLoading(true);
    try {
      await api.post(`/chats/${activeChatId}/upload`, form, { headers: { "Content-Type": "multipart/form-data" } });
      await loadDetail(activeChatId);
      await refreshChats();
    } catch (err) {
      setError(err.response?.data?.detail || "Failed to upload file");
    } finally { e.target.value = ""; setUploadLoading(false); }
  };

  const togglePin = async (id, e) => {
    e.stopPropagation();
    await api.post(`/chats/${id}/pin`);
    await refreshChats();
  };

  const archiveChat = async (id, e) => {
    e.stopPropagation();
    await api.post(`/chats/${id}/archive`);
    if (activeChatId === id) setActiveChatId(null);
    await refreshChats();
  };

  const deleteChat = async (id, e) => {
    e.stopPropagation();
    await api.delete(`/chats/${id}`);
    if (activeChatId === id) setActiveChatId(null);
    await refreshChats();
  };
  const logout = () => { localStorage.removeItem("token"); navigate("/login"); };

  const onKeyDown = (e) => { if (e.key === "Enter" && !e.shiftKey) sendMessage(e); };

  return (
    <div className="layout">
      {/* SIDEBAR */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <div className="sidebar-brand">
            <span className="brand-icon">🏠</span>
            <span className="brand-name">PropAI</span>
          </div>
          <div className="sidebar-user">Signed in as {me?.username}</div>
        </div>

        <div className="sidebar-actions">
          <button id="btn-new-chat" className="btn-new-chat" onClick={createChat}>
            ＋ New Chat
          </button>
          <div>
            <label className={`upload-label${!activeChat || uploadLoading ? " disabled" : ""}`} htmlFor="file-upload">
              {uploadLoading ? "⏳ Uploading…" : "📎 Upload Document"}
              <input id="file-upload" type="file" onChange={upload}
                disabled={!activeChat || uploadLoading}
                accept=".pdf,.txt,.md,.csv,.json,.png,.jpg,.jpeg,.bmp,.tiff,.tif,.webp" />
            </label>
            <p className="upload-hint">PDF · TXT · MD · CSV · JSON</p>
          </div>
        </div>

        <div className="chat-list">
          {chats.length > 0 && <p className="chat-list-label">Conversations</p>}
          {chats.map(chat => (
            <div key={chat.id} className={`chat-row${chat.id === activeChatId ? " active" : ""}`}>
              <div className="chat-row-top">
                <button className="chat-title" onClick={() => setActiveChatId(chat.id)} title={chat.title}>
                  {chat.title}
                </button>
                <span className={`mode-badge ${chat.mode}`}>
                  {chat.mode === "rag" ? "RAG" : "Chat"}
                </span>
              </div>
              <div className="row-actions">
                <button onClick={e => togglePin(chat.id, e)}>{chat.is_pinned ? "📌" : "Pin"}</button>
                <button onClick={e => archiveChat(chat.id, e)}>Archive</button>
                <button className="btn-danger" onClick={e => deleteChat(chat.id, e)}>Delete</button>
              </div>
            </div>
          ))}
        </div>

        <div className="sidebar-footer">
          <Link to="/about">About</Link>
          <button className="btn-logout" onClick={logout}>Logout</button>
        </div>
      </aside>

      {/* MAIN */}
      <main className="chat-main">
        {activeChat ? (
          <>
            <div className="chat-header">
              <h1>{activeChat.title}</h1>
              <span className={`header-badge ${activeChat.mode}`}>
                {activeChat.mode === "rag" ? "📄 Document Analysis" : "💬 Chat"}
              </span>
            </div>

            {files.length > 0 && (
              <div className="files-panel">
                {files.map(f => (
                  <span key={f.id} className="file-chip">📄 {f.filename}</span>
                ))}
              </div>
            )}

            <section className="messages">
              {messages.map(msg => (
                <div
                  key={msg.id}
                  className={`bubble ${msg.role}${msg.role === "ai" && msg.content.split("\n").length > 10 ? " long" : ""}`}
                >
                  <TypewriterText
                    text={msg.content}
                    animate={msg.role === "ai" && msg.id === typingMessageId}
                    onDone={() => setTypingMessageId(cur => cur === msg.id ? null : cur)}
                  />
                  {msg.role === "ai" && <MetricChips metrics={msg.metrics} />}
                </div>
              ))}

              {isLoading && (
                <div className="thinking-bubble">
                  <span className="thinking-dot" />
                  <span className="thinking-dot" />
                  <span className="thinking-dot" />
                </div>
              )}
              <div ref={messagesEndRef} />
            </section>

            {error && <p className="error-banner">{error}</p>}

            <form className="message-form" onSubmit={sendMessage}>
              <input
                id="message-input"
                className="message-input"
                value={text}
                onChange={e => setText(e.target.value)}
                onKeyDown={onKeyDown}
                placeholder={activeChat.mode === "rag"
                  ? "Ask about your document…"
                  : "Ask about properties, market trends…"}
                autoComplete="off"
              />
              <button id="btn-send" type="submit" className="btn-send" disabled={isLoading || !text.trim()}>
                {isLoading ? "…" : "Send ➤"}
              </button>
            </form>
          </>
        ) : (
          <WelcomeScreen onNew={createChat} />
        )}
      </main>
    </div>
  );
}


function ProtectedRoute({ children }) {
  return localStorage.getItem("token") ? children : <Navigate to="/login" replace />;
}


export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<AuthPage mode="login" />} />
      <Route path="/register" element={<AuthPage mode="register" />} />
      <Route path="/" element={<ProtectedRoute><ChatPage /></ProtectedRoute>} />
      <Route path="/about" element={<AboutPage />} />
    </Routes>
  );
}
