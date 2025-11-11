import express from "express";
import cors from "cors";
import mongoose from "mongoose";
import dotenv from "dotenv";
import morgan from "morgan";
import helmet from "helmet";
import rateLimit from "express-rate-limit";

import authRoutes from "./src/routes/auth.js";
import recommendRoutes from "./src/routes/recommend.js";
import seasonalRoutes from "./src/routes/seasonal.js";

dotenv.config();

const app = express();

/* ---------------------- Basic Security & Logging ---------------------- */
app.use(helmet()); // secure HTTP headers

// log requests in development
if (process.env.NODE_ENV !== "production") {
  app.use(morgan("dev"));
}

// trust reverse proxies (only if needed)
if (String(process.env.TRUST_PROXY).toLowerCase() === "true") {
  app.set("trust proxy", 1);
}

/* ---------------------- CORS ---------------------- */
/*
  FRONTEND_URL can contain one or more comma-separated URLs.
  Example:
    FRONTEND_URL=http://localhost:5173,http://127.0.0.1:5173
*/
const rawOrigins = process.env.FRONTEND_URL || "http://localhost:5173";
const allowedOrigins = rawOrigins
  .split(",")
  .map((s) => s.trim().replace(/\/$/, ""))
  .filter(Boolean);

function normalizeOrigin(origin) {
  if (!origin) return origin;
  try {
    const u = new URL(origin);
    return `${u.protocol}//${u.hostname}${u.port ? `:${u.port}` : ""}`;
  } catch {
    return origin.trim().replace(/\/$/, "");
  }
}

// Expand common local hostnames so `localhost` and `127.0.0.1` are both allowed
function expandLocalVariants(origins) {
  const extra = [];
  for (const o of origins) {
    try {
      const u = new URL(o);
      if (u.hostname === "localhost") {
        extra.push(`${u.protocol}//127.0.0.1${u.port ? `:${u.port}` : ""}`);
      }
      if (u.hostname === "127.0.0.1") {
        extra.push(`${u.protocol}//localhost${u.port ? `:${u.port}` : ""}`);
      }
    } catch {
      // ignore parse errors
    }
  }
  return [...new Set([...origins, ...extra])];
}

const expandedAllowed = expandLocalVariants(allowedOrigins);
const normalizedAllowed = expandedAllowed.map(normalizeOrigin);

app.use(
  cors({
    origin: (origin, callback) => {
      if (!origin) return callback(null, true); // allow curl, Postman, mobile apps
      const norm = normalizeOrigin(origin);
      if (normalizedAllowed.includes(norm)) return callback(null, true);
      console.warn(`🚫 CORS blocked for origin: ${origin}`);
      return callback(new Error("CORS blocked: origin not allowed"));
    },
    credentials: true,
    optionsSuccessStatus: 200,
  })
);

/* ---------------------- Rate Limiting & Body Parsing ---------------------- */
const limiter = rateLimit({
  windowMs: 60 * 1000, // 1 minute
  max: Number(process.env.RATE_LIMIT) || 100, // max requests per minute
});
app.use(limiter);

// parse JSON body (limit to 50kb)
app.use(express.json({ limit: "50kb" }));
app.use(express.urlencoded({ extended: true }));

/* ---------------------- Routes ---------------------- */
app.use("/api/auth", authRoutes);                // login/register
app.use("/api/advisory", recommendRoutes);      // ML crop recommendation
app.use("/api/advisory/seasonal", seasonalRoutes); // seasonal crops

// Health check route
app.get("/", (req, res) =>
  res.json({
    status: "ok",
    service: "AgroAware API",
    env: process.env.NODE_ENV || "development",
  })
);

/* ---------------------- Error Handling ---------------------- */
// 404
app.use((req, _res, next) => {
  const err = new Error("Not Found");
  err.status = 404;
  next(err);
});

// Central error handler
app.use((err, req, res, _next) => {
  console.error("⚠ Unhandled error:", err.message);
  const status = err.status || 500;
  res.status(status).json({
    error: err.message || "Internal Server Error",
    ...(process.env.NODE_ENV !== "production" && { stack: err.stack }),
  });
});

/* ---------------------- MongoDB Connection & Server Start ---------------------- */
const PORT = Number(process.env.PORT || 5000);
const MONGO_URI = process.env.MONGO_URI;

if (!MONGO_URI) {
  console.error("❌ Missing MONGO_URI in .env — cannot start server");
  process.exit(1);
}

async function start() {
  try {
    await mongoose.connect(MONGO_URI, {
      useNewUrlParser: true,
      useUnifiedTopology: true,
    });
    console.log("✅ MongoDB connected successfully");

    const server = app.listen(PORT, () => {
      console.log(`🚀 Server running at http://localhost:${PORT}`);
      console.log("🌍 Allowed CORS origins:", allowedOrigins);
    });

    // graceful shutdown
    const shutdown = async () => {
      console.log("🛑 Shutting down server...");
      await mongoose.disconnect();
      server.close(() => {
        console.log("✅ Server closed gracefully");
        process.exit(0);
      });
    };
    process.on("SIGINT", shutdown);
    process.on("SIGTERM", shutdown);
  } catch (err) {
    console.error("❌ Failed to start server:", err);
    process.exit(1);
  }
}

start();

export default app;