// server/src/routes/recommend.js
import express from "express";
import axios from "axios";

const router = express.Router();

// required numeric fields expected by ML service
const REQUIRED_FIELDS = ["N", "P", "K", "ph", "temperature", "rainfall"];

// ML service base URL (resolved at request time so .env can be loaded before use)
function getMLBase() {
  // support either ML_SERVICE_URL (preferred) or legacy ML_API_URL
  return process.env.ML_SERVICE_URL || process.env.ML_API_URL || "http://localhost:8000";
}

// helper to check numbers
function isNumberLike(v) {
  return v !== null && v !== undefined && !Number.isNaN(Number(v));
}

router.post("/crop", async (req, res) => {
  try {
    const body = req.body || {};

    // Build payload with exactly the keys ML expects
    // Accept lowercase keys too (robustness)
    const normalized = {};
    for (const key of REQUIRED_FIELDS) {
      // support both uppercase/lowercase incoming (just in case)
      let value = body[key];
      if (value === undefined) value = body[key.toLowerCase()];
      normalized[key] = value;
    }

    // Validate presence + numeric
    const missing = [];
    const notNumber = [];
    for (const k of REQUIRED_FIELDS) {
      if (normalized[k] === undefined || normalized[k] === null || normalized[k] === "") {
        missing.push(k);
      } else if (!isNumberLike(normalized[k])) {
        notNumber.push(k);
      }
    }

    if (missing.length > 0 || notNumber.length > 0) {
      return res.status(400).json({
        error: "Validation failed",
        details: {
          missing,
          notNumber
        },
        // echo what we received so you can debug quickly
        received: req.body
      });
    }

    // Convert to numbers (ML expects numeric types)
    const payload = {};
    for (const k of REQUIRED_FIELDS) payload[k] = Number(normalized[k]);

    // Debug log so you can inspect what Node will send to ML
    console.log("Calling ML service with payload:", payload);

    // call ML service predict endpoint
    const mlBase = getMLBase();
    const mlUrl = `${mlBase.replace(/\/$/, "")}/predict`; // ensures no double slashes
    const { data } = await axios.post(mlUrl, payload, {
      headers: { "Content-Type": "application/json" },
      timeout: 15000
    });

    // ✅ NEW: Fetch fertilizer recommendations for the predicted crop
    if (data.predicted_crop) {
      try {
        const fertilizerUrl = `${mlBase.replace(/\/$/, "")}/fertilizer`;
        const fertilizerPayload = {
          crop: data.predicted_crop,
          N: payload.N,
          P: payload.P,
          K: payload.K
        };
        console.log("🌾 Fetching fertilizer for:", fertilizerPayload);
        const { data: fertilizerData } = await axios.post(fertilizerUrl, fertilizerPayload, {
          headers: { "Content-Type": "application/json" },
          timeout: 10000
        });
        console.log("✅ Fertilizer data received:", fertilizerData);
        // Attach fertilizer data to response
        data.fertilizer = fertilizerData;
      } catch (fertErr) {
        console.warn("⚠️ Fertilizer fetch failed (non-blocking):", fertErr?.message);
        console.warn("Response:", fertErr?.response?.data);
        // Don't fail the whole request, just skip fertilizer
      }
    }

    // send result back to frontend
    console.log("📤 Sending response with fertilizer:", { predicted_crop: data.predicted_crop, has_fertilizer: !!data.fertilizer });
    return res.json(data);
  } catch (err) {
    console.error("Error in /api/advisory/crop:", err?.message || err);
    // if ML responded with details, forward useful bits
    if (err.response?.data) {
      return res.status(err.response.status || 500).json({
        error: "ML service error",
        mlResponse: err.response.data
      });
    }
    return res.status(500).json({ error: "Internal server error" });
  }
});

export default router;