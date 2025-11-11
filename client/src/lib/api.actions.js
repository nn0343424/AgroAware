// client/src/lib/api.actions.js
import API from "./api"; // your configured axios instance

/* ======================================================
   AUTH
====================================================== */
export const loginApi = (email, password) =>
  API.post("/api/auth/login", { email, password });

export const registerApi = (payload) =>
  API.post("/api/auth/register", payload);

/* ======================================================
   CROP ADVISORY (EXPERT MODE)
====================================================== */
export const getCropRecommendation = (payload) =>
  API.post("/api/advisory/crop", payload);

/* ======================================================
   SEASONAL ADVISORY (BEGINNER MODE)
====================================================== */
export const getSeasonalCrops = (payload) =>
  API.post("/api/advisory/seasonal", payload);

export const getSeasonalList = () =>
  API.get("/api/advisory/seasonal/list");

/* ======================================================
   FERTILIZER ADVICE (NEW - CALLS ML SERVICE DIRECT)
====================================================== */

// IMPORTANT: This endpoint goes **directly to ML service**, not Node backend
const ML_SERVICE_URL = "http://localhost:8800";

export const getFertilizerAdvice = async (payload) => {
  // payload = { crop, N, P, K }
  try {
    const resp = await fetch(`${ML_SERVICE_URL}/fertilizer`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    return await resp.json();
  } catch (err) {
    console.error("Fertilizer API error:", err);
    return { error: "Unable to fetch fertilizer advice" };
  }
};
