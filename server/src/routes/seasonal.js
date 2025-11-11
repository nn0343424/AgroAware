// server/src/routes/seasonal.js
import express from "express";
import fs from "fs";
import path from "path";
import csv from "csv-parser";
import { fileURLToPath } from "url";

const router = express.Router();

// Resolve path with ESM
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Adjust this relative path if your CSV is somewhere else
const CSV_PATH = path.resolve(__dirname, "../../ml_service/data/district_season_map.csv");

// in-memory cache
let cachedRows = null;
let lastLoaded = 0;

function loadCsvIntoMemory() {
  return new Promise((resolve, reject) => {
    if (cachedRows && cachedRows.length > 0) return resolve(cachedRows);

    if (!fs.existsSync(CSV_PATH)) {
      return resolve([]); // treat missing as empty dataset
    }

    const rows = [];
    fs.createReadStream(CSV_PATH)
      .pipe(csv())
      .on("data", (r) => {
        // normalize keys to lower-case to avoid header case issues
        const normalized = {};
        for (const k of Object.keys(r)) {
          normalized[k.trim().toLowerCase()] = (r[k] || "").trim();
        }
        rows.push(normalized);
      })
      .on("end", () => {
        cachedRows = rows;
        lastLoaded = Date.now();
        resolve(rows);
      })
      .on("error", (err) => reject(err));
  });
}

// Helper to build metadata
async function buildMeta() {
  const rows = await loadCsvIntoMemory();
  const statesSet = new Set();
  const seasonsSet = new Set();
  const districtsByState = {};

  for (const r of rows) {
    const state = (r.state || "").trim();
    const district = (r.district || "").trim();
    const season = (r.season || "").trim();

    if (!state) continue;
    statesSet.add(state);

    if (district) {
      districtsByState[state] = districtsByState[state] || new Set();
      districtsByState[state].add(district);
    }
    if (season) seasonsSet.add(season);
  }

  // convert sets to arrays (sorted)
  const states = Array.from(statesSet).sort();
  const seasons = Array.from(seasonsSet).sort();
  const districtsObj = {};
  for (const s of Object.keys(districtsByState)) {
    districtsObj[s] = Array.from(districtsByState[s]).sort();
  }

  return { states, districtsByState: districtsObj, seasons };
}

/* ---------- Metadata endpoint: GET /list ---------- */
router.get("/list", async (req, res) => {
  try {
    const meta = await buildMeta();
    return res.json(meta);
  } catch (err) {
    console.error("seasonal /list error:", err);
    return res.status(500).json({ error: "Failed to load seasonal metadata" });
  }
});

/* ---------- Lookup endpoint: POST /  ---------- 
   body: { state, district, season }
   returns: matched row (recommended_crops split into array)
*/
router.post("/", async (req, res) => {
  const { state, district, season } = req.body || {};
  if (!state || !district || !season) {
    return res.status(400).json({ error: "State, district and season are required" });
  }

  try {
    const rows = await loadCsvIntoMemory();
    if (!rows || rows.length === 0) {
      return res.status(404).json({ error: "Seasonal dataset not found" });
    }

    const match = rows.find((r) =>
      (r.state || "").toLowerCase() === String(state).toLowerCase()
      && (r.district || "").toLowerCase() === String(district).toLowerCase()
      && (r.season || "").toLowerCase() === String(season).toLowerCase()
    );

    if (!match) {
      return res.status(404).json({ error: `No data found for ${district}, ${season} in ${state}` });
    }

    const crops = (match.recommended_crops || match.recommended || match.crops || "")
      .split(/;|,/)
      .map(s => s.trim())
      .filter(Boolean);

    return res.json({
      state: match.state,
      district: match.district,
      season: match.season,
      recommended_crops: crops,
      primary_crop: match.primary_crop || "",
      notes: match.notes || ""
    });
  } catch (err) {
    console.error("seasonal POST error:", err);
    return res.status(500).json({ error: "Failed to query seasonal dataset" });
  }
});

export default router;