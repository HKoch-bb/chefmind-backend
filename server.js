require("dotenv").config({ override: true });
const express = require("express");
const cors = require("cors");
const rateLimit = require("express-rate-limit");
const OpenAI = require("openai");

const app = express();
app.use(cors());
app.use(express.json({ limit: "15mb" }));

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

/* ── Rate Limiters ───────────────────────────────────────────────────────────
 *
 *  Four tiers based on cost and abuse surface:
 *
 *  heavyLimiter   — Full recipe/meal-plan generation (GPT-4.1-mini, long prompts)
 *                   15 requests per 15 min per IP. ~1 call/min max burst.
 *
 *  standardLimiter — Lighter AI calls: details, subs, swaps, nutrition summary
 *                   30 requests per 15 min per IP.
 *
 *  chatLimiter    — Streaming chat. Shorter window (5 min) to prevent spam.
 *                   20 messages per 5 min per IP.
 *
 *  visionLimiter  — Image identify (GPT-4o vision). Most expensive endpoint.
 *                   8 requests per 15 min per IP.
 *
 *  imageLimiter   — Pexels proxy. Not an LLM call, but rate-limit anyway.
 *                   60 requests per 15 min per IP.
 */

const heavyLimiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 15,
  standardHeaders: true,
  legacyHeaders: false,
  message: {
    error: "Too many generation requests. Please wait a few minutes and try again.",
  },
  skip: (req) => process.env.NODE_ENV === "test",
});

const standardLimiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 30,
  standardHeaders: true,
  legacyHeaders: false,
  message: {
    error: "Too many requests. Please slow down and try again shortly.",
  },
  skip: (req) => process.env.NODE_ENV === "test",
});

const chatLimiter = rateLimit({
  windowMs: 5 * 60 * 1000,
  max: 20,
  standardHeaders: true,
  legacyHeaders: false,
  message: {
    error: "Too many chat messages. Give the chef a moment! 🧑‍🍳",
  },
  skip: (req) => process.env.NODE_ENV === "test",
});

const visionLimiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 8,
  standardHeaders: true,
  legacyHeaders: false,
  message: {
    error: "Too many image scans. Please wait before scanning another photo.",
  },
  skip: (req) => process.env.NODE_ENV === "test",
});

const imageLimiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 60,
  standardHeaders: true,
  legacyHeaders: false,
  message: { error: "Too many image requests." },
  skip: (req) => process.env.NODE_ENV === "test",
});

/* ── Cache ── */
const cache = new Map();
const CACHE_TTL = 1000 * 60 * 10;
const setCache = (key, data) => cache.set(key, { data, expiry: Date.now() + CACHE_TTL });
const getCache = (key) => {
  const item = cache.get(key);
  if (!item) return null;
  if (Date.now() > item.expiry) { cache.delete(key); return null; }
  return item.data;
};

/* ── Retry ── */
async function withRetry(fn, retries = 2, attempt = 0) {
  try { return await fn(); }
  catch (err) {
    if (retries === 0) throw err;
    // Exponential backoff: 500ms, 1000ms, …
    await new Promise(r => setTimeout(r, 500 * Math.pow(2, attempt)));
    return withRetry(fn, retries - 1, attempt + 1);
  }
}

/* ── Helpers ── */
const formatIngredient = (ing) => [ing.qty, ing.unit, ing.name].filter(Boolean).join(" ");

function buildFilterLines(filters = {}) {
  const lines = [];
  if (filters.cuisine?.length)
    lines.push(`- Cuisine style (MUST match): ${filters.cuisine.join(", ")}`);
  if (filters.foodTypes?.length)
    lines.push(`- Meal types to use: ${filters.foodTypes.join(", ")}`);
  if (filters.diet?.length)
    lines.push(`- Must satisfy ALL dietary requirements: ${filters.diet.join(", ")}`);
  if (filters.difficulty)
    lines.push(`- Difficulty: ALL recipes MUST be strictly ${filters.difficulty}. Easy=<20min/minimal steps. Medium=~30min. Hard=advanced.`);
  return lines.length ? lines.join("\n") : null;
}

const buildLanguageLine = (language) => {
  if (!language || language === "English") return "";
  return `
##LANGUAGE RULE — NON-NEGOTIABLE##
You MUST write ALL output text in ${language}. This is mandatory.
This includes: overview, step descriptions, notes, tips, ingredient names (translated naturally), previews, filter_notes, and any other descriptive content.
Do NOT write any descriptive text in English. Translate everything naturally into ${language}.
ONLY the JSON keys themselves (like "overview", "steps", "name", "text") must remain in English — their VALUES must be in ${language}.
If you respond in English when ${language} is required, your response will be rejected.
##END LANGUAGE RULE##
`;
};

const normalizeStrict = (r, i) => ({ id: i, title: r?.title || "Untitled", preview: r?.preview || "" });
const normalizeFlexible = (r, i) => ({
  id: i, title: r?.title || "Untitled", preview: r?.preview || "",
  missing_ingredients: (r?.missing_ingredients || []).map(m => ({ name: m.name || "", qty: m.qty || "", unit: m.unit || "" })),
});

/* ── Health ── */
app.get("/", (req, res) => res.send("✅ ChefMind API running"));

/* ── Pexels Image Proxy ── */
app.get("/image", imageLimiter, async (req, res) => {
  try {
    const query = (req.query.q || "food").trim();
    const cacheKey = `img:${query.toLowerCase()}`;
    const cached = cache.get(cacheKey);
    if (cached && Date.now() < cached.expiry) return res.json({ url: cached.data });

    const pexelsKey = process.env.PEXELS_API_KEY;
    if (!pexelsKey) return res.status(500).json({ error: "PEXELS_API_KEY not set" });

    const r = await fetch(
      `https://api.pexels.com/v1/search?query=${encodeURIComponent(query + " food dish")}&per_page=5&orientation=landscape`,
      { headers: { Authorization: pexelsKey } }
    );
    if (!r.ok) return res.status(502).json({ error: "Pexels failed" });
    const data = await r.json();
    const url = (data.photos || [])[0]?.src.large || null;
    if (url) cache.set(cacheKey, { data: url, expiry: Date.now() + 1000 * 60 * 60 * 24 });
    res.json({ url });
  } catch (err) {
    res.status(500).json({ error: "Image fetch failed" });
  }
});

/* ── Identify Ingredients from Image ── */
app.post("/identify-image", visionLimiter, async (req, res) => {
  try {
    const { base64, mimeType = "image/jpeg" } = req.body;
    if (!base64) return res.status(400).json({ error: "No image data" });

    const response = await openai.chat.completions.create({
      model: "gpt-4o",
      messages: [{
        role: "user",
        content: [
          { type: "image_url", image_url: { url: `data:${mimeType};base64,${base64}` } },
          { type: "text", text: `Identify all visible food ingredients in this image. Estimate quantities if visible.

RETURN VALID JSON ONLY:
{
  "ingredients": [
    { "name": "eggs", "qty": "4", "unit": "piece" },
    { "name": "tomatoes", "qty": "2", "unit": "piece" }
  ],
  "description": "Brief one-sentence description of what you see"
}` }
        ]
      }],
      max_tokens: 600,
    });

    const text = response.choices[0].message.content.replace(/```json|```/g, "").trim();
    const data = JSON.parse(text);
    res.json(data);
  } catch (err) {
    console.error("Image identify:", err);
    res.status(500).json({ error: "Failed to identify ingredients" });
  }
});

/* ── Generate Recipes ── */
app.post("/generate-recipes", heavyLimiter, async (req, res) => {
  try {
    const { ingredients = [], filters = {}, language = "English" } = req.body;
    if (!ingredients.length) return res.status(400).json({ error: "No ingredients" });

    const formattedIngredients = ingredients.map(formatIngredient).join(", ");
    const cacheKey = `recipes:${formattedIngredients}:${JSON.stringify({ ...filters, language })}`;
    const cached = getCache(cacheKey);
    if (cached) return res.json(cached);

    const filterLines = buildFilterLines(filters);
    const filterBlock = filterLines ? `\nFILTER CONSTRAINTS:\n${filterLines}\n` : "";
    const langLine = buildLanguageLine(language);
    const systemMsg = `You are a professional chef generating recipe suggestions.${langLine}`;

    const prompt = `${filterBlock}
=== SECTION 1: STRICT RECIPES (exactly 3) ===
- Use ONLY listed ingredients. No substitutions or additions.${filterLines ? "\n- Must satisfy every filter above." : ""}

=== SECTION 2: FLEXIBLE RECIPES (exactly 3) ===
- Use listed as BASE. May add extras (list in missing_ingredients).${filterLines ? "\n- Must satisfy every filter above." : ""}

Available: ${formattedIngredients}
No duplicate titles. 1-line preview each.

RETURN VALID JSON ONLY:
{
  "strict": [{ "title": "", "preview": "" }],
  "flexible": [{ "title": "", "preview": "", "missing_ingredients": [{ "name": "", "qty": "", "unit": "" }] }]
}`;

    const response = await withRetry(() =>
      openai.chat.completions.create({
        model: "gpt-4.1-mini",
        response_format: { type: "json_object" },
        messages: [
          { role: "system", content: systemMsg },
          { role: "user", content: prompt },
        ],
      })
    );

    const data = JSON.parse(response.choices[0].message.content);
    const result = {
      strict: (data.strict || []).map(normalizeStrict),
      flexible: (data.flexible || []).map(normalizeFlexible),
    };
    setCache(cacheKey, result);
    res.json(result);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Failed to generate recipes" });
  }
});

/* ── Generate Meal Plan ── */
app.post("/generate-meal-plan", heavyLimiter, async (req, res) => {
  try {
    const { ingredients = [], filters = {}, mode = "pantry", language = "English" } = req.body;
    if (!ingredients.length) return res.status(400).json({ error: "No ingredients" });

    const formattedIngredients = ingredients.map(formatIngredient).join(", ");
    const cacheKey = `mealplan:${mode}:${formattedIngredients}:${JSON.stringify({ ...filters, language })}`;
    const cached = getCache(cacheKey);
    if (cached) return res.json(cached);

    const filterLines = buildFilterLines(filters);
    const filterBlock = filterLines ? `\nFILTER CONSTRAINTS:\n${filterLines}\n` : "";
    const langLine = buildLanguageLine(language);
    const systemMsg = `You are a professional meal planner.${langLine}`;

    const ingredientRules = mode === "pantry"
      ? `STRICT RULES: Use ONLY: ${formattedIngredients}. No assumptions, no staples unless listed.`
      : `GROCERY RULES: Full haul: ${formattedIngredients}. Distribute across 5 days, maximize variety. Basic staples (salt, oil) assumed available.`;

    const prompt = `${ingredientRules}
${filterBlock}
Create a 5-day meal plan (Mon-Fri), 4 meals/day (Breakfast, Lunch, Dinner, Snack).
Each meal: a name + 1-line note.

RETURN VALID JSON ONLY:
{
  "plan": [
    { "day": "Monday",    "meals": { "Breakfast": { "name": "", "note": "" }, "Lunch": { "name": "", "note": "" }, "Dinner": { "name": "", "note": "" }, "Snack": { "name": "", "note": "" } } },
    { "day": "Tuesday",   "meals": { "Breakfast": { "name": "", "note": "" }, "Lunch": { "name": "", "note": "" }, "Dinner": { "name": "", "note": "" }, "Snack": { "name": "", "note": "" } } },
    { "day": "Wednesday", "meals": { "Breakfast": { "name": "", "note": "" }, "Lunch": { "name": "", "note": "" }, "Dinner": { "name": "", "note": "" }, "Snack": { "name": "", "note": "" } } },
    { "day": "Thursday",  "meals": { "Breakfast": { "name": "", "note": "" }, "Lunch": { "name": "", "note": "" }, "Dinner": { "name": "", "note": "" }, "Snack": { "name": "", "note": "" } } },
    { "day": "Friday",    "meals": { "Breakfast": { "name": "", "note": "" }, "Lunch": { "name": "", "note": "" }, "Dinner": { "name": "", "note": "" }, "Snack": { "name": "", "note": "" } } }
  ]
}`;

    const response = await withRetry(() =>
      openai.chat.completions.create({
        model: "gpt-4.1-mini",
        response_format: { type: "json_object" },
        messages: [
          { role: "system", content: systemMsg },
          { role: "user", content: prompt },
        ],
        max_tokens: 2000,
      })
    );

    const data = JSON.parse(response.choices[0].message.content);
    const DAYS_LIST = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"];
    const MEALS_LIST = ["Breakfast", "Lunch", "Dinner", "Snack"];

    const plan = DAYS_LIST.map(day => {
      const found = (data.plan || []).find(d => d.day === day) || {};
      const meals = {};
      MEALS_LIST.forEach(meal => {
        meals[meal] = { name: found.meals?.[meal]?.name || "", note: found.meals?.[meal]?.note || "" };
      });
      return { day, meals };
    });

    const result = { plan };
    setCache(cacheKey, result);
    res.json(result);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Failed to generate meal plan" });
  }
});

/* ── Recipe Details ── */
app.post("/recipe-details", standardLimiter, async (req, res) => {
  try {
    const { recipeName, language = "English" } = req.body;
    const cacheKey = `details:${recipeName}:${language}`;
    const cached = getCache(cacheKey);
    if (cached) return res.json(cached);

    const langLine = buildLanguageLine(language);
    const systemMsg = `You are a professional chef and nutritionist.${langLine}`;
    const prompt = `Generate a HIGH-QUALITY, PRACTICAL recipe for: ${recipeName}
Clear step-by-step instructions with timing and heat levels. Include beginner tips. Each step must include "time_min".

RETURN VALID JSON ONLY:
{
  "overview": "",
  "servings": "",
  "prep_time": "",
  "cook_time": "",
  "ingredients": { "main": [{ "name": "", "quantity": "", "qty_number": 0, "unit": "" }] },
  "steps": [{ "text": "Detailed step...", "time_min": 5 }],
  "nutrition": { "calories": "", "protein": "", "carbs": "", "fat": "" }
}`;

    const response = await withRetry(() =>
      openai.chat.completions.create({
        model: "gpt-4.1-mini",
        response_format: { type: "json_object" },
        messages: [
          { role: "system", content: systemMsg },
          { role: "user", content: prompt },
        ],
      })
    );

    const data = JSON.parse(response.choices[0].message.content);
    if (Array.isArray(data.steps)) {
      data.steps = data.steps.map(s => typeof s === "string" ? { text: s, time_min: null } : s);
    }
    setCache(cacheKey, data);
    res.json(data);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Failed to load recipe details" });
  }
});

/* ── Generate By Name ── */
app.post("/generate-by-name", heavyLimiter, async (req, res) => {
  try {
    const { recipeName, filters = {}, language = "English" } = req.body;
    if (!recipeName?.trim()) return res.status(400).json({ error: "No recipe name" });

    const cacheKey = `byname:${recipeName.trim().toLowerCase()}:${JSON.stringify({ ...filters, language })}`;
    const cached = getCache(cacheKey);
    if (cached) return res.json(cached);

    const filterLines = [];
    if (filters.cuisine?.length) filterLines.push(`- Cuisine: ${filters.cuisine.join(", ")} (NON-NEGOTIABLE)`);
    if (filters.diet?.length)    filterLines.push(`- Diet: ${filters.diet.join(", ")} (NON-NEGOTIABLE)`);
    if (filters.foodTypes?.length) filterLines.push(`- Meal type: ${filters.foodTypes.join(", ")}`);
    if (filters.difficulty)      filterLines.push(`- Difficulty: ${filters.difficulty}`);

    const filterBlock = filterLines.length ? `\nFILTERS:\n${filterLines.join("\n")}\n` : "";
    const langLine = buildLanguageLine(language);
    const systemMsg = `You are a professional chef and nutritionist.${langLine}`;

    const prompt = `Generate a HIGH-QUALITY, PRACTICAL recipe for: "${recipeName}"
${filterBlock}
Clear step-by-step instructions with timing. Include beginner tips. Each step must include "time_min".

RETURN VALID JSON ONLY:
{
  "overview": "",
  "difficulty_label": "",
  "servings": "",
  "prep_time": "",
  "cook_time": "",
  "ingredients": { "main": [{ "name": "", "quantity": "", "qty_number": 0, "unit": "" }] },
  "steps": [{ "text": "Detailed step...", "time_min": 5 }],
  "nutrition": { "calories": "", "protein": "", "carbs": "", "fat": "" },
  "filter_notes": ""
}`;

    const response = await withRetry(() =>
      openai.chat.completions.create({
        model: "gpt-4.1-mini",
        response_format: { type: "json_object" },
        messages: [
          { role: "system", content: systemMsg },
          { role: "user", content: prompt },
        ],
        max_tokens: 2000,
      })
    );

    const data = JSON.parse(response.choices[0].message.content);
    if (Array.isArray(data.steps)) {
      data.steps = data.steps.map(s => typeof s === "string" ? { text: s, time_min: null } : s);
    }
    const result = { ...data, _title: recipeName.trim() };
    setCache(cacheKey, result);
    res.json(result);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Failed to generate recipe" });
  }
});

/* ── Ingredient Substitution ── */
app.post("/ingredient-sub", standardLimiter, async (req, res) => {
  try {
    const { ingredient, recipeName, language = "English" } = req.body;
    if (!ingredient) return res.status(400).json({ error: "No ingredient" });

    const cacheKey = `sub:${ingredient.toLowerCase()}:${(recipeName||"").toLowerCase()}:${language}`;
    const cached = getCache(cacheKey);
    if (cached) return res.json(cached);

    const langLine = buildLanguageLine(language);
    const context = recipeName ? ` in the context of making "${recipeName}"` : "";
    const prompt = `You are a chef. Suggest 3 substitutions for "${ingredient}"${context}.
${langLine}

RETURN VALID JSON ONLY:
{ "substitutions": [{ "name": "", "ratio": "", "note": "" }, { "name": "", "ratio": "", "note": "" }, { "name": "", "ratio": "", "note": "" }] }`;

    const response = await withRetry(() =>
      openai.chat.completions.create({
        model: "gpt-4.1-mini",
        response_format: { type: "json_object" },
        messages: [{ role: "user", content: prompt }],
        max_tokens: 400,
      })
    );

    const data = JSON.parse(response.choices[0].message.content);
    setCache(cacheKey, data);
    res.json(data);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Failed to get substitutions" });
  }
});

/* ── Nutrition Summary ── */
app.post("/nutrition-summary", standardLimiter, async (req, res) => {
  try {
    const { mealNames = [] } = req.body;
    if (!mealNames.length) return res.status(400).json({ error: "No meals" });

    const cacheKey = `nutsummary:${mealNames.sort().join(",")}`;
    const cached = getCache(cacheKey);
    if (cached) return res.json(cached);

    const mealList = mealNames.map((n, i) => `${i + 1}. ${n}`).join("\n");
    const prompt = `You are a nutritionist. Estimate nutrition per serving:
${mealList}

RETURN VALID JSON ONLY:
{ "meals": [{ "name": "", "calories": 0, "protein_g": 0, "carbs_g": 0, "fat_g": 0 }] }`;

    const response = await withRetry(() =>
      openai.chat.completions.create({
        model: "gpt-4.1-mini",
        response_format: { type: "json_object" },
        messages: [{ role: "user", content: prompt }],
        max_tokens: 1000,
      })
    );

    const data = JSON.parse(response.choices[0].message.content);
    setCache(cacheKey, data);
    res.json(data);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Failed to get nutrition" });
  }
});

/* ── Swap Single Meal ── */
app.post("/swap-meal", standardLimiter, async (req, res) => {
  try {
    const { day, mealType, currentMeal, ingredients = [], filters = {}, language = "English" } = req.body;
    if (!day || !mealType) return res.status(400).json({ error: "Missing day or mealType" });

    const filterLines = buildFilterLines(filters);
    const filterBlock = filterLines ? `\nFILTER CONSTRAINTS:\n${filterLines}\n` : "";
    const langLine = buildLanguageLine(language);
    const systemMsg = `You are a professional meal planner.${langLine}`;
    const ingredientHint = ingredients.length
      ? `Available ingredients: ${ingredients.map(i => [i.qty, i.unit, i.name].filter(Boolean).join(" ")).join(", ")}.`
      : "";

    const prompt = `${filterBlock}
Suggest ONE new ${mealType} meal for ${day}.
${ingredientHint}
${currentMeal ? `Do NOT suggest "${currentMeal}" — it must be a different meal.` : ""}
Keep it practical and quick.

RETURN VALID JSON ONLY:
{ "name": "", "note": "" }`;

    const response = await withRetry(() =>
      openai.chat.completions.create({
        model: "gpt-4.1-mini",
        response_format: { type: "json_object" },
        messages: [
          { role: "system", content: systemMsg },
          { role: "user", content: prompt },
        ],
        max_tokens: 200,
      })
    );
    const data = JSON.parse(response.choices[0].message.content);
    res.json({ name: data.name || "", note: data.note || "" });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Failed to swap meal" });
  }
});

/* ── Generate By Nutrition ── */
app.post("/generate-by-nutrition", heavyLimiter, async (req, res) => {
  try {
    const { targets = {}, filters = {}, language = "English" } = req.body;
    const targetLines = [
      targets.calories && `- Calories per serving: ~${targets.calories} kcal`,
      targets.protein  && `- Protein per serving: ~${targets.protein}g`,
      targets.carbs    && `- Carbs per serving: ~${targets.carbs}g`,
      targets.fat      && `- Fat per serving: ~${targets.fat}g`,
      targets.fiber    && `- Dietary Fiber per serving: ~${targets.fiber}g`,
    ].filter(Boolean);
    if (!targetLines.length) return res.status(400).json({ error: "No nutrition targets provided" });

    const filterLines = buildFilterLines(filters);
    const filterBlock = filterLines ? `\nFILTER CONSTRAINTS:\n${filterLines}\n` : "";
    const langLine = buildLanguageLine(language);
    const systemMsg = `You are a professional nutritionist and chef.${langLine}`;

    const prompt = `${filterBlock}
Generate exactly 4 distinct recipes that match these per-serving nutrition targets as closely as possible:
${targetLines.join("\n")}
${filterLines ? "All filter constraints above must be strictly followed." : ""}
No duplicate titles. Each recipe must be practical and cookable.

RETURN VALID JSON ONLY:
{
  "recipes": [
    {
      "title": "",
      "preview": "",
      "calories": 0,
      "protein_g": 0,
      "carbs_g": 0,
      "fat_g": 0,
      "fiber_g": 0,
      "match_note": "one sentence on how well this matches the targets"
    }
  ]
}`;

    const response = await withRetry(() =>
      openai.chat.completions.create({
        model: "gpt-4.1-mini",
        response_format: { type: "json_object" },
        messages: [
          { role: "system", content: systemMsg },
          { role: "user", content: prompt },
        ],
        max_tokens: 1200,
      })
    );
    const data = JSON.parse(response.choices[0].message.content);
    res.json({ recipes: (data.recipes || []).slice(0, 4) });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Failed to generate nutrition-based recipes" });
  }
});

/* ── AI Chef Chat ── */
app.post("/chat", chatLimiter, async (req, res) => {
  try {
    const {
      messages = [],
      pantry = [],
      savedRecipeNames = [],
      language = "English",
    } = req.body;

    if (!messages.length) return res.status(400).json({ error: "No messages" });

    const pantryInStock = pantry.filter(i => i.inStock !== false);
    const pantryStr = pantryInStock.length
      ? pantryInStock.map(i => [i.qty, i.unit, i.name].filter(Boolean).join(" ")).join(", ")
      : "No pantry items added yet";

    const savedStr = savedRecipeNames.length
      ? savedRecipeNames.slice(0, 20).join(", ")
      : "No saved recipes yet";

    const langInstruction = language !== "English"
      ? `\nIMPORTANT: Respond in ${language} only.`
      : "";

    const systemPrompt = `You are ChefMind AI — a warm, expert personal chef assistant built into the ChefMind cooking app. You have full knowledge of the user's kitchen context.

## User's Current Pantry
${pantryStr}

## User's Saved Recipes
${savedStr}

## Your Capabilities
- Suggest recipes using pantry ingredients (reference them by name)
- Help adapt any recipe (make it vegan, gluten-free, lower calorie, etc.)
- Answer cooking technique questions with clear, practical advice
- Suggest food & drink pairings (wine, cocktails, sides)
- Help troubleshoot cooking mistakes ("my sauce is too salty — how do I fix it?")
- Provide substitution ideas for any ingredient
- Give meal prep tips and time-saving strategies
- Explain food science in simple terms
- Help scale recipes up or down

## Response Style
- Be warm, conversational and encouraging — like a knowledgeable friend in the kitchen
- When listing steps or options, use clear numbered lists or bullet points
- Reference the user's actual pantry ingredients when relevant ("Since you have rice and eggs, you could make...")
- Keep responses focused and practical — avoid unnecessary filler
- For recipes, always include rough cook time and difficulty
- If asked something outside cooking/food, gently redirect back to culinary topics${langInstruction}`;

    const openaiMessages = messages.map(m => ({ role: m.role, content: m.content }));

    const response = await withRetry(() =>
      openai.chat.completions.create({
        model: "gpt-4.1-mini",
        messages: [
          { role: "system", content: systemPrompt },
          ...openaiMessages,
        ],
        max_tokens: 1000,
        temperature: 0.8,
      })
    );

    res.json({ reply: response.choices[0].message.content });
  } catch (err) {
    console.error("Chat error:", err);
    res.status(500).json({ error: "Chef is unavailable right now — please try again" });
  }
});

/* ── Start ── */
const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`🚀 ChefMind running on http://localhost:${PORT}`));