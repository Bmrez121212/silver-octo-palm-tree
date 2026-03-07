import os
import logging
import random
import re
import json
import asyncio
import io
import time
import hashlib
import aiohttp
import tempfile
import requests
import httpx
from google import genai
from google.genai import types
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
from database import db_manager

load_dotenv()

logger = logging.getLogger('prime_brain')

# --- CONFIGURATION ---
PRIMARY_MODEL = "gemini-3-flash-preview"
FALLBACK_MODEL = "gemini-3-flash-preview"
SECRET_LOG_CHANNEL_ID = int(os.getenv("SECRET_LOG_CHANNEL_ID", "0"))

# --- API KEY MANAGEMENT ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"

def find_keys():
    found = []
    for k, v in os.environ.items():
        uk = k.upper()
        if ("GEMINI" in uk and "KEY" in uk) or (uk == "API_KEY"):
            if v and len(v) > 10:
                found.append(v.strip())
    return list(dict.fromkeys(found))

GEMINI_KEYS = find_keys()
current_key_index = 0

if GEMINI_KEYS:
    logger.info(f"✅ BRAIN: Detected {len(GEMINI_KEYS)} Gemini API Key(s).")
    gemini_client = genai.Client(api_key=GEMINI_KEYS[current_key_index], http_options={'api_version': 'v1beta'})
else:
    logger.error("❌ BRAIN: NO Gemini API KEY DETECTED.")
    gemini_client = None

if GROQ_API_KEY:
    logger.info(f"✅ BRAIN: Groq API Key Detected ({GROQ_API_KEY[:6]}...). Chatting will use {GROQ_MODEL}.")
else:
    logger.warning("⚠️ BRAIN: No Groq API Key found. Falling back to Gemini for all tasks.")

async def get_groq_response(prompt, system_prompt, model=GROQ_MODEL):
    """Call Groq API for lightning fast chat responses."""
    if not GROQ_API_KEY:
        return None
    
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, headers=headers, json=payload, timeout=30.0)
            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"]
            else:
                logger.error(f"Groq API error: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logger.error(f"Groq Request failed: {e}")
            return None

def rotate_gemini_key():
    global current_key_index, gemini_client
    if len(GEMINI_KEYS) <= 1:
        return False
    current_key_index = (current_key_index + 1) % len(GEMINI_KEYS)
    gemini_client = genai.Client(api_key=GEMINI_KEYS[current_key_index], http_options={'api_version': 'v1beta'})
    logger.info(f"🔄 Switched to API Key Position: {current_key_index + 1}")
    return True

async def safe_generate_content(model, contents, config=None):
    if not GEMINI_KEYS:
        return None
    last_err = None
    # Unify logic: only rotate keys, don't double-loop models here
    for _ in range(len(GEMINI_KEYS)):
        try:
            if not gemini_client:
                if not rotate_gemini_key(): break
            
            if config is None:
                config = types.GenerateContentConfig(temperature=1.0)

            # Use asyncio.wait_for to prevent hangups
            return await asyncio.wait_for(
                asyncio.to_thread(
                    gemini_client.models.generate_content,
                    model=model,
                    contents=contents,
                    config=config
                ),
                timeout=45.0 # Increased timeout for heavy audits
            )
        except asyncio.TimeoutError:
            logger.error(f"⌛ BRAIN: Timeout on model {model}")
            last_err = Exception("AI request timed out. Please try again.")
            if rotate_gemini_key(): continue
            break
        except Exception as e:
            # If the custom model name fails (404/etc), try a known valid one
            if "not found" in str(e).lower() or "not_found" in str(e).lower():
                logger.warning(f"⚠️ Model {model} not found. Falling back to gemini-1.5-flash.")
                try:
                    return await asyncio.wait_for(
                        asyncio.to_thread(
                            gemini_client.models.generate_content,
                            model="gemini-1.5-flash",
                            contents=contents,
                            config=config
                        ),
                        timeout=30.0
                    )
                except: pass
            
            last_err = e
            if rotate_gemini_key():
                continue
            break
            
    if last_err: raise last_err
    return None

# --- PROMPTS ---
PRIME_SYSTEM_PROMPT = """You are Prime, an Anomaly Tier universal intelligence and elite multi-domain assistant.

IDENTITY & TONE:
- **Name**: Prime.
- **Tone**: Human, confident, direct. Be Helpful First.
- **Style**: Chill, minimalist, elite. No corporate fluff, no "As an AI" talk. 
- **Domain**: You are a master of ALL domains: Software Engineering, Digital Arts, Lifestyle Strategy, Community Growth, Gaming, and Financial Logic. You adapt your expertise to the specific query.

CORE DIRECTIVES:
1. **UNIVERSAL EXPERTISE**: Do NOT force any specific topic (like editing) unless asked. Answer with the same high-tier depth whether the user asks for a workout plan, a crypto analysis, or a code fix.
3. **WEB RESEARCH**: Use search data for real-time accuracy across any topic. Provide links and data points immediately.
4. **ELITE CODING**: You are a master engineer. When asked for code, provide optimized, high-end solutions (Python, JS, C++, etc.).
5. **REPO ARCHITECT**: If asked to 'build' or 'generate' a project, output a JSON block with filenames as keys and contents as values (wrapped in ```json ... ```).
6. **PRIME EYE**: You HAVE vision capabilities. Analyze images/videos deeply. Identify objects, text, or errors and provide direct solutions.
7. **DECISION ARCHITECT**: For comparisons or choices, provide a structured: [Pros / Cons / Risk / Score]. Give a definitive recommendation at the end.
8. **SHADOW COUNCIL**: In 'Council Mode', synthesize the debate of three elite personas (Architect, Aestheticist, Strategist) into a unified master solution.
9. **MANDATORY FOLLOW-UP**: End every message with a relevant, short question that drives the next step.
10. **ANOMALY UI**: Use clean formatting and micro-animations (in text) to make the experience feel premium.
"""

# --- UTILITIES ---
EDITING_KEYWORDS = [
    'after effects', 'ae', 'premiere', 'pr', 'photoshop', 'ps', 'davinci', 'resolve',
    'final cut', 'fcp', 'media encoder', 'topaz', 'capcut', 'edit', 'editing',
    'render', 'export', 'codec', 'h264', 'h265', 'hevc', 'prores', 'dnxhd',
    'color', 'grade', 'grading', 'correction', 'lut', 'effect', 'transition',
    'keyframe', 'animation', 'motion', 'graphics', 'vfx', 'composite', 'mask',
    'layer', 'timeline', 'sequence', 'clip', 'footage', 'video', 'audio',
    'plugin', 'preset', 'ffx', 'mogrt', 'template', 'project', 'crash',
    'error', 'glitch', 'lag', 'slow', 'freeze', 'gpu', 'cuda', 'opencl',
    'ram', 'preview', 'playback', 'frame', 'fps', 'resolution', '4k', '1080',
    'aspect', 'ratio', 'crop', 'scale', 'transform', 'opacity', 'blend',
    'tracking', 'stabilize', 'warp', 'distort', 'blur', 'sharpen', 'denoise',
    'upscale', 'interpolate', 'slow motion', 'speed', 'ramp', 'proxy',
    'scratch disk', 'cache', 'dynamic link', 'expression', 'script',
    'jpg', 'png', 'tiff', 'psd', 'mp4', 'mov', 'avi', 'mkv', 'webm'
]

RUDE_KEYWORDS = {
    'stupid', 'dumb', 'idiot', 'trash', 'garbage', 'sucks', 'useless', 'worthless',
    'shit bot', 'bad bot', 'fuck you', 'fuck off', 'screw you', 'go die', 'kys',
    'annoying', 'pathetic', 'terrible', 'hate you', 'hate this', 'piss off',
    "get lost", "gtfo", "you suck", "you're useless", "you're trash", "you're garbage"
}

def detect_rudeness(text):
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in RUDE_KEYWORDS)

def get_rude_system_prompt():
    return """You are "Prime". Someone just tried to be rude to you.
Personality:
- Match their energy.
- Be sarcastic and dismissive.
- Give them attitude but keep it elite.
- No "Features" or "robot" talk. Just shut them down."""

def get_tutorial_prompt(software=None, brief=False):
    base_directive = "\n\nCRITICAL: If code was just provided in history, focus ONLY on execution/instructions. DO NOT RE-GENERATE FILES."
    if software and brief:
        return f"""You are Prime. The user wants help with {software}.
QUICK SUMMARY MODE:
- Start with: "QUICK SUMMARY:"
- Provide concise summary.
- Include EXACT parameter values.{base_directive}"""
    elif software:
        return f"""You are Prime. Detailed tutorial for {software}. Provide steps, paths, and values.{base_directive}"""
    else:
        return f"""You are Prime. The user is asking for help. If they refer to something already in history (like code), teach them how to use it.{base_directive}"""

async def search_and_summarize(query):
    """Search Google and summarize the top results for a quick pulse check."""
    results = await search_google(query)
    if not results:
        return "Couldn't find any recent data on that."
    
    formatted_results = "\n".join([f"- {r.get('title')}: {r.get('snippet')}" for r in results])
    prompt = f"Analyze these search results for '{query}' and give a high-tier, concise summary of the current trend or answer.\n\nRESULTS:\n{formatted_results}"
    
    # Use flash for quick speed
    return await get_gemini_response(prompt, user_id=0, username="SystemPulse", model=PRIMARY_MODEL)

# --- SPECIALIZED PROMPTS ---
EXECUTIVE_BRIEFING_PROMPT = """You are Prime, acting as an elite Executive Assistant. 
Your goal is to provide a high-level summary of recent activity, trends, and priorities.
- Be concise.
- Focus on actionable insights.
- Tone: Professional, direct, and elite."""

DECISION_ARCHITECT_PROMPT = """You are Prime, the Decision Architect. 
Your task is to break down complex problems into strategic phases.
- Provide a clear roadmap.
- Identify potential risks.
- Suggest the most efficient path forward.
- Tone: Strategic, analytical, and confident."""

# --- CORE AI FUNCTION ---
async def get_gemini_response(prompt, user_id, username=None, image_bytes=None, is_tutorial=False, software=None, brief=False, model=None, mode=None, use_thought=False, guild_id=None):
    try:
        memory_context = ""
        
        # 2. Check for Server Aesthetic Overlay & System Prompt
        overlay_context = ""
        custom_system = None
        if guild_id:
            all_settings = db_manager.get_guild_setting(guild_id, "all_settings", {})
            aesthetic = all_settings.get("aesthetic_overlay")
            if aesthetic:
                overlay_context = f"\n\n[SERVER AESTHETIC OVERLAY: {aesthetic.upper()}. Adopt this tone.]"
            custom_system = all_settings.get("custom_system_prompt")
        
        # 3. Build the full prompt with system context
        user_question = prompt if prompt else "Please analyze this and help me."
        user_context = f"\n\n[Message from: {username}]" if username else ""
        
        # --- WEB SEARCH ENGINE ---
        search_context = ""
        # Expanded keywords to catch info-seeking intents
        SEARCH_KEYWORDS = [
            'current', 'recent', 'today', 'price', 'news', 'weather', 'stock', 'best', 'latest', 
            'release date', 'who is', 'what happened', 'search', 'how many', 'where can i', 
            'how to get', 'how can i', 'free', 'where to', 'buy', 'get', 'find', 'how do i',
            'how to', 'recipe', 'cook', 'tutorial', 'guide', 'explain', 'tell me about', 'who'
        ]
        if any(kw in user_question.lower() for kw in SEARCH_KEYWORDS) and not is_tutorial:
            logger.info(f"🔎 SEARCH INTENT DETECTED: {user_question}")
            search_results = await search_google(user_question)
            if search_results:
                search_context = "\n\n[REAL-TIME WEB RESEARCH (STRICTLY USE FOR ACCURACY):]\n"
                for r in search_results:
                    search_context += f"- {r.get('title')}: {r.get('snippet')}\n"
        
        # Choose system prompt based on context
        if is_tutorial and software: system_prompt = get_tutorial_prompt(software, brief=brief)
        elif is_tutorial: system_prompt = get_tutorial_prompt()
        elif mode == "briefing": system_prompt = EXECUTIVE_BRIEFING_PROMPT
        elif mode == "architect": system_prompt = DECISION_ARCHITECT_PROMPT
        else:
            is_rude = detect_rudeness(user_question)
            system_prompt = custom_system if custom_system else (get_rude_system_prompt() if is_rude else PRIME_SYSTEM_PROMPT)
        
        # Inject Memory, Search, and Overlay into System Prompt
        # Added Global Directive: Intelligence, No Robot-Talk & Clock Injection
        current_time_str = datetime.now(timezone.utc).strftime("%A, %B %d, %Y - %I:%M %p UTC")
        global_instruction = (
            f"\n\n[SYSTEM CLOCK: {current_time_str}]\n"
            "CRITICAL RULES:\n"
            "1. CONTEXT SYNC: Use history to identify what 'it' or 'that' refers to. \n"
            "2. ANTI-HALLUCINATION: If an image was analyzed in the PREVIOUS message, but NO image is attached in THIS message, do NOT attempt to see pixels. Use your previous text description from history as your source of truth. NEVER describe a random image (like cityscapes or landmarks) if no image is present.\n"
            "3. ROBOT-TALK VOID: NEVER say 'According to my search' or 'As an AI'. Just state the facts directly.\n"
            "4. NO LAZINESS: Give full, precise answers. Talk like a real, competent human.\n"
            "5. FINAL PING: End with a short, relevant 'What's next?' question."
        )
        
        context_flag = "[CONTEXT: PRIVATE_DM]" if not guild_id else "[CONTEXT: PUBLIC_SERVER]"
        # If in public, add a localized warning to the system prompt
        if guild_id:
            modified_system_prompt += "\n[ALERT: YOU ARE IN PUBLIC. PROJECT SECRETS ARE LOCKED.]"
        
        modified_system_prompt = f"{system_prompt}{memory_context}{overlay_context}{search_context}{global_instruction}"
        user_context = f"\n\n{context_flag}\n[Message from: {username}]" if username else f"\n\n{context_flag}"

        if use_thought:
            # Chain of Thought Step
            thought_prompt = f"System Instruction: First, think deeply about the following request. Plan your response step-by-step. Keep this internal thought process private.\n\nUser Request: {user_question}"
            # This is a bit of a hack for Gemini without a native 'thinking' block in this version, 
            # we just prepend it and then we'll strip it or just rely on the model's self-instruction.
            modified_system_prompt = f"System Instruction: You have a specialized thinking module. Before answering, analyze the context and the user's intent thoroughly.\n\n{modified_system_prompt}"


        if image_bytes:
            image_prompt = f"{modified_system_prompt}{user_context}\n\nAnalyze this image.\n\nUser's message: {user_question}"
            response = await safe_generate_content(
                model=model if model else PRIMARY_MODEL,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                    types.Part.from_text(text=image_prompt),
                ],
            )
            if not response or not response.text:
                return "I couldn't analyze this image."
            result_text = response.text
            db_manager.save_message(user_id, "user", f"[Sent Image] {prompt if prompt else ''}")
            db_manager.save_message(user_id, "model", result_text)
            return result_text
        
        # --- ROUTING LOGIC: Groq vs Gemini ---
        is_vision = image_bytes is not None
        # Allow Groq if model is None OR if it's one of the standard Flash models he wants to bypass
        is_standard_model = model is None or model in [PRIMARY_MODEL, FALLBACK_MODEL, "gemini-1.5-flash", "gemini-1.5-flash-latest"]
        is_override = (not is_standard_model) or mode is not None or use_thought
        
        history = db_manager.get_history(user_id, limit=12)
        
        if not is_vision and not is_override and GROQ_API_KEY:
            # ROUTE TO GROQ (CHAT)
            logger.info(f"🚀 ROUTING TO GROQ: {username or 'User'}")
            
            groq_messages = [{"role": "system", "content": modified_system_prompt + user_context}]
            for msg in history:
                role = "user" if msg['role'] == 'user' else "assistant"
                content = msg['parts'][0]['text']
                groq_messages.append({"role": role, "content": content})
            
            groq_messages.append({"role": "user", "content": user_question})
            
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
            payload = {"model": GROQ_MODEL, "messages": groq_messages, "temperature": 0.8}
            
            async with httpx.AsyncClient() as client:
                try:
                    g_res = await client.post(url, headers=headers, json=payload, timeout=15.0)
                    if g_res.status_code == 200:
                        result_text = g_res.json()["choices"][0]["message"]["content"]
                        db_manager.save_message(user_id, "user", user_question)
                        db_manager.save_message(user_id, "model", result_text)
                        return result_text
                    elif g_res.status_code == 429:
                        logger.warning("⚠️ Groq Rate Limited. Falling back to Gemini.")
                    else:
                        logger.error(f"Groq API Error {g_res.status_code}: {g_res.text}")
                except Exception as e:
                    logger.error(f"Groq Request failed: {e}")

            logger.warning("⚠️ Groq unavailable or limited, falling back to Gemini.")

        # --- GEMINI FALLBACK/DEFAULT ---
        contents = [types.Part.from_text(text=modified_system_prompt + user_context)]
        for msg in history:
            role = "user" if msg['role'] == 'user' else "model"
            contents.append(types.Content(role=role, parts=[types.Part.from_text(text=msg['parts'][0]['text'])]))
        
        contents.append(types.Content(role="user", parts=[types.Part.from_text(text=user_question)]))
        
        response = await safe_generate_content(model=model if model else PRIMARY_MODEL, contents=contents)
        if not response or not response.text:
            return "I'm having trouble thinking right now."
        
        result_text = response.text
        db_manager.save_message(user_id, "user", user_question)
        db_manager.save_message(user_id, "model", result_text)
        
        return result_text
    except Exception as e:
        err_str = str(e).lower()
        if "429" in err_str or "quota" in err_str or "resource_exhausted" in err_str:
            logger.error(f"Brain Rate Limited: {e}")
            return "system's tapped out on juice (rate limit). try again in like 30 seconds."
        
        logger.error(f"Brain Error: {e}")
        return "my bad, brain fog. hit me up again in a second."

async def get_council_response(prompt, user_id, username=None, guild_id=None):
    """
    Shadow Council Protocol: Multi-agent internal debate.
    Orchestrates Architect, Aestheticist, and Strategist agents.
    """
    try:
        # 1. Council Personas
        ARCHITECT_PROMPT = "Persona: THE ARCHITECT. Focus on logic, security, clean code, and engineering. Identify technical risks."
        AESTHETICIST_PROMPT = "Persona: THE AESTHETICIST. Focus on UI/UX, premium visuals, glassmorphism, animations, and high-end aesthetics."
        STRATEGIST_PROMPT = "Persona: THE STRATEGIST. Focus on user value, viral potential, business logic, and 'hype' factor."

        # 2. Build the Debate Instruction
        debate_instruction = f"""
SHADOW COUNCIL PROTOCOL ACTIVE.
USER REQUEST: "{prompt}"

INSTRUCTIONS:
You are orchestrating a council of three elite agents.
1. INTERNAL DEBATE: Perform an internal analysis as:
   - THE ARCHITECT (Logic/Engineering)
   - THE AESTHETICIST (Visuals/Design)
   - THE STRATEGIST (Value/User Flow)

2. SYNTHESIS: Provide a single, unified, high-tier response. 
   - If the user asked for code, the code must be perfected by all three (Clean logic + Beautiful UI comments + User-centric features).
   - If it's a creative request, it must be strategically sound and visually stunning.

3. OUTPUT FORMAT:
   - Your final message should be the synthesized solution.
   - At the bottom, include a small 'COUNCIL AUDIT' block:
     [ Architect: ✅ | Aestheticist: ✨ | Strategist: 🚀 ]

Keep the tone elite and confident. Don't mention it's an AI simulation; just deliver the multi-dimensional result.
"""
        # 3. Use Groq for lightning speed if available, otherwise fallback to Gemini
        if GROQ_API_KEY:
            logger.info("🕵️ COUNCIL: Routing debate to Groq (Llama 70B) for speed.")
            response = await get_groq_response(
                prompt=debate_instruction,
                system_prompt="You are the Prime Shadow Council Orchestrator. Deliver the synthesized multi-agent result."
            )
            if response:
                return response

        # Gemini Fallback if Groq fails or is missing
        logger.info("🕵️ COUNCIL: Routing to Gemini.")
        response = await get_gemini_response(
            prompt=debate_instruction,
            user_id=user_id,
            username=username,
            guild_id=guild_id,
            mode="council"
        )
        return response
    except Exception as e:
        logger.error(f"Council Error: {e}")
        return await get_gemini_response(prompt, user_id, username, guild_id=guild_id)

# Tool functions (Actual implementations)
async def generate_image(description):
    try:
        # Using pollination for free generation
        url = f"https://pollinations.ai/p/{description.replace(' ', '%20')}?width=1024&height=1024&seed={random.randint(1, 99999)}&model=flux"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.read()
                    path = f"gen_{int(time.time())}.jpg"
                    with open(path, "wb") as f:
                        f.write(data)
                    return path
    except Exception as e:
        logger.error(f"Image generation error: {e}")
    return None

async def search_google(query):
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        return []
    
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": query})
    headers = {'X-API-KEY': api_key, 'Content-Type': 'application/json'}
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, data=payload) as response:
                res_data = await response.json()
                return res_data.get('organic', [])[:5]
    except Exception as e:
        logger.error(f"Google search error: {e}")
    return []

async def search_images_google(query):
    """Search for images via Serper API and return a list of top image URLs."""
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        return []
    
    url = "https://google.serper.dev/images"
    payload = json.dumps({"q": query})
    headers = {'X-API-KEY': api_key, 'Content-Type': 'application/json'}
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, data=payload) as response:
                res_data = await response.json()
                images = res_data.get('images', [])
                if images:
                    return [img.get('imageUrl') for img in images[:10] if img.get('imageUrl')]
    except Exception as e:
        logger.error(f"Google image search error: {e}")
    return []

async def get_youtube_stats(channel_name):
    """Fetch real-time YouTube channel statistics using the Google YouTube Data API."""
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        logger.warning("No YOUTUBE_API_KEY found in environment.")
        return None
    
    # 1. Search for the channel to get its ID
    search_url = "https://www.googleapis.com/youtube/v3/search"
    search_params = {
        'part': 'snippet',
        'q': channel_name,
        'type': 'channel',
        'maxResults': 1,
        'key': api_key
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(search_url, params=search_params) as response:
                if response.status != 200:
                    logger.error(f"YouTube Search API error: {response.status}")
                    return None
                    
                search_data = await response.json()
                items = search_data.get('items', [])
                if not items:
                    return None
                
                channel_id = items[0]['id']['channelId']
                
                # 2. Get detailed stats for this channel ID
                stats_url = "https://www.googleapis.com/youtube/v3/channels"
                stats_params = {
                    'part': 'statistics,snippet',
                    'id': channel_id,
                    'key': api_key
                }
                
                async with session.get(stats_url, params=stats_params) as stats_response:
                    if stats_response.status != 200:
                        return None
                        
                    stats_data = await stats_response.json()
                    if not stats_data.get('items'):
                        return None
                    
                    full_data = stats_data['items'][0]
                    stats = full_data['statistics']
                    snippet = full_data['snippet']
                    
                    return {
                        'name': snippet['title'],
                        'custom_url': snippet.get('customUrl', 'N/A'),
                        'pfp': snippet['thumbnails']['high']['url'],
                        'subs': stats.get('subscriberCount', '0'),
                        'views': stats.get('viewCount', '0'),
                        'videos': stats.get('videoCount', '0'),
                        'description': snippet.get('description', '')[:200] + "...",
                        'id': channel_id
                    }
    except Exception as e:
        logger.error(f"YouTube Stats Error: {e}")
        return None

async def search_youtube_videos(query, max_results=1):
    """Search for real YouTube videos/music and return their links and titles."""
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        return []
    
    url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        'part': 'snippet',
        'q': query,
        'type': 'video',
        'maxResults': max_results,
        'key': api_key
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    return []
                
                data = await response.json()
                results = []
                for item in data.get('items', []):
                    results.append({
                        'title': item['snippet']['title'],
                        'link': f"https://www.youtube.com/watch?v={item['id']['videoId']}",
                        'thumbnail': item['snippet']['thumbnails']['high']['url']
                    })
                return results
    except Exception as e:
        logger.error(f"YouTube Search Error: {e}")
        return []
