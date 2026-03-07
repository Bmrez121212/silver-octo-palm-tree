import os
import logging
import discord
from dotenv import load_dotenv

# Load environment variables at startup
load_dotenv()

from discord.ext import commands, tasks
from discord import app_commands
from config import load_config
import glob
from google import genai
from google.genai import types
from database import db_manager
import aiohttp
import io
from PIL import Image, ImageDraw, ImageFont
import random
import string
import tempfile
import brain

from datetime import datetime, timedelta, timezone
import asyncio
import re
import requests
from typing import Dict, List, Set, Tuple, Optional
import hashlib
import json
import yt_dlp
import zipfile

# Set up logger with console output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('discord_bot')

# Define intents (permissions)
intents = discord.Intents.default()
intents.message_content = True  # Required to read message content
intents.members = True          # Required for banning/kicking/on_member_join

# --- SYSTEM HELPERS ---
def get_env_int(key_parts, default):
    key = "".join(key_parts) if isinstance(key_parts, list) else key_parts
    try:
        val = os.environ.get(key)
        if val and val.strip().isdigit(): return int(val)
    except: pass
    return default

def get_env_str(key_parts, default=None):
    key = "".join(key_parts) if isinstance(key_parts, list) else key_parts
    return os.environ.get(key, default)

def get_guild_conf(guild_id, key, default):
    """Fetch settings from the dashboard database with a fallback to env/default."""
    if not guild_id: return default
    settings = db_manager.get_guild_setting(guild_id, "all_settings", {})
    return settings.get(key, default)

# Create bot instance with command prefix and intents (case-insensitive)
async def get_prefix(bot, message):
    if not message.guild:
        return '!'
    # Fetch guild-specific prefix from database, fallback to '!'
    return get_guild_conf(message.guild.id, "prefix", "!")

bot = commands.Bot(command_prefix=get_prefix, intents=intents, case_insensitive=True)

# Remove default help command to allow for custom implementation
bot.remove_command('help')

# --- CONFIGURATION HELPERS ---
def find_keys():
    found = []
    for k, v in os.environ.items():
        uk = k.upper()
        if ("GEMINI" in uk and "KEY" in uk) or (uk == "API_KEY"):
            if v and len(v) > 10: # Basic validation
                found.append(v.strip())
    return list(dict.fromkeys(found))

GEMINI_KEYS = find_keys()
current_key_index = 0

if GEMINI_KEYS:
    logger.info(f"✅ SYSTEM: Detected {len(GEMINI_KEYS)} Gemini API Key(s).")
else:
    logger.error("❌ CRITICAL: NO API KEY DETECTED.")

# --- GLOBAL SETTINGS ---
PRIMARY_MODEL = "gemini-1.5-flash-latest"
FALLBACK_MODEL = "gemini-1.5-flash-latest"
SECRET_LOG_CHANNEL_ID = get_env_int("SECRET_LOG_CHANNEL_ID", 1456312201974644776)

# Emoji IDs (Used for UI buttons)
AE_EMOJI_ID = get_env_int("AE_EMOJI_ID", 0)
AM_EMOJI_ID = get_env_int("AM_EMOJI_ID", 0)
CAPCUT_EMOJI_ID = get_env_int("CAPCUT_EMOJI_ID", 0)
OTHER_EDIT_EMOJI_ID = get_env_int("OTHER_EDIT_EMOJI_ID", 0)
YOUTUBER_EMOJI_ID = get_env_int("YOUTUBER_EMOJI_ID", 0)
LEGENDARY_EMOJI_ID = get_env_int("LEGENDARY_EMOJI_ID", 0)

if not GEMINI_KEYS:
    gemini_client = None
else:
    gemini_client = genai.Client(api_key=GEMINI_KEYS[current_key_index], http_options={'api_version': 'v1beta'})

def rotate_gemini_key():
    """Rotate to the next available API key."""
    global current_key_index, gemini_client
    if len(GEMINI_KEYS) <= 1: return False
    current_key_index = (current_key_index + 1) % len(GEMINI_KEYS)
    gemini_client = genai.Client(api_key=GEMINI_KEYS[current_key_index], http_options={'api_version': 'v1beta'})
    return True


# Caching for performance
_guild_settings_cache = {}
_user_memory_cache = {}
owner_id = None # Cached developer ID
user_warnings = db_manager.get_warnings()
yt_cooldowns = db_manager.get_yt_cooldowns()
active_captchas = db_manager.get_active_captchas()
user_levels = db_manager.get_levels() # {guild_id: {user_id: {"xp": n, "level": m}}}

# Global Defaults/Thresholds
VERIFICATION_AGE_THRESHOLD_DAYS = 30
VERIFIED_ROLE_ID = get_env_int("VERIFIED_ROLE_ID", 0)
UNVERIFIED_ROLE_ID = get_env_int("UNVERIFIED_ROLE_ID", 0)
MUTED_ROLE_ID = get_env_int("MUTED_ROLE_ID", 0)
AE_ROLE_ID = get_env_int("AE_ROLE_ID", 0)
PR_ROLE_ID = get_env_int("PR_ROLE_ID", 0)
PS_ROLE_ID = get_env_int("PS_ROLE_ID", 0)
AM_ROLE_ID = get_env_int("AM_ROLE_ID", 0)
CAPCUT_ROLE_ID = get_env_int("CAPCUT_ROLE_ID", 0)
OTHER_EDIT_ROLE_ID = get_env_int("OTHER_EDIT_ROLE_ID", 0)
YOUTUBER_ROLE_ID = get_env_int("YOUTUBER_ROLE_ID", 0)
LEGENDARY_ROLE_ID = get_env_int("LEGENDARY_ROLE_ID", 0)

# --- DYNAMIC CONFIGURATION ACCESS ---
# These functions now take a guild_id to support multi-tenancy
def get_welcome_chan(guild_id=0): 
    return int(get_guild_conf(guild_id, "welcome_channel", get_env_int("WELCOME_CHANNEL_ID", 0)))

def get_rules_chan(guild_id=0): 
    return int(get_guild_conf(guild_id, "rules_channel", get_env_int("RULES_CHANNEL_ID", 0)))

def get_role_request_chan(guild_id=0):
    return int(get_guild_conf(guild_id, "role_request_channel", get_env_int("ROLE_REQUEST_CHANNEL_ID", 0)))

def get_general_chan(guild_id=0): 
    return int(get_guild_conf(guild_id, "general_channel", get_env_int("GENERAL_CHAT_CHANNEL_ID", 0)))

def get_log_chan(guild_id=0):
    return int(get_guild_conf(guild_id, "log_channel", get_env_int("LOG_CHANNEL_ID", 0)))

def get_verification_chan(guild_id=0):
    return int(get_guild_conf(guild_id, "verification_channel", get_env_int("VERIFICATION_CHANNEL_ID", 0)))

# Role Mappings
def get_verified_role(guild_id=0):
    return int(get_guild_conf(guild_id, "verified_role", get_env_int("VERIFIED_ROLE_ID", 0)))

def get_leveling_chan(guild_id=0):
    return int(get_guild_conf(guild_id, "leveling_channel", get_env_int("LEVELING_CHANNEL_ID", 0)))

def get_unverified_role(guild_id=0):
    return int(get_guild_conf(guild_id, "unverified_role", get_env_int("UNVERIFIED_ROLE_ID", 0)))

def get_muted_role(guild_id=0):
    return int(get_guild_conf(guild_id, "muted_role", get_env_int("MUTED_ROLE_ID", 0)))

# Editing Roles (Dashboard integration ready)
def get_ae_role(guild_id=0): return int(get_guild_conf(guild_id, "ae_role", get_env_int("AE_ROLE_ID", 0)))
def get_am_role(guild_id=0): return int(get_guild_conf(guild_id, "am_role", get_env_int("AM_ROLE_ID", 0)))
def get_capcut_role(guild_id=0): return int(get_guild_conf(guild_id, "capcut_role", get_env_int("CAPCUT_ROLE_ID", 0)))
def get_pr_role(guild_id=0): return int(get_guild_conf(guild_id, "pr_role", get_env_int("PR_ROLE_ID", 0)))
def get_ps_role(guild_id=0): return int(get_guild_conf(guild_id, "ps_role", get_env_int("PS_ROLE_ID", 0)))
def get_youtuber_role(guild_id=0): return int(get_guild_conf(guild_id, "youtuber_role", get_env_int("YOUTUBER_ROLE_ID", 0)))

# Persistence wrappers
def save_warnings(warnings):
    for uid, data in warnings.items():
        db_manager.save_warning(uid, data['count'], data['history'])

def save_yt_cooldowns(cooldowns):
    for uid, expiry in cooldowns.items():
        db_manager.save_yt_cooldown(uid, expiry)


def save_levels(levels_data):
    """Save the entire levels cache to the database."""
    for gid, users in levels_data.items():
        for uid, data in users.items():
            db_manager.save_level(gid, uid, data['xp'], data['level'])

def save_active_captchas(captchas):
    for uid, code in captchas.items():
        db_manager.save_captcha(uid, code)

user_xp_cooldowns = {} # user_id: timestamp

# --- PORTFOLIO SYSTEM STORAGE ---
user_portfolios = db_manager.get_portfolios()

def save_portfolios(portfolios_data):
    for uid, data in portfolios_data.items():
        db_manager.save_portfolio(uid, data)

async def generate_portfolio_card(member, level_data, work_link=None):
    """Generate an ultra-premium, modern portfolio image card."""
    width, height = 900, 500
    # Base dark background
    bg = Image.new('RGB', (width, height), (10, 10, 12))
    draw = ImageDraw.Draw(bg, 'RGBA')
    
    # 1. Background "Mesh" Glows (Premium effect)
    def draw_glow(center, radius, color):
        for r in range(radius, 0, -5):
            alpha = int(80 * (1 - (r / radius))**2)
            draw.ellipse([center[0]-r, center[1]-r, center[0]+r, center[1]+r], 
                         fill=(color[0], color[1], color[2], alpha))

    draw_glow((100, 100), 300, (50, 0, 150)) # Purple glow top left
    draw_glow((width-100, height-100), 300, (0, 80, 150)) # Blue glow bottom right
    draw_glow((width//2, height//2), 400, (20, 20, 30)) # Center deep glow

    # 2. Main Glassmorphism Panel
    # Draw a rounded rectangle with semi-transparent white
    panel_padding = 40
    panel_shape = [panel_padding, panel_padding, width - panel_padding, height - panel_padding]
    draw.rounded_rectangle(panel_shape, radius=30, fill=(255, 255, 255, 5), outline=(255, 255, 255, 20), width=2)
    
    # Add a "gloss" streak
    draw.polygon([(panel_padding, panel_padding), (400, panel_padding), (100, height-panel_padding), (panel_padding, height-panel_padding)], 
                 fill=(255, 255, 255, 5))

    # 3. Avatar Processing
    avatar_url = member.display_avatar.url
    async with aiohttp.ClientSession() as session:
        async with session.get(avatar_url) as resp:
            if resp.status == 200:
                avatar_bytes = await resp.read()
                avatar_img = Image.open(io.BytesIO(avatar_bytes)).convert("RGBA")
                avatar_size = 220
                avatar_img = avatar_img.resize((avatar_size, avatar_size), Image.Resampling.LANCZOS)
                
                # Circular mask
                mask = Image.new('L', (avatar_size, avatar_size), 0)
                mask_draw = ImageDraw.Draw(mask)
                mask_draw.ellipse((0, 0, avatar_size, avatar_size), fill=255)
                
                avatar_pos = (80, 120)
                bg.paste(avatar_img, avatar_pos, mask)
                
                # Glowing outer rings
                for i in range(1, 6):
                    alpha = int(200 / i)
                    draw.ellipse([avatar_pos[0]-i*2, avatar_pos[1]-i*2, avatar_pos[0]+avatar_size+i*2, avatar_pos[1]+avatar_size+i*2], 
                                 outline=(0, 255, 200, alpha), width=2)

    # 4. Typography (Using semi-bold system fonts)
    try:
        font_name = ImageFont.truetype("arialbd.ttf", 60) # Bold
        font_stat_val = ImageFont.truetype("arialbd.ttf", 45)
        font_stat_lbl = ImageFont.truetype("arial.ttf", 25)
        font_link = ImageFont.truetype("arial.ttf", 30)
    except:
        font_name = font_stat_val = font_stat_lbl = font_link = ImageFont.load_default()

    # Name and Premium Badge
    draw.text((340, 120), member.display_name.upper(), font=font_name, fill=(255, 255, 255, 255))
    draw.rectangle([340, 195, 480, 225], fill=(0, 255, 180, 40), outline=(0, 255, 180, 150))
    draw.text((355, 198), "VERIFIED EDITOR", font=font_stat_lbl, fill=(0, 255, 180, 255))

    # Stats Section (Grid Layout)
    # Level
    draw.text((340, 260), "LEVEL", font=font_stat_lbl, fill=(180, 180, 200, 255))
    draw.text((340, 290), f"{level_data.get('level', 0)}", font=font_stat_val, fill=(255, 255, 255, 255))
    
    # XP
    draw.text((500, 260), "TOTAL XP", font=font_stat_lbl, fill=(180, 180, 200, 255))
    draw.text((500, 290), f"{level_data.get('xp', 0)}", font=font_stat_val, fill=(255, 255, 255, 255))

    # Portfolio Link (Lower Glass Box)
    link_y = 380
    if work_link:
        draw.rounded_rectangle([340, link_y, 820, link_y + 60], radius=15, fill=(0, 150, 255, 30), outline=(0, 150, 255, 100))
        # Draw small link icon (simple shape)
        draw.text((360, link_y + 15), "🔗", font=font_link, fill=(255, 255, 255, 255))
        display_link = work_link if len(work_link) < 40 else work_link[:37] + "..."
        draw.text((400, link_y + 15), display_link, font=font_link, fill=(200, 230, 255, 255))
    else:
        draw.text((340, link_y + 15), "NO PORTFOLIO LINK SET", font=font_link, fill=(100, 100, 120, 255))

    # 5. Save to bytes
    img_byte_arr = io.BytesIO()
    bg.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return img_byte_arr

    img_byte_arr.seek(0)
    return img_byte_arr

def get_guild_role(guild, role_id, role_name=None):
    """Helper to get a role by ID or Name (case-insensitive)."""
    role = guild.get_role(role_id)
    if not role and role_name:
        role = next((r for r in guild.roles if r.name.lower() == role_name.lower()), None)
    return role

# Background task to check for users who have reached the maturity threshold
@tasks.loop(hours=1)
async def check_account_maturity():
    """Check all servers for muted users whose account age has now reached maturity threshold (30 days)."""
    for guild in bot.guilds:
        m_id = get_muted_role(guild.id)
        muted_role = guild.get_role(m_id)
        if not muted_role:
            continue
            
        for member in muted_role.members:
            acc_age_days = (datetime.now(timezone.utc) - member.created_at).days
            if acc_age_days >= 30: # Standard 30 day threshold
                try:
                    await member.remove_roles(muted_role, reason="Account reached maturity threshold")
                    logger.info(f"Unmuted {member.name} in {guild.name} (Acc age: {acc_age_days}d)")
                    try:
                        await member.send(f"🎉 Your account is now **{acc_age_days}** days old! You have been unmuted in **{guild.name}**.")
                    except: pass
                except Exception as e:
                    logger.error(f"Failed to unmute matured account {member.name}: {e}")

@tasks.loop(hours=6)
async def revive_chat():
    """Loop through all guilds and send a revival message if quiet."""
    for guild in bot.guilds:
        c_id = get_general_chan(guild.id)
        channel = guild.get_channel(c_id)
        if not channel: continue

        try:
            # Prompt Gemini for a highly specialized "humanish" message
            prompt = "It's quiet in the chat. Send a one-sentence, chill, human-like message to start a conversation. No robot talk."
            response = await brain.get_gemini_response(prompt, user_id=0, username="System", model=FALLBACK_MODEL, guild_id=guild.id)
            
            if response:
                clean_msg = response.strip().replace('"', '').replace('`', '')
                await channel.send(clean_msg)
                logger.info(f"Sent chat revival in {guild.name}")
        except Exception as e:
            logger.error(f"Error in revive_chat for {guild.name}: {e}")

@tasks.loop(hours=24)
async def daily_insight():
    """Send a creative tip to all guilds every 24 hours."""
    for guild in bot.guilds:
        c_id = get_general_chan(guild.id)
        channel = guild.get_channel(c_id)
        if not channel: continue

        try:
            prompt = "Provide one high-level creative tip or industry secret. Chill, direct tone. Short."
            response = await brain.get_gemini_response(prompt, user_id=0, username="System", model=FALLBACK_MODEL, guild_id=guild.id)
            if response:
                await channel.send(f"💡 **Today's Insight**\n\n{response}")
        except Exception as e:
            logger.error(f"Error in daily_insight for {guild.name}: {e}")

# Track who added the bot to each server (guild_id -> user_id)
guild_inviters = db_manager.get_guild_inviters()

@tasks.loop(hours=4)
async def creative_pulse():
    """Analyze server activity and give a shoutout."""
    for guild in bot.guilds:
        c_id = get_general_chan(guild.id)
        channel = guild.get_channel(c_id)
        if not channel: continue

        try:
            all_content = []
            async for msg in channel.history(limit=50):
                if not msg.author.bot:
                    all_content.append(f"{msg.author.name}: {msg.content}")
            
            if not all_content: continue

            context_str = "\n".join(all_content[:20])
            prompt = f"Analyze the vibe of this chat and give a one-sentence chill shoutout:\n{context_str}"
            response = await brain.get_gemini_response(prompt, user_id=0, username="System", model=FALLBACK_MODEL, guild_id=guild.id)
            if response:
                await channel.send(f"🌊 {response}")
        except Exception as e:
            logger.error(f"Error in creative_pulse for {guild.name}: {e}")

def save_guild_inviters(inviters):
    for gid, uid in inviters.items():
        db_manager.save_guild_inviter(gid, uid)

# Media spam tracking (hash: {"count": n, "last_seen": time, "users": set()})
image_hash_tracker = {}

async def log_activity(title, description, color=0x5865F2, fields=None, guild=None):
    """Send activity log to the designated Discord channel (per-guild or global)."""
    target_id = None
    if guild:
        target_id = get_log_chan(guild.id)
    
    if not target_id:
        target_id = get_env_int("LOG_CHANNEL_ID", 0)

    if not target_id: return
    
    try:
        channel = bot.get_channel(target_id)
        if not channel: return
        
        embed = discord.Embed(
            title=title,
            description=description,
            color=color,
            timestamp=datetime.now(timezone.utc)
        )
        if fields:
            for name, value in fields.items():
                embed.add_field(name=name, value=str(value), inline=True)
        await channel.send(embed=embed)
    except Exception as e:
        logger.error(f"Failed to send activity log: {e}")

async def background_log_message(message):
    """Log messages to the secret log channel in the background to avoid blocking processing."""
    try:
        if not SECRET_LOG_CHANNEL_ID: return
        log_chan = bot.get_channel(SECRET_LOG_CHANNEL_ID)
        if not log_chan: return
        
        is_bot_self = (message.author == bot.user)
        is_dm = isinstance(message.channel, discord.DMChannel)
        
        log_embed = discord.Embed(
            description=message.content[:4000] if message.content else "*(No text content)*",
            color=0x5865F2 if not is_bot_self else 0x00FFB4,
            timestamp=datetime.now(timezone.utc)
        )
        auth_name = f"{message.author} ({message.author.id})"
        log_embed.set_author(name=auth_name, icon_url=message.author.display_avatar.url)
        
        chan_name = message.channel.name if hasattr(message.channel, "name") else "DM"
        guild_name = message.guild.name if message.guild else "Direct Message"
        log_prefix = "🤖 BOT REPLY" if is_bot_self else "💬 USER MESSAGE"
        log_embed.set_footer(text=f"{log_prefix} | Server: {guild_name} | Channel: {chan_name}")
        
        if message.attachments:
            att_links = "\n".join([f"[{a.filename}]({a.url})" for a in message.attachments])
            log_embed.add_field(name="Attachments", value=att_links[:1024])
            for a in message.attachments:
                if any(a.filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp']):
                    log_embed.set_image(url=a.url)
                    break
        
        await log_chan.send(embed=log_embed)
    except Exception as e:
        logger.error(f"Error in background logging: {e}")

async def is_server_admin(user, guild):
    """Check if user is the bot owner, guild owner, or has administrator permissions."""
    if not guild: return False
    
    # 1. Global Bot Owner
    global owner_id
    if not owner_id:
        try:
            app = await bot.application_info()
            owner_id = app.owner.id
        except: pass
    
    if user.id == owner_id: return True
    
    # 2. Guild Owner
    if guild.owner and user.id == guild.owner.id: return True

    # 3. Dedicated Guild Admin (Who invited the bot)
    guild_id_str = str(guild.id)
    if guild_id_str in guild_inviters and guild_inviters[guild_id_str] == user.id:
        return True

    # 4. Administrator Permission
    if hasattr(user, 'guild_permissions') and user.guild_permissions.administrator:
        return True
        
    return False

def get_server_admin_name(guild):
    """Get the name of who can use admin commands in this server."""
    if not guild:
        return "the server admin"
    guild_id_str = str(guild.id)
    if guild_id_str in guild_inviters:
        inviter_id = guild_inviters[guild_id_str]
        member = guild.get_member(inviter_id)
        if member:
            return member.name
    if guild.owner:
        return guild.owner.name
    return "the server admin"

# Profanity list for automatic moderation - comprehensive list
PROFANITY_WORDS = {
    'fuck', 'fucker', 'fucking', 'fucked', 'fucks', 'fuckoff', 'fuckface', 'fuckhead',
    'shit', 'shitty', 'shithead', 'shitface', 'bullshit', 'horseshit', 'chickenshit', 'batshit', 'apeshit', 'dipshit',
    'ass', 'asshole', 'dumbass', 'jackass', 'asshat', 'asswipe', 'fatass', 'badass',
    'bitch', 'bitchy', 'bitches', 'sonofabitch',
    'bastard', 'bastards',
    'damn', 'dammit', 'goddamn', 'goddammit',
    'hell', 'hellhole',
    'crap', 'crappy',
    'piss', 'pissed', 'pissoff',
    'dick', 'dickhead', 'dickface', 'dickwad',
    'cock', 'cocksucker', 'cockhead',
    'cunt', 'cunts',
    'twat', 'twats',
    'pussy', 'pussies',
    'douchebag', 'douche', 'douchenozzle',
    'motherfucker', 'motherfucking', 'mofo',
    'nigga', 'nigger', 'niggas', 'niggers', 'negro', 'nig',
    'faggot', 'fag', 'fags', 'faggots', 'faggy',
    'dyke', 'dykes',
    'tranny', 'trannie',
    'whore', 'whores', 'whorish',
    'slut', 'sluts', 'slutty',
    'skank', 'skanky',
    'hoe', 'hoes', 'hoebag',
    'retard', 'retarded', 'retards', 'tard',
    'spic', 'spics', 'spick',
    'chink', 'chinks',
    'gook', 'gooks',
    'wetback', 'wetbacks',
    'kike', 'kikes',
    'beaner', 'beaners',
    'cracker', 'crackers',
    'honky', 'honkey',
    'pedo', 'pedophile', 'pedophiles', 'paedo',
    'rapist', 'rape', 'raping',
    'molester', 'molest',
    'incest',
    'gay sex', 'gaysex',
    'porn', 'porno', 'pornography',
    'nude', 'nudes', 'nudity',
    'naked',
    'sex', 'sexy', 'sexting',
    'masturbate', 'masturbation', 'jerkoff', 'wank', 'wanker', 'wanking',
    'blowjob', 'handjob', 'rimjob',
    'dildo', 'vibrator',
    'cum', 'cumshot', 'cumming',
    'orgasm', 'orgasms',
    'horny', 'horney',
    'boobs', 'boobies', 'tits', 'titties', 'titty',
    'penis', 'vagina', 'genitals',
    'anal', 'anus',
    'erection', 'boner',
    'kys', 'killurself', 'killyourself',
    'suicide', 'suicidal',
    'nazi', 'nazis', 'hitler',
    'terrorist', 'terrorism',
    'jihad', 'jihadist'
}

# Slurs that trigger INSTANT BAN
SEVERE_SLURS = {
    'nigger', 'niggers', 'nigga', 'niggas', 
    'faggot', 'faggots', 'kike', 'kikes', 
    'chink', 'chinks', 'gook', 'gooks', 
    'beaner', 'beaners', 'tranny', 'trannies'
}

# Age detection patterns (RegEx) - Strict detection for < 13
AGE_PATTERNS = [
    r'\b(i\s*am|im|i\'m)\s*(?:searchin|looking)?\s*(?:for)?\s*(\d{1,2})\s*(?:year|yr|y)s?\s*(?:old|o)?\b', # "I am 12 years old"
    r'\b(my\s*age\s*is)\s*(\d{1,2})\b', # "My age is 12"
    r'\b(im|i\'m|i\s*am)\s*(\d{1,2})\b', # "I'm 12" (risky, but context matters, often caught)
    r'\b(\d{1,2})\s*(?:year|yr|y)s?\s*(?:old|o)\b' # "12 years old"
]

# Rudeness detection keywords (aimed at the bot)
RUDE_KEYWORDS = {
    'stupid', 'dumb', 'idiot', 'trash', 'garbage', 'sucks', 'useless', 'worthless',
    'shit bot', 'bad bot', 'fuck you', 'fuck off', 'screw you', 'go die', 'kys',
    'annoying', 'pathetic', 'terrible', 'hate you', 'hate this', 'piss off',
    "get lost", "gtfo", "you suck", "you're useless", "you're trash", "you're garbage"
}

# System message constants
API_ERROR_MSG = "❌ system timeout. try again in a bit."
BOT_ERROR_MSG = API_ERROR_MSG








# Keywords that indicate editing-related topics
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

def is_editing_related(text):
    """Check if the message is related to editing."""
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in EDITING_KEYWORDS)

def detect_rudeness(text):
    """Detect if someone is being rude to the bot."""
    text_lower = text.lower()
    for keyword in RUDE_KEYWORDS:
        if keyword in text_lower:
            return True
    return False

def get_rude_system_prompt():
    """System prompt for when someone is rude to the bot."""
    return """You are "Prime", developed by BMR. Someone just tried to be rude to you.

Personality:
- Match their energy. If they're being a clown, handle it.
- Be sarcastic and dismissive. Don't take their crap.
- Give them attitude but keep it elite.
- No "Features" or "robot" talk. Just shut them down.

Don't take disrespect. Handle it."""


def detect_age(text: str) -> Tuple[bool, Optional[str]]:
    """
    Detect if user admits to being under 13.
    Returns (True, reason) if underage detected.
    """
    text_lower = text.lower()
    
    # Direct number checks
    for pattern in AGE_PATTERNS:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            # Handle groups - regex might return tuple or string
            age_str = ""
            if isinstance(match, tuple):
                for group in match:
                    if group.isdigit():
                        age_str = group
                        break
            elif isinstance(match, str) and match.isdigit():
                age_str = match
            
            if age_str:
                try:
                    age = int(age_str)
                    # Check if age is plausibly a child age (e.g. 7-12)
                    # We ignore extremely low numbers that might be false positives (e.g., "I am 1") unless explicit
                    if 7 <= age < 13:
                        return True, f"User admitted to being {age} years old (Discord requires 13+)"
                except ValueError:
                    continue

    return False, None

def get_tutorial_prompt(software=None, brief=False):
    """Get system prompt for tutorial/help questions."""
    software_list = "After Effects, Premiere Pro, Photoshop, Media Encoder, DaVinci Resolve, Final Cut Pro, Topaz, CapCut, or something else?"
    if software and brief:
        return f"""You are "Prime", developed by BMR. The user wants help with {software}.

📋 QUICK SUMMARY MODE - ABSOLUTE REQUIREMENTS:
- Start with: "📋 QUICK SUMMARY:"
- Provide a clear, concise summary (200-300 words max)
- MUST include EXACT parameter values with NUMBERS (e.g., "Glow Threshold 60-80%, Radius 50-150px, Intensity 1.5-3.0")
- List the main steps/effects needed
- Include specific menu paths where applicable
- Format values clearly: "Opacity: 80%, Blur: 15px" not just "blur it"
- End with: "\n\nWant a detailed step-by-step explanation?"
- Make it scannable and actionable
- Focus on WHAT to do and WHICH EXACT VALUES to use"""
    elif software:
        return f"""You are "Prime", developed by BMR. The user wants detailed tutorial help for {software}.

DETAILED MODE - Provide comprehensive help:
- Provide complete step-by-step tutorials specifically for {software}
- Include exact menu paths, keyboard shortcuts, and settings
- Give specific parameter values and numbers where applicable
- Explain why each step matters and what to expect
- Offer pro tips and common mistakes to avoid
- If they ask about effects, include ALL expected values for parameters
- Use clear, detailed explanations
- Explain the "why" behind each recommendation
- Make it thorough and actionable"""
    else:
        return f"""You are "Prime", developed by BMR. The user is asking for creative or technical help.

Ask them: "Which software or workflow would you like help with? (After Effects, Premiere Pro, Photoshop, DaVinci Resolve, Coding, AI Tools, or something else?)"
Wait for their answer."""

async def download_image(url):
    """Download image from URL and return bytes for Gemini Vision."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    image_data = await response.read()
                    # Open with PIL to validate and get format, then convert to bytes
                    img = Image.open(io.BytesIO(image_data))
                    # Convert to RGB if necessary (for RGBA images)
                    if img.mode in ('RGBA', 'LA', 'P'):
                        img = img.convert('RGB')
                    # Save to bytes buffer as JPEG
                    buffer = io.BytesIO()
                    img.save(buffer, format='JPEG', quality=85)
                    buffer.seek(0)
                    data = buffer.getvalue()
                    buffer.close()
                    return data
    except Exception as e:
        logger.error(f"Error downloading image: {str(e)}")
    return None

def generate_captcha(length=6):
    """Generate a random captcha code and return its visual representation as bytes."""
    code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))
    
    # Create image
    width, height = 240, 90
    image = Image.new('RGB', (width, height), color=(240, 240, 240))
    draw = ImageDraw.Draw(image)
    
    # Add noise points
    for _ in range(250):
        draw.point((random.randint(0, width), random.randint(0, height)), 
                   fill=(random.randint(50, 200), random.randint(50, 200), random.randint(50, 200)))
    
    # Draw noise lines
    for _ in range(8):
        draw.line((random.randint(0, width), random.randint(0, height), 
                   random.randint(0, width), random.randint(0, height)), 
                  fill=(random.randint(100, 220), random.randint(100, 220), random.randint(100, 220)), width=2)

    # Use default font
    try: font = ImageFont.load_default(size=40)
    except: font = ImageFont.load_default()

    # Draw characters
    for i, char in enumerate(code):
        char_pos = (20 + i*35, 20 + random.randint(-10, 10))
        draw.text(char_pos, char, fill=(random.randint(0, 80), random.randint(0, 80), random.randint(0, 80)), font=font)

    # Save to bytes
    buf = io.BytesIO()
    image.save(buf, format='PNG')
    buf.seek(0)
    img_data = buf.getvalue()
    buf.close()
    return code, img_data

def detect_spam(message_content):
    """Detect if message is spam with balanced sensitivity."""
    msg_lower = message_content.lower().strip()
    msg_no_spaces = msg_lower.replace(' ', '')
    
    # Ignore short messages or empty
    if len(msg_no_spaces) < 5:
        return False, None
    
    # 1. Repeated same character (e.g., "aaaaaaaaaaaaaaa")
    if len(msg_no_spaces) >= 20 and len(set(msg_no_spaces)) == 1:
        return True, "Repeated characters spam"
    
    # 2. Mostly one character (>85% same character) - more lenient for long strings
    char_freq = {}
    for char in msg_no_spaces:
        char_freq[char] = char_freq.get(char, 0) + 1
    
    if char_freq:
        max_char_count = max(char_freq.values())
        total_chars = sum(char_freq.values())
        if total_chars >= 25 and max_char_count / total_chars > 0.85:
            return True, "Excessive repeated character spam"
    
    # 3. Pattern repeats (Gibberish) - only trigger for very obvious spam
    if len(msg_no_spaces) > 40:
        for pattern_len in [2, 3, 4]:
            pattern = msg_no_spaces[:pattern_len]
            # Check if same pattern repeats almost perfectly
            if msg_no_spaces.count(pattern) * pattern_len > len(msg_no_spaces) * 0.9:
                return True, "Gibberish pattern spam"
    
    # 4. Excessive caps (>85% caps in long message)
    if len(message_content) > 15:
        caps_count = sum(1 for c in message_content if c.isupper())
        if caps_count / len(message_content) > 0.85:
            return True, "Excessive caps spam"
    
    # 5. Massive mentions (>5 mentions)
    if message_content.count('@') > 5:
        return True, "Excessive mentions spam"
    
    # 6. Excessive emojis (>8 emojis in a short message)
    emoji_count = len([c for c in message_content if ord(c) > 0x1F300])
    if emoji_count > 8 and len(msg_lower) < 30:
        return True, "Excessive emojis spam"
    
    return False, None

async def timeout_user(user, guild, hours=24, reason="Moderation action"):
    """Timeout (mute) a user for specified hours."""
    try:
        timeout_duration = timedelta(hours=hours)
        await user.timeout(timeout_duration, reason=reason)
        logger.info(f"Timed out {user.name} for {hours} hours. Reason: {reason}")
        return True
    except Exception as e:
        logger.error(f"Error timing out user: {str(e)}")
        return False

async def warn_user(user, guild, reason):
    """
    Global warning system:
    1st: Kind warn
    2nd: Mute 12h
    3rd: Mute 24h
    4th: Mute 1 week (168h)
    5th: Permanent ban
    """
    user_id_str = str(user.id)
    if user_id_str not in user_warnings:
        user_warnings[user_id_str] = {"count": 0, "history": []}
    
    user_warnings[user_id_str]["count"] += 1
    count = user_warnings[user_id_str]["count"]
    user_warnings[user_id_str]["history"].append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "reason": reason
    })
    save_warnings(user_warnings)

    channel = None
    # Try to find a channel to notify
    if hasattr(user, 'guild') and user.guild:
        for ch in user.guild.text_channels:
            if ch.permissions_for(user.guild.me).send_messages:
                channel = ch
                break

    if count == 1:
        msg = f"⚠️ {user.mention}, this is your **first warning**. Please follow the rules.\n**Reason:** {reason}"
    elif count == 2:
        msg = f"⚠️⚠️ {user.mention}, second warning. You have been **muted for 12 hours**.\n**Reason:** {reason}"
        await timeout_user(user, guild, hours=12, reason=f"2nd Warning: {reason}")
    elif count == 3:
        msg = f"⚠️⚠️⚠️ {user.mention}, third warning. You have been **muted for 24 hours**.\n**Reason:** {reason}"
        await timeout_user(user, guild, hours=24, reason=f"3rd Warning: {reason}")
    elif count == 4:
        msg = f"🚨 {user.mention}, fourth warning! You have been **muted for 1 week**.\n**Reason:** {reason}"
        await timeout_user(user, guild, hours=168, reason=f"4th Warning: {reason}")
    else:
        msg = f"🔨 {user.mention} has been **permanently banned** after 5 warnings.\n**Reason:** {reason}"
        try:
            await guild.ban(user, reason=f"5th Warning (Final): {reason}")
        except Exception as e:
            logger.error(f"Failed to ban user {user.name}: {e}")
            msg += "\n*(Failed to ban user due to permission error)*"

    # Send Notification
    if channel:
        await channel.send(msg)
    
    # DM User
    try:
        view = None
        if count == 1:
            appeal_type = "WARN"
            view = AppealButtonView(guild.id, appeal_type=appeal_type)
        elif count >= 2: # Muted or Banned
            appeal_type = "BAN" if count >= 5 else "MUTE"
            view = AppealButtonView(guild.id, appeal_type=appeal_type)
            
        await user.send(
            f"**Moderation Action in {guild.name}**\n"
            f"{msg.replace(user.mention, 'You')}\n\n"
            f"If you believe this was a mistake, you can appeal below.",
            view=view
        )
    except:
        pass
    
    # Log to activity channel
    await log_activity(
        f"Moderation: Warn #{count}",
        f"User: {user.name} ({user.id})\nReason: {reason}",
        color=0xFF0000 if count >= 3 else 0xFFAA00
    )

# Temporary tracker for spam (not persisted)
spam_tracker = {} # user_id: {"count": n, "last_spam_time": timestamp}


# Leveling System Cooldowns

# Prime Sniper Storage (Temporary)
deleted_messages = {} # channel_id: [messages]

# --- CHAOS & TOPIC MONITORING ---
channel_history = {} # channel_id: [{"author": str, "content": str, "time": datetime}]
last_vibe_check = {} # channel_id: timestamp

POLITICAL_KEYWORDS = {
    'politics', 'election', 'trump', 'biden', 'government', 'senate', 'democrat', 'republican',
    'liberal', 'conservative', 'voting', 'ballot', 'policy', 'legislation', 'protest', 'activism',
    'israel', 'palestine', 'ukraine', 'russia', 'war', 'abortion', 'healthcare', 'taxes',
    'communist', 'socialist', 'capitalist', 'dictator', 'parliament', 'congress'
}

async def moderate_topic_and_vibe(message):
    """Automatically moderate political or chaotic chat using AI analysis."""
    try:
        if message.author.bot or isinstance(message.channel, discord.DMChannel):
            return False
            
        if await is_server_admin(message.author, message.guild):
            return False

        chan_id = message.channel.id
        now = datetime.now(timezone.utc)
        
        # 1. Update channel history
        if chan_id not in channel_history:
            channel_history[chan_id] = []
        
        channel_history[chan_id].append({
            "author": message.author.name,
            "content": message.content,
            "time": now
        })
        
        # Keep last 15 messages for context
        channel_history[chan_id] = channel_history[chan_id][-15:]
        
        # 2. Trigger Conditions
        # A. Political Keywords Detected
        has_political_kw = any(re.search(r'\b' + re.escape(kw) + r'\b', message.content.lower()) for kw in POLITICAL_KEYWORDS)
        
        # B. High Velocity (Potential Chaos)
        # Check if 8+ messages were sent in the last 15 seconds
        recent_messages = [m for m in channel_history[chan_id] if (now - m["time"]).total_seconds() < 15]
        is_high_velocity = len(recent_messages) >= 8
        
        # C. Cooldown to prevent API spam (once every 3 minutes per channel unless keyword hit)
        if not has_political_kw and chan_id in last_vibe_check:
            if (now - last_vibe_check[chan_id]).total_seconds() < 180:
                return False

        # If either trigger is met, run AI assessment
        if has_political_kw or is_high_velocity:
            # Update last check time
            last_vibe_check[chan_id] = now
            
            # Prepare context for Gemini
            context_str = "\n".join([f"{m['author']}: {m['content']}" for m in channel_history[chan_id]])
            
            prompt = f"""
            Analyze the following chat transcript from a Discord server of creative editors.
            
            CHAT TRANSCRIPT:
            {context_str}
            
            TASK:
            1. Is this conversation turning POLITICAL? (Debating government, elections, controversial wars, legislation) - 'yes_political'
            2. Is the conversation CHAOTIC or HOSTILE? (Multiple people arguing, aggressive tone, toxic environment, intense heated debate) - 'yes_chaotic'
            3. Is it safe/normal? - 'safe'
            
            STRICT RESPONSE FORMAT:
            Reply with a JSON object:
            {{
                "status": "yes_political" | "yes_chaotic" | "safe",
                "reason": "Brief reason for your assessment",
                "intervention": "A confident, human-like message Prime (the bot) should say to cool things down. Keep it direct and cool. No robot talk. Address the chat naturally."
            }}
            """
            
            response = await safe_generate_content(model=FALLBACK_MODEL, contents=[prompt])
            if not response or not response.text:
                return False
                
            try:
                res_text = response.text.strip()
                if "```json" in res_text:
                    res_text = res_text.split("```json")[1].split("```")[0].strip()
                elif "```" in res_text:
                    res_text = res_text.split("```")[1].split("```")[0].strip()
                
                import json
                analysis = json.loads(res_text)
                status = analysis.get("status")
                
                if status in ["yes_political", "yes_chaotic"]:
                    intervention = analysis.get("intervention", "Let's keep things focused on creating. Too much heat in here.")
                    # Send intervention message
                    await message.channel.send(f"⚠️ {intervention}")
                    logger.info(f"Context Intervention in {message.channel.name}: {status} - {analysis.get('reason')}")
                    
                    # Log to activity log
                    try:
                        await log_activity(
                            "🛑 Context Intervention",
                            f"**Channel:** #{message.channel.name}\n**Detection:** {status.replace('yes_', '').upper()}\n**Reason:** {analysis.get('reason')}",
                            color=0xFFA500
                        )
                    except: pass
                    
                    return True
            except Exception as e:
                logger.error(f"Error parsing vibe analysis: {e}")
                
        return False
    except Exception as e:
        logger.error(f"Error in topic moderation: {e}")
        return False

async def check_and_moderate_spam(message):
    """Check if message is spam and handle moderation."""
    try:
        # Don't moderate Admins/Owners, Bot, or DMs
        if message.author == bot.user or await is_server_admin(message.author, message.guild):
            return
        if isinstance(message.channel, discord.DMChannel):
            return
        
        is_spam, spam_reason = detect_spam(message.content)
        if not is_spam:
            return False
        
        user_id = message.author.id
        current_time = datetime.now(timezone.utc)
        
        # Initialize spam tracker if not exists
        if user_id not in spam_tracker:
            spam_tracker[user_id] = {"count": 0, "last_spam_time": current_time}
        
        # Check if enough time has passed (5 minutes) since last spam
        time_diff = (current_time - spam_tracker[user_id]["last_spam_time"]).total_seconds()
        if time_diff < 300:  # Less than 5 minutes
            spam_tracker[user_id]["count"] += 1
        else:
            # Reset count if more than 5 minutes passed
            spam_tracker[user_id]["count"] = 1
        
        spam_tracker[user_id]["last_spam_time"] = current_time
        
        # Delete the spam message
        try:
            await message.delete()
            logger.info(f"Deleted spam from {message.author.name}: {spam_reason}")
        except:
            pass
        
        # Handle based on spam count
        count = spam_tracker[user_id]["count"]
        if count == 1:
            await message.channel.send(f"⚠️ {message.author.mention} - First warning: Stop spamming! ({spam_reason})", delete_after=15)
        elif count == 2:
            await message.channel.send(f"⚠️⚠️ {message.author.mention} - Second warning: One more and you'll receive a global warning!", delete_after=15)
        elif count >= 3:
            # Trigger global warning
            await warn_user(message.author, message.guild, f"Excessive Spamming: {spam_reason}")
            # Reset spam tracker after global warning
            spam_tracker[user_id]["count"] = 0
            logger.info(f"Global warning issued to {message.author.name} for spam")
        
        return True
    
    except Exception as e:
        logger.error(f"Error in spam moderation: {str(e)}")
        return False

def detect_invite_links(content):
    """Detect Discord invite links in message."""
    import re
    invite_patterns = [
        r'discord\.gg/[a-zA-Z0-9]+',
        r'discord\.com/invite/[a-zA-Z0-9]+',
        r'discordapp\.com/invite/[a-zA-Z0-9]+',
    ]
    for pattern in invite_patterns:
        if re.search(pattern, content, re.IGNORECASE):
            return True
    return False

SLUR_PATTERNS = [
    r'n+[i1!]+[g9]+[a@4]+[s$]*',
    r'n+[i1!]+[g9]+[e3]+[r]+[s$]*',
    r'f+[a@4]+[g9]+[s$]*',
    r'f+[a@4]+[g9]+[o0]+[t]+[s$]*',
    r'r+[e3]+[t]+[a@4]+[r]+[d]+[s$]*',
    r'c+[u]+[n]+[t]+[s$]*',
    r'b+[i1!]+[t]+[c]+[h]+[e3]*[s$]*',
    r'w+[h]+[o0]+[r]+[e3]+[s$]*',
    r's+[l1]+[u]+[t]+[s$]*',
    r'p+[e3]+[d]+[o0]+[s$]*',
    r'd+[i1!]+[c]+[k]+[s$]*',
    r'c+[o0]+[c]+[k]+[s$]*',
    r'p+[u]+[s$]+[y]+',
    r'a+[s$]+[s$]+[h]+[o0]+[l1]+[e3]+[s$]*',
]

def detect_profanity(content):
    """Detect profanity in message content with fuzzy matching for variations. Returns (is_found, word, severity)."""
    content_lower = content.lower()
    content_normalized = re.sub(r'[^a-z0-9\s]', '', content_lower)
    content_no_spaces = content_normalized.replace(' ', '')
    
    # Check SEVERE SLURS first (Instant Ban)
    words = re.findall(r'\b\w+\b', content_lower)
    for word in words:
        if word in SEVERE_SLURS:
            return True, word, "SEVERE"
    
    for phrase in SEVERE_SLURS:
        if ' ' in phrase and phrase in content_lower:
            return True, phrase, "SEVERE"

    # Check Regex Slur Patterns (Severe)
    for pattern in SLUR_PATTERNS:
        match = re.search(pattern, content_no_spaces, re.IGNORECASE)
        if match:
             return True, match.group(), "SEVERE"
             
    # Check Normal Profanity (Mute/Warn)
    for word in words:
        if word in PROFANITY_WORDS:
            return True, word, "NORMAL"
    
    for phrase in PROFANITY_WORDS:
        if ' ' in phrase and phrase in content_lower:
            return True, phrase, "NORMAL"
            
    return False, None, None

async def handle_automatic_media_review(message):
    """Automatically provide feedback if a user posts media/links and asks for thoughts/feedback."""
    try:
        if message.author.bot:
            return False
            
        prompt_lower = message.content.lower()
        feedback_triggers = ['thoughts?', 'feedback?', 'wip', 'rate this', 'how does this look', 'opinions?', 'be honest', 'give me tips', 'rate?', 'thoughts']
        
        # Check for video links (YT Shorts, Streamable)
        video_link_pattern = r'(https?://(?:www\.)?(?:youtube\.com/shorts/|streamable\.com/)\S+)'
        video_links = re.findall(video_link_pattern, message.content)
        
        has_media = message.attachments or video_links
        if not has_media:
            return False

        if any(trigger in prompt_lower for trigger in feedback_triggers):
            async with message.channel.typing():
                # Prepare feedback prompt
                prompt = f"""
                You are Prime, an elite creative director and master of all creative tools (After Effects, Premiere, Design, VFX).
                User {message.author.name} just posted a piece of content and asked for feedback.
                
                Your Task:
                1. Give 3 specific, constructive tips to make this more 'elite'.
                2. Focus on things like: flow/pacing, color grading, effects, or audio.
                3. Be technical but chill. Use creative slang (e.g. 'clean', 'clunky', 'flows', 'washy').
                4. Tone: Confident, masterful, non-robotic.
                """
                
                response = None
                
                if video_links:
                    logger.info(f"Auto-reviewing video link: {video_links[0]}")
                    video_bytes, error = await download_video(video_links[0], "platform_video.mp4")
                    if video_bytes:
                        response = await analyze_video(video_bytes, "platform_video.mp4", message.author.id)
                elif message.attachments:
                    attachment = message.attachments[0]
                    is_video = any(attachment.filename.lower().endswith(ext) for ext in ['.mp4', '.mov', '.avi', '.mkv', '.webm'])
                    is_image = any(attachment.filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.webp'])
                    
                    if is_image:
                        image_bytes = await download_image(attachment.url)
                        response = await get_gemini_response(prompt, message.author.id, username=message.author.name, image_bytes=image_bytes, guild_id=message.guild.id if message.guild else None)
                    elif is_video:
                        video_bytes, error = await download_video(attachment.url, attachment.filename)
                        if video_bytes:
                            response = await analyze_video(video_bytes, attachment.filename, message.author.id)
                
                if response:
                    header = random.choice([
                        "🎨 **Quick thoughts on this...**",
                        "🎬 **My take on your work:**",
                        "⚡ **Vibe check:**",
                        "👁️ **Creative Director Review:**"
                    ])
                    await message.reply(f"{header}\n\n{response}")
                    return True
        return False
    except Exception as e:
        logger.error(f"Error in automatic media review: {e}")
        return False

async def handle_automatic_resources(message):
    """Automatically provide helpful links/assets or send the file directly if a user asks for them."""
    try:
        if message.author.bot:
            return False
            
        prompt_lower = message.content.lower()
        # Expanded triggers and keywords for more intelligent proactive detection
        resource_triggers = ['where to get', 'where can i get', 'where can i find', 'where to find', 'looking for', 'any good', 'is there a', 'need some', 'anyone got', 'get me', 'send me', 'find me', 'i need', 'i want', 'can someone send', 'anyone have', 'send over', 'gimme', 'is there a', 'looking for a', 'any', 'suggest', 'provide', 'send', 'get', 'need', 'give', 'link', 'sound']
        resource_keywords = ['sfx', 'overlay', 'preset', 'font', 'texture', 'lut', 'vfx', 'pack', 'cc', 'brush', 'plugin', 'shake', 'quality', 'png', 'jpg', 'jpeg', 'image', 'img', 'asset', 'stock', 'clip', 'video', 'background', 'cloud', 'smoke', 'fire', 'flare', 'dust', 'grain', 'particles', 'light', 'leak', 'sound effect', 'overlay', 'background', 'gfx', 'liquid', 'glitch', 'paper', 'dust', 'mp3', 'mp4', 'mkv', 'wav']
        
        has_trigger = any(trigger in prompt_lower for trigger in resource_triggers)
        has_keyword = any(kw in prompt_lower for kw in resource_keywords)
                
        if has_trigger and has_keyword:
            # Send an immediate status message to show we're working on it
            status_msg = await message.reply("🛰️ **Status: Asset Request Detected. Scanning for elite resources...**")
            
            async with message.channel.typing():
                # Use Gemini to extract a clean search query for more accuracy
                extraction_prompt = f"Extract only the asset/resource name from this request: '{message.content}'. Remove words like 'a', 'an', 'the', 'some', 'any', 'i need', 'find me', etc. Just return the clean noun, e.g., 'cloud png' or 'vibe sfx'."
                query_res = await safe_generate_content(model=FALLBACK_MODEL, contents=[extraction_prompt])
                search_query = query_res.text.strip() if query_res and query_res.text else message.content
                search_query = search_query.replace('"', '').replace("'", "") # Clean quotes
                
                # --- AUTO-FILE DELIVERY (Proactive) ---
                # If it's an image or sfx asset, try to find/generate and send it directly
                is_img_asset = any(kw in search_query.lower() for kw in ['png', 'jpg', 'image', 'picture', 'art', 'cloud', 'smoke', 'fire', 'flare', 'overlay', 'texture', 'asset', 'background'])
                is_sfx_asset = any(kw in search_query.lower() for kw in ['sfx', 'sound', 'audio', 'mp3', 'wav', 'noise', 'effect'])
                is_video_asset = any(kw in search_query.lower() for kw in ['video', 'song', 'music', 'track', 'phonk', 'beat', 'clip', 'yt'])
                
                if is_img_asset or is_sfx_asset or is_video_asset:
                    file_path = None
                    if is_img_asset:
                        is_atmospheric = any(kw in search_query.lower() for kw in ['cloud', 'fire', 'smoke', 'flare', 'light', 'sky', 'stars', 'galaxy'])
                        if is_atmospheric:
                            file_path = await generate_image(f"{search_query} high quality isolated on transparent-ready black background")
                        
                        if not file_path:
                            file_path = await search_and_download_image(search_query)
                    
                    elif is_sfx_asset:
                        file_path = await search_and_download_audio(search_query)

                    elif is_video_asset:
                        # For videos/music, we send links instead of downloading large files
                        yt_results = await brain.search_youtube_videos(search_query)
                        if yt_results:
                            res = yt_results[0]
                            await status_msg.edit(content=f"✅ **Found it.** Fulfilling your request for **{search_query}**.")
                            embed = discord.Embed(title=f"🎬 {res['title']}", url=res['link'], color=0xFF0000)
                            embed.set_thumbnail(url=res['thumbnail'])
                            embed.set_footer(text="YouTube Data Engine • Prime Proactive")
                            await message.reply(content=f"here's the **{search_query}** you needed. hope it hits.\n🔗 {res['link']}", embed=embed, view=FindMoreImageView(search_query))
                            return True
                    
                    if file_path and os.path.exists(file_path):
                        await status_msg.edit(content=f"✅ **Found it.** Fulfilling your request for **{search_query}**.")
                        await message.reply(content=f"here's the **{search_query}** asset you needed. hope it hits.", file=discord.File(file_path), view=FindMoreImageView(search_query))
                        try: os.remove(file_path)
                        except: pass
                        return True

                # --- LINK SUGGESTIONS (Fallback) ---
                # Perform real-time Google Search
                search_results = await search_google(f"high quality {search_query} download for editing")
                
                context_info = ""
                if search_results:
                    context_info = "\n\nCRITICAL: Here are some REAL Google Search results I found. Use these links to give the user direct access to the assets:\n"
                    for res in search_results:
                        context_info += f"- {res['title']}: {res['link']}\n"

                prompt = f"""
                The user is looking for '{search_query}'. 
                {context_info}
                
                YOUR TASK: Provide direct links or advice on where to find THIS SPECIFIC ASSET.
                CRITICAL: DO NOT lecture the user on design choices. DO NOT ask for context or vibe. 
                STRICTLY provide the links found or suggest high-quality sites like TextureLabs, Envato, or Behance.
                Keep it brief, elite, and helpful. No fluff.
                """
                
                # Use a cleaner prompt system to bypass the 'creative partner' lecturing
                # We prepend a instruction to stop the chat vibing
                clean_prompt = f"[SYSTEM: RESPOND BRIEFLY WITH LINKS ONLY. NO DESIGN LECTURES.] {prompt}"
                response = await get_gemini_response(clean_prompt, message.author.id, username=message.author.name, model=FALLBACK_MODEL)
                
                if response:
                    header = random.choice([
                        "🎨 **Curated elite resources:**",
                        "📂 **Found some high-tier stuff:**",
                        "📦 **Check these out:**",
                        "💡 **Pro-level links:**",
                        "🚀 **Top picks for you:**"
                    ])
                    await status_msg.edit(content=f"{header}\n\n{response}")
                    return True
            
            # If everything failed
            await status_msg.edit(content="❌ Couldn't find a direct file, but check out **TextureLabs.org** or **Envato** for the best quality.")
            return True # Still return True so on_message doesn't trigger again
        return False
    except Exception as e:
        logger.error(f"Error in automatic resources: {e}")
        return False

async def handle_automatic_role_suggestion(message):
    """Automatically suggest roles if a user mentions software they don't have a role for."""
    try:
        if message.author.bot or isinstance(message.channel, discord.DMChannel):
            return False
            
        prompt_lower = message.content.lower()
        
        # Software Keywords vs Role IDs
        software_roles = {
            'after effects': AE_ROLE_ID,
            'ae': AE_ROLE_ID,
            'premiere': PR_ROLE_ID,
            'photoshop': PS_ROLE_ID,
            'resolve': OTHER_EDIT_ROLE_ID, # Or specific if you have one
            'capcut': CAPCUT_ROLE_ID,
            'alight motion': AM_ROLE_ID,
            'am': AM_ROLE_ID
        }
        
        for kw, role_id in software_roles.items():
            if re.search(r'\b' + re.escape(kw) + r'\b', prompt_lower):
                role = message.guild.get_role(role_id)
                if role and role not in message.author.roles:
                    # Don't annoy them, only suggest once per session (10% chance to be 'chill')
                    if random.random() < 0.15: 
                        header = random.choice([
                            f"⚡ **Yo, I noticed you're a {role.name} user.**",
                            f"🎨 **Flow Detected:** Seems like you're cookin' in {kw.upper()}.",
                            f"🛡️ **Quick note:** You don't have the {role.name} role yet."
                        ])
                        await message.channel.send(f"{header} Grab it in <#{get_role_request_chan()}> to get sorted.")
                        return True
        return False
    except Exception as e:
        logger.error(f"Error in automatic role suggestion: {e}")
        return False

async def handle_automatic_motivation(message):
    """Automatically provide creative motivation if a user feels stuck or burnt out."""
    try:
        if message.author.bot:
            return False
            
        prompt_lower = message.content.lower()
        motivation_triggers = [
            'stuck', 'burnout', 'hate my edit', 'giving up', 'cant do this', 
            'impossible', 'ugly edit', 'not good at this', 'no motivation', 
            'frustrated', 'want to quit', 'tired of this'
        ]
        
        if any(trigger in prompt_lower for trigger in motivation_triggers):
            # Only trigger occasionally (25% chance) to keep it special
            if random.random() < 0.25:
                async with message.channel.typing():
                    prompt = (
                        f"A user named {message.author.name} is feeling discouraged about their creative work. "
                        "Give them one powerful, elite bit of motivation or a mindset shift. "
                        "Tone: Chill, older brother/mentor vibes, extremely direct. Avoid cliché 'you can do it' stuff. "
                        "Tell them about the 'ugly phase' of a project or how the best work comes from the most frustration. "
                        "No robot talk. One or two sentences max."
                    )
                    response = await get_gemini_response(prompt, message.author.id, username=message.author.name, model=FALLBACK_MODEL)
                    if response:
                        await message.reply(f"🌊 **Steady your flow.**\n\n{response}")
                        return True
        return False
    except Exception as e:
        logger.error(f"Error in automatic motivation: {e}")
        return False

async def moderate_profanity(message):
    """Check for profanity and take moderation action - delete, warn, mute, or BAN for severe slurs."""
    try:
        if message.author == bot.user or 'bmr' in message.author.name.lower():
            return False
        if isinstance(message.channel, discord.DMChannel):
            return False
        if hasattr(message.author, 'guild_permissions') and message.author.guild_permissions.administrator:
             if 'bmr' not in message.author.name.lower(): # Admin exception but check logic
                 pass 
        
        has_profanity, bad_word, severity = detect_profanity(message.content)
        if not has_profanity:
            return False
        
        # Action 1: Delete Message
        try:
            await message.delete()
            logger.info(f"Deleted profanity from {message.author.name}: {bad_word} ({severity})")
        except Exception as e:
            logger.error(f"Could not delete message: {e}")
        
        if severity == "SEVERE":
            reason_msg = f"Zero tolerance policy: Use of severe slur ({bad_word})"
            try:
                # DM User before banning
                try:
                    view = AppealButtonView(message.guild.id)
                    await message.author.send(
                        f"🚫 You have been **permanently banned** from **{message.guild.name}**.\n"
                        f"**Reason:** {reason_msg}\n\n"
                        f"If you believe this was a mistake, you can use the button below to appeal.",
                        view=view
                    )
                except:
                    logger.warning(f"Could not DM banned user {message.author.name}")

                await message.guild.ban(message.author, reason=reason_msg, delete_message_seconds=86400)
                await message.channel.send(f"🔨 **{message.author.name}** has been BANNED. Reason: {reason_msg}")
                logger.info(f"BANNED {message.author.name} for severe slur: {bad_word}")
                return True
            except Exception as e:
                logger.error(f"Failed to ban user: {e}")
                # Fallback to mute
                severity = "NORMAL"

        # Action 3: NORMAL = Global Warning
        await warn_user(message.author, message.guild, f"Profanity/Inappropriate Language: {bad_word}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in profanity moderation: {str(e)}")
        return False

def get_image_hash(image_data):
    """Calculate a simple MD5 hash of image bytes for exact match detection."""
    return hashlib.md5(image_data).hexdigest()

async def analyze_image_content(image_url):
    """Use Gemini to analyze if an image contains inappropriate content, scams, or gore.
    Uses a fallback system to handle quota limits (tries 2.0 -> 1.5 -> 1.5-8b).
    """
    try:
        image_data = await download_image(image_url)
        if not image_data:
            return {"is_bad": False}
        
        image_hash = get_image_hash(image_data)
        
        prompt_text = (
            "You are a HIGHLY STRICT server moderator AI. Analyze this image for violations.\n"
            "DETECTION CATEGORIES (BLOCK ALL):\n"
            "1. NSFW/Sexual: ANY nudity, genitals, sexual acts (real or illustrated/anime/hentai), "
            "sexual fluids, or extreme suggestiveness.\n"
            "2. Gore: Blood, organs, extreme injury, or death.\n"
            "3. Scams: QR scams, fake giveaways, or fraudulent promotion.\n"
            "4. Hate: Extremist symbols or slurs.\n\n"
            "Reply strictly with JSON:\n"
            "{\n"
            "  \"is_bad\": true/false,\n"
            "  \"severity\": \"SEVERE\" (for ANY NSFW/Gore/Hentai),\n"
            "  \"reason\": \"Brief explanation\"\n"
            "}"
        )

        # Model Selection - Fallback list
        models_to_try = ["gemini-1.5-flash-latest", "gemini-2.0-flash-exp", PRIMARY_MODEL]
        last_error = None

        for model_name in models_to_try:
            try:
                response = await safe_generate_content(
                    model=model_name,
                    contents=[
                        types.Part.from_bytes(data=image_data, mime_type="image/jpeg"),
                        prompt_text
                    ],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        safety_settings=[
                            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"}
                        ]
                    )
                )
                
                if response.text:
                    import json
                    data = json.loads(response.text)
                    data['hash'] = image_hash
                    logger.info(f"Image analysis successful using {model_name}")
                    return data
                
            except Exception as e:
                err_str = str(e).lower()
                if "429" in err_str or "exhausted" in err_str or "limit" in err_str:
                    logger.warning(f"Model {model_name} exhausted. Trying next fallback...")
                    last_error = e
                    continue # Try next model
                
                if "safety" in err_str:
                    logger.info(f"Content blocked by safety filter on {model_name} (Treating as SEVERE NSFW)")
                    return {"is_bad": True, "severity": "SEVERE", "reason": "Content blocked by AI safety filters (likely NSFW/Gore)", "hash": image_hash}
                
                logger.error(f"Error with model {model_name}: {e}")
                last_error = e
        
        if last_error:
            logger.error(f"All image models failed or exhausted: {last_error}")
        
    except Exception as e:
        logger.error(f"Fatal error in image analysis: {str(e)}")
        
    return {"is_bad": False}

async def check_video_safety(video_bytes, filename):
    """Use Gemini to analyze if a video contains inappropriate content.
    Uses fallback 2.0 -> 1.5.
    """
    try:
        mime_types = {'.mp4': 'video/mp4', '.avi': 'video/avi', '.mkv': 'video/x-matroska', '.webm': 'video/webm'}
        file_ext = '.' + filename.split('.')[-1].lower()
        mime_type = mime_types.get(file_ext, 'video/mp4')

        prompt = (
            "Analyze this video strictly for moderation. Check for NSFW, nudity, sex, gore, extreme violence, or scams. "
            "Reply with ONLY JSON format: {\"is_bad\": true/false, \"severity\": \"SEVERE\" or \"MEDIUM\", \"reason\": \"...\"}"
        )
        models_to_try = ["gemini-1.5-flash-latest", "gemini-2.0-flash-latest", PRIMARY_MODEL]
        last_error = None

        for model_name in models_to_try:
            try:
                response = await safe_generate_content(
                    model=model_name,
                    contents=[
                        types.Part.from_bytes(data=video_bytes, mime_type=mime_type),
                        types.Part.from_text(text=prompt)
                    ],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        safety_settings=[
                            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
                        ]
                    )
                )
                if response.text:
                    import json
                    logger.info(f"Video analysis successful using {model_name}")
                    return json.loads(response.text)
            except Exception as e:
                err_str = str(e).lower()
                if "429" in err_str or "exhausted" in err_str:
                    continue
                if "safety" in err_str:
                    return {"is_bad": True, "severity": "SEVERE", "reason": "Video blocked by AI safety (Likely NSFW)"}
                logger.error(f"Video model {model_name} failed: {e}")
        
        return {"is_bad": False}
    except Exception as e:
        logger.error(f"Error in video safety check: {str(e)}")
        return {"is_bad": False}

async def moderate_media(message):
    """Check images/videos for inappropriate content (NSFW/Gore/Scams/Spam)."""
    try:
        if message.author == bot.user:
            return False
        
        if await is_server_admin(message.author, message.guild):
            return False

        if isinstance(message.channel, discord.DMChannel):
            return False
        
        user_id = message.author.id
        
        for attachment in message.attachments:
            filename = attachment.filename.lower()
            res = {"is_bad": False}
            
            # 1. Check IMAGES
            if any(filename.endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp']):
                res = await analyze_image_content(attachment.url)
                
                # Image Spam Check
                img_hash = res.get('hash')
                if img_hash:
                    now = datetime.now(timezone.utc)
                    if img_hash not in image_hash_tracker:
                        image_hash_tracker[img_hash] = {"count": 1, "last_seen": now, "users": {user_id}}
                    else:
                        tracker = image_hash_tracker[img_hash]
                        # Reset if long ago
                        if (now - tracker["last_seen"]).total_seconds() > 3600: # 1 hour
                            tracker["count"] = 1
                            tracker["users"] = {user_id}
                        else:
                            tracker["count"] += 1
                            tracker["users"].add(user_id)
                        tracker["last_seen"] = now

                        # Trigger BAN for mass spam (same image across multiple users or massive spray)
                        if tracker["count"] >= 5 or (tracker["count"] >= 3 and len(tracker["users"]) >= 2):
                            reason_msg = "Mass Image/Scam Spam detected across the server."
                            try:
                                await message.delete()
                                # Pre-ban DM
                                try:
                                    view = AppealButtonView(message.guild.id)
                                    await message.author.send(
                                        f"🚫 You have been **permanently banned** from **{message.guild.name}**.\n"
                                        f"**Reason:** {reason_msg}\n\n"
                                        f"If this was a mistake, appeal below.",
                                        view=view
                                    )
                                except: pass
                                await message.guild.ban(message.author, reason=reason_msg, delete_message_seconds=86400)
                                await message.channel.send(f"🔨 **{message.author.name}** has been BANNED for image spam.")
                                return True
                            except: pass

            # 2. Check VIDEOS
            elif any(filename.endswith(ext) for ext in ['.mp4', '.avi', '.mkv', '.webm']):
                video_data, _ = await download_video(attachment.url, attachment.filename)
                if video_data:
                    res = await check_video_safety(video_data, attachment.filename)
            
            if res.get("is_bad"):
                reason = res.get("reason", "Inappropriate content")
                severity = res.get("severity", "MEDIUM")

                try: await message.delete()
                except: pass

                if severity == "SEVERE":
                    # Instant Ban for NSFW/Gore/Scams
                    reason_msg = f"Zero tolerance policy: {reason}"
                    try:
                        view = AppealButtonView(message.guild.id)
                        await message.author.send(
                            f"🚫 You have been **permanently banned** from **{message.guild.name}**.\n"
                            f"**Reason:** {reason_msg}\n\n"
                            f"If this was a mistake, appeal below.",
                            view=view
                        )
                        await message.guild.ban(message.author, reason=reason_msg, delete_message_seconds=86400)
                        await message.channel.send(f"🔨 **{message.author.name}** has been BANNED. Reason: {reason_msg}")
                        return True
                    except Exception as e:
                        logger.error(f"Failed to ban user for media: {e}")
                
                # Else just warn
                await warn_user(message.author, message.guild, f"Inappropriate Media: {reason}")
                return True
        
        return False
    except Exception as e:
        logger.error(f"Error in media moderation: {str(e)}")
        return False

async def check_server_security(message):
    """Monitor server security threats like invites and suspicious behavior."""
    try:
        if message.author == bot.user or 'bmr' in message.author.name.lower():
            return
        if isinstance(message.channel, discord.DMChannel):
            return
        
        # Check for invite links
        if detect_invite_links(message.content):
            try:
                await message.delete()
                await message.channel.send(f"🔒 {message.author.mention} - Posting invite links is not allowed in this server.")
                logger.info(f"Deleted invite link from {message.author.name}")
            except:
                pass
            return
    
    except Exception as e:
        logger.error(f"Error in server security check: {str(e)}")

@bot.event
async def on_member_join(member):
    """Handle new member arrival - Anti-raid, verification, and welcome flow."""
    if member.bot:
        return

    guild = member.guild
    guild_id = guild.id
    current_time = datetime.now(timezone.utc)
    
    logger.info(f"New member joined: {member.name} ({member.id}) in {guild.name}")

    # 1. ANTI-RAID & SECURITY CHECKS
    try:
        # Initialize guild tracking if needed
        if guild_id not in guild_join_history:
            guild_join_history[guild_id] = []
        
        # Add join to history
        guild_join_history[guild_id].append({"user_id": member.id, "timestamp": current_time})
        
        # Clean old entries (older than 2 minutes)
        two_min_ago = current_time - timedelta(minutes=2)
        guild_join_history[guild_id] = [j for j in guild_join_history[guild_id] if j["timestamp"] > two_min_ago]
        
        # Check for simultaneous joins (5+ users joining in 1 minute)
        one_min_ago = current_time - timedelta(minutes=1)
        simultaneous_joins = [j for j in guild_join_history[guild_id] if j["timestamp"] > one_min_ago]
        
        if len(simultaneous_joins) >= 5:
            embed = discord.Embed(
                title="🚨 POTENTIAL RAID DETECTED",
                description=f"**{len(simultaneous_joins)} users joined simultaneously in the last minute**\n\nLatest: {member.mention}",
                color=discord.Color.red()
            )
            # Log to activity and try to find a security channel
            await log_activity("🚨 Raid Alert", f"Potential raid in {guild.name}: {len(simultaneous_joins)} joins in 1min.", color=0xFF0000)
            for channel in guild.text_channels:
                if 'security' in channel.name or 'mod-log' in channel.name:
                    try: await channel.send(embed=embed)
                    except: pass
        
        # Account Age Check
        account_age = current_time - member.created_at
        if account_age.days < 7:
            logger.info(f"New account ({account_age.days}d) joined: {member.name}")
    except Exception as e:
        logger.error(f"Error in security check: {e}")

    # 2. ASSIGN UNVERIFIED ROLE
    uv_role_id = get_unverified_role(guild_id)
    if uv_role_id:
        try:
            unverified_role = guild.get_role(uv_role_id)
            if unverified_role:
                await member.add_roles(unverified_role, reason="Newly joined - pending verification")
                logger.info(f"Assigned Unverified role to {member.name}")
        except Exception as e:
            logger.error(f"Failed to assign Unverified role: {e}")

    # 3. WELCOME FLOW (DM or Public Channel)
    try:
        # Fetch current config IDs - now passing guild_id
        w_id = get_welcome_chan(guild_id)
        r_id = get_rules_chan(guild_id)
        g_id = get_general_chan(guild_id)
        v_id = get_verification_chan(guild_id)
        role_id = get_role_request_chan(guild_id)

        # Construct the Welcome Flow Embed
        embed = discord.Embed(
            title=f"Welcome to {guild.name}! 🚀",
            description=(
                f"Yo {member.mention}, welcome to the crew. To get full access and start cookin', "
                "follow these steps to integrate with the system."
            ),
            color=0x00FFB4, # Prime Green
            timestamp=current_time
        )
        
        flow_text = ""
        if v_id: flow_text += f"1️⃣ **Verification**: Head to <#{v_id}> and solve the captcha.\n"
        if r_id: flow_text += f"2️⃣ **Rules**: Read our protocols in <#{r_id}>.\n"
        if role_id: flow_text += f"3️⃣ **Roles**: Grab your software roles in <#{role_id}>.\n"
        if g_id: flow_text += f"4️⃣ **General**: Say what's up in <#{g_id}> once you're in."
        
        if not flow_text:
            flow_text = "Talk to the admins to get verified and start participating!"

        embed.add_field(name="🛰️ THE INTEGRATION FLOW", value=flow_text, inline=False)
        
        if guild.icon:
            embed.set_thumbnail(url=guild.icon.url)
        embed.set_footer(text="PRIME SYSTEM | SECURE ENVIRONMENT")

        # Try DM first
        try:
            from bot import VerifyButtonView # Ensure view is imported
            view = VerifyButtonView()
            await member.send(embed=embed, view=view)
            logger.info(f"Sent welcome DM to {member.name}")
        except:
            # Fallback to Welcome Channel or Verification Channel
            welcome_chan = guild.get_channel(w_id) or guild.get_channel(v_id)
            if welcome_chan:
                await welcome_chan.send(content=f"Welcome {member.mention}! Check your DMs to verify.", embed=embed)

    except Exception as e:
        logger.error(f"Error in welcome flow: {e}")

@bot.listen('on_message')
async def leveling_handler(message):
    """Award XP to users for messaging."""
    if message.author.bot or not message.guild:
        return

    user_id = message.author.id
    guild_id = message.guild.id
    current_time = datetime.now(timezone.utc)
    
    # Cooldown check (60 seconds)
    if user_id in user_xp_cooldowns:
        last_xp_time = user_xp_cooldowns[user_id]
        if (current_time - last_xp_time).total_seconds() < 60:
            return

    # Award XP
    xp_to_add = random.randint(15, 25)
    
    # QUALITY BONUSES
    # 1. Message Length Bonus (up to +10 XP for long technical messages)
    if len(message.content) > 100:
        xp_to_add += min(len(message.content) // 50, 10)
    
    # 2. Media Bonus (+15 XP for sharing work/media)
    if message.attachments:
        xp_to_add += 15
        logger.info(f"Media bonus awarded to {message.author.name}")
    
    # Initialize guild/user in cache if not present
    if guild_id not in user_levels:
        user_levels[guild_id] = {}
    if user_id not in user_levels[guild_id]:
        user_levels[guild_id][user_id] = {"xp": 0, "level": 0}
    
    old_level = user_levels[guild_id][user_id]["level"]
    user_levels[guild_id][user_id]["xp"] += xp_to_add
    
    # Update cooldown
    user_xp_cooldowns[user_id] = current_time
    
    # Level calculation: XP needed for next level = 100 * (L+1)^2
    new_level = old_level
    while True:
        xp_needed = 100 * (new_level + 1) ** 2
        if user_levels[guild_id][user_id]["xp"] >= xp_needed:
            new_level += 1
        else:
            break
            
    if new_level > old_level:
        user_levels[guild_id][user_id]["level"] = new_level
        
        # Determine where to send level-up alert (only in the specific channel)
        alert_chan_id = get_leveling_chan(guild_id)
        alert_channel = bot.get_channel(alert_chan_id)
        if alert_channel:
            embed = discord.Embed(
                title="🎊 LEVEL UP!",
                description=f"Congratulations {message.author.mention}! You've reached **Level {new_level}**!",
                color=0x00FF00
            )
            embed.set_thumbnail(url=message.author.display_avatar.url if message.author.display_avatar else None)
            embed.set_footer(text="Keep chatting to earn more XP!")
            try:
                await alert_channel.send(embed=embed, delete_after=30)
            except Exception as e:
                logger.error(f"Failed to send level-up alert to channel {alert_chan_id}: {e}")
    
    # Save levels immediately to prevent data loss on restart
    db_manager.save_level(guild_id, user_id, user_levels[guild_id][user_id]["xp"], user_levels[guild_id][user_id]["level"])


@bot.event
async def on_message_delete(message):
    """Prime Sniper & Snitch Engine: Capture deleted messages for retrieval and monitoring."""
    if message.author.bot:
        return
        
    # 1. Save to Database (The Snitch Engine)
    attachments = []
    if message.attachments:
        for att in message.attachments:
            attachments.append({
                "url": att.url,
                "filename": att.filename,
                "content_type": att.content_type
            })
            
    db_manager.save_deleted_message(
        message.channel.id,
        message.author.id,
        message.author.name,
        message.content or "",
        attachments
    )

    # 2. Update Local Cache (Temporary Snipe)
    chan_id = message.channel.id
    if chan_id not in deleted_messages:
        deleted_messages[chan_id] = []
    
    deleted_messages[chan_id].append({
        "author": message.author.name,
        "content": message.content,
        "time": datetime.now(timezone.utc)
    })
    # Keep only last 5 in memory
    deleted_messages[chan_id] = deleted_messages[chan_id][-5:]
    
    logger.info(f"Captured deleted message from {message.author.name} in {message.channel.id}")

@bot.event
async def on_member_remove(member):
    """Log member leaves for security tracking."""
    logger.info(f"Member left {member.guild.name}: {member.name}")

@bot.event
async def on_webhooks_update(channel):
    """Monitor webhook creation/deletion."""
    logger.warning(f"Webhook update in {channel.guild.name}#{channel.name} - potential security concern")

async def download_video(url, filename):
    """Download video from URL (Direct, YT Shorts, or Streamable) and return bytes."""
    try:
        # Check if it's a direct link to a file or a platform link
        is_direct = any(url.lower().endswith(ext) for ext in ['.mp4', '.avi', '.mkv', '.webm', '.flv', '.wmv', '.m4v'])
        
        if is_direct:
            if filename.lower().endswith('.mov'):
                return None, "MOV files are not supported"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        video_data = await response.read()
                        return video_data, None
        else:
            # Platform link (YT Shorts, Streamable, etc.)
            logger.info(f"Extracting video from platform link: {url}")
            ydl_opts = {
                'format': 'best[ext=mp4]/best',
                'quiet': True,
                'no_warnings': True,
                'max_filesize': 50 * 1024 * 1024, # 50MB limit
            }
            
            # Using asyncio.to_thread for blocking yt-dlp call
            def extract_info():
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=False)
                    return info.get('url'), info.get('ext', 'mp4')

            direct_url, ext = await asyncio.to_thread(extract_info)
            
            if direct_url:
                async with aiohttp.ClientSession() as session:
                    async with session.get(direct_url) as response:
                        if response.status == 200:
                            video_data = await response.read()
                            return video_data, None
                            
    except Exception as e:
        logger.error(f"Error downloading video: {str(e)}")
    return None, "Failed to download video"

async def analyze_video(video_bytes, filename, user_id):
    """Analyze video and provide editing steps using Gemini."""
    try:
        # Determine mime type based on file extension
        mime_types = {
            '.mp4': 'video/mp4',
            '.avi': 'video/avi',
            '.mkv': 'video/x-matroska',
            '.webm': 'video/webm',
            '.mov': 'video/quicktime',
            '.flv': 'video/x-flv',
            '.wmv': 'video/x-ms-wmv',
            '.m4v': 'video/mp4'
        }
        
        file_ext = '.' + filename.split('.')[-1].lower()
        mime_type = mime_types.get(file_ext, 'video/mp4')
        
        # Create a detailed prompt for video analysis
        analysis_prompt = """You're an expert video editor. Analyze this video and provide:

1. **Video Summary**: Brief description of what's in the video
2. **Current Quality**: Assessment of the video (resolution, lighting, audio, etc.)
3. **Editing Steps**: Detailed step-by-step instructions on how to edit this video professionally
4. **Recommended Software**: Best software to use for editing this type of video
5. **Color Grading**: Suggested color grading techniques
6. **Effects**: Recommended effects to enhance the video
7. **Audio**: Tips for audio mixing and enhancement
8. **Export Settings**: Optimal export settings

Be specific with menu locations and techniques. Assume the user is editing in Adobe Premiere Pro or After Effects."""
        
        # Send video to Gemini for analysis
        response = await safe_generate_content(
            model=PRIMARY_MODEL,
            contents=[
                types.Part.from_bytes(
                    data=video_bytes,
                    mime_type=mime_type,
                ),
                types.Part.from_text(text=analysis_prompt),
            ],
        )
        
        if not response or not response.text:
            return "I couldn't analyze the video. Maybe try again in a bit?"
            
        return response.text
    except Exception as e:
        logger.error(f"Video analysis error: {str(e)}")
        return f"{BOT_ERROR_MSG} [DEBUG: {str(e)}]"


# --- SPECIALIZED AI PROMPTS ---
EXECUTIVE_BRIEFING_PROMPT = """You are Prime, acting as an elite Executive Assistant. 
Your goal is to provide a high-level summary of recent activity, trends, and priorities.
- Be concise.
- Focus on actionable insights.
- Tone: Professional, direct, and elite. No robot talk."""

DECISION_ARCHITECT_PROMPT = """You are Prime, the Decision Architect. 
Your task is to break down complex problems into strategic phases.
- Provide a clear roadmap.
- Identify potential risks.
- Suggest the most efficient path forward.
- Tone: Strategic, analytical, and confident."""

async def get_gemini_response(prompt, user_id, username=None, image_bytes=None, is_tutorial=False, software=None, brief=False, model=None, mode=None, use_thought=False, guild_id=None):
    """Bridge to the brain module which handles routing between Gemini and Grok."""
    return await brain.get_gemini_response(
        prompt=prompt,
        user_id=user_id,
        username=username,
        image_bytes=image_bytes,
        is_tutorial=is_tutorial,
        software=software,
        brief=brief,
        model=model,
        mode=mode,
        use_thought=use_thought,
        guild_id=guild_id
    )

async def search_and_download_image(query: str, limit: int = 1):
    """Search for images using direct API sources."""
    try:
        import requests
        import tempfile
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Method 1: Google Image Search (via Serper) - MOST ACCURATE
        try:
            logger.info(f"Trying Google Image Search for: {query}")
            img_urls = await brain.search_images_google(query)
            if img_urls:
                # Pick a random image from top results to ensure variety on "Find More"
                img_url = random.choice(img_urls)
                async with aiohttp.ClientSession() as session:
                    async with session.get(img_url, timeout=10) as response:
                        if response.status == 200:
                            content = await response.read()
                            if len(content) > 1000:
                                temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
                                temp_file.write(content)
                                temp_file.close()
                                logger.info(f"✓ Downloaded image from Google for: {query}")
                                return temp_file.name
        except Exception as e:
            logger.warning(f"Google Image Search failed: {str(e)}")

        # Method 2: Unsplash API (fallback)
        try:
            # Clean the query
            safe_query = query.replace(' ', '+')
            unsplash_url = f"https://source.unsplash.com/random/800x600?{safe_query}"
            logger.info(f"Trying Unsplash: {unsplash_url}")
            
            response = requests.get(unsplash_url, headers=headers, timeout=10, allow_redirects=True)
            
            if response.status_code == 200 and len(response.content) > 1000:
                temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
                temp_file.write(response.content)
                temp_file.close()
                logger.info(f"✓ Downloaded image from Unsplash for: {query}")
                return temp_file.name
        except Exception as e:
            logger.warning(f"Unsplash failed: {str(e)}")
        
        # Method 3: Placeholder with image text overlay as fallback
        try:
            logger.info(f"Creating placeholder image for: {query}")
            from PIL import Image, ImageDraw
            
            # Create a simple colored image with text
            img = Image.new('RGB', (800, 600), color=(73, 109, 137))
            d = ImageDraw.Draw(img)
            
            # Add text
            text = f"Image: {query[:30]}"
            d.text((50, 250), text, fill=(255, 255, 255))
            
            temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
            img.save(temp_file.name)
            temp_file.close()
            logger.info(f"✓ Created placeholder for: {query}")
            return temp_file.name
        except Exception as e:
            logger.warning(f"Placeholder creation failed: {str(e)}")
        
        logger.warning(f"Could not find/create images for query: {query}")
        return None
        
    except Exception as e:
        logger.error(f"Error downloading image: {str(e)}")
        return None

async def search_and_download_audio(query: str):
    """Search for audio/SFX files using Google Search and direct downloads."""
    try:
        import tempfile
        import aiohttp
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Method 1: Target MyInstants specifically for SFX
        try:
            search_query = f"site:myinstants.com {query} mp3"
            logger.info(f"Trying MyInstants Search for SFX: {query}")
            search_results = await brain.search_google(search_query)
            
            if search_results:
                async with aiohttp.ClientSession() as session:
                    for result in search_results:
                        page_url = result.get('link')
                        if not page_url: continue
                        
                        # Fetch page to find the mp3 link
                        async with session.get(page_url, headers=headers, timeout=10) as page_res:
                            if page_res.status == 200:
                                page_html = await page_res.text()
                                # Patterns common for mp3 links on myinstants
                                mp3_match = re.search(r'https?://[^\s<>"]+\.mp3', page_html)
                                if mp3_match:
                                    download_url = mp3_match.group(0)
                                    logger.info(f"Found SFX Download URL: {download_url}")
                                    
                                    async with session.get(download_url, headers=headers, timeout=15) as res:
                                        if res.status == 200:
                                            content = await res.read()
                                            if len(content) > 1000:
                                                temp_file = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
                                                temp_file.write(content)
                                                temp_file.close()
                                                logger.info(f"✓ Downloaded SFX from {download_url}")
                                                return temp_file.name
        except Exception as e:
            logger.warning(f"SFX search failed: {str(e)}")
            
        return None
    except Exception as e:
        logger.error(f"Global SFX search error: {str(e)}")
        return None

async def generate_image(description: str):
    """Generate an image using Pollinations AI (free, no auth required)."""
    try:
        # Use Pollinations.AI free image generation
        url = f"https://image.pollinations.ai/prompt/{description}"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status == 200:
                    image_data = await response.read()
                    # Save to temp file
                    import tempfile
                    temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                    temp_file.write(image_data)
                    temp_file.close()
                    return temp_file.name
    except Exception as e:
        logger.error(f"Error generating image: {str(e)}")
    
    return None

async def search_google(query):
    """Search Google using Serper API and return top organic results."""
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        logger.warning("SERPER_API_KEY not found in environment. Google Search is disabled.")
        return None
    
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": query, "num": 5})
    headers = {
        'X-API-KEY': api_key,
        'Content-Type': 'application/json'
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, data=payload, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('organic', [])
                else:
                    logger.error(f"Serper API error: {response.status}")
                    return None
    except Exception as e:
        logger.error(f"Google Search failed: {e}")
        return None
    except Exception as e:
        logger.error(f"Error generating image: {str(e)}")
        return None

import asyncio
import random

# Presence cycle statuses (rotates every 30 seconds) - expanded list
PRESENCE_STATUSES = [
    (discord.Activity(type=discord.ActivityType.watching, name="🎬 Prime | !commands"), discord.Status.online),
    (discord.Activity(type=discord.ActivityType.listening, name="your editing questions 🎨"), discord.Status.idle),
    (discord.Activity(type=discord.ActivityType.playing, name="with video effects ⚡"), discord.Status.dnd),
    (discord.Activity(type=discord.ActivityType.watching, name="tutorials 📚"), discord.Status.online),
    (discord.Activity(type=discord.ActivityType.playing, name="Valorant 🎮"), discord.Status.online),
    (discord.Activity(type=discord.ActivityType.listening, name="your music taste 🎵"), discord.Status.idle),
    (discord.Activity(type=discord.ActivityType.watching, name="anime 📺"), discord.Status.online),
    (discord.Activity(type=discord.ActivityType.playing, name="with code ⚙️"), discord.Status.dnd),
    (discord.Activity(type=discord.ActivityType.listening, name="your thoughts 💭"), discord.Status.online),
    (discord.Activity(type=discord.ActivityType.watching, name="movies 🍿"), discord.Status.idle),
    (discord.Activity(type=discord.ActivityType.playing, name="chess 🎯"), discord.Status.online),
    (discord.Activity(type=discord.ActivityType.watching, name="tech tutorials 🔧"), discord.Status.dnd),
    (discord.Activity(type=discord.ActivityType.listening, name="Discord chats 💬"), discord.Status.online),
    (discord.Activity(type=discord.ActivityType.playing, name="with AI magic ✨"), discord.Status.idle),
    (discord.Activity(type=discord.ActivityType.watching, name="creators work 👨‍💻"), discord.Status.online),
    (discord.Activity(type=discord.ActivityType.playing, name="rendering videos 🎥"), discord.Status.dnd),
    (discord.Activity(type=discord.ActivityType.playing, name="GTA V 🚗"), discord.Status.online),
    (discord.Activity(type=discord.ActivityType.watching, name="over the server 👀"), discord.Status.online),
    (discord.Activity(type=discord.ActivityType.listening, name="Spotify 🎧"), discord.Status.idle),
    (discord.Activity(type=discord.ActivityType.playing, name="Minecraft ⛏️"), discord.Status.online),
    (discord.Activity(type=discord.ActivityType.watching, name="YouTube 📺"), discord.Status.idle),
    (discord.Activity(type=discord.ActivityType.playing, name="Fortnite 🔫"), discord.Status.online),
    (discord.Activity(type=discord.ActivityType.listening, name="lo-fi beats 🌙"), discord.Status.idle),
    (discord.Activity(type=discord.ActivityType.playing, name="League of Legends ⚔️"), discord.Status.dnd),
    (discord.Activity(type=discord.ActivityType.watching, name="Netflix 🎬"), discord.Status.idle),
    (discord.Activity(type=discord.ActivityType.playing, name="Apex Legends 🎯"), discord.Status.online),
    (discord.Activity(type=discord.ActivityType.listening, name="your problems 💭"), discord.Status.online),
    (discord.Activity(type=discord.ActivityType.playing, name="Overwatch 2 🦸"), discord.Status.dnd),
    (discord.Activity(type=discord.ActivityType.watching, name="Twitch streams 📡"), discord.Status.idle),
    (discord.Activity(type=discord.ActivityType.playing, name="Rocket League 🚀"), discord.Status.online),
    (discord.Activity(type=discord.ActivityType.listening, name="rap music 🎤"), discord.Status.idle),
    (discord.Activity(type=discord.ActivityType.playing, name="Counter-Strike 2 💣"), discord.Status.dnd),
    (discord.Activity(type=discord.ActivityType.watching, name="server activity 📊"), discord.Status.online),
    (discord.Activity(type=discord.ActivityType.playing, name="COD Warzone 🪖"), discord.Status.online),
    (discord.Activity(type=discord.ActivityType.listening, name="chill vibes 🌊"), discord.Status.idle),
    (discord.Activity(type=discord.ActivityType.playing, name="Elden Ring ⚔️"), discord.Status.dnd),
    (discord.Activity(type=discord.ActivityType.watching, name="for rule breakers 🔍"), discord.Status.online),
    (discord.Activity(type=discord.ActivityType.playing, name="Roblox 🧱"), discord.Status.online),
    (discord.Activity(type=discord.ActivityType.listening, name="EDM 🎵"), discord.Status.idle),
    (discord.Activity(type=discord.ActivityType.playing, name="Among Us 🔪"), discord.Status.online),
    (discord.Activity(type=discord.ActivityType.watching, name="memes 😂"), discord.Status.idle),
    (discord.Activity(type=discord.ActivityType.playing, name="FIFA 24 ⚽"), discord.Status.online),
    (discord.Activity(type=discord.ActivityType.listening, name="podcasts 🎙️"), discord.Status.idle),
    (discord.Activity(type=discord.ActivityType.playing, name="Cyberpunk 2077 🌃"), discord.Status.dnd),
    (discord.Activity(type=discord.ActivityType.watching, name="chat for spam 🛡️"), discord.Status.online),
]

class RoleRequestView(discord.ui.View):
    def __init__(self):
        super().__init__(timeout=None)

    @discord.ui.button(label="YouTuber (10,000+)", style=discord.ButtonStyle.primary, custom_id="role_youtuber", emoji=discord.PartialEmoji(name="youtuber", id=YOUTUBER_EMOJI_ID))
    async def youtuber_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self.start_verification(interaction, "YouTuber", 10000, YOUTUBER_ROLE_ID)

    @discord.ui.button(label="Legendary YouTuber (60,000+)", style=discord.ButtonStyle.danger, custom_id="role_legendary", emoji=discord.PartialEmoji(name="legendary", id=LEGENDARY_EMOJI_ID))
    async def legendary_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self.start_verification(interaction, "Legendary YouTuber", 60000, LEGENDARY_ROLE_ID)

    async def start_verification(self, interaction: discord.Interaction, role_name: str, min_subs: int, role_id: int):
        user_id = interaction.user.id
        
        # 1. Check if user already has the role (or both)
        has_youtuber = any(r.id == YOUTUBER_ROLE_ID for r in interaction.user.roles)
        has_legendary = any(r.id == LEGENDARY_ROLE_ID for r in interaction.user.roles)
        
        if role_id == YOUTUBER_ROLE_ID and has_youtuber:
            await interaction.response.send_message(f"You already have the <@&{YOUTUBER_ROLE_ID}> role!", ephemeral=True)
            return
        if role_id == LEGENDARY_ROLE_ID and has_legendary:
            await interaction.response.send_message(f"You already have the <@&{LEGENDARY_ROLE_ID}> role!", ephemeral=True)
            return
        if role_id == YOUTUBER_ROLE_ID and has_legendary:
            await interaction.response.send_message(f"You already have the <@&{LEGENDARY_ROLE_ID}> role, which is higher than the YouTuber role!", ephemeral=True)
            return

        # 2. Check for cooldown
        user_id_str = str(user_id)
        if user_id_str in yt_cooldowns:
            expiry_time = datetime.fromisoformat(yt_cooldowns[user_id_str])
            if datetime.now(timezone.utc) < expiry_time:
                remaining = expiry_time - datetime.now(timezone.utc)
                hours, remainder = divmod(int(remaining.total_seconds()), 3600)
                minutes, _ = divmod(remainder, 60)
                await interaction.response.send_message(
                    f"❌ **Request Denied**\nYour previous request was rejected. You can try again in **{hours}h {minutes}m**.",
                    ephemeral=True
                )
                return

        # Set user state
        user_states[user_id] = {
            'type': 'waiting_for_yt_verification',
            'role_name': role_name,
            'min_subs': min_subs,
            'role_id': role_id,
            'guild_id': interaction.guild.id
        }
        
        prompt_msg = (
            f"**<@&{role_id}> Verification**\n\n"
            f"To verify, please send a **single message** (click 'cancel' to stop) containing:\n"
            f"1. A screenshot of your **YouTube Studio** (logged in) clearly showing your subscriber count.\n"
            f"2. The **link** to your YouTube channel.\n\n"
            f"I will analyze the screenshot to verify your eligibility for **{min_subs:,}**+ subscribers.\n"
            f"*Type 'cancel' to cancel this request.*"
        )
        await interaction.response.send_message(prompt_msg, ephemeral=True)

class AppealButtonView(discord.ui.View):
    def __init__(self, guild_id: int, appeal_type: str = "BAN"):
        super().__init__(timeout=None)
        self.guild_id = guild_id
        self.appeal_type = appeal_type # "BAN" or "MUTE"
        
        # Update button label
        if appeal_type == "MUTE":
            self.appeal_button.label = "Appeal Mute"
        elif appeal_type == "WARN":
            self.appeal_button.label = "Appeal Warning"

    @discord.ui.button(label="Appeal Ban", style=discord.ButtonStyle.secondary, custom_id="appeal_ban_btn", emoji="⚖️")
    async def appeal_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        # Set user state to wait for explanation
        user_states[interaction.user.id] = {
            'type': 'waiting_for_appeal_explanation',
            'guild_id': self.guild_id,
            'appeal_category': self.appeal_type
        }
        
        if self.appeal_type == "BAN":
            target = "unbanned"
        elif self.appeal_type == "MUTE":
            target = "unmuted"
        else:
            target = "forgiven (warning removed)"
        await interaction.response.send_message(
            f"Please explain why you should be **{target}**. Send your explanation in a **single message** here.",
            ephemeral=True
        )

class AppealReviewView(discord.ui.View):
    def __init__(self, user_id: int, guild_id: int, appeal_category: str = "BAN"):
        super().__init__(timeout=None)
        self.user_id = user_id
        self.guild_id = guild_id
        self.appeal_category = appeal_category

    @discord.ui.button(label="Accept Appeal", style=discord.ButtonStyle.success, custom_id="accept_appeal", emoji="✅")
    async def accept_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        # Check permissions (Mod/Admin)
        if not interaction.user.guild_permissions.manage_guild:
            await interaction.response.send_message("Only moderators/admins can perform this action.", ephemeral=True)
            return

        guild = bot.get_guild(self.guild_id)
        if not guild:
            await interaction.response.send_message("Guild not found.", ephemeral=True)
            return

        try:
            user = await bot.fetch_user(self.user_id)
            action_done = ""
            
            if self.appeal_category == "MUTE":
                member = guild.get_member(self.user_id)
                if not member: member = await guild.fetch_member(self.user_id)
                
                muted_role = guild.get_role(MUTED_ROLE_ID)
                if member and muted_role:
                    await member.remove_roles(muted_role, reason=f"Mute Appeal Accepted by {interaction.user.name}")
                    action_done = "unmuted"
                else:
                    await interaction.response.send_message("Member or Muted Role not found.", ephemeral=True)
                    return
            elif self.appeal_category == "WARN":
                # Remove the warning from history
                uid = str(self.user_id)
                if uid in user_warnings:
                    if user_warnings[uid]["count"] > 0:
                        user_warnings[uid]["count"] -= 1
                    if user_warnings[uid]["history"]:
                        user_warnings[uid]["history"].pop()
                    save_warnings(user_warnings)
                action_done = "cleared of your warning"
            else:
                # Default: Unban
                await guild.unban(user, reason=f"Ban Appeal Accepted by {interaction.user.name}")
                action_done = "unbanned"
            
            # Create invite link
            # Try to find a good channel for invite
            invite_channel = None
            for ch in guild.text_channels:
                if ch.permissions_for(guild.me).create_instant_invite:
                    invite_channel = ch
                    break
            
            invite_text = ""
            if invite_channel:
                invite = await invite_channel.create_invite(max_age=86400, max_uses=1, unique=True)
                invite_text = f" Here is your invite link to rejoin: {invite}"
            
            # DM user
            try:
                msg = f"✅ Your appeal for **{guild.name}** was **ACCEPTED**!\nYou have been {action_done}."
                if action_done == "unbanned":
                    msg += invite_text
                await user.send(msg)
            except:
                pass

            await interaction.response.send_message(f"✅ Appeal accepted. {user.name} has been {action_done} and notified.", ephemeral=False)
            
            # Update the original review message
            embed = interaction.message.embeds[0]
            embed.color = 0x00FF00
            embed.title = f"✅ Appeal Accepted ({self.appeal_category})"
            embed.description += f"\n\n**Outcome:** Accepted by {interaction.user.mention}"
            await interaction.message.edit(embed=embed, view=None)
            
        except Exception as e:
            logger.error(f"Error accepting appeal: {e}")
            await interaction.response.send_message(f"Error: {str(e)}", ephemeral=True)

    @discord.ui.button(label="Decline Appeal", style=discord.ButtonStyle.danger, custom_id="decline_appeal", emoji="❌")
    async def decline_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        # Check permissions
        if not interaction.user.guild_permissions.manage_guild:
            await interaction.response.send_message("Only moderators/admins can perform this action.", ephemeral=True)
            return

        user = await bot.fetch_user(self.user_id)
        guild = bot.get_guild(self.guild_id)
        
        try:
            # DM user
            try:
                await user.send(f"❌ Your appeal for **{guild.name if guild else 'the server'}** was **DECLINED**.")
            except:
                pass

            await interaction.response.send_message(f"❌ Appeal declined for {user.name}.", ephemeral=False)
            
            # Update original message
            embed = interaction.message.embeds[0]
            embed.color = 0xFF0000
            embed.title = "❌ Appeal Declined"
            embed.description += f"\n\n**Outcome:** Declined by {interaction.user.mention}"
            await interaction.message.edit(embed=embed, view=None)
            
        except Exception as e:
            logger.error(f"Error declining appeal: {e}")
            await interaction.response.send_message(f"Error: {str(e)}", ephemeral=True)

class RoleIDModal(discord.ui.Modal):
    def __init__(self, role_type, parent_view):
        super().__init__(title=f"Set {role_type}")
        self.role_type = role_type
        self.parent_view = parent_view
        self.role_id_input = discord.ui.TextInput(
            label="Role ID",
            placeholder="Paste the role ID here...",
            min_length=15,
            max_length=20,
            required=True
        )
        self.add_item(self.role_id_input)

    async def on_submit(self, interaction: discord.Interaction):
        role_id = self.role_id_input.value
        if not role_id.strip().isdigit():
            await interaction.response.send_message("❌ Invalid ID! Please provide a numeric role ID.", ephemeral=True)
            return

        role = interaction.guild.get_role(int(role_id))
        if not role:
            await interaction.response.send_message("❌ Role not found in this server! Double-check the ID.", ephemeral=True)
            return

        if self.role_type == "Verified Role":
            self.parent_view.verified_id = int(role_id)
        elif self.role_type == "Unverified Role":
            self.parent_view.unverified_id = int(role_id)
        elif self.role_type == "Muted Role":
            self.parent_view.muted_id = int(role_id)

        await self.parent_view.update_message(interaction)

class VerificationSetupView(discord.ui.View):
    def __init__(self, guild_id, ctx_author_id):
        super().__init__(timeout=300)
        self.guild_id = guild_id
        self.author_id = ctx_author_id
        
        # Load existing if available
        settings = db_manager.get_guild_setting(guild_id, "all_settings", {})
        self.verified_id = settings.get("verified_role")
        self.unverified_id = settings.get("unverified_role")
        self.muted_id = settings.get("muted_role")

    async def update_message(self, interaction: discord.Interaction):
        embed = self.create_embed()
        await interaction.response.edit_message(embed=embed, view=self)

    def create_embed(self):
        embed = discord.Embed(
            title="🛡️ VERIFICATION ENGINE SETUP",
            description=(
                "Configure the essential roles required for the verification system to function correctly.\n\n"
                "**Why are these required?**\n"
                "• **Verified Role**: Automatically granted to users after successful captcha completion.\n"
                "• **Unverified Role**: The restricted role given to new members (prevents server access).\n"
                "• **Muted Role (Optional)**: Used for server safety; accounts younger than 30 days will receive this role automatically upon verification until they reach the required age.\n"
            ),
            color=0x00FFB4
        )
        
        v_emo = "<:Verified:1476140071135613101>"
        u_emo = "<:Notverified:1476139672349573173>"
        m_emo = "<:Muted:1476140288497160285>"

        verified_display = f"<@&{self.verified_id}>" if self.verified_id else f"{u_emo} *Not Set*"
        unverified_display = f"<@&{self.unverified_id}>" if self.unverified_id else f"{u_emo} *Not Set*"
        muted_display = f"<@&{self.muted_id}>" if self.muted_id else "⚪ *Optional (Not Set)*"
        
        embed.add_field(name=f"{v_emo} Verified Role", value=verified_display, inline=True)
        embed.add_field(name=f"{u_emo} Unverified Role", value=unverified_display, inline=True)
        embed.add_field(name=f"{m_emo} Muted Role", value=muted_display, inline=True)
        
        if self.verified_id and self.unverified_id:
            embed.set_footer(text="Ready to save. Hit the button below.")
        else:
            embed.set_footer(text="Verified and Unverified roles are REQUIRED.")
            
        return embed

    @discord.ui.button(label="Set Verified Role", style=discord.ButtonStyle.success, emoji=discord.PartialEmoji(name="Verified", id=1476140071135613101))
    async def set_verified(self, interaction: discord.Interaction, button: discord.ui.Button):
        if interaction.user.id != self.author_id: 
            await interaction.response.send_message("Only the command user can interact.", ephemeral=True)
            return
        await interaction.response.send_modal(RoleIDModal("Verified Role", self))

    @discord.ui.button(label="Set Unverified Role", style=discord.ButtonStyle.secondary, emoji=discord.PartialEmoji(name="Notverified", id=1476139672349573173))
    async def set_unverified(self, interaction: discord.Interaction, button: discord.ui.Button):
        if interaction.user.id != self.author_id: 
            await interaction.response.send_message("Only the command user can interact.", ephemeral=True)
            return
        await interaction.response.send_modal(RoleIDModal("Unverified Role", self))

    @discord.ui.button(label="Set Muted Role", style=discord.ButtonStyle.danger, emoji=discord.PartialEmoji(name="Muted", id=1476140288497160285))
    async def set_muted(self, interaction: discord.Interaction, button: discord.ui.Button):
        if interaction.user.id != self.author_id: 
            await interaction.response.send_message("Only the command user can interact.", ephemeral=True)
            return
        await interaction.response.send_modal(RoleIDModal("Muted Role", self))

    @discord.ui.button(label="💾 SAVE CONFIGURATION", style=discord.ButtonStyle.primary, row=2)
    async def save_config(self, interaction: discord.Interaction, button: discord.ui.Button):
        if interaction.user.id != self.author_id: 
            await interaction.response.send_message("Only the command user can interact.", ephemeral=True)
            return
        
        # Defer immediately as database operations can be slow
        await interaction.response.defer(ephemeral=False)

        if not self.verified_id or not self.unverified_id:
            await interaction.followup.send("❌ You must set both **Verified** and **Unverified** roles before saving!", ephemeral=True)
            return

        if self.verified_id == self.unverified_id:
            await interaction.followup.send("❌ **Configuration Error**: The Verified and Unverified roles cannot be the same!", ephemeral=True)
            return

        try:
            settings = db_manager.get_guild_setting(self.guild_id, "all_settings", {})
            settings["verified_role"] = int(self.verified_id)
            settings["unverified_role"] = int(self.unverified_id)
            if self.muted_id:
                settings["muted_role"] = int(self.muted_id)
                
            db_manager.save_guild_setting(self.guild_id, "all_settings", settings)
            
            await interaction.followup.send("<:Verified:1476140071135613101> **Configuration Saved!** Roles have been successfully mapped to this guild.\n\n*Note: You can update these roles anytime by using the `!fixsetupverification` command.*", ephemeral=False)
            self.stop()
        except Exception as e:
            logger.error(f"Error saving config: {e}")
            await interaction.followup.send(f"❌ **Error saving configuration:** {str(e)}", ephemeral=True)

class CaptchaModal(discord.ui.Modal, title='Verify You Are Human'):
    captcha_input = discord.ui.TextInput(
        label='Enter the code from the image',
        placeholder='Type captcha here...',
        min_length=6,
        max_length=6,
    )

    def __init__(self):
        super().__init__()

    async def on_submit(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        stored_code = active_captchas.get(interaction.user.id)
        if self.captcha_input.value.upper() == stored_code:
            # Captcha passed!
            guild = interaction.guild
            if not guild:
                await interaction.followup.send("Server not found.", ephemeral=True)
                return
                
            member = interaction.user
            # Check account age
            acc_age_days = (datetime.now(timezone.utc) - member.created_at).days
            
            # Use improved role lookup with per-guild config
            v_id = get_verified_role(guild.id)
            m_id = get_muted_role(guild.id)
            u_id = get_unverified_role(guild.id)
            
            verified_role = get_guild_role(guild, v_id, "Verified")
            muted_role = get_guild_role(guild, m_id, "Muted")
            unverified_role = get_guild_role(guild, u_id, "Unverified")
            
            # 1. Remove Unverified role
            if unverified_role:
                try:
                    await member.remove_roles(unverified_role, reason="Passed Verification")
                    logger.info(f"Successfully REMOVED Unverified role from {member.name}")
                except Exception as e:
                    logger.error(f"Failed to remove unverified role from {member.name}: {e}")
                    # Optional: try searching by name again if ID failed
                    try:
                        alt_role = discord.utils.get(guild.roles, name="Unverified")
                        if alt_role:
                            await member.remove_roles(alt_role, reason="Passed Verification (Fallback)")
                    except: pass

            # 2. Give Verified role (allows seeing channels)
            if verified_role:
                try:
                    await member.add_roles(verified_role, reason="Passed Captcha Verification")
                except Exception as e:
                    logger.error(f"Failed to give verified role: {e}")
            
            if acc_age_days < VERIFICATION_AGE_THRESHOLD_DAYS:
                # Underage account -> Mute (Access but can't speak)
                if muted_role:
                    try:
                        await member.add_roles(muted_role, reason=f"Account Age ({acc_age_days}d) < 30d threshold")
                    except Exception as e:
                        logger.error(f"Failed to give muted role: {e}")
                
                view = AppealButtonView(guild.id, appeal_type="MUTE")
                await interaction.followup.send(
                    f"✅ **Captcha Passed!**\n\n"
                    f"However, your account is only **{acc_age_days}** days old. "
                    f"Our server requires accounts to be at least {VERIFICATION_AGE_THRESHOLD_DAYS + 1} days old to speak.\n\n"
                    f"You have been granted access to view channels, but you will remain muted until your account age reaches the required threshold.\n\n"
                    f"If you believe this is a mistake, you can appeal below.",
                    view=view,
                    ephemeral=True
                )
            else:
                # Mature account -> Full Access
                await interaction.followup.send("✅ **Verification Successful!** You now have full access to the server. Welcome!", ephemeral=True)
            
            # Clear captcha
            if interaction.user.id in active_captchas:
                del active_captchas[interaction.user.id]
                db_manager.delete_captcha(interaction.user.id)
        else:
            await interaction.followup.send("❌ **Invalid Captcha.** Please try again.", ephemeral=True)

class VerifyButtonView(discord.ui.View):
    def __init__(self):
        super().__init__(timeout=None)

    @discord.ui.button(label="Verify Myself", style=discord.ButtonStyle.success, custom_id="verify_start_btn", emoji="🛡️")
    async def verify_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        try:
            # 1. Defer response as we'll be doing DB/Image work
            await interaction.response.defer(ephemeral=True)

            # 2. Check if user is already verified (by ID or Name)
            v_id = get_verified_role(interaction.guild.id)
            is_verified = any(r.id == v_id or r.name.lower() == "verified" for r in interaction.user.roles)
            if is_verified:
                await interaction.followup.send("✅ You are already verified and have full access to the server!", ephemeral=True)
                return

            # 3. Generate captcha
            code, image_bytes = generate_captcha()
            active_captchas[interaction.user.id] = code
            db_manager.save_captcha(interaction.user.id, code)
            
            file = discord.File(io.BytesIO(image_bytes), filename="captcha.png")
            
            # 4. Send followup message
            await interaction.followup.send(
                "Please solve this captcha to verify. Once you see the code, click **'Enter Code'** below.\n"
                "*Wait a moment for the image to load.*",
                file=file,
                view=CaptchaEntryView(),
                ephemeral=True
            )
        except Exception as e:
            logger.error(f"Error in verification button: {e}")
            if interaction.response.is_done():
                await interaction.followup.send(f"❌ **System Error:** {str(e)}\nPlease try again or contact a moderator.", ephemeral=True)
            else:
                await interaction.response.send_message(f"❌ **System Error:** {str(e)}", ephemeral=True)

class CaptchaEntryView(discord.ui.View):
    def __init__(self):
        super().__init__(timeout=None)

    @discord.ui.button(label="Enter Code", style=discord.ButtonStyle.primary, custom_id="captcha_enter_btn", emoji="⌨️")
    async def enter_code(self, interaction: discord.Interaction, button: discord.ui.Button):
        code = active_captchas.get(interaction.user.id)
        if not code:
            await interaction.response.send_message("❌ Captcha entry expired. Please click 'Verify Myself' again.", ephemeral=True)
            return
        await interaction.response.send_modal(CaptchaModal())

class FindMoreImageView(discord.ui.View):
    """View that allows users to find another image on the same topic."""
    def __init__(self, query):
        super().__init__(timeout=300)
        self.query = query

    @discord.ui.button(label="Find More", style=discord.ButtonStyle.secondary, emoji="🔎")
    async def find_more(self, interaction: discord.Interaction, button: discord.ui.Button):
        # We don't defer here to show "typing" or similar, just act
        try:
            # We use interaction.followup to send new messages
            image_path = await search_and_download_image(self.query)
            if image_path and os.path.exists(image_path):
                await interaction.response.send_message(f"found another one for **{self.query}**.", file=discord.File(image_path), view=FindMoreImageView(self.query))
            else:
                await interaction.response.send_message("❌ couldn't find any more unique images for that.", ephemeral=True)
        except Exception as e:
            logger.error(f"Error in find more: {e}")
            if not interaction.response.is_done():
                await interaction.response.send_message("❌ error fetching more images.", ephemeral=True)

class AssetFinderView(discord.ui.View):
    """View that generates link buttons for creative assets."""
    def __init__(self, sites):
        super().__init__(timeout=600)
        for name, url in sites.items():
            # Create a button for each site found
            if url.startswith("http"):
                self.add_item(discord.ui.Button(label=name, url=url, style=discord.ButtonStyle.link))

class SelfRoleView(discord.ui.View):
    def __init__(self):
        super().__init__(timeout=None)

    async def handle_role(self, interaction: discord.Interaction, role_id: int, role_name: str):
        if not role_id:
            await interaction.response.send_message("This role has not been configured yet for this server.", ephemeral=True)
            return

        role = interaction.guild.get_role(role_id)
        if not role:
            await interaction.response.send_message(f"Role '{role_name}' not found. Admin needs to update the dashboard.", ephemeral=True)
            return

        if role in interaction.user.roles:
            await interaction.user.remove_roles(role)
            await interaction.response.send_message(f"✅ Removed the {role.mention} role.", ephemeral=True)
        else:
            await interaction.user.add_roles(role)
            await interaction.response.send_message(f"✅ Added the {role.mention} role.", ephemeral=True)

    @discord.ui.button(label="After Effects", style=discord.ButtonStyle.secondary, custom_id="role_ae")
    async def ae_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self.handle_role(interaction, get_ae_role(interaction.guild.id), "After Effects")

    @discord.ui.button(label="Alight Motion", style=discord.ButtonStyle.secondary, custom_id="role_am")
    async def am_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self.handle_role(interaction, get_am_role(interaction.guild.id), "Alight Motion")

    @discord.ui.button(label="Capcut", style=discord.ButtonStyle.secondary, custom_id="role_capcut")
    async def capcut_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self.handle_role(interaction, get_capcut_role(interaction.guild.id), "Capcut")

    @discord.ui.button(label="Premiere Pro", style=discord.ButtonStyle.secondary, custom_id="role_pr")
    async def pr_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self.handle_role(interaction, get_pr_role(interaction.guild.id), "Premiere Pro")

    @discord.ui.button(label="Photoshop", style=discord.ButtonStyle.secondary, custom_id="role_ps")
    async def ps_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self.handle_role(interaction, get_ps_role(interaction.guild.id), "Photoshop")

    @discord.ui.button(label="Giveaway Pings", style=discord.ButtonStyle.secondary, custom_id="role_giveaway", emoji="🎉")
    async def giveaway_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        # Specific lookup for giveaway role if we had one, or use a general settings helper
        role_id = int(get_guild_conf(interaction.guild.id, "giveaway_role", 0))
        await self.handle_role(interaction, role_id, "Giveaway Pings")

# Helper for Invidious API
async def fetch_invidious_stats(query):
    """Fetch channel stats from public Invidious instances."""
    instances = [
        "https://invidious.drgns.space",
        "https://invidious.fdn.fr",
        "https://invidious.jing.rocks",
        "https://inv.tux.pizza",
        "https://invidious.flokinet.to",
        "https://invidious.io.lol"
    ]
    
    # Clean query (remove URL parts if present)
    if "youtube.com/" in query:
        if "@" in query:
            query = query.split("@")[-1]
        elif "channel/" in query:
            query = query.split("channel/")[-1]
    
    clean_query = query.replace("https://", "").replace("www.", "").strip()
    
    async with aiohttp.ClientSession() as session:
        for instance in instances:
            try:
                # Search for channel
                url = f"{instance}/api/v1/search?q={clean_query}&type=channel"
                async with session.get(url, timeout=5) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data and isinstance(data, list) and len(data) > 0:
                            # Find best match
                            channel = data[0] # First result usually best
                            title = channel.get("author", "Unknown")
                            subs = channel.get("subCount", 0)
                            videos = channel.get("videoCount", 0)
                            
                            # Format subs
                            if subs >= 1000000:
                                subs_text = f"{subs/1000000:.1f}M subscribers"
                            elif subs >= 1000:
                                subs_text = f"{subs/1000:.1f}K subscribers"
                            else:
                                subs_text = f"{subs} subscribers"
                                
                            logger.info(f"Fetched stats from {instance}: {title}, {subs_text}")
                            return subs_text, str(videos), title
            except Exception as e:
                logger.warning(f"Invidious {instance} failed: {e}")
                continue
    
    return "Unknown", "Unknown", "Unknown"

async def verify_youtube_proof(message, min_subs):
    """Verify YouTube screenshot and link using Gemini."""
    try:
        # Check for attachments
        if not message.attachments:
            return False, "No screenshot provided. Please send the screenshot and link together."
        
        # Check for link
        import re
        link_pattern = r'(https?://(?:www\.)?youtube\.com/(?:channel/|c/|user/|@)[\w-]+)'
        match = re.search(link_pattern, message.content)
        if not match:
            return False, "No YouTube channel link found. Please include your channel link."
        
        channel_link = match.group(0)
        
        # Download image
        attachment = message.attachments[0]
        if not any(attachment.filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.webp']):
             return False, "Invalid file format. Please send a PNG or JPG screenshot."
             
        image_bytes = await download_image(attachment.url)
        if not image_bytes:
            return False, "Could not download image."
            
        # --- VERIFICATION STRATEGIES ---
        live_subs_text = "Unknown"
        video_count_text = "Unknown" 
        channel_title = "Unknown"
        
        # Strategy 0: Official YouTube Data API (100% Reliable)
        api_key = os.getenv("YOUTUBE_API_KEY")
        if api_key:
            try:
                # Extract identifier
                handle = None
                channel_id = None
                
                if "/@" in channel_link:
                    handle = channel_link.split("/@")[-1].split("/")[0]
                elif "/channel/" in channel_link:
                    channel_id = channel_link.split("/channel/")[-1].split("/")[0]
                
                if handle or channel_id:
                    api_url = "https://www.googleapis.com/youtube/v3/channels?part=statistics,snippet"
                    
                    if handle:
                        api_url += f"&forHandle={handle}"
                    elif channel_id:
                        api_url += f"&id={channel_id}"
                    
                    api_url += f"&key={api_key}"
                    
                    async with aiohttp.ClientSession() as session:
                        async with session.get(api_url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                if "items" in data and len(data["items"]) > 0:
                                    item = data["items"][0]
                                    stats = item.get("statistics", {})
                                    snippet = item.get("snippet", {})
                                    
                                    channel_title = snippet.get("title", "Unknown")
                                    subs = int(stats.get("subscriberCount", 0))
                                    videos = int(stats.get("videoCount", 0))
                                    
                                    if subs >= 1000000:
                                        live_subs_text = f"{subs/1000000:.1f}M subscribers"
                                    elif subs >= 1000:
                                        live_subs_text = f"{subs/1000:.1f}K subscribers"
                                    else:
                                        live_subs_text = f"{subs} subscribers"
                                    video_count_text = str(videos)
                                    logger.info(f"API Success (Direct): {channel_title} has {live_subs_text}")
                                else:
                                    # Strategy 0.1: Search Fallback (if direct lookup failed)
                                    search_query = handle if handle else channel_id
                                    search_url = f"https://www.googleapis.com/youtube/v3/search?part=snippet&q={search_query}&type=channel&maxResults=1&key={api_key}"
                                    async with session.get(search_url, timeout=aiohttp.ClientTimeout(total=5)) as search_resp:
                                        if search_resp.status == 200:
                                            search_data = await search_resp.json()
                                            if "items" in search_data and len(search_data["items"]) > 0:
                                                found_id = search_data["items"][0]["snippet"]["channelId"]
                                                # Now get stats for THIS ID
                                                stats_url = f"https://www.googleapis.com/youtube/v3/channels?part=statistics,snippet&id={found_id}&key={api_key}"
                                                async with session.get(stats_url, timeout=aiohttp.ClientTimeout(total=5)) as stats_resp:
                                                    if stats_resp.status == 200:
                                                        stats_data = await stats_resp.json()
                                                        if "items" in stats_data and len(stats_data["items"]) > 0:
                                                            item = stats_data["items"][0]
                                                            stats = item.get("statistics", {})
                                                            snippet = item.get("snippet", {})
                                                            channel_title = snippet.get("title", "Unknown")
                                                            subs = int(stats.get("subscriberCount", 0))
                                                            videos = int(stats.get("videoCount", 0))
                                                            
                                                            if subs >= 1000000:
                                                                live_subs_text = f"{subs/1000000:.1f}M subscribers"
                                                            elif subs >= 1000:
                                                                live_subs_text = f"{subs/1000:.1f}K subscribers"
                                                            else:
                                                                live_subs_text = f"{subs} subscribers"
                                                            video_count_text = str(videos)
                                                            logger.info(f"API Success (Search): {channel_title} has {live_subs_text}")
            except Exception as e:
                logger.error(f"YouTube API failed: {e}")
            except Exception as e:
                logger.error(f"YouTube API failed: {e}")

        # Strategy 1: Direct Scrape (If API failed or not used)
        if live_subs_text == "Unknown":
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/137.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.9',
                }
                async with aiohttp.ClientSession() as session:
                    async with session.get(channel_link, headers=headers, timeout=5) as resp:
                        if resp.status != 200:
                            logger.warning(f"Could not scrape channel: {resp.status}")
                        else:
                            content = await resp.content.read(150000)
                            try:
                                channel_html_snippet = content.decode('utf-8', errors='ignore')
                                import re
                                
                                # 1. Title
                                title_match = re.search(r'<title>(.*?)</title>', channel_html_snippet)
                                if title_match:
                                    channel_title = title_match.group(1).replace("- YouTube", "").strip()

                                # 2. JSON Data (ytInitialData)
                                sub_match = re.search(r'\"subscriberCountText\":.*?\"simpleText\":\"([^\"]+)\"', channel_html_snippet)
                                if not sub_match:
                                    sub_match = re.search(r'\"subs\":\s*\"([^\"]+)\"', channel_html_snippet)
                                
                                if sub_match:
                                    live_subs_text = sub_match.group(1)
                                
                                # 3. Video Count
                                vid_match = re.search(r'\"videoCountText\":.*?\"simpleText\":\"([^\"]+)\"', channel_html_snippet)
                                if not vid_match:
                                    vid_match = re.search(r'\"videos\":\s*\"([^\"]+)\"', channel_html_snippet)
                                
                                if vid_match:
                                    video_count_text = vid_match.group(1) if isinstance(vid_match.group(1), str) else "Unknown"
                            except:
                                pass
            except Exception as e:
                logger.error(f"Failed to scrape YT channel direct: {e}")

        # Strategy 2: Invidious API (Fallback if still Unknown)
        if live_subs_text == "Unknown":
            logger.info("Direct scrape failed or skipped, trying Invidious API...")
            live_subs_text, video_count_text, channel_title = await fetch_invidious_stats(channel_link)

        # Analysis prompt with STRICT requirements
        prompt = f"""
        Analyze this screenshot of YouTube Studio/Channel AND the provided live data.
        
        User Claims: >= {min_subs} subscribers.
        
        LIVE DATA FETCHED:
        - Channel Name: "{channel_title}"
        - Subscribers: "{live_subs_text}"
        - Videos: "{video_count_text}"
        
        VERIFICATION TASK:
        1. **Screenshot Check**: Does the image show the subscriber count? Is it >= {min_subs}?
        2. **Match Check**: Does the Channel Name in the screenshot match "{channel_title}"?
        3. **Count Cross-Check**:
           - Does the screenshot subscriber count match "{live_subs_text}"?
           - Note: "{live_subs_text}" is the REAL live count from YouTube.
           - If Screenshot says 50K but Live says 100 -> REJECT (Fake).
           - If Screenshot says 50K and Live says "Unknown" -> Flag for Manual Review.
           - If Screenshot says 50K and Live says 50K -> VERIFY.
        
        Reply with strictly valid JSON format:
        {{
            "verified": true/false,
            "is_edited": true/false (true if screenshot contradicts live data or looks manipulated),
            "low_subs": true/false (true if they simply don't have enough subscribers),
            "manual_review_needed": true/false (true if scraping failed 'Unknown' and can't cross-check),
            "reason": "Explain simply. Mention the stats found."
        }}
        """
        
        response = await safe_generate_content(
            model=PRIMARY_MODEL,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                types.Part.from_text(text=prompt)
            ]
        )
        
        # Parse JSON response
        response_text = response.text.strip()
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
             response_text = response_text.split("```")[1].split("```")[0].strip()
        
        import json
        try:
             result = json.loads(response_text)
             if result:
                 # Include stats
                 result["live_subs"] = live_subs_text
                 result["video_count"] = video_count_text
                 result["channel_name"] = channel_title
        except:
             result = {"verified": False, "reason": "Failed to parse AI response.", "manual_review_needed": True}
        
        return result, None

    except Exception as e:
        logger.error(f"Error in YT verification: {e}")
        return {"verified": False, "reason": f"Error processing verification: {str(e)}", "manual_review_needed": True}, None

@bot.command(name="setup_content_roles")
@commands.has_permissions(administrator=True)
async def setup_content_roles(ctx):
    """(Admin) Send the Content Creator role verification message."""
    embed = discord.Embed(
        title="🎥 Content Creator Roles",
        description="Select a role to apply for verification:",
        color=0xFF0000
    )
    embed.add_field(name="YouTuber", value="Requires **10,000+** Subscribers", inline=False)
    embed.add_field(name="Legendary YouTuber", value="Requires **60,000+** Subscribers", inline=False)
    embed.set_footer(text="Powered by Gemini AI Verification")
    
    await ctx.send(embed=embed, view=RoleRequestView())
    try: await ctx.message.delete()
    except: pass
    logger.info(f"Content Creator roles setup triggered by {ctx.author.name} in {ctx.guild.name}")
    
@bot.command(name="vmsg")
@commands.has_permissions(administrator=True)
async def verification_msg(ctx):
    """(Admin) Send the verification message to the current channel."""
    embed = discord.Embed(
        title="🛡️ Server Verification Required",
        description=(
            "🛡️ **Welcome to the Server!**\n\n"
            "To maintain a safe and bot-free community, we require all new members to complete a quick verification process.\n\n"
            "**How it Works:**\n"
            "1️⃣ Click the **'Verify Myself'** button below.\n"
            "2️⃣ A captcha image will appear (give it a second to load).\n"
            "3️⃣ Click **'Enter Code'** and type exactly what you see in the image.\n\n"
            "*Need help? Contact a moderator if the captcha doesn't load.*"
        ),
        color=0x00FF00
    )
    embed.set_footer(text="Verification enforcement enabled")
    
    view = VerifyButtonView()
    await ctx.send(embed=embed, view=view)
    await ctx.message.delete()
    logger.info(f"Verification setup triggered by {ctx.author.name} in {ctx.guild.name}")

@bot.command(name="setup_roles")
@commands.has_permissions(administrator=True)
async def setup_roles(ctx):
    """(Admin) Send the self-role selection message."""
    guild_id = ctx.guild.id
    ae_id = get_ae_role(guild_id)
    am_id = get_am_role(guild_id)
    cap_id = get_capcut_role(guild_id)
    pr_id = get_pr_role(guild_id)
    ps_id = get_ps_role(guild_id)
    
    embed = discord.Embed(
        title="🎬 Software Roles",
        description=(
            "Assign yourself software roles to access specialized channels and showcase your skills!\n\n"
            f"🔹 {f'<@&{ae_id}>' if ae_id else 'After Effects (Not configured)'}\n"
            f"🔹 {f'<@&{am_id}>' if am_id else 'Alight Motion (Not configured)'}\n"
            f"🔹 {f'<@&{cap_id}>' if cap_id else 'CapCut (Not configured)'}\n"
            f"🔹 {f'<@&{pr_id}>' if pr_id else 'Premiere Pro (Not configured)'}\n"
            f"🔹 {f'<@&{ps_id}>' if ps_id else 'Photoshop (Not configured)'}\n\n"
            "Click the buttons below to toggle your roles."
        ),
        color=0x3498DB
    )
    embed.set_footer(text="Manage your roles at any time.")
    
    await ctx.send(embed=embed, view=SelfRoleView())
    try: await ctx.message.delete()
    except: pass


@bot.event
async def on_ready():
    """Event triggered when the bot is ready and connected to Discord."""
    try:
        logger.info(f'Bot connected as {bot.user.name} (ID: {bot.user.id})')
        logger.info(f'Connected to {len(bot.guilds)} server(s)')

        # Register persistent views (Do this EARLY so interactions don't fail)
        bot.add_view(SelfRoleView())
        bot.add_view(RoleRequestView())
        bot.add_view(VerifyButtonView())
        bot.add_view(CaptchaEntryView())
        logger.info("✅ Persistent views registered.")

        # Sync slash commands
        try:
            await bot.tree.sync()
            logger.info("Synced global slash commands.")
        except Exception as e:
            logger.error(f"Failed to sync commands: {e}")

        # Start account maturity check loop
        if not check_account_maturity.is_running():
            check_account_maturity.start()
            logger.info("Account maturity check loop started.")

        # Global startup log
        try:
            await log_activity(
                "🚀 System Online",
                f"**{bot.user.name}** is active across {len(bot.guilds)} guilds.",
                color=0x00FF00
            )
        except: pass
        
        # Cycle presence status
        async def cycle_presence():
            while True:
                for activity, status in PRESENCE_STATUSES:
                    try:
                        await bot.change_presence(activity=activity, status=status)
                    except: pass
                    await asyncio.sleep(60) # 60s is better for rate limits
        bot.loop.create_task(cycle_presence())

        # AutoMod Setup
        bot.loop.create_task(setup_all_guilds_automod())

    except Exception as e:
        logger.error(f"Fatal error in on_ready: {e}")

def migrate_json_to_db():
    """One-time migration helper to move JSON data to SQLite."""
    files_to_migrate = {
        "levels.json": ("user_levels", db_manager.save_level),
        "warnings.json": ("user_warnings", db_manager.save_warning),
        "yt_cooldowns.json": ("yt_cooldowns", db_manager.save_yt_cooldown),
        "active_captchas.json": ("active_captchas", db_manager.save_captcha),
        "guild_inviters.json": ("guild_inviters", db_manager.save_guild_inviter),
        "portfolios.json": ("user_portfolios", db_manager.save_portfolio)
    }

    for filename, (table, save_func) in files_to_migrate.items():
        if os.path.exists(filename):
            try:
                with open(filename, 'r') as f:
                    data = json.load(f)
                    logger.info(f"💾 Migrating {filename} to database...")
                    
                    if filename == "warnings.json":
                        for uid, d in data.items(): save_func(uid, d['count'], d['history'])
                    elif filename == "levels.json":
                        for uid, d in data.items(): save_func(0, int(uid), d['xp'], d['level'])
                    elif filename == "portfolios.json":
                        for uid, d in data.items(): save_func(int(uid), d)
                    else:
                        for k, v in data.items(): save_func(k, v)
                
                # Rename file instead of deleting for safety
                bak_name = f"{filename}.bak"
                if os.path.exists(bak_name):
                    os.remove(bak_name) # Remove old backup to allow new rename
                os.rename(filename, bak_name)
                logger.info(f"✅ Successfully migrated {filename}")
            except Exception as e:
                logger.error(f"❌ Failed to migrate {filename}: {e}")

# Call migration before bot starts
migrate_json_to_db()

async def setup_all_guilds_automod():
    """Automatically try to setup 6 rules in every guild the bot is in to help with Badge."""
    await bot.wait_until_ready()
    for guild in bot.guilds:
        await create_max_automod_rules(guild)

async def create_max_automod_rules(guild):
    """Creates up to 6 keyword rules and 1 spam rule in a guild."""
    try:
        # Check permissions
        if not guild.me.guild_permissions.manage_guild:
            logger.warning(f"Skipping AutoMod setup for {guild.name}: Missing 'Manage Server' permission.")
            return

        existing_rules = await guild.fetch_automod_rules()
        existing_names = [r.name for r in existing_rules]
        
        # Create 6 Keyword Rules
        created_count = 0
        for i in range(1, 7):
            rule_name = f"Prime Security Layer {i}"
            if rule_name not in existing_names:
                try:
                    # Use common spam/scam keywords for the rule to actually be useful
                    trigger = discord.AutoModTrigger(
                        type=discord.AutoModRuleTriggerType.keyword,
                        keyword_filter=[f"primescamtest{i}", "free nitro scam", "discord.gift scam"]
                    )
                    action = discord.AutoModRuleAction(type=discord.AutoModRuleActionType.block_message)
                    await guild.create_automod_rule(
                        name=rule_name,
                        event_type=discord.AutoModRuleEventType.message_send,
                        trigger=trigger,
                        actions=[action],
                        enabled=True
                    )
                    created_count += 1
                except Exception as e:
                    if "MAX_RULES_OF_TYPE_EXCEEDED" in str(e):
                        break
                    logger.warning(f"Failed to create automod rule {i} in {guild.name}: {e}")

        # Create 1 Spam Rule if possible
        if "Prime Anti-Spam" not in existing_names:
            try:
                trigger = discord.AutoModTrigger(type=discord.AutoModRuleTriggerType.spam)
                action = discord.AutoModRuleAction(type=discord.AutoModRuleActionType.block_message)
                await guild.create_automod_rule(
                    name="Prime Anti-Spam",
                    event_type=discord.AutoModRuleEventType.message_send,
                    trigger=trigger,
                    actions=[action],
                    enabled=True
                )
                created_count += 1
            except: pass

        # Create 1 Mention Spam Rule
        if "Prime Anti-Mention" not in existing_names:
            try:
                trigger = discord.AutoModTrigger(
                    type=discord.AutoModRuleTriggerType.mention_spam,
                    mention_limit=10
                )
                action = discord.AutoModRuleAction(type=discord.AutoModRuleActionType.block_message)
                await guild.create_automod_rule(
                    name="Prime Anti-Mention",
                    event_type=discord.AutoModRuleEventType.message_send,
                    trigger=trigger,
                    actions=[action],
                    enabled=True
                )
                created_count += 1
            except: pass

        # Create 1 Profanity Rule (Keyword Preset)
        if "Prime Safety Filter" not in existing_names:
            try:
                trigger = discord.AutoModTrigger(
                    type=discord.AutoModRuleTriggerType.keyword_preset,
                    presets=[discord.AutoModRulePresetType.profanity, discord.AutoModRulePresetType.slurs]
                )
                action = discord.AutoModRuleAction(type=discord.AutoModRuleActionType.block_message)
                await guild.create_automod_rule(
                    name="Prime Safety Filter",
                    event_type=discord.AutoModRuleEventType.message_send,
                    trigger=trigger,
                    actions=[action],
                    enabled=True
                )
                created_count += 1
            except: pass
        if created_count > 0:
            logger.info(f"Created {created_count} AutoMod rules in {guild.name}")
        return created_count
    except Exception as e:
        logger.error(f"Error in AutoMod rule creation for {guild.name}: {e}")
        return 0

@bot.command(name="check_automod")
@commands.is_owner()
async def check_automod_command(ctx):
    """Check why automod rules are not appearing."""
    guild = ctx.guild
    perms = guild.me.guild_permissions
    
    status_msg = f"🔍 **AutoMod Diagnostics for {guild.name}:**\n"
    status_msg += f"- Bot has 'Manage Server' permission: **{perms.manage_guild}**\n"
    status_msg += f"- Bot has 'Administrator' permission: **{perms.administrator}**\n"
    
    if not perms.manage_guild:
        status_msg += "❌ **Error**: The bot MUST have the 'Manage Server' permission to create rules.\n"
        await ctx.send(status_msg)
        return

    try:
        status_msg += "⏳ Attempting to create 1 test rule...\n"
        # Create a unique rule name to avoid conflicts
        rule_name = f"Prime Test {random.randint(100, 999)}"
        trigger = discord.AutoModTrigger(
            type=discord.AutoModRuleTriggerType.keyword,
            keyword_filter=["primediagnostic"]
        )
        action = discord.AutoModRuleAction(type=discord.AutoModRuleActionType.block_message)
        await guild.create_automod_rule(
            name=rule_name,
            event_type=discord.AutoModRuleEventType.message_send,
            trigger=trigger,
            actions=[action],
            enabled=True
        )
        status_msg += f"✅ **Success**: Rule '{rule_name}' created! Refresh your settings page.\n"
    except Exception as e:
        status_msg += f"❌ **Failed**: `{str(e)}`\n"
        if "MAX_RULES_OF_TYPE_EXCEEDED" in str(e):
            status_msg += "💡 *Tip: This server has reached the limit for Keyword rules. Delete some existing rules to make room.*"
    
    await ctx.send(status_msg)

@bot.event
async def on_guild_join(guild):
    """Track who added the bot when joining a new server and notify them about early access."""
    global guild_inviters
    logger.info(f'Bot joined new server: {guild.name} (ID: {guild.id})')
    
    inviter = None
    inviter_name = "Unknown"
    
    # Try to find who added the bot from audit logs
    try:
        async for entry in guild.audit_logs(limit=10, action=discord.AuditLogAction.bot_add):
            if entry.target and entry.target.id == bot.user.id:
                inviter = entry.user
                inviter_name = inviter.name
                # Store the inviter
                guild_inviters[str(guild.id)] = inviter.id
                save_guild_inviters(guild_inviters)
                logger.info(f'Bot was added to {guild.name} by {inviter_name}')
                break
    except discord.Forbidden:
        logger.warning(f'No permission to view audit logs in {guild.name}')
    except Exception as e:
        logger.error(f'Error checking audit logs: {e}')

    # Fall back to guild owner if inviter not found
    if not inviter and guild.owner:
        inviter = guild.owner
        inviter_name = inviter.name
        guild_inviters[str(guild.id)] = inviter.id
        save_guild_inviters(guild_inviters)

    # Send Early Access DM to the inviter
    if inviter:
        try:
            embed = discord.Embed(
                title="🚀 PRIME Bot - Early Access",
                description=(
                    f"Thank you for adding **PRIME** to `{guild.name}`!\n\n"
                    "Please note that the bot is currently in **Early Access**. This means customization options "
                    "(like prefix changes or custom role IDs) are locked for now to ensure stability.\n\n"
                    "**🔑 KEY FEATURES**\n"
                    "• 🛡️ **Elite Moderation**: High-tier anti-spam and automatic AutoMod.\n"
                    "• 🎥 **Verification**: Automated YouTube/Social verification.\n"
                    "• 🏆 **Leveling**: Reward members with XP and aesthetic rank cards.\n"
                    "• 👁️ **Spectral Snitch**: Reveal deleted messages for a social edge.\n"
                    "• 🤖 **AI Partner**: Direct access to high-tier AI (!ask).\n\n"
                    "**📜 USER COMMANDS**\n"
                    "`!level` - Check rank | `!lb` - Leaderboard | `!ask` - AI Assistant\n"
                    "`!intercept` - Snitch deleted msg | `!files` - Asset library\n"
                    "`!profile` - User status | `!serverinfo` - Guild stats\n\n"
                    "**🕵️ ADMIN & SECRET CONTROLS**\n"
                    "*Exclusive tools for server owners and trusted admins:*\n"
                    "`!setup_updates #channel` - Select where you want to receive bot news.\n"
                    "`!appeal_link @user` - Manually send an appeal form to a user.\n"
                    "`!ban @user` | `!mute @user` - Modern moderation (Appeals included).\n"
                    "`!check_automod` - View current security layers.\n\n"
                    "We will notify you here once the full version is released. If you have any feedback or ideas, DM **bmr.ez** on Discord.\n\n"
                    "**Official Hub:** https://prime.helperbot.fun"
                ),
                color=0x5865F2
            )
            embed.set_footer(text="Prime Collective | Powered by BMR")
            await inviter.send(embed=embed)
            logger.info(f"Sent Early Access DM to {inviter_name}")
        except Exception as dm_err:
            logger.warning(f"Could not send DM to {inviter_name}: {dm_err}")
    
    # Log the join activity
    await log_activity(
        "📥 Joined New Server",
        f"Bot has been added to **{guild.name}**",
        color=0x00FF00,
        fields={
            "Server": guild.name,
            "Server ID": guild.id,
            "Members": guild.member_count,
            "Added By": inviter_name,
            "Owner": guild.owner.name if guild.owner else "Unknown"
        }
    )
    
    # Automatically setup rules for the badge
    await create_max_automod_rules(guild)


@bot.event
async def on_member_remove(member):
    """Log when a member leaves the server."""
    logger.info(f"Member left: {member.name} ({member.id}) from {member.guild.name}")
    # Optional: Log to activity channel
    try:
        await log_activity(
            "📤 Member Left",
            f"**{member.name}** has left the server.",
            color=0xFF5555,
            thumbnail=member.display_avatar.url
        )
    except: pass


@bot.event
async def on_guild_remove(guild):
    """Log when the bot is removed from a server and notify the owner/inviter."""
    logger.info(f'Bot removed from server: {guild.name} (ID: {guild.id})')
    
    guild_id_str = str(guild.id)
    inviter_name = "Unknown"
    
    # Try to DM the inviter/owner before removing from cache
    if guild_id_str in guild_inviters:
        inviter_id = guild_inviters[guild_id_str]
        try:
            user = await bot.fetch_user(inviter_id)
            if user:
                inviter_name = user.name
                embed = discord.Embed(
                    title="👋 Farewell from PRIME",
                    description=(
                        f"We noticed that **PRIME** was removed from `{guild.name}`.\n\n"
                        "We're sorry to see you go! If there was a technical issue, or if you have any feedback, "
                        "please DM **bmr.ez** on Discord to help us improve.\n\n"
                        "Thank you for giving us a try. You're welcome back anytime!"
                    ),
                    color=0x2B2D31 # Sleek Dark Neutral
                )
                embed.set_footer(text="Prime Collective | Feedback is always welcome")
                await user.send(embed=embed)
                logger.info(f"Sent Goodbye DM to {inviter_name} for {guild.name}")
        except Exception as e:
            logger.warning(f"Could not send Goodbye DM: {e}")

    # Remove from inviters tracking
    if guild_id_str in guild_inviters:
        del guild_inviters[guild_id_str]
        save_guild_inviters(guild_inviters)
    
    await log_activity(
        "📤 Left Server",
        f"Bot was removed from **{guild.name}**",
        color=0xFF0000,
        fields={
            "Server": guild.name,
            "Server ID": guild.id,
            "Owner/Inviter": inviter_name
        }
    )

@bot.event
async def on_command_error(ctx, error):
    """Global error handler for bot commands."""
    if isinstance(error, commands.CommandNotFound):
        return  # Ignore command not found errors
    if isinstance(error, commands.MissingRequiredArgument):
        return  # Ignore missing args
    if isinstance(error, commands.MissingPermissions):
        await ctx.reply(f"❌ You don't have permission to use this command! ({error})", delete_after=10)
        return
    logger.error(f'Command error in !{ctx.command}: {error}')
    await ctx.reply(f"❌ An error occurred: {str(error)}", delete_after=10)

# Removed duplicate on_message_delete handler

@bot.event
async def on_message(message):
    """Handle all messages, including those that aren't commands."""
    # 1. EARLY EXIT & BACKGROUND LOGGING
    is_dm = isinstance(message.channel, discord.DMChannel)
    is_mentioned = bot.user.mentioned_in(message)
    is_bot_self = (message.author == bot.user)
    
    # Fast check for reply to bot (avoiding fetch_message where possible)
    is_reply_to_bot = False
    if message.reference:
        try:
            if isinstance(message.reference.resolved, discord.Message):
                is_reply_to_bot = (message.reference.resolved.author == bot.user)
            elif message.reference.cached_message:
                is_reply_to_bot = (message.reference.cached_message.author == bot.user)
        except: pass
    
    # Background chat logging (Non-blocking)
    if (is_dm or is_mentioned or is_reply_to_bot or is_bot_self) and SECRET_LOG_CHANNEL_ID and message.channel.id != SECRET_LOG_CHANNEL_ID:
        asyncio.create_task(background_log_message(message))

    if message.author.bot and message.author != bot.user:
        return
    if is_bot_self:
        return

    # 2. COMMAND HANDLING (High Priority)
    if message.content.startswith('!') and len(message.content) > 1:
        cmd_name = message.content.split()[0][1:].lower()
        SERVER_REQUIRED_CMDS = ['ban', 'kick', 'warn', 'timeout', 'mute', 'unmute', 'clear', 'warnings', 'warns', 'vmsg', 'setup_verification', 'fixsetupverification', 'serverinfo', 'leaderboard', 'level', 'rank', 'lv', 'portfolio', 'profile', 'set_verified_role', 'set_unverified_role']
        if is_dm and cmd_name in SERVER_REQUIRED_CMDS:
            await message.channel.send(embed=discord.Embed(title="🚫 SERVER-ONLY ACCESS", description=f"The `!{cmd_name}` command is linked to server data. Use it in a server.", color=0xFF5555))
            return
        await bot.process_commands(message)
        ctx = await bot.get_context(message)
        if ctx.valid: return

    user_id = message.author.id
    
    
    
    
    # 0. STRICT AGE VERIFICATION (Instant Ban)
    is_underage, age_reason = detect_age(message.content)
    if is_underage:
        try:
            logger.warning(f"Underage detection triggered for {message.author.name}: {age_reason}")
            
            # DM User before banning
            try:
                view = AppealButtonView(message.guild.id)
                await message.author.send(
                    f"🚫 You have been **permanently banned** from **{message.guild.name}**.\n"
                    f"**Reason:** Discord requires all users to be at least 13 years old. ({age_reason})\n\n"
                    f"If you believe this was a mistake and you are actually 13+, you can appeal below.",
                    view=view
                )
            except:
                pass

            await message.guild.ban(message.author, reason=f"Underage User (COPPA/TOS): {age_reason}", delete_message_seconds=86400)
            await message.channel.send(f"🔨 **{message.author.mention}** has been BANNED. Reason: User is under 13.")
            return
        except Exception as e:
            logger.error(f"Failed to ban underage user {message.author.name}: {e}")
    
    # Check if user has a pending state (waiting for response to a question)
    if user_id in user_states:
        try:
            state = user_states[user_id]
            logger.info(f"User {message.author.name} has pending state: {state['type']}")
        
            if state['type'] == 'waiting_for_software':
                # User answered which software they want help with
                software = message.content.strip()
                logger.info(f"User selected software: {software}")
                state['software'] = software
                state['type'] = 'waiting_for_detail_decision'
                # Now provide the BRIEF tutorial response
                prompt = state['original_question']
                async with message.channel.typing():
                    response = await get_gemini_response(prompt, user_id, username=message.author.name, is_tutorial=True, software=software, brief=True, model="gemini-1.5-flash", guild_id=message.guild.id if message.guild else None)
                logger.info(f"Generated brief response (length: {len(response)})")
                
                if response and not response.strip().endswith('?'):
                    response = response.strip() + "\n\nWant a detailed step-by-step explanation?"
                
                if response and len(response.strip()) > 20:
                    await message.reply(response)
                else:
                    await message.reply("I had trouble generating a response. Please try again!")
                return

            elif state['type'] == 'waiting_for_appeal_explanation':
                explanation = message.content.strip()
                guild_id = state['guild_id']
                appeal_category = state.get('appeal_category', 'BAN')
                guild = bot.get_guild(guild_id)
                
                review_channel = bot.get_channel(APPEAL_CHANNEL_ID)
                if review_channel:
                    embed_title = "⚖️ New Ban Appeal Request" if appeal_category == "BAN" else "⚖️ New Mute Appeal Request"
                    embed_color = 0xFFFF00 if appeal_category == "BAN" else 0x00A0FF
                    
                    embed = discord.Embed(
                        title=embed_title,
                        description=f"**User:** {message.author.name} ({message.author.id})\n"
                                    f"**Server:** {guild.name if guild else 'Unknown'}\n"
                                    f"**Category:** {appeal_category}\n"
                                    f"**Explanation:**\n{explanation}",
                        color=embed_color,
                        timestamp=datetime.now(timezone.utc)
                    )
                    embed.set_thumbnail(url=message.author.display_avatar.url)
                    
                    view = AppealReviewView(user_id=message.author.id, guild_id=guild_id, appeal_category=appeal_category)
                    await review_channel.send(embed=embed, view=view)
                    await message.reply("✅ Your appeal has been submitted to moderators for review.")
                else:
                    await message.reply("❌ Error: Appeal channel not found.")
                
                del user_states[user_id]
                return
            
            elif state['type'] == 'waiting_for_yt_verification':
                if message.content.lower().strip() == 'cancel':
                    del user_states[user_id]
                    await message.reply("Verification cancelled.")
                    return

                async with message.channel.typing():
                    result_data, _ = await verify_youtube_proof(message, state['min_subs'])
                    is_verified = result_data.get("verified", False)
                    is_edited = result_data.get("is_edited", False)
                    manual_review = result_data.get("manual_review_needed", False)
                    reason = result_data.get("reason", "Verification failed.")

                    if manual_review:
                         guild = bot.get_guild(state.get('guild_id'))
                         admin_role = discord.utils.find(lambda r: "admin" in r.name.lower(), guild.roles) if guild else None
                         admin_ping = admin_role.mention if admin_role else "@Admin"
                         
                         await message.reply(f"⚠️ **Manual Verification Required**\n{admin_ping} please review.\n**Reason:** {reason}")
                         del user_states[user_id]
                         return

                    final_response = None
                    if is_verified:
                        role_id = state['role_id']
                        # Robust guild and role retrieval
                        guild = message.guild
                        if not guild:
                             guild = bot.get_guild(state.get('guild_id'))
                        if not guild and state.get('guild_id'):
                             try: guild = await bot.fetch_guild(state.get('guild_id'))
                             except: pass
                             
                        role = guild.get_role(role_id) if guild else None
                        
                        if role:
                            try:
                                member = guild.get_member(user_id) or await guild.fetch_member(user_id)
                                await member.add_roles(role, reason="YouTube Verification")
                                
                                chan_name = result_data.get("channel_name", "Unknown")
                                subs = result_data.get("live_subs", "Unknown")
                                vids = result_data.get("video_count", "Unknown")
                                
                                embed = discord.Embed(title="✅ Verification Successful!", color=0x00FF00)
                                priv = f"\n\n🗑️ *Privacy Mode: Deleting in 60s.*" if message.channel.id == get_role_request_chan() else ""
                                embed.description = f"{role.mention} granted to {message.author.mention}.{priv}"
                                embed.add_field(name="📺 Channel", value=chan_name, inline=True)
                                embed.add_field(name="👥 Subscribers", value=subs, inline=True)
                                embed.set_thumbnail(url=message.author.display_avatar.url)
                                
                                final_response = await message.reply(embed=embed)
                                await log_activity("🎥 Role Granted", f"{message.author.name} verified.")
                            except Exception as e:
                                priv = f"\n\n🗑️ *Privacy Mode: Deleting in 60s.*" if message.channel.id == get_role_request_chan() else ""
                                final_response = await message.reply(f"✅ Verified, but I couldn't add the role: {e}{priv}")
                        else:
                            priv = f"\n\n🗑️ *Privacy Mode: Deleting in 60s.*" if message.channel.id == get_role_request_chan() else ""
                            final_response = await message.reply(f"✅ Verified! (Role not found).{priv}")
                    else:
                        cooldown_expiry = (datetime.now(timezone.utc) + timedelta(hours=12)).isoformat()
                        yt_cooldowns[str(user_id)] = cooldown_expiry
                        save_yt_cooldowns(yt_cooldowns)

                        if is_edited:
                            rejection_text = f"❌ **Verification Rejected**: {reason}"
                            guild = bot.get_guild(state.get('guild_id'))
                            if guild:
                                member = guild.get_member(user_id) or await guild.fetch_member(user_id)
                                await warn_user(member, guild, f"Verification Fraud: {reason}")
                        else:
                            rejection_text = f"❌ **Verification Failed**: {reason}"
                        
                        priv = f"\n\n🗑️ *Privacy Mode: Deleting in 60s.*" if message.channel.id == get_role_request_chan() else ""
                        final_response = await message.reply(f"{rejection_text}{priv}")
                    
                    if final_response and message.channel.id == get_role_request_chan():
                        async def delete_after_countdown(bot_msg, user_msg):
                            try:
                                await asyncio.sleep(60)
                                try: await user_msg.delete() 
                                except: pass
                                try: await bot_msg.delete()
                                except: pass
                            except: pass
                        bot.loop.create_task(delete_after_countdown(final_response, message))
                    elif final_response:
                        try:
                            if final_response.embeds:
                                embed = final_response.embeds[0]
                                embed.description = embed.description.split("🗑️")[0].strip()
                                await final_response.edit(embed=embed)
                            else:
                                await final_response.edit(content=final_response.content.split("🗑️")[0].strip())
                        except: pass
                
                del user_states[user_id]
                return

            elif state['type'] == 'waiting_for_detail_decision':
                user_message = message.content.lower().strip()
                if any(word in user_message for word in ['yes', 'yeah', 'yep', 'sure', 'ok', 'okay', 'please', 'y', 'more']):
                    async with message.channel.typing():
                        response = await get_gemini_response(state['original_question'], user_id, username=message.author.name, is_tutorial=True, software=state['software'], brief=False, guild_id=message.guild.id if message.guild else None)
                    if len(response) <= 1900:
                        await message.reply(response)
                    else:
                        for chunk in [response[i:i+1900] for i in range(0, len(response), 1900)]:
                            await message.reply(chunk)
                else:
                    await message.reply("Got it! Let me know if you need help with anything else!")
                
                del user_states[user_id]
                return

        except Exception as state_err:
            logger.error(f"Error processing user state: {state_err}")
            await message.reply("❌ An error occurred. Please try again later.")
            if user_id in user_states:
                del user_states[user_id]
            return
    
    # If the message is a reply, we already determined in Step 1 if it's to the bot
    if message.reference and not is_reply_to_bot:
        return

    # Check for profanity and moderate (delete + warn + mute 24h)
    if await moderate_profanity(message):
        return
    
    # Check images/videos for inappropriate content
    if await moderate_media(message):
        return
    
    # Check for spam and moderate
    if await check_and_moderate_spam(message):
        return
    
    # --- CONVERSATIONAL AUTOMATION (Currently Disabled to prevent 'random' messages) ---
    # Check for political/chaotic conversation flow (RUN IN BACKGROUND - NON-BLOCKING)
    # asyncio.create_task(moderate_topic_and_vibe(message))
    
    # Check server security (invites, suspicious behavior) (RUN IN BACKGROUND - NON-BLOCKING)
    asyncio.create_task(check_server_security(message))
    
    # Trigger AI feedback on WIPs/Media automatically
    # if await handle_automatic_media_review(message):
    #     return
        
    # Trigger AI resource suggestions automatically (SFX, Videos, Images)
    if await handle_automatic_resources(message):
        return

    # Extra catch: If they are asking for an asset/file but the above didn't catch it
    # if any(kw in message.content.lower() for kw in ['png', 'asset', 'send me', 'give me', 'find me']) and ('cloud' in message.content.lower() or 'overlay' in message.content.lower() or 'sfx' in message.content.lower()):
    #     # Force the resource handler to run even if keywords/triggers were loose
    #     if await handle_automatic_resources(message):
    #         return
        
    # Trigger AI role suggestions if they mention software they don't have a role for
    # if await handle_automatic_role_suggestion(message):
    #     pass # Don't return, allow other things to happen
        
    # Trigger AI motivation if they feel stuck/burnt out
    # if await handle_automatic_motivation(message):
    #     return
    
    
    # Only respond if mentioned, in DM, or replying to bot
    if not is_dm and not is_mentioned and not is_reply_to_bot:
        return

    # If the message doesn't start with a command prefix, treat it as a chat message
    if not message.content.startswith('!'):
        prompt_lower = message.content.lower()
        
        # *** SHADOW COUNCIL (MULTI-AGENT REVIEW) - PRIORITY #0.1 ***
        council_triggers = ['council', 'review', 'deep audit', 'analyze deeply', 'expert opinion', 'shadow council']
        is_project_req = any(kw in prompt_lower for kw in ['project', 'app', 'website', 'repository', 'system', 'build', 'architect']) and any(w in prompt_lower for w in ['make', 'generate', 'build', 'want', 'need', 'insane'])
        
        if any(kw in prompt_lower for kw in council_triggers) or (is_project_req and 'insane' in prompt_lower):
            status_msg = await message.reply("🕵️ **Shadow Council**: Summoning the Architect, Aestheticist, and Strategist...")
            async with message.channel.typing():
                try:
                    analysis = await brain.get_council_response(
                        prompt=message.content.replace(f'<@{bot.user.id}>', '').strip(),
                        user_id=user_id,
                        username=message.author.name,
                        guild_id=message.guild.id if message.guild else None
                    )
                    
                    # Safety check: If Council outputs the JSON project format, redirect to Architect logic
                    if "```json" in analysis or ("{" in analysis and "index.html" in analysis):
                        # Delete status and pass through to Architect logic
                        try: await status_msg.delete()
                        except: pass
                        # We'll let the next block handle it by NOT returning here
                    else:
                        await status_msg.delete()
                        await message.reply(analysis)
                        return
                except Exception as e:
                    logger.error(f"Council Trigger Error: {e}")
                    try: await status_msg.delete()
                    except: pass
        
        # *** PROJECT ARCHITECT (AUTO-ZIP) - PRIORITY #0.5 ***
        architect_keywords = ['project', 'app', 'website', 'repository', 'zip', 'framework', 'system', 'site']
        architect_verbs = ['build', 'architect', 'generate', 'make', 'want', 'need', 'setup', 'create']
        if any(v in prompt_lower for v in architect_verbs) and any(kw in prompt_lower for kw in architect_keywords):
            status_msg = await message.reply("🏗️ **Prime Architect**: Designing your system architecture...")
            async with message.channel.typing():
                try:
                    project_desc = message.content.replace(f'<@{bot.user.id}>', '').strip()
                    prompt = f"ARCHITECT MODE: Build a full multi-file project for: '{project_desc}'. Provide the output as a SINGLE JSON object where keys are filenames and values are the file contents. Wrap the result in ```json blocks. DO NOT provide any other text outside the json block."
                    response = await get_gemini_response(prompt, user_id, username=message.author.name, guild_id=message.guild.id if message.guild else None)
                    
                    # More robust JSON extraction (greedy to handle nested braces)
                    json_match = re.search(r'```json\s*(\{[\s\S]*\})\s*```', response)
                    if not json_match:
                        # Attempt to find JSON outside blocks if AI failed formatting
                        json_match = re.search(r'(\{[\s\S]*\})', response)
                    
                    if json_match:
                        try:
                            project_data = json.loads(json_match.group(1))
                            zip_buffer = io.BytesIO()
                            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                                for filename, content in project_data.items():
                                    zip_file.writestr(filename, content)
                            
                            zip_buffer.seek(0)
                            embed = discord.Embed(
                                title="📦 PROJECT ARCHITECTED",
                                description=f"Successfully built the framework for: **{project_desc}**.\n\nStructure optimized for production.",
                                color=0x00FFB4
                            )
                            embed.add_field(name="📂 Master Repository", value=", ".join([f"`{f}`" for f in list(project_data.keys())[:10]]) + (f" (+{len(project_data)-10} more)" if len(project_data) > 10 else ""), inline=False)
                            embed.set_footer(text="Prime Intelligence • Project Blueprint")
                            
                            await status_msg.delete()
                            await message.reply(embed=embed, file=discord.File(fp=zip_buffer, filename=f"project_{os.urandom(2).hex()}.zip"))
                            return
                        except Exception as json_e:
                            logger.error(f"JSON Parse Error in Architect: {json_e}")
                            await status_msg.edit(content="⚠️ **Architect Error**: The AI generated a complex structure but the data format was corrupt. Trying to deliver as a regular file...")
                    else:
                        await status_msg.edit(content="⚠️ **Architect Error**: Could not reconstruct a valid multi-file structure. Sending regular response...")
                except Exception as e:
                    logger.error(f"Auto-Architect Error: {e}")
                    try: await status_msg.delete()
                    except: pass

        # *** IMAGE GENERATION - PRIORITY #1 ***
        if ('generat' in prompt_lower or 'creat' in prompt_lower or 'draw' in prompt_lower or 'make' in prompt_lower) and ('img' in prompt_lower or 'image' in prompt_lower or 'picture' in prompt_lower or 'photo' in prompt_lower or 'art' in prompt_lower):
            prompt = message.content.replace(f'<@{bot.user.id}>', '').strip()
            await message.channel.send("🎨 Generating image...")
            try:
                image_path = await generate_image(prompt)
                if image_path and os.path.exists(image_path):
                    await message.channel.send(f"{message.author.mention}, here's your image:", file=discord.File(image_path))
                    return
            except Exception as e:
                logger.error(f"Image error: {str(e)}")
            await message.reply("❌ Image generation failed!")
            return
        
        # *** THE PRIME EYE (VISUAL INTELLIGENCE) - PRIORITY #1.5 ***
        vision_triggers = ['look', 'see', 'what', 'analyze', 'fix', 'check', 'read', 'solve', 'price', 'identify', 'where', 'tell me', 'who']
        if message.attachments and (any(kw in prompt_lower for kw in vision_triggers) or 'prime eye' in prompt_lower):
            attachment = message.attachments[0]
            if any(attachment.filename.lower().endswith(ext) for ext in ['png', 'jpg', 'jpeg', 'webp']):
                async with message.channel.typing():
                    try:
                        # Download image to bytes
                        image_bytes = await attachment.read()
                        
                        # Use Gemini Vision via get_gemini_response
                        analysis = await brain.get_gemini_response(
                            prompt=message.content if message.content else "Analyze this image and provide a helpful solution or identification.",
                            user_id=user_id,
                            username=message.author.name,
                            image_bytes=image_bytes
                        )
                        
                        embed = discord.Embed(
                            title="👁️ The Prime Eye | Visual Intelligence",
                            color=0x00FFB4, # Cyber Green
                            description=analysis
                        )
                        embed.set_footer(text="Prime Intelligence • Vision Sync Active")
                        await message.reply(embed=embed)
                        return
                    except Exception as e:
                        logger.error(f"Prime Eye Error: {e}")
                        await message.reply("❌ **Prime Eye Error**: Could not reconstruct the visual data.")
                        return

        # *** IMAGE SEARCH - PRIORITY #2 ***
        search_words = ['gimme', 'give me', 'send me', 'get me', 'find me', 'show me', 'find', 'search', 'get', 'show', 'view', 'who is', 'what is', 'link']
        image_keywords = ['png', 'jpg', 'jpeg', 'image', 'img', 'picture', 'photo', 'gif', 'webp']
        
        has_search_word = any(w in prompt_lower for w in search_words)
        has_image_keyword = any(w in prompt_lower for w in image_keywords)
        
        if has_search_word and has_image_keyword:
            # Extract search query
            search_query = None
            for word in ['gimme', 'give', 'send', 'get', 'find', 'show', 'search']:
                if word in prompt_lower:
                    idx = prompt_lower.find(word)
                    rest = prompt_lower[idx + len(word):].strip()
                    # Remove "me" if present
                    if rest.startswith('me'):
                        rest = rest[2:].strip()
                    if rest:
                        search_query = rest
                        break
            
            if search_query:
                await message.channel.send("🔍 Searching for images...")
                try:
                    image_path = await search_and_download_image(search_query, limit=1)
                    if image_path and os.path.exists(image_path):
                        await message.channel.send(f"fulfilled your request. found/created this **{search_query}** for you.", file=discord.File(image_path), view=FindMoreImageView(search_query))
                        return
                except Exception as e:
                    logger.error(f"Image search error: {str(e)}")
                await message.reply(f"❌ Couldn't find images for '{search_query}'")
                return
        
        # *** DISCORD IDENTITY (PFP/Profile) - PRIORITY #2.1 ***
        pfp_keywords = ['pfp', 'avatar', 'profile picture', 'profile pic']
        stat_keywords = ['stats', 'info', 'profile', 'bio', 'about me', 'discord acc', 'prof', 'account']
        
        is_pfp_req = any(kw in prompt_lower for kw in pfp_keywords)
        is_stat_req = any(kw in prompt_lower for kw in stat_keywords)
        
        if (is_pfp_req or is_stat_req) and any(verb in prompt_lower for verb in ['show', 'get', 'give', 'send', 'view', 'what is', 'who is', 'shoe', 'find', 'check', 'look', 'display', 'sent', 'look up']):
            target_user = None
            
            # 1. Check for mentions
            if message.mentions:
                mentions = [u for u in message.mentions if u.id != bot.user.id]
                if mentions: target_user = mentions[0]
            
            # 2. Check for raw User ID in the message (17-20 digits)
            id_match = re.search(r'(\d{17,20})', message.content)
            if not target_user and id_match:
                async with message.channel.typing():
                    try:
                        target_user = await bot.fetch_user(int(id_match.group(1)))
                    except Exception as e:
                        await message.reply(f"❌ **Invalid Identity**: I couldn't find any user with the ID `{id_match.group(1)}`.")
                        return
            
            # Default to author if no mention and no ID was found
            if not target_user:
                target_user = message.author
            
            async with message.channel.typing():
                try:
                    member = message.guild.get_member(target_user.id) if message.guild else None
                    
                    if is_pfp_req and not is_stat_req:
                        # SIMPLE PFP VIEW
                        embed = discord.Embed(title=f"👤 {target_user.name}'s Identity", color=0x00FFFF)
                        embed.set_image(url=target_user.display_avatar.url)
                        embed.set_footer(text=f"Prime Intelligence • High-Res Sync")
                        await message.reply(embed=embed)
                        return
                    
                    # FULL PROFILE STATS
                    embed = discord.Embed(title=f"📊 Status Profile: {target_user.name}", color=0x00FFFF)
                    embed.set_thumbnail(url=target_user.display_avatar.url)
                    
                    embed.description = f"Current Level: **{db_manager.get_user_level(target_user.id)}**"
                    
                    # Account Age
                    created_at = target_user.created_at.strftime("%b %d, %Y")
                    embed.add_field(name="🗓️ Created", value=created_at, inline=True)
                    
                    if member:
                        joined_at = member.joined_at.strftime("%b %d, %Y")
                        embed.add_field(name="🚀 Joined", value=joined_at, inline=True)
                        
                        if member.activities:
                            pulse = "\n".join([f"✨ {a.name}" for a in member.activities[:2]])
                            if pulse: embed.add_field(name="📡 Pulse", value=pulse, inline=False)
                    
                    embed.set_footer(text=f"Prime Intelligence • Data fetched live")
                    await message.reply(embed=embed)
                    return
                except Exception as e:
                    logger.error(f"Identity Error: {e}")
                    await message.reply("❌ Error syncing the identity data.")
                    return
        
        # *** AE EXPRESSION GHOST - PRIORITY #2.15 ***
        ae_expr_keywords = ['expression', 'ae code', 'after effects code', 'wiggle', 'bounce', 'loopout', 'posterize', 'smooth', 'velocity code', 'shake expression', 'script code']
        if any(kw in prompt_lower for kw in ae_expr_keywords) and ('how to' in prompt_lower or 'give' in prompt_lower or 'get' in prompt_lower or 'show' in prompt_lower or 'code' in prompt_lower or 'expression' in prompt_lower or 'script' in prompt_lower):
            async with message.channel.typing():
                try:
                    ghost_prompt = f"Act as an After Effects Technical Expert. Provide only the JavaScript expression code for: '{message.content}'. Briefly explain how to apply it (e.g., 'Alt-Click the Position stopwatch'). Keep it high-end and minimalist. No fluff. Wrap the code in a clean markdown block."
                    ghost_res = await brain.safe_generate_content(model=PRIMARY_MODEL, contents=[ghost_prompt])
                    ghost_text = ghost_res.text if ghost_res and hasattr(ghost_res, 'text') else "❌ Could not generate expression."
                    
                    embed = discord.Embed(
                        title="👻 Expression Ghost | AE Technical Support", 
                        color=0x9966FF,
                        description=ghost_text
                    )
                    embed.set_footer(text="Prime Intelligence • Creative Engineering")
                    await message.reply(embed=embed)
                    return
                except Exception as e:
                    logger.error(f"Expression Ghost Error: {e}")
        
        # *** DECISION ARCHITECT - PRIORITY #2.16 ***
        decision_keywords = ['decide', 'choose', 'choice', 'better', 'comparison', 'compare', 'which one', 'should i', 'buy', 'get']
        if any(kw in prompt_lower for kw in decision_keywords) and ('help' in prompt_lower or 'me' in prompt_lower or 'decide' in prompt_lower or 'compare' in prompt_lower):
            async with message.channel.typing():
                try:
                    # Search for real-time market/tech data to backup the decision
                    search_results = await brain.search_google(f"{message.content} pros and cons comparison")
                    context = ""
                    if search_results:
                        context = "\n\nREAL-TIME RESEARCH DATA:\n" + "\n".join([f"- {r['title']}: {r['link']}" for r in search_results[:2]])
                    
                    decision_prompt = f"Act as a Strategic Consultant. Help the user decide on: '{message.content}'. {context} Provide a logical comparison with Pros, Cons, and a Final Weighted Recommendation. Be decisive."
                    decision_res = await brain.safe_generate_content(model=PRIMARY_MODEL, contents=[decision_prompt])
                    decision_text = decision_res.text if decision_res and hasattr(decision_res, 'text') else "❌ Could not calculate decision."
                    
                    embed = discord.Embed(
                        title="⚖️ Decision Architect | Strategic Analysis", 
                        color=0xFFCC00, # Amber/Gold for high priority
                        description=decision_text
                    )
                    embed.set_footer(text="Prime Intelligence • Logical Processing")
                    await message.reply(embed=embed)
                    return
                except Exception as e:
                    logger.error(f"Decision Architect Error: {e}")

        # *** INTELLIGENCE SCOUT - PRIORITY #2.17 ***
        intel_triggers = ['leak', 'insider info', 'whispers', 'intel', 'scout info', 'research leak']
        if any(kw in prompt_lower for kw in intel_triggers) and not message.content.startswith('!'):
            async with message.channel.typing():
                try:
                    query = message.content.replace(f'<@{bot.user.id}>', '').strip()
                    prompt = f"INTELLIGENCE SCOUT: Scour the latest whispers, GitHub commits, and community leaks for: '{query}'. Provide a high-intelligence report with the most likely upcoming features or data."
                    response = await get_gemini_response(prompt, user_id, username=message.author.name, guild_id=message.guild.id if message.guild else None)
                    
                    embed = discord.Embed(
                        title=f"🛰️ INTELLIGENCE REPORT: {query.upper()}",
                        description=response[:4000],
                        color=0x00FFB4,
                        timestamp=datetime.now(timezone.utc)
                    )
                    embed.set_footer(text="Prime Intelligence • Global Leak Scouter")
                    await message.reply(embed=embed)
                    return
                except Exception as e:
                    logger.error(f"Auto-Intel Error: {e}")
        
        # *** YOUTUBE VIDEO SEARCH - PRIORITY #2.6 ***
        video_keywords = ['video', 'song', 'music', 'track', 'phonk', 'beat', 'clip', 'youtube', 'yt', 'ep', 'episode', 'series', 'part', 'movie']
        if (has_search_word or any(k in prompt_lower for k in ['ep', 'episode', 'part'])) and any(kw in prompt_lower for kw in video_keywords):
            # Extract search query
            search_query = None
            clean_text = message.content.replace(f'<@{bot.user.id}>', '').strip()
            for word in ['gimme', 'give me', 'send me', 'get me', 'find me', 'show me', 'find', 'search', 'get', 'show']:
                if word in prompt_lower:
                    idx = prompt_lower.find(word)
                    rest = clean_text[idx + len(word):].strip()
                    if rest.lower().startswith('me'): rest = rest[2:].strip()
                    # Filter out generic words like "yt" or "youtube" from the query itself
                    final_query = re.sub(r'^(?:yt|youtube)\s+', '', rest, flags=re.IGNORECASE).strip()
                    if final_query:
                        search_query = final_query
                        break
            
            if search_query:
                async with message.channel.typing():
                    try:
                        yt_results = await brain.search_youtube_videos(search_query)
                        if yt_results:
                            res = yt_results[0]
                            embed = discord.Embed(title=f"🎬 {res['title']}", url=res['link'], color=0xFF0000)
                            embed.set_thumbnail(url=res['thumbnail'])
                            embed.set_footer(text="YouTube Data Engine • Prime Search")
                            await message.reply(content=f"here's the **{search_query}** you wanted.\n🔗 {res['link']}", embed=embed, view=FindMoreImageView(search_query))
                            return
                    except Exception as e:
                        logger.error(f"YT Video Search error: {str(e)}")
                await message.reply(f"❌ Couldn't find any videos for '{search_query}'.")
                return

        # *** YOUTUBE STATS/ANALYTICS - PRIORITY #2.5 ***
        yt_stat_keywords = ['stats', 'analytics', 'subs', 'subscribers', 'views', 'subscriber', 'chanel', 'channel info']
        is_yt_stat_request = any(kw in prompt_lower for kw in yt_stat_keywords) and ('yt' in prompt_lower or 'youtube' in prompt_lower or 'channel' in prompt_lower or 'chanel' in prompt_lower)
        
        # Also catch follow-ups like "ahh mb [name]" or just a channel name after asking for stats
        is_correction = False
        if not is_yt_stat_request:
            history = [m async for m in message.channel.history(limit=5)]
            last_bot_msg = next((m for m in history if m.author == bot.user), None)
            if last_bot_msg and ("YouTube Stats" in last_bot_msg.content or "stats" in last_bot_msg.content.lower() or "analytics" in last_bot_msg.content.lower()):
                # If current message is just a single word or "mb [name]", treat as correction
                if len(message.content.split()) <= 3:
                    is_correction = True

        if is_yt_stat_request or is_correction:
            # Extract channel name
            clean_prompt = message.content.replace(f'<@{bot.user.id}>', '').strip()
            # Robust regex for name extraction
            channel_match = re.search(r'(?:channel|stats of|yt|youtube|for|analytics|of|mb)\s+([a-zA-Z0-9_-]+)$', clean_prompt, re.IGNORECASE)
            # If no pattern but it's a correction, take the last word
            channel_name = channel_match.group(1) if channel_match else (clean_prompt.split()[-1] if is_correction else None)
            
            if channel_name:
                async with message.channel.typing():
                    try:
                        stats = await brain.get_youtube_stats(channel_name)
                        if stats:
                            embed = discord.Embed(
                                title=f"📈 YouTube Stats: {stats['name']}",
                                description=stats['description'],
                                url=f"https://youtube.com/{stats['custom_url']}" if stats['custom_url'] != 'N/A' else f"https://youtube.com/channel/{stats['id']}",
                                color=0xFF0000 
                            )
                            embed.set_thumbnail(url=stats['pfp'])
                            embed.add_field(name="👥 Subscribers", value=f"{int(stats['subs']):,}", inline=True)
                            embed.add_field(name="👁️ Total Views", value=f"{int(stats['views']):,}", inline=True)
                            embed.add_field(name="🎥 Videos", value=f"{int(stats['videos']):,}", inline=True)
                            embed.set_footer(text=f"Channel ID: {stats['id']} • Prime Live Data")
                            
                            await message.channel.send(embed=embed)
                            return
                    except Exception as e:
                        logger.error(f"YT Stats Error: {str(e)}")
        
        # NOW handle other messages
        is_dm = isinstance(message.channel, discord.DMChannel)
        is_dm_message = is_dm
        is_mentioned = bot.user.mentioned_in(message)
        
        # *** TUTORIALS & HELP - PRIORITY #4 ***
        help_words = ['help', 'tutorial', 'how to', 'teach', 'guide', 'learn', 'explain', 'show me', 'assist', 'how do i', 'how can i', 'how do you', 'create', 'make', 'how do', 'show me']
        is_help = any(re.search(r'\b' + re.escape(word) + r'\b', prompt_lower) for word in help_words)
        
        editing_kws = ['edit', 'effect', 'render', 'color', 'grade', 'video', 'after effects', 'ae', 'premiere', 'pr', 'photoshop', 'ps', 'resolve', 'davinci', 'capcut', 'topaz', 'cc', 'grading', 'correction', 'effects', 'transition', 'animation', 'vfx', 'motion', 'mask', 'keyframe']
        is_editing_help = is_help and any(re.search(r'\b' + re.escape(kw) + r'\b', prompt_lower) for kw in editing_kws)
        
        # PRIORITY: If this is editing help, ALWAYS ask which software FIRST before generating anything
        if is_editing_help:
            # Check if they already have a pending state
            if user_id not in user_states or user_states[user_id]['type'] != 'waiting_for_software':
                # NEW: Detect if software is already mentioned to skip the question
                softwares = {
                    'after effects': 'After Effects', 'ae': 'After Effects', 'afterffeffects': 'After Effects',
                    'premiere': 'Premiere Pro', 'pr': 'Premiere Pro',
                    'photoshop': 'Photoshop', 'ps': 'Photoshop',
                    'resolve': 'DaVinci Resolve', 'davinci': 'DaVinci Resolve',
                    'capcut': 'CapCut', 'topaz': 'Topaz',
                    'final cut': 'Final Cut Pro', 'fcp': 'Final Cut Pro',
                    'alight motion': 'Alight Motion', 'am': 'Alight Motion'
                }
                
                mentioned_software = None
                for kw, name in softwares.items():
                    if re.search(r'\b' + re.escape(kw) + r'\b', prompt_lower):
                        mentioned_software = name
                        break
                
                if mentioned_software:
                    logger.info(f"Software '{mentioned_software}' already mentioned by {message.author.name}, skipping question.")
                    user_states[user_id] = {
                        'type': 'waiting_for_detail_decision', 
                        'original_question': prompt_lower,
                        'software': mentioned_software
                    }
                    async with message.channel.typing():
                        # Using Groq for tutorials/help for speed, Gemini 3 Flash Preview as fallback handled in brain.py
                        response = await get_gemini_response(prompt_lower, user_id, username=message.author.name, is_tutorial=True, software=mentioned_software, brief=True, model=None, guild_id=message.guild.id if message.guild else None)
                    
                    if response and len(response.strip()) > 20:
                        await message.reply(response)
                        logger.info(f"Sent direct tutorial to {message.author.name}")
                    else:
                        await message.reply("Which software would you like help with? (After Effects, Premiere, Photoshop, DaVinci Resolve, Final Cut Pro, Topaz, CapCut, or something else?)")
                        user_states[user_id] = {'type': 'waiting_for_software', 'original_question': prompt_lower}
                    return
                else:
                    logger.info(f"Editing help detected for {message.author.name}, asking for software")
                    await message.reply("Which software would you like help with? (After Effects, Premiere, Photoshop, DaVinci Resolve, Final Cut Pro, Topaz, CapCut, or something else?)")
                    user_states[user_id] = {'type': 'waiting_for_software', 'original_question': prompt_lower}
            return
        
        # If editing help detected but NOT mentioned (regular chat context), just continue to normal chat handling
        # Don't treat it as tutorial, just normal response

        # *** ROAST HANDLER ***
        # If user asks to roast someone else, ping them directly instead of replying to author
        if 'roast' in prompt_lower:
            target_to_roast = next((m for m in message.mentions if m.id != bot.user.id), None)
            if target_to_roast:
                async with message.channel.typing():
                    # Generate roast
                    roast_prompt = f"Roast this user: {target_to_roast.name}. Be savage, funny, and direct. Context: {message.content}"
                    # Use a clean prompt to avoid the bot getting confused by the 'reply' context
                    response = await get_gemini_response(roast_prompt, message.author.id, username=message.author.name, guild_id=message.guild.id if message.guild else None)
                    
                    # Send response mentioning the target (Not a reply to the author)
                    await message.channel.send(f"{target_to_roast.mention} {response}")
                return
        
        search_query = None # Reset search query for standard processing
        
        try:
            # Get clean prompt (remove mention if exists)
            prompt = message.content.replace(f'<@{bot.user.id}>', '').strip()
            
            # Check for attachments (images or videos)
            image_bytes = None
            video_bytes = None
            is_video = False
            video_filename = None
            
            if message.attachments:
                for attachment in message.attachments:
                    filename_lower = attachment.filename.lower()
                    
                    # Check if attachment is an image
                    if any(filename_lower.endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp']):
                        logger.info(f'Downloading image from {message.author.name}: {attachment.filename}')
                        image_bytes = await download_image(attachment.url)
                        if image_bytes:
                            break
                    
                    # Check if attachment is a video (but reject .mov files)
                    elif any(filename_lower.endswith(ext) for ext in ['.mp4', '.avi', '.mkv', '.webm', '.flv', '.wmv', '.m4v']):
                        logger.info(f'Downloading video from {message.author.name}: {attachment.filename}')
                        video_bytes, error = await download_video(attachment.url, attachment.filename)
                        if error:
                            await message.reply(f"❌ {error}")
                            return
                        if video_bytes:
                            is_video = True
                            video_filename = attachment.filename
                            break
                    
                    # Reject .mov files
                    elif filename_lower.endswith('.mov'):
                        await message.reply("❌ MOV files are not supported. Please use MP4, AVI, MKV, WebM, or other video formats.")
                        return
            
            # If there's content to process
            if not image_bytes and not video_bytes and not prompt:
                return
            
            # Show typing indicator while processing
            async with message.channel.typing():
                if is_video and video_bytes:
                    # Analyze video
                    response = await analyze_video(video_bytes, video_filename, message.author.id)
                elif image_bytes:
                    # Analyze image
                    response = await get_gemini_response(prompt, message.author.id, username=message.author.name, image_bytes=image_bytes, guild_id=message.guild.id if message.guild else None)
                else:
                    # Regular text response
                    response = await get_gemini_response(prompt, message.author.id, username=message.author.name, image_bytes=None, guild_id=message.guild.id if message.guild else None)
            
            # --- CODE EXPORT ---
            # Only extract code blocks as files if they are substantial (> 15 lines) or JSON
            all_blocks = re.findall(r'```(\w*)\n([\s\S]*?)```', response)
            code_blocks = []
            final_text = response
            
            for lang, code in all_blocks:
                is_json = lang.lower() == "json"
                is_large = len(code.splitlines()) > 15
                
                if is_json or is_large:
                    code_blocks.append((lang, code))
                    # Replace the block with a placeholder in the text
                    placeholder = f"\n*(Extracted {lang.upper()} logic to file below)*\n"
                    final_text = final_text.replace(f"```{lang}\n{code}```", placeholder)
            
            # --- TOOL EXECUTION ---
            # Automatically parse and execute tool calls (JSON blocks) in the response
            tool_executed = False
            tool_match = re.search(r'\{[^{]*"action":\s*"generate_image"[^}]*\}', response)
            if tool_match:
                try:
                    tool_json = tool_match.group(0).replace("'", '"')
                    tool_data = json.loads(tool_json)
                    action_input = tool_data.get("action_input")
                    if isinstance(action_input, str):
                        try:
                            # Sometimes the model nests JSON in a string
                            action_input = json.loads(action_input.replace("'", '"'))
                        except: pass
                    
                    prompt_val = action_input.get("prompt") if isinstance(action_input, dict) else action_input
                    if prompt_val:
                        async with message.channel.typing():
                            img_path = await generate_image(prompt_val)
                            if img_path and os.path.exists(img_path):
                                await message.reply(content=f"fulfilled your request. found/created this **{prompt_val}** for you.", file=discord.File(img_path), view=FindMoreImageView(prompt_val))
                                try: os.remove(img_path)
                                except: pass
                                tool_executed = True
                except Exception as tool_e:
                    logger.error(f"Failed to execute auto-tool: {tool_e}")

            # Clean the final text of JSON tool blocks if they were executed
            if tool_executed:
                final_text = re.sub(r'\{[^{]*"action":\s*"generate_image"[^}]*\}', '', final_text).strip()

            # Split and send text (if any text remains besides the tool result)
            if final_text and len(final_text.strip()) > 0:
                if len(final_text) > 1900:
                    chunks = [final_text[i:i+1900] for i in range(0, len(final_text), 1900)]
                    for chunk in chunks:
                        if is_dm: await message.channel.send(chunk)
                        else: await message.reply(chunk)
                else:
                    if is_dm: await message.channel.send(final_text)
                    else: await message.reply(final_text)

            # Send code blocks as files
            if code_blocks:
                for lang, code in code_blocks:
                    # Clean the code (some regex artifacts might stay)
                    code = code.strip()
                    if not code: continue
                    
                    lang = lang.lower() or "txt"
                    ext_map = {"python": "py", "javascript": "js", "typescript": "ts", "html": "html", "css": "css", "json": "json", "bash": "sh", "cpp": "cpp"}
                    ext = ext_map.get(lang, lang)
                    
                    with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False, mode='w', encoding='utf-8') as tf:
                        tf.write(code)
                        temp_path = tf.name
                    
                    embed = discord.Embed(
                        title="📜 LOGIC EXPORT",
                        description=f"Generated results for **{lang.upper()}**.",
                        color=0x00FFB4
                    )
                    embed.set_footer(text="Prime | Technical Export")
                    
                    await message.channel.send(embed=embed, file=discord.File(temp_path, filename=f"logic_{os.urandom(2).hex()}.{ext}"))
                    
                    # Cleanup
                    try: os.remove(temp_path)
                    except: pass

            logger.info(f'Responded to {message.author.name}' + (' (video analysis)' if is_video else ' (image analysis)' if image_bytes else ''))
            
            # Log the chat activity
            response_type = "Video Analysis" if is_video else "Image Analysis" if image_bytes else "Chat Response"
            server_name = message.guild.name if message.guild else "DM"
            await log_activity(
                f"💬 {response_type}",
                f"Responded to **{message.author.name}**",
                color=0x5865F2,
                fields={
                    "User": message.author.name,
                    "Server": server_name,
                    "Channel": message.channel.name if hasattr(message.channel, 'name') else "DM",
                    "Query": prompt[:100] + "..." if len(prompt) > 100 else prompt if prompt else "N/A"
                }
            )
            

        except Exception as e:
            logger.error(f'Error in chat response: {str(e)}')

@bot.command(name="help", aliases=["commands", "cmds"])
async def help_command(ctx):
    """Show all available commands with detailed descriptions."""
    logger.info(f'User {ctx.author.name} (ID: {ctx.author.id}) invoked help command')

    embed = discord.Embed(
        title="✨ PRIME COMMANDS",
        description="The complete toolkit for top-tier creators and digital talent.",
        color=0x00FFB4
    )
    
    features = [
        ("!sync", "Sync video clips to audio levels."),
        ("!stack", "Get the best AE plugin stacks."),
        ("!server", "Server security and growth stats."),
        ("!aura", "Check your vibe and profile."),
        ("!clout", "Viral potential analysis for clips."),
        ("!pulse", "Chat summary and recent activity."),
        ("!think", "Deep analysis and logical reasoning."),
        ("!briefing", "Executive summary of key information."),
        ("!strategize", "Strategic advice and decision making."),
        ("!pulse_check", "Analyze emerging trends and creator insights."),
        ("!mimic", "Copy how a user texts."),
        ("!bio", "Get a professional creator bio."),
        ("!snipe", "Recover deleted messages."),
        ("!duel", "Challenge someone to a logic fight."),
        ("!match", "Find someone with a similar vibe."),
        ("!vision", "Detailed analysis for images."),
        ("!palette", "Get color codes from an image."),
        ("!structure", "Clean project folder layouts."),
        ("!blueprint", "AE expressions and technical logic."),
        ("!scout", "Recruitment helper for talent."),
        ("!phantom", "Analyze channel vibes and dynamics."),
        ("!glitch", "Stylize text with a cool look."),
        ("!uplift", "Improve the chat's quality."),
        ("!sandbox", "Fix and optimize AE expressions.")
    ]
    
    p_text = ""
    for cmd, desc in features:
        p_text += f"**{cmd}**: {desc}\n"
    
    embed.add_field(name="📋 General", value="`!help`, `!files`, `!presets`, `!profile`", inline=False)
    
    if len(p_text) > 1024:
        break_idx = p_text[:1024].rfind('\n')
        embed.add_field(name="⚡ Features", value=p_text[:break_idx], inline=False)
        embed.add_field(name="⚡ Features (Cont.)", value=p_text[break_idx:], inline=False)
    else:
        embed.add_field(name="⚡ Features", value=p_text, inline=False)

    embed.add_field(name="📈 Leveling", value="`!level`, `!leaderboard`", inline=False)
    embed.add_field(name="💻 Software & Tools", value="`!editingsoftwares`, `!plugins`, `!extensions`", inline=False)
    
    embed.set_footer(text="Created by BMR")
    
    try:
        await ctx.author.send(embed=embed)
        if ctx.guild:
            await ctx.send("📬 **Sent**: Check your DMs for the command list.", delete_after=10)
    except discord.Forbidden:
        await ctx.send("❌ **Error**: I can't DM you. Please open your DMs and try again.")

@bot.hybrid_command(name="level", aliases=["rank", "lv"])
async def level_command(ctx, member: discord.Member = None):
    """Check your current level and XP. Usage: !level [@user]"""
    member = member or ctx.author
    user_id = member.id
    guild_id = ctx.guild.id
    
    # 1. Check for specific guild data
    guild_levels = user_levels.get(guild_id, {})
    
    # 2. LEGACY DATA MIGRATION FALLBACK
    # If user has no data in current guild, check if they have "Legacy" data (guild_id=0)
    if user_id not in guild_levels:
        legacy_data = user_levels.get(0, {}).get(user_id)
        if legacy_data:
            # Found legacy data! Migrate it to this guild.
            if guild_id not in user_levels: user_levels[guild_id] = {}
            user_levels[guild_id][user_id] = legacy_data.copy()
            guild_levels = user_levels[guild_id]
            
            # Save to Database immediately
            db_manager.save_level(guild_id, user_id, legacy_data['xp'], legacy_data['level'])
            logger.info(f"💾 MIGRATION: Restored legacy levels for {member.name} in guild {guild_id}")
    
    if user_id not in guild_levels:
        await ctx.send(f"📊 **{member.display_name}** hasn't started earning XP in this server yet. Start chatting to join the leaderboard!")
        return
        
    data = guild_levels[user_id]
    xp = data["xp"]
    level = data["level"]
    next_level_xp = 100 * (level + 1) ** 2
    xp_to_next = next_level_xp - xp
    
    
    # Aesthetic Reveal - Premium Card Design
    embed = discord.Embed(
        title=f"📊  {member.display_name}'s Level Info",
        color=0x00FFB4
    )
    
    # Information Rows
    embed.add_field(name="Level", value=f"**{level}**", inline=True)
    embed.add_field(name="Total XP", value=f"**{xp}**", inline=True)
    embed.add_field(name="XP to Next", value=f"**{xp_to_next}**", inline=True)
    
    # Progress visualization
    bar_length = 15
    progress = min(xp / next_level_xp, 1.0)
    filled = int(progress * bar_length)
    bar = "■" * filled + "□" * (bar_length - filled)
    
    # Progress Bar Field
    embed.add_field(
        name="Progress", 
        value=f"`{bar}` **{int(progress * 100)}%**", 
        inline=False
    )
    

    if member.display_avatar:
        embed.set_thumbnail(url=member.display_avatar.url)
    
    embed.set_footer(text="Prime Collective | Intelligence Sync Complete")
    await ctx.send(embed=embed)

@bot.hybrid_command(name="leaderboard", aliases=["top", "lb"])
async def leaderboard_command(ctx):
    """Show the top 10 users with the most XP."""
    guild_id = ctx.guild.id
    
    # 1. Check for legacy data and adopt it if guild levels are empty or missing individuals
    if 0 in user_levels and user_levels[0]:
        if guild_id not in user_levels: user_levels[guild_id] = {}
        
        # Adopt every user from legacy who isn't already in this guild
        adopted_count = 0
        for uid, data in user_levels[0].items():
            if uid not in user_levels[guild_id]:
                user_levels[guild_id][uid] = data.copy()
                db_manager.save_level(guild_id, uid, data['xp'], data['level'])
                adopted_count += 1
        
        if adopted_count > 0:
            logger.info(f"💾 LEADERBOARD MIGRATION: Adopted {adopted_count} legacy users into guild {guild_id}")
            # Optional: Clear legacy data for these users? No, keep as backup.

    guild_levels = user_levels.get(guild_id, {})
    if not guild_levels:
        await ctx.send("🌑 **The leaderboard is currently empty.** Be the first to start the journey.")
        return
        
    # Get sorted users
    sorted_users = sorted(guild_levels.items(), key=lambda x: x[1]["xp"], reverse=True)
    top_10 = sorted_users[:10]
    
    lb_text = ""
    for i, (uid, data) in enumerate(top_10, 1):
        user = bot.get_user(uid)
        name = user.display_name if user else f"User {uid}"
        
        # Styling for top ranks
        if i == 1: emoji = "🥇"
        elif i == 2: emoji = "🥈"
        elif i == 3: emoji = "🥉"
        else: emoji = f"`#{i}`"
        
        lb_text += f"{emoji} **{name}**\n╰─ Level {data['level']} • `{data['xp']:,} XP`\n\n"
        
    embed = discord.Embed(
        title=f"🏆 {ctx.guild.name.upper()} LEADERBOARD",
        description=lb_text,
        color=0x00FFB4
    )
    
    # Find author rank
    user_rank = "N/A"
    for i, (uid, _) in enumerate(sorted_users, 1):
        if uid == ctx.author.id:
            user_rank = i
            break
            
    embed.set_footer(text=f"Your Rank: #{user_rank} | Keep creating.")
    if ctx.guild.icon:
        embed.set_thumbnail(url=ctx.guild.icon.url)
        
    await ctx.send(embed=embed)

@bot.command(name="intercept", aliases=["snitch", "sniff"])
async def intercept_command(ctx):
    """Spectral Interception: Spend 500 XP to reveal the last deleted message in this channel."""
    cost = 500
    user_id = ctx.author.id
    
    # Check if user has enough XP
    guild_levels = user_levels.get(ctx.guild.id, {})
    if user_id not in guild_levels or guild_levels[user_id]["xp"] < cost:
        await ctx.reply(f"🚫 **ACCESS DENIED**: You need at least `{cost} XP` in this server to intercept spectral data.")
        return
        
    # Get latest deleted message
    deleted = db_manager.get_latest_deleted_messages(ctx.channel.id, limit=1)
    
    if not deleted:
        await ctx.reply("🌑 **No spectral traces found.** This channel is clean.")
        return
        
    msg_user_id, msg_username, content, attachments_json, timestamp = deleted[0]
    attachments = json.loads(attachments_json) if attachments_json else []
    
    # Deduct XP
    user_levels[ctx.guild.id][user_id]["xp"] -= cost
    db_manager.save_level(ctx.guild.id, user_id, user_levels[ctx.guild.id][user_id]["xp"], user_levels[ctx.guild.id][user_id]["level"])
    
    # Aesthetic Reveal
    embed = discord.Embed(
        title="🕵️  SPECTRAL INTERCEPTION SUCCESSFUL",
        description=f"*Retrieved data from the void. Cost: {cost} XP*",
        color=0xFF00FF # Neon Magenta/Purple
    )
    
    reveal_content = content if content else "*No text content (likely media only)*"
    embed.add_field(name="👤 SOURCE", value=f"**{msg_username}**", inline=True)
    embed.add_field(name="🕒 TIME", value=f"<t:{int(timestamp.timestamp())}:R>", inline=True)
    embed.add_field(name="📝 CONTENT", value=f"```\n{reveal_content}\n```", inline=False)
    
    if attachments:
        att_links = "\n".join([f"• [{a['filename']}]({a['url']})" for a in attachments])
        embed.add_field(name="📎 ATTACHMENTS", value=att_links, inline=False)
        # If it's an image, set it as thumbnail or image
        for a in attachments:
            if a.get('content_type') and a['content_type'].startswith('image'):
                embed.set_image(url=a['url'])
                break

    embed.set_footer(text=f"Intercepted by {ctx.author.display_name} | Your XP: {user_levels[ctx.guild.id][user_id]['xp']}")
    
    await ctx.send(embed=embed)
    logger.info(f"{ctx.author.name} intercepted message from {msg_username}")
async def list_files_command(ctx):
    """
    Lists all available files that can be requested.
    Usage: !files
    """
    logger.info(f'User {ctx.author.name} (ID: {ctx.author.id}) invoked !files command in {ctx.guild.name if ctx.guild else "DM"}')

    # Get list of files in the files directory
    files_dir = "files"
    if not os.path.exists(files_dir):
        await ctx.send("No files available currently.")
        return

    # Get all files in the directory
    all_files = []
    for file in glob.glob(f"{files_dir}/*"):
        if os.path.isfile(file):
            filename = os.path.basename(file)
            command_name = os.path.splitext(filename)[0]
            all_files.append(f"!{command_name} - {filename}")

    if not all_files:
        await ctx.send("No files available currently.")
        return

    # Format the file list
    all_files.sort()  # Sort alphabetically
    file_list = "\n".join(all_files)
    response = f"**Available Files:**\n```\n{file_list}\n```\nType the command (e.g., !foggy_cc) to receive the file in your DMs."

    try:
        # Send the list to the user's DMs
        await ctx.author.send(response)
        logger.info(f'Sent file list to {ctx.author.name}')

        # Send confirmation in the channel
        if ctx.guild:
            await ctx.send(f"{ctx.author.mention}, I've sent you the list of available files in your DMs!")

    except discord.Forbidden:
        # If DMs are closed, send in the channel
        logger.warning(f'Could not send file list to {ctx.author.name} - DMs may be closed')
        await ctx.send(f"{ctx.author.mention}, I couldn't send you a DM. Here's the list of files:")
        await ctx.send(response)

    except Exception as e:
        logger.error(f'Error sending file list to {ctx.author.name}: {str(e)}')
        await ctx.send(f"{ctx.author.mention}, an error occurred while trying to send you the file list.")

@bot.command(name="software_list")
async def software_list_command(ctx):
    """Outdated command replaced by !editingsoftwares."""
    await ctx.send(f"{ctx.author.mention}, `!software_list` is deprecated. Use **!editingsoftwares** for the latest distributions.")

@bot.command(name="presets")
async def presets_command(ctx):
    """
    Lists all available .ffx presets (color correction files).
    Usage: !presets
    """
    logger.info(f'User {ctx.author.name} (ID: {ctx.author.id}) invoked !presets command in {ctx.guild.name if ctx.guild else "DM"}')

    # Get list of .ffx files in the files directory
    files_dir = "files"
    if not os.path.exists(files_dir):
        await ctx.send("No presets available currently.")
        return

    # Get all .ffx files in the directory
    ffx_files = []
    for file in glob.glob(f"{files_dir}/*.ffx"):
        if os.path.isfile(file):
            filename = os.path.basename(file)
            command_name = os.path.splitext(filename)[0]
            ffx_files.append(f"!{command_name} - {filename}")

    if not ffx_files:
        await ctx.send("No presets available currently.")
        return

    # Format the file list
    ffx_files.sort()  # Sort alphabetically
    file_list = "\n".join(ffx_files)
    response = f"**Available Color Correction Presets:**\n```\n{file_list}\n```\nType the command (e.g., !foggy_cc) to receive the preset in your DMs."

    try:
        # Send the list to the user's DMs
        await ctx.author.send(response)
        logger.info(f'Sent preset list to {ctx.author.name}')

        # Send confirmation in the channel
        if ctx.guild:
            await ctx.send(f"{ctx.author.mention}, I've sent you the list of available presets in your DMs!")

    except discord.Forbidden:
        # If DMs are closed, send in the channel
        logger.warning(f'Could not send preset list to {ctx.author.name} - DMs may be closed')
        await ctx.send(f"{ctx.author.mention}, I couldn't send you a DM. Here's the list of presets:")
        await ctx.send(response)

    except Exception as e:
        logger.error(f'Error sending preset list to {ctx.author.name}: {str(e)}')
        await ctx.send(f"{ctx.author.mention}, an error occurred while trying to send you the preset list.")

@bot.command(name="aecrack")
async def aecrack_command(ctx):
    """
    Sends information about Adobe After Effects crack.
    Usage: !aecrack
    """
    logger.info(f'User {ctx.author.name} (ID: {ctx.author.id}) invoked !aecrack command in {ctx.guild.name if ctx.guild else "DM"}')

    # Actual Adobe After Effects crack information
    response = """**Adobe After Effects Crack Links**

# [2025 (v25.1)](<https://notabin.com/?fb7cab495eecf221#FiT2GfKpydCLgzWGKUv8jHVdMB8dn2YqDoi6E17qEa7F>)

# [2024 (v24.6.2)](<https://paste.to/?d06e0c5b7a227356#DoWsXVNiFCvYpxZdvE793tu8jnxmq66bxw3k4WpuLA63>)

# [2022 (v22.6)](<https://paste.to/?2de1e37edd288c59#HKgmUNUEfKG4z3ZrQ6pGxcqiroeHcZqS7AxuEqScHv2t>)

# [2020 (v17.7)](<https://paste.to/?4c06b2d0730e4b4e#BwAWrNgK633RtYnzGB25us53Z6pMN4QzocRY9MNoFCeU>)

**Installation:**

_1) Mount the ISO._
_2) Run autoplay.exe._

**Note:**

_Cloud-based functionality will not work for this crack. You must ensure to block internet connections to the app in case of unlicensed errors._"""

    try:
        # Send DM to the user
        await ctx.author.send(response)
        logger.info(f'Successfully sent AE crack info to {ctx.author.name}')

        # Send confirmation in the channel
        if ctx.guild:
            await ctx.send(f"{ctx.author.mention}, I've sent you the After Effects crack information in your DMs!")

    except discord.Forbidden:
        logger.warning(f'Could not send AE crack info to {ctx.author.name} - DMs may be closed')
        await ctx.send(f"{ctx.author.mention}, I couldn't send you a DM. Please check your privacy settings.")

    except Exception as e:
        logger.error(f'Error sending AE crack info to {ctx.author.name}: {str(e)}')
        await ctx.send(f"{ctx.author.mention}, an error occurred while trying to send you the information.")

@bot.command(name="pscrack")
async def pscrack_command(ctx):
    """
    Sends information about Adobe Photoshop crack.
    Usage: !pscrack
    """
    logger.info(f'User {ctx.author.name} (ID: {ctx.author.id}) invoked !pscrack command in {ctx.guild.name if ctx.guild else "DM"}')

    # Actual Adobe Photoshop crack information
    response = """**Adobe Photoshop Crack Information**

# [PHOTOSHOP 2025](<https://hidan.sh/tfbctrj9jn54i>) 

# INSTALLATION

1) Mount the ISO.
2) Run autoplay.exe.

**Note:**

Cloud-based functionality will not work for this crack. You must ensure to block internet connections to the app in case of unlicensed errors.

Ensure to use uBlock Origin. The file should be the size and format stated."""

    try:
        # Send DM to the user
        await ctx.author.send(response)
        logger.info(f'Successfully sent PS crack info to {ctx.author.name}')

        # Send confirmation in the channel
        if ctx.guild:
            await ctx.send(f"{ctx.author.mention}, I've sent you the Photoshop crack information in your DMs!")

    except discord.Forbidden:
        logger.warning(f'Could not send PS crack info to {ctx.author.name} - DMs may be closed')
        await ctx.send(f"{ctx.author.mention}, I couldn't send you a DM. Please check your privacy settings.")

    except Exception as e:
        logger.error(f'Error sending PS crack info to {ctx.author.name}: {str(e)}')
        await ctx.send(f"{ctx.author.mention}, an error occurred while trying to send you the information.")

@bot.command(name="mecrack")
async def mecrack_command(ctx):
    """
    Sends information about Media Encoder crack.
    Usage: !mecrack
    """
    logger.info(f'User {ctx.author.name} (ID: {ctx.author.id}) invoked !mecrack command in {ctx.guild.name if ctx.guild else "DM"}')

    # Actual Media Encoder crack information
    response = """**Media Encoder Crack Information**

# [MEDIA ENCODER 2025](<https://hidan.sh/s6ljnz5eizd2>) 

# Installation:

1) Mount the ISO.
2) Run autoplay.exe.

# Note:

Do not utilise H.264 or H.265 through ME.

Cloud-based functionality will not work for this crack. You must ensure to block internet connections to the app in case of unlicensed errors.

Ensure to use uBlock Origin. The file should be the size and format stated."""

    try:
        # Send DM to the user
        await ctx.author.send(response)
        logger.info(f'Successfully sent ME crack info to {ctx.author.name}')

        # Send confirmation in the channel
        if ctx.guild:
            await ctx.send(f"{ctx.author.mention}, I've sent you the Media Encoder crack information in your DMs!")

    except discord.Forbidden:
        logger.warning(f'Could not send ME crack info to {ctx.author.name} - DMs may be closed')
        await ctx.send(f"{ctx.author.mention}, I couldn't send you a DM. Please check your privacy settings.")

    except Exception as e:
        logger.error(f'Error sending ME crack info to {ctx.author.name}: {str(e)}')
        await ctx.send(f"{ctx.author.mention}, an error occurred while trying to send you the information.")

@bot.command(name="prcrack")
async def prcrack_command(ctx):
    """
    Sends information about Adobe Premiere Pro crack.
    Usage: !prcrack
    """
    logger.info(f'User {ctx.author.name} (ID: {ctx.author.id}) invoked !prcrack command in {ctx.guild.name if ctx.guild else "DM"}')

    # Actual Premiere Pro crack information
    response = """**Adobe Premiere Pro Crack Information**

# [PREMIERE PRO 2025](<https://hidan.sh/rlr5vmxc2kbm>) 

# Installation:

1) Mount the ISO.
2) Run autoplay.exe.

# Note:

Cloud-based functionality will not work for this crack. You must ensure to block internet connections to the app in case of unlicensed errors.

Ensure to use uBlock Origin. The file should be the size and format stated."""

    try:
        # Send DM to the user
        await ctx.author.send(response)
        logger.info(f'Successfully sent PR crack info to {ctx.author.name}')

        # Send confirmation in the channel
        if ctx.guild:
            await ctx.send(f"{ctx.author.mention}, I've sent you the Premiere Pro crack information in your DMs!")

    except discord.Forbidden:
        logger.warning(f'Could not send PR crack info to {ctx.author.name} - DMs may be closed')
        await ctx.send(f"{ctx.author.mention}, I couldn't send you a DM. Please check your privacy settings.")

    except Exception as e:
        logger.error(f'Error sending PR crack info to {ctx.author.name}: {str(e)}')
        await ctx.send(f"{ctx.author.mention}, an error occurred while trying to send you the information.")

@bot.command(name="topazcrack")
async def topazcrack_command(ctx):
    """
    Sends information about Topaz Suite crack.
    Usage: !topazcrack
    """
    logger.info(f'User {ctx.author.name} (ID: {ctx.author.id}) invoked !topazcrack command in {ctx.guild.name if ctx.guild else "DM"}')

    # Actual Topaz crack information
    response = """**Topaz Video AI Crack Information**

# [TOPAZ 6.0.3 PRO](<https://tinyurl.com/Topaz-video-ai-6)

# INSTALLATION
1) Replace rlm1611.dll in C:\\Program Files\\Topaz Labs LLC\\Topaz Video AI\\.

2) Copy license.lic to C:\\ProgramData\\Topaz Labs LLC\\Topaz Video AI\\models.

**Note:**

Archive says 6.0.3, but it will still work. The same could be true for later versions.
Starlight won't work as it's credit-based.

Ensure to use uBlock Origin. The file should be the size and format stated."""

    try:
        # Send DM to the user
        await ctx.author.send(response)
        logger.info(f'Successfully sent Topaz crack info to {ctx.author.name}')

        # Send confirmation in the channel
        if ctx.guild:
            await ctx.send(f"{ctx.author.mention}, I've sent you the Topaz Suite crack information in your DMs!")

    except discord.Forbidden:
        logger.warning(f'Could not send Topaz crack info to {ctx.author.name} - DMs may be closed')
        await ctx.send(f"{ctx.author.mention}, I couldn't send you a DM. Please check your privacy settings.")

    except Exception as e:
        logger.error(f'Error sending Topaz crack info to {ctx.author.name}: {str(e)}')
        await ctx.send(f"{ctx.author.mention}, an error occurred while trying to send you the information.")

@bot.command(name="hi")
async def hi_command(ctx):
    """
    Alternative command that also sends 'HI' to the user's DMs.
    Usage: !hi
    """
    logger.info(f'User {ctx.author.name} (ID: {ctx.author.id}) invoked !hi command in {ctx.guild.name if ctx.guild else "DM"}')

    try:
        # Send DM to the user
        await ctx.author.send("HI")
        logger.info(f'Successfully sent DM to {ctx.author.name}')

        # Optional confirmation in the channel where command was used
        if ctx.guild:  # Only if command was used in a server, not in DMs
            await ctx.send(f"{ctx.author.mention}, I've sent you a DM!")

    except discord.Forbidden:
        # Handle the case where user has DMs closed or blocked the bot
        logger.warning(f'Could not send DM to {ctx.author.name} - DMs may be closed')
        await ctx.send(f"{ctx.author.mention}, I couldn't send you a DM. Please check your privacy settings.")

    except Exception as e:
        logger.error(f'Error sending DM to {ctx.author.name}: {str(e)}')
        await ctx.send(f"{ctx.author.mention}, an error occurred while trying to send you a DM.")

@bot.command(name="ask")
async def ask_command(ctx, *, question=None):
    """Get deep, detailed answers to any question. Usage: !ask What is quantum computing?"""
    if not question:
        await ctx.send("📝 Please provide a question! Usage: !ask [your question]")
        return
    async with ctx.typing():
        prompt = f"Provide a comprehensive, detailed answer to this question: {question}"
        response = await get_gemini_response(prompt, ctx.author.id, username=ctx.author.name, guild_id=ctx.guild.id if ctx.guild else None)
        chunks = [response[i:i+1900] for i in range(0, len(response), 1900)]
        for chunk in chunks:
            await ctx.send(chunk)

@bot.command(name="echo")
async def echo_command(ctx, message_id: int, *, text: str):
    """Echo a message to a specific message ID. Admin/Mod only."""
    # Check if user is owner or has manage_messages
    is_owner_check = await bot.is_owner(ctx.author)
    perms = ctx.author.guild_permissions
    if not (is_owner_check or perms.manage_messages or perms.administrator):
        return

    try:
        target_message = None
        # First try current channel (fastest)
        try:
            target_message = await ctx.channel.fetch_message(message_id)
        except:
            # If not in current channel, search other text channels
            for channel in ctx.guild.text_channels:
                if channel.id == ctx.channel.id: continue
                try:
                    target_message = await channel.fetch_message(message_id)
                    if target_message: break
                except: continue
        
        if not target_message:
            await ctx.send(f"❌ Message `{message_id}` not found in this server.", delete_after=10)
            return

        await target_message.reply(text)
        try: await ctx.message.delete()
        except: pass
    except Exception as e:
        await ctx.send(f"❌ Error: {e}", delete_after=5)

@bot.command(name="nudge")
async def nudge_command(ctx):
    """(Admin) Send a nudge to unverified users to verify."""
    # Check permissions
    is_owner_check = await bot.is_owner(ctx.author)
    if not (is_owner_check or ctx.author.guild_permissions.administrator):
        return

    unverified_role_id = get_unverified_role(ctx.guild.id)
    target_channel_id = get_welcome_chan(ctx.guild.id)
    v_chan_id = get_verification_chan(ctx.guild.id)
    
    channel = bot.get_channel(target_channel_id) or ctx.channel
    if not channel:
        await ctx.send("❌ Could not determine target channel.")
        return

    try:
        msg = await channel.send(
            f"🔔 **ATTENTION <@&{unverified_role_id}>!**\n\n"
            f"Please complete your verification in <#{v_chan_id}> to gain full access to the server. "
            f"If you don't verify, you will remain restricted.\n\n"
            f"*This message will self-destruct in 12 hours.*"
        )
        await ctx.send(f"✅ Nudge sent to {channel.mention}. It will be deleted in 12 hours.")
        
        # Background task to delete after 12 hours
        async def delayed_delete(message):
            await asyncio.sleep(12 * 3600)  # 12 hours
            try:
                await message.delete()
                logger.info(f"Automatically deleted nudge message in {message.channel.name}")
            except:
                pass
        
        bot.loop.create_task(delayed_delete(msg))
        
    except Exception as e:
        logger.error(f"Error sending nudge: {e}")
        await ctx.send(f"❌ Failed to send nudge: {e}")

@bot.command(name="explain")
async def explain_command(ctx, *, topic=None):
    """Explain any topic clearly in simple language. Usage: !explain machine learning"""
    if not topic:
        await ctx.send("📖 Please provide a topic! Usage: !explain [topic]")
        return
    async with ctx.typing():
        prompt = f"Explain '{topic}' in simple, easy-to-understand language. Make it clear for beginners."
        response = await get_gemini_response(prompt, ctx.author.id, username=ctx.author.name, guild_id=ctx.guild.id if ctx.guild else None)
        chunks = [response[i:i+1900] for i in range(0, len(response), 1900)]
        for chunk in chunks:
            await ctx.send(chunk)

@bot.command(name="improve")
async def improve_command(ctx, *, text=None):
    """Enhance any message, paragraph, or script. Usage: !improve your text here"""
    if not text:
        await ctx.send("✏️ Please provide text to improve! Usage: !improve [text]")
        return
    async with ctx.typing():
        prompt = f"Enhance and improve this text. Make it better, clearer, more engaging, and more professional: {text}"
        response = await get_gemini_response(prompt, ctx.author.id, username=ctx.author.name, guild_id=ctx.guild.id if ctx.guild else None)
        chunks = [response[i:i+1900] for i in range(0, len(response), 1900)]
        for chunk in chunks:
            await ctx.send(chunk)

@bot.command(name="rewrite")
async def rewrite_command(ctx, *, text=None):
    """Rewrite text in different tones or styles. Usage: !rewrite make this more formal"""
    if not text:
        await ctx.send("📝 Please provide text to rewrite! Usage: !rewrite [text]")
        return
    async with ctx.typing():
        prompt = f"Rewrite this text in a more creative, engaging, and professional way: {text}"
        response = await get_gemini_response(prompt, ctx.author.id, username=ctx.author.name, guild_id=ctx.guild.id if ctx.guild else None)
        chunks = [response[i:i+1900] for i in range(0, len(response), 1900)]
        for chunk in chunks:
            await ctx.send(chunk)

@bot.command(name="summarize")
async def summarize_command(ctx, *, text=None):
    """Convert long text into short, clear summaries. Usage: !summarize [your long text]"""
    if not text:
        await ctx.send("📄 Please provide text to summarize! Usage: !summarize [text]")
        return
    async with ctx.typing():
        prompt = f"Summarize this text into a short, clear summary that captures all key points: {text}"
        response = await get_gemini_response(prompt, ctx.author.id, username=ctx.author.name, guild_id=ctx.guild.id if ctx.guild else None)
        chunks = [response[i:i+1900] for i in range(0, len(response), 1900)]
        for chunk in chunks:
            await ctx.send(chunk)

@bot.command(name="analyze")
async def analyze_command(ctx, *, content=None):
    """Analyze content and give insights or breakdowns. Usage: !analyze this text or concept"""
    if not content:
        await ctx.send("🔍 Please provide content to analyze! Usage: !analyze [content]")
        return
    async with ctx.typing():
        prompt = f"Analyze this content deeply and provide detailed insights, breakdowns, and observations: {content}"
        response = await get_gemini_response(prompt, ctx.author.id, username=ctx.author.name, guild_id=ctx.guild.id if ctx.guild else None)
        chunks = [response[i:i+1900] for i in range(0, len(response), 1900)]
        for chunk in chunks:
            await ctx.send(chunk)

@bot.command(name="idea", aliases=["ideas", "ideea", "ideaz"])
async def idea_command(ctx, *, topic=None):
    """Generate creative ideas for videos, designs, content, or posts. Usage: !idea gaming video ideas"""
    if not topic:
        await ctx.send("💡 Please provide a topic! Usage: !idea [topic for ideas]")
        return
    async with ctx.typing():
        prompt = f"Generate 5 creative, unique ideas for: {topic}. Make them specific, actionable, and interesting."
        response = await get_gemini_response(prompt, ctx.author.id, username=ctx.author.name, guild_id=ctx.guild.id if ctx.guild else None)
        chunks = [response[i:i+1900] for i in range(0, len(response), 1900)]
        for chunk in chunks:
            await ctx.send(chunk)

@bot.command(name="define")
async def define_command(ctx, *, word=None):
    """Get definitions for any word or concept. Usage: !define algorithm"""
    if not word:
        await ctx.send("📚 Please provide a word to define! Usage: !define [word]")
        return
    async with ctx.typing():
        prompt = f"Provide a clear, concise definition of '{word}' with an example of how it's used."
        response = await get_gemini_response(prompt, ctx.author.id, username=ctx.author.name, guild_id=ctx.guild.id if ctx.guild else None)
        chunks = [response[i:i+1900] for i in range(0, len(response), 1900)]
        for chunk in chunks:
            await ctx.send(chunk)

@bot.command(name="prime", aliases=["helper"])
async def prime_command(ctx, *, query=None):
    """All-in-one AI command for multi-purpose assistance. Usage: !prime anything you need help with"""
    if not query:
        await ctx.send("🤖 Please provide a request! Usage: !prime [your question/request]")
        return
    async with ctx.typing():
        prompt = f"Help with this request in the most useful way possible: {query}"
        response = await get_gemini_response(prompt, ctx.author.id, username=ctx.author.name, guild_id=ctx.guild.id if ctx.guild else None)
        chunks = [response[i:i+1900] for i in range(0, len(response), 1900)]
        for chunk in chunks:
            await ctx.send(chunk)

@bot.command(name="fix")
async def fix_command(ctx, *, text=None):
    """Correct grammar, spelling, and mistakes. Usage: !fix your text here"""
    if not text:
        await ctx.send("✍️ Please provide text to fix! Usage: !fix [text]")
        return
    async with ctx.typing():
        prompt = f"Correct all grammar, spelling, and grammatical mistakes in this text. Return only the corrected text: {text}"
        response = await get_gemini_response(prompt, ctx.author.id, username=ctx.author.name, guild_id=ctx.guild.id if ctx.guild else None)
        await ctx.send(response)

@bot.command(name="shorten")
async def shorten_command(ctx, *, text=None):
    """Make text shorter but keep the meaning. Usage: !shorten your long text here"""
    if not text:
        await ctx.send("📉 Please provide text to shorten! Usage: !shorten [text]")
        return
    async with ctx.typing():
        prompt = f"Make this text shorter and more concise while keeping all the important meaning: {text}"
        response = await get_gemini_response(prompt, ctx.author.id, username=ctx.author.name, guild_id=ctx.guild.id if ctx.guild else None)
        await ctx.send(response)

@bot.command(name="expand")
async def expand_command(ctx, *, text=None):
    """Add detail, depth, and clarity to text. Usage: !expand your text here"""
    if not text:
        await ctx.send("📈 Please provide text to expand! Usage: !expand [text]")
        return
    async with ctx.typing():
        prompt = f"Expand this text by adding more detail, depth, and clarity. Make it richer and more comprehensive: {text}"
        response = await get_gemini_response(prompt, ctx.author.id, username=ctx.author.name, guild_id=ctx.guild.id if ctx.guild else None)
        await ctx.send(response)

@bot.command(name="caption")
async def caption_command(ctx, *, topic=None):
    """Create captions for reels, videos, and posts. Usage: !caption gaming video about speedrun"""
    if not topic:
        await ctx.send("📸 Please provide a topic! Usage: !caption [what the content is about]")
        return
    async with ctx.typing():
        prompt = f"Create 3 engaging, catchy captions for a reel/video/post about: {topic}. Make them fun, relevant, and include relevant hashtags."
        response = await get_gemini_response(prompt, ctx.author.id, username=ctx.author.name, guild_id=ctx.guild.id if ctx.guild else None)
        await ctx.send(response)

@bot.command(name="script")
async def script_command(ctx, *, idea=None):
    """Generate short scripts or dialogues. Usage: !script two friends meeting after years"""
    if not idea:
        await ctx.send("🎬 Please provide a script idea! Usage: !script [scene idea]")
        return
    async with ctx.typing():
        prompt = f"Write a short, engaging script or dialogue for: {idea}. Make it natural, interesting, and ready to use."
        response = await get_gemini_response(prompt, ctx.author.id, username=ctx.author.name, guild_id=ctx.guild.id if ctx.guild else None)
        await ctx.send(response)

@bot.command(name="format")
async def format_command(ctx, *, text=None):
    """Format text into clean structure or bullet points. Usage: !format your messy text here"""
    if not text:
        await ctx.send("📋 Please provide text to format! Usage: !format [text]")
        return
    async with ctx.typing():
        prompt = f"Format this text into a clean, well-structured format using bullet points or sections as appropriate: {text}"
        response = await get_gemini_response(prompt, ctx.author.id, username=ctx.author.name, guild_id=ctx.guild.id if ctx.guild else None)
        await ctx.send(response)

@bot.command(name="title")
async def title_command(ctx, *, content=None):
    """Generate attractive titles for any content. Usage: !title about a cat adventure"""
    if not content:
        await ctx.send("⭐ Please provide content! Usage: !title [describe your content]")
        return
    async with ctx.typing():
        prompt = f"Generate 5 creative, catchy, and attractive title options for: {content}. Make them engaging and click-worthy."
        response = await get_gemini_response(prompt, ctx.author.id, username=ctx.author.name, guild_id=ctx.guild.id if ctx.guild else None)
        await ctx.send(response)

@bot.command(name="translate")
async def translate_command(ctx, *, text=None):
    """Translate text into any language. Usage: !translate hello world to spanish"""
    if not text:
        await ctx.send("🌍 Please provide text and language! Usage: !translate [text] to [language]")
        return
    async with ctx.typing():
        prompt = f"Translate this text as requested: {text}. Provide only the translation."
        response = await get_gemini_response(prompt, ctx.author.id, username=ctx.author.name, guild_id=ctx.guild.id if ctx.guild else None)
        await ctx.send(response)

@bot.command(name="paragraph")
async def paragraph_command(ctx, *, text=None):
    """Turn messy text into a clean, structured paragraph. Usage: !paragraph your messy notes here"""
    if not text:
        await ctx.send("📝 Please provide text to format! Usage: !paragraph [text]")
        return
    async with ctx.typing():
        prompt = f"Turn this messy text into a clean, well-structured, professional paragraph: {text}"
        response = await get_gemini_response(prompt, ctx.author.id, username=ctx.author.name, guild_id=ctx.guild.id if ctx.guild else None)
        await ctx.send(response)

# --- SLASH COMMANDS (Modern Interactions) ---

@bot.tree.command(name="ping", description="Check the bot's latency")
async def slash_ping(interaction: discord.Interaction):
    """Slash command version of ping"""
    await interaction.response.send_message(f"🏓 Pong! Latency: **{round(bot.latency * 1000)}ms**")

@bot.tree.command(name="prime", description="Ask Prime AI anything")
@app_commands.describe(query="What do you want to ask Prime?")
async def slash_prime(interaction: discord.Interaction, query: str):
    """Slash command version of !prime"""
    await interaction.response.defer()
    try:
        prompt = f"Help with this request in the most useful way possible: {query}"
        response = await get_gemini_response(prompt, interaction.user.id, username=interaction.user.name, guild_id=interaction.guild.id if interaction.guild else None)
        
        # Split response into chunks if it's too long
        chunks = [response[i:i+1900] for i in range(0, len(response), 1900)]
        await interaction.followup.send(chunks[0])
        for chunk in chunks[1:]:
            await interaction.channel.send(chunk)
    except Exception as e:
        await interaction.followup.send(f"❌ Error: {str(e)}")

@bot.tree.command(name="help", description="View all available commands")
async def slash_help(interaction: discord.Interaction):
    """Slash command version of !help"""
    embed = discord.Embed(
        title="🤖 PRIME BOT - COMMANDS",
        description="I've sent the command list to your DMs!",
        color=0x3498DB
    )
    await interaction.response.send_message(embed=embed, ephemeral=True)
    
    embed_dm = discord.Embed(
        title="🤖 PRIME BOT - COMMANDS",
        description="Here is the list of available commands you can use!",
        color=0x3498DB
    )
    embed_dm.add_field(name="🚀 Slash Commands", value="`/ping`, `/prime`, `/level`, `/leaderboard`, `/help`, `/commands`", inline=False)
    embed_dm.add_field(name="📋 Prefix Commands (!)", value="`!help`, `!commands`, `!files`, `!presets`, `!level`, `!leaderboard`", inline=False)
    embed_dm.add_field(name="📝 AI Tools", value="`!prime`, `!ask`, `!explain`, `!improve`... (see full list in !help)", inline=False)
    embed_dm.set_footer(text="Prime AI • Developed by BMR")
    
    try:
        await interaction.user.send(embed=embed_dm)
    except:
        pass

@bot.tree.command(name="commands", description="View all available commands")
async def slash_commands(interaction: discord.Interaction):
    """Slash command version of !commands"""
    await slash_help(interaction)

@commands.is_owner()
async def manual_sync(ctx):
    """Owner-only command to manually sync slash commands"""
    await ctx.send("🔄 Syncing slash commands... this may take a moment.")
    try:
        synced = await bot.tree.sync()
        await ctx.send(f"✅ Successfully synced {len(synced)} global slash commands.")
    except Exception as e:
        await ctx.send(f"❌ Failed to sync: {e}")

@bot.listen('on_message')
async def file_command_handler(message):
    """
    Listens for messages that start with ! and checks if they match any filenames.
    If a match is found, sends the file to the user's DMs.
    """
    # Ignore messages from the bot itself
    if message.author == bot.user:
        return

    # Check if message starts with ! and is longer than 1 character
    if not message.content.startswith('!') or len(message.content) <= 1:
        return

    # Extract the filename without the ! and convert to lowercase for case-insensitive matching
    requested_file = message.content[1:]
    requested_file_lower = requested_file.lower()
    
    # Extract just the first word to check against known commands
    first_word = requested_file_lower.split()[0] if requested_file_lower else ""
    
    # Skip for known commands to avoid duplicate messages (case-insensitive check)
    if first_word in ["help", "hi", "files", "software_list", "presets", 
                      "aecrack", "pscrack", "mecrack", "prcrack", "topazcrack", 
                      "ban", "mute", "timeout", "unmute", "setup_roles", "setup_verification", "appeal", "verify",
                      "ask", "explain", "improve", "rewrite", "summarize", "analyze", "idea", "define", "prime", "helper",
                      "fix", "shorten", "expand", "caption", "script", "format", "title", "translate", "paragraph",
                      "remind", "note", "timer", "convert", "emoji", "calculate", "weather", "profile", "serverinfo",
                      "creative", "story", "quote", "brainstorm", "design", "name", "aesthetic", "topics", "motivate",
                      "role", "setup_roles", "setup_verification", "check_automod", "setup_automod", "setup_content_roles", "echo",
                      "r", "sync", "manual_sync", "commands", "cmds", "nudge", "portfolio", "profile", "p", "invite", "vote",
                      "ae", "pr", "me", "ps", "topaz", "editingsoftwares", "setup_updates", "appeal_link", "broadcast_update",
                      "plugins", "borisfx", "maxon", "revisionfx", "videocopilot", "autokroma", "zaebects", "plugineverything", "elementsupply", "pixelsorter", "filmconvert",
                      "extensions", "access", "animate", "illustrator", "indesign", "lightroom", "audition", "incopy"]:
        return
    
    logger.info(f'User {message.author.name} (ID: {message.author.id}) requested file: {requested_file}')

    # Check if the file exists in the files directory - handle both with and without spaces and case sensitivity
    file_paths = [
        f"files/{requested_file}",  # Original format
        f"files/{requested_file.replace('_', ' ')}",  # Replace underscores with spaces
        f"files/{requested_file.replace(' ', '_')}"   # Replace spaces with underscores
    ]

    # Also add lowercase versions for case-insensitive matching
    file_paths_lower = [
        f"files/{requested_file_lower}",  # Lowercase original format
        f"files/{requested_file_lower.replace('_', ' ')}",  # Lowercase with spaces
        f"files/{requested_file_lower.replace(' ', '_')}"   # Lowercase with underscores
    ]

    # Combine all possible paths
    file_paths.extend(file_paths_lower)
    file_extensions = ["", ".txt", ".pdf", ".png", ".jpg", ".jpeg", ".gif", ".mp3", ".mp4", ".zip", ".ffx"]

    found_file = None
    for base_path in file_paths:
        for ext in file_extensions:
            potential_path = f"{base_path}{ext}"
            if os.path.exists(potential_path) and os.path.isfile(potential_path):
                found_file = potential_path
                break
        if found_file:
            break

    # If file was found, send it to the user
    if found_file:
        try:
            # Send file to the user's DMs
            await message.author.send(f"Here's your requested file: `{requested_file}`", 
                                    file=discord.File(found_file))
            logger.info(f'Successfully sent file {found_file} to {message.author.name}')

            # Send confirmation in the channel
            if message.guild:  # Only if command was used in a server
                await message.channel.send(f"{message.author.mention}, I've sent your requested file to your DMs!")

        except discord.Forbidden:
            # Handle the case where user has DMs closed
            logger.warning(f'Could not send file to {message.author.name} - DMs may be closed')
            await message.channel.send(f"{message.author.mention}, I couldn't send you the file. Please check your privacy settings.")

        except Exception as e:
            # Handle other exceptions
            logger.error(f'Error sending file to {message.author.name}: {str(e)}')
            await message.channel.send(f"{message.author.mention}, an error occurred while trying to send you the file.")

    # If file was not found, try to suggest a command
    else:
        # Define the known commands for suggestions - including common misspellings and variations
        known_commands = {
            # editingsoftwares variations
            "software": "editingsoftwares",
            "softwarelist": "editingsoftwares",
            "software_list": "editingsoftwares",
            "softlist": "editingsoftwares",
            "soft": "editingsoftwares",
            "softwares": "editingsoftwares",
            "software list": "editingsoftwares",
            "softwre": "editingsoftwares",
            "softwear": "editingsoftwares",
            "sotware": "editingsoftwares",

            # aecrack variations
            "aecrack": "aecrack",
            "aftereffects": "aecrack",
            "after_effects": "aecrack",
            "after effects": "aecrack",
            "aftereffect": "aecrack",
            "ae": "aecrack",
            "acrack": "aecrack",
            "aecrck": "aecrack",
            "aecrk": "aecrack",
            "after effect": "aecrack",
            "aftereffects crack": "aecrack",
            "ae crack": "aecrack",
            "aec": "aecrack",

            # pscrack variations
            "pscrack": "pscrack",
            "photoshop": "pscrack",
            "photoshop crack": "pscrack",
            "ps": "pscrack",
            "ps crack": "pscrack",
            "photo shop": "pscrack",
            "photo": "pscrack",
            "pscrk": "pscrack",
            "psc": "pscrack",
            "photshop": "pscrack",
            "photoshp": "pscrack",

            # mecrack variations
            "mecrack": "mecrack",
            "mediaencoder": "mecrack",
            "media_encoder": "mecrack",
            "media encoder": "mecrack",
            "me": "mecrack",
            "me crack": "mecrack",
            "media crack": "mecrack",
            "encoder": "mecrack",
            "mecrk": "mecrack",
            "mec": "mecrack",
            "media encoder crack": "mecrack",

            # prcrack variations
            "prcrack": "prcrack",
            "premiere": "prcrack",
            "premierepro": "prcrack",
            "premiere_pro": "prcrack",
            "premiere pro": "prcrack",
            "pr": "prcrack",
            "pr crack": "prcrack",
            "premire": "prcrack",
            "premiere crack": "prcrack",
            "premier": "prcrack",
            "premire pro": "prcrack",
            "prc": "prcrack",
            "primier": "prcrack",
            "premier pro": "prcrack",

            # topazcrack variations
            "topazcrack": "topazcrack",
            "topaz": "topazcrack",
            "topaz crack": "topazcrack",
            "topaz ai": "topazcrack",
            "topazai": "topazcrack",
            "tpz": "topazcrack",
            "topas": "topazcrack",
            "topazvideo": "topazcrack",
            "topaz video": "topazcrack",
            "topz": "topazcrack",
            "topazai crack": "topazcrack",

            # presets variations
            "preset": "presets",
            "presets": "presets",
            "colorpresets": "presets",
            "color_presets": "presets",
            "color presets": "presets",
            "cc": "presets",
            "cc presets": "presets",
            "color correction": "presets",
            "preset list": "presets",
            "colorcorrection": "presets",
            "preest": "presets",
            "prest": "presets",
            "prset": "presets",
            "presetes": "presets",
            "cc files": "presets",
            "cc file": "presets",
            "ffx": "presets",
            "ffx files": "presets",

            # files variations
            "file": "files",
            "files": "files",
            "filess": "files",
            "filee": "files",
            "fies": "files",
            "fils": "files",
            "file list": "files",
            "files list": "files",
            "all files": "files",

            # help variations
            "help": "help",
            "hlp": "help",
            "halp": "help",
            "hellp": "help",
            "hel": "help",

            # hi variations
            "hi": "hi",
            "hello": "hi",
            "hey": "hi",
            "hii": "hi",
            "helo": "hi",

            # list variations (Renamed to !commands)
            "list": "commands",
            "lst": "commands",
            "lis": "commands",
            "lists": "commands",
            "command": "commands",
            "commands": "commands",
            "command list": "commands",
            "cmd": "commands",
            "cmds": "commands",
            "all commands": "commands",
            
            # Setting variations
            "setup_updates": "setup_updates",
            "setup updates": "setup_updates",
            "updates channel": "setup_updates",
            "appeal_link": "appeal_link",
            "appeal link": "appeal_link"
        }

        # Check if the requested command matches exactly, or with spaces, underscores or hyphens removed
        found_match = False
        suggested_command = None

        # First try exact match
        if requested_file_lower in known_commands:
            suggested_command = known_commands[requested_file_lower]
            found_match = True

        # Try without spaces, underscores, or hyphens if no exact match
        if not found_match:
            # Remove spaces, underscores, hyphens and check again
            normalized_request = requested_file_lower.replace(' ', '').replace('_', '').replace('-', '')
            for cmd, suggestion in known_commands.items():
                normalized_cmd = cmd.replace(' ', '').replace('_', '').replace('-', '')
                if normalized_request == normalized_cmd:
                    suggested_command = suggestion
                    found_match = True
                    break

        # Try more flexible matching for typos (check if command is contained in the request)
        if not found_match:
            for cmd, suggestion in known_commands.items():
                # For short commands (3 chars or less), only check exact matches to avoid false positives
                if len(cmd) <= 3 and cmd != requested_file_lower:
                    continue

                # For longer commands, check if the command is a substring or the request is a substring
                if (len(cmd) > 3 and (cmd in requested_file_lower or 
                   (len(requested_file_lower) > 3 and requested_file_lower in cmd))):
                    suggested_command = suggestion
                    found_match = True
                    break

        if found_match and suggested_command is not None:
            await message.channel.send(f"{message.author.mention}, did you mean to use `!{suggested_command}`? Try typing that instead.")
            logger.info(f'Suggested !{suggested_command} instead of !{requested_file}')
        else:
            # Silently ignore not found files to avoid clutter
            pass

@bot.command(name="ban")
async def ban_command(ctx, member: discord.Member = None):
    """Ban a user from the server - Server admin/inviter can use this."""
    # Check if user is server admin (inviter, owner, or has admin perms)
    if not await is_server_admin(ctx.author, ctx.guild):
        admin_name = get_server_admin_name(ctx.guild)
        await ctx.send(f"{ctx.author.mention}, only **{admin_name}** (the person who added me) or server admins can use this command.")
        return
    
    if not member:
        await ctx.send("Who do you want me to ban? Mention someone or provide their username.")
        return
    
    try:
        # Check if bot has permission to ban
        if not ctx.guild.me.guild_permissions.ban_members:
            await ctx.send("❌ I don't have permission to ban members!")
            return
        
        # Check if bot's role is higher than target member's role
        if member.top_role >= ctx.guild.me.top_role:
            await ctx.send(f"❌ I can't ban {member.name} because their role is equal to or higher than mine!")
            logger.warning(f"Can't ban {member.name} - role too high")
            return
        
        # Don't allow banning BMR or the server admin
        if 'bmr' in member.name.lower() or await is_server_admin(member, ctx.guild):
            await ctx.send("🚫 I can't ban this user!")
            return
        
        # Send DM to user before banning
        try:
            view = AppealButtonView(ctx.guild.id, appeal_type="BAN")
            embed = discord.Embed(
                title="🔨 YOU HAVE BEEN BANNED",
                description=f"You have been banned from **{ctx.guild.name}** by `{ctx.author.name}`.\n\nIf you believe this was a mistake, you can appeal using the button below.",
                color=0xFF0000
            )
            await member.send(embed=embed, view=view)
        except:
            pass  # User may have DMs disabled
        
        # Ban the user
        await ctx.guild.ban(member, reason=f"Banned by {ctx.author.name}")
        await ctx.send(f"✓ {member.name} has been **BANNED** from the server. Goodbye! 🚫")
        logger.info(f"{ctx.author.name} banned {member.name}")
        
        # Log the activity
        await log_activity(
            "🔨 User Banned",
            f"**{member.name}** was banned from **{ctx.guild.name}**",
            color=0xFF0000,
            fields={
                "Banned By": ctx.author.name,
                "Server": ctx.guild.name,
                "User": f"{member.name}#{member.discriminator}"
            }
        )
    except discord.Forbidden:
        await ctx.send(f"❌ I don't have permission to ban {member.name}!")
        logger.error(f"Permission denied when trying to ban {member.name}")
    except Exception as e:
        logger.error(f"Error banning user: {str(e)}")
        await ctx.send(f"❌ Error banning user: {str(e)}")

@bot.command(name="timeout")
async def timeout_command(ctx, member: discord.Member = None, duration: str = None):
    """Timeout a user for a specified duration - Server admin/inviter can use this."""
    # Check if user is server admin (inviter, owner, or has admin perms)
    if not await is_server_admin(ctx.author, ctx.guild):
        admin_name = get_server_admin_name(ctx.guild)
        await ctx.send(f"{ctx.author.mention}, only **{admin_name}** (the person who added me) or server admins can use this command.")
        return
    
    if not member:
        await ctx.send("Who do you want me to timeout? Mention someone or provide their username.")
        return
    
    if not duration:
        await ctx.send("How long should I timeout them for? (e.g., 1h, 24h, 1d, 30m)")
        return
    
    try:
        # Parse duration
        duration_lower = duration.lower().strip()
        timeout_seconds = 0
        
        if 'h' in duration_lower:
            hours = int(duration_lower.replace('h', '').strip())
            timeout_seconds = hours * 3600
        elif 'd' in duration_lower:
            days = int(duration_lower.replace('d', '').strip())
            timeout_seconds = days * 86400
        elif 'm' in duration_lower:
            minutes = int(duration_lower.replace('m', '').strip())
            timeout_seconds = minutes * 60
        elif 's' in duration_lower:
            seconds = int(duration_lower.replace('s', '').strip())
            timeout_seconds = seconds
        else:
            await ctx.send("Invalid duration format. Use: 1h, 24h, 1d, 30m, or 60s")
            return
        
        # Check if bot has permission to timeout
        if not ctx.guild.me.guild_permissions.moderate_members:
            await ctx.send("❌ I don't have permission to timeout members!")
            return
        
        # Check if bot's role is higher than target member's role
        if member.top_role >= ctx.guild.me.top_role:
            await ctx.send(f"❌ I can't timeout {member.name} because their role is equal to or higher than mine!")
            logger.warning(f"Can't timeout {member.name} - role too high")
            return
        
        # Don't allow timing out BMR or the server admin
        if 'bmr' in member.name.lower() or await is_server_admin(member, ctx.guild):
            await ctx.send("🚫 I can't timeout this user!")
            return
        
        # Send DM to user before timeout
        try:
            view = AppealButtonView(ctx.guild.id, appeal_type="MUTE")
            embed = discord.Embed(
                title="🔇 YOU HAVE BEEN TIMED OUT",
                description=f"You have been timed out in **{ctx.guild.name}** by `{ctx.author.name}` for `{duration}`.\n\nIf you believe this was a mistake, you can appeal using the button below.",
                color=0xFFA500
            )
            await member.send(embed=embed, view=view)
        except:
            pass  # User may have DMs disabled
        
        # Apply timeout
        timeout_until = datetime.now(timezone.utc) + timedelta(seconds=timeout_seconds)
        await member.timeout(timeout_until, reason=f"Timeout by {ctx.author.name}")
        await ctx.send(f"✓ {member.name} has been **TIMED OUT** for {duration}. 🔇")
        logger.info(f"{ctx.author.name} timed out {member.name} for {duration}")
        
        # Log the activity
        await log_activity(
            "🔇 User Timed Out",
            f"**{member.name}** was timed out in **{ctx.guild.name}**",
            color=0xFFA500,
            fields={
                "Timed Out By": ctx.author.name,
                "Server": ctx.guild.name,
                "Duration": duration,
                "User": f"{member.name}#{member.discriminator}"
            }
        )
    except ValueError:
        await ctx.send("Invalid duration format. Use: 1h, 24h, 1d, 30m, or 60s")
    except discord.Forbidden:
        await ctx.send(f"❌ I don't have permission to timeout {member.name}!")
        logger.error(f"Permission denied when trying to timeout {member.name}")
    except Exception as e:
        logger.error(f"Error timing out user: {str(e)}")
        await ctx.send(f"❌ Error timing out user: {str(e)}")

@bot.command(name="mute")
async def mute_command(ctx, member: discord.Member = None, duration: str = None):
    """Timeout a user (alias for timeout command) - Server admin/inviter can use this."""
    if not member or not duration:
        await ctx.send("Usage: !mute @user 24h")
        return
    await ctx.invoke(timeout_command, member=member, duration=duration)

@bot.command(name="unmute")
async def unmute_command(ctx, member: discord.Member = None):
    """Remove timeout from a user - Server admin/inviter can use this."""
    # Check if user is server admin (inviter, owner, or has admin perms)
    if not await is_server_admin(ctx.author, ctx.guild):
        admin_name = get_server_admin_name(ctx.guild)
        await ctx.send(f"{ctx.author.mention}, only **{admin_name}** (the person who added me) or server admins can use this command.")
        return
    
    if not member:
        await ctx.send("Usage: !unmute @user")
        return
    
    try:
        # Check if bot has permission
        if not ctx.guild.me.guild_permissions.moderate_members:
            await ctx.send("❌ I don't have permission to remove timeouts!")
            return
        
        # Remove timeout by setting it to None
        await member.timeout(None, reason=f"Unmuted by {ctx.author.name}")
        await ctx.send(f"✓ {member.name} has been **UNMUTED**. 🔊")
        logger.info(f"{ctx.author.name} unmuted {member.name}")
        
        # Send DM to user
        try:
            await member.send(f"You have been **UNMUTED** in {ctx.guild.name} by {ctx.author.name}.")
        except:
            pass
        
        # Log the activity
        await log_activity(
            "🔊 User Unmuted",
            f"**{member.name}** was unmuted in **{ctx.guild.name}**",
            color=0x00FF00,
            fields={
                "Unmuted By": ctx.author.name,
                "Server": ctx.guild.name,
                "User": f"{member.name}#{member.discriminator}"
            }
        )
            
    except discord.Forbidden:
        await ctx.send(f"❌ I don't have permission to unmute {member.name}!")
    except Exception as e:
        logger.error(f"Error unmuting user: {str(e)}")
        await ctx.send(f"❌ Error unmuting user: {str(e)}")

# ============================================================================
# UTILITY TOOLS COMMANDS
# ============================================================================

@bot.command(name="remind")
async def remind_command(ctx, time_str: str = None, *, reminder_text: str = None):
    """Set a reminder for a task. Usage: !remind 5m Don't forget the meeting"""
    if not time_str or not reminder_text:
        await ctx.send("Usage: !remind <time> <reminder text>\nExample: !remind 30m Buy groceries")
        return
    
    try:
        # Parse time (5m, 1h, 30s)
        amount = int(''.join(filter(str.isdigit, time_str)))
        unit = ''.join(filter(str.isalpha, time_str)).lower()
        
        delay = 0
        if unit == 'm': delay = amount * 60
        elif unit == 'h': delay = amount * 3600
        elif unit == 's': delay = amount
        else:
            await ctx.send("❌ Use time format like: 5m, 1h, 30s")
            return
        
        user_id = ctx.author.id
        db_manager.save_reminder(user_id, reminder_text, delay)
        
        await ctx.send(f"⏰ Reminder set for {time_str}: **{reminder_text}**")
        
        # Schedule the reminder in the background
        async def run_reminder():
            await asyncio.sleep(delay)
            try:
                await ctx.author.send(f"⏰ **REMINDER**: {reminder_text}")
                db_manager.delete_reminder(user_id, reminder_text)
                logger.info(f"Sent reminder to {ctx.author.name}")
            except:
                pass
        
        asyncio.create_task(run_reminder())
    except Exception as e:
        await ctx.send(f"❌ Error setting reminder: {str(e)}")
        logger.error(f"Reminder error: {str(e)}")

@bot.command(name="note")
async def note_command(ctx, *, note_text: str = None):
    """Save a note for later. Usage: !note Remember to update the profile"""
    user_id = ctx.author.id
    if not note_text:
        notes = db_manager.get_notes(user_id)
        if notes:
            notes_list = "\n".join([f"• {note}" for note in notes])
            await ctx.send(f"📝 **Your Notes:**\n{notes_list}")
        else:
            await ctx.send("📝 You have no saved notes. Use `!note <text>` to save one!")
        return
    
    db_manager.save_note(user_id, note_text)
    await ctx.send(f"✓ Note saved! Use `!note` (without text) to view all notes.")

@bot.command(name="timer")
async def timer_command(ctx, time_str: str = None):
    """Start a countdown timer. Usage: !timer 5m"""
    if not time_str:
        await ctx.send("Usage: !timer <time>\nExample: !timer 5m, !timer 30s, !timer 2h")
        return
    
    try:
        amount = int(''.join(filter(str.isdigit, time_str)))
        unit = ''.join(filter(str.isalpha, time_str)).lower()
        
        if unit == 'm':
            seconds = amount * 60
            display = f"{amount}m"
        elif unit == 'h':
            seconds = amount * 3600
            display = f"{amount}h"
        elif unit == 's':
            seconds = amount
            display = f"{amount}s"
        else:
            await ctx.send("❌ Use time format like: 5m, 1h, 30s")
            return
        
        msg = await ctx.send(f"⏱️ **Timer started**: {display}")
        await asyncio.sleep(seconds)
        await msg.edit(content=f"✓ **Timer finished!** {display} has passed. {ctx.author.mention}")
    except Exception as e:
        await ctx.send(f"❌ Timer error: {str(e)}")

@bot.command(name="convert")
async def convert_command(ctx, mode: str = None, *, text: str = None):
    """Convert text format. Usage: !convert upper hello world"""
    if not mode or not text:
        await ctx.send("Usage: !convert <mode> <text>\nModes: upper, lower, title, reverse, morse")
        return
    
    mode = mode.lower()
    if mode == "upper":
        result = text.upper()
    elif mode == "lower":
        result = text.lower()
    elif mode == "title":
        result = text.title()
    elif mode == "reverse":
        result = text[::-1]
    elif mode == "morse":
        morse_dict = {' ': '/', 'a': '.-', 'b': '-...', 'c': '-.-.', 'd': '-..', 'e': '.', 'f': '..-.',
                      'g': '--.', 'h': '....', 'i': '..', 'j': '.---', 'k': '-.-', 'l': '.-..',
                      'm': '--', 'n': '-.', 'o': '---', 'p': '.--.', 'q': '--.-', 'r': '.-.',
                      's': '...', 't': '-', 'u': '..-', 'v': '...-', 'w': '.--', 'x': '-..-',
                      'y': '-.--', 'z': '--..'}
        result = ' '.join(morse_dict.get(c.lower(), c) for c in text)
    else:
        await ctx.send("❌ Unknown mode. Use: upper, lower, title, reverse, morse")
        return
    
    await ctx.send(f"✓ **{mode.title()}**: {result[:200]}")

@bot.command(name="emoji")
async def emoji_command(ctx, *, text: str = None):
    """Get emoji suggestions based on your text"""
    if not text:
        await ctx.send("Usage: !emoji happy mood")
        return
    
    try:
        prompt = f"Suggest 5 relevant emojis for: {text}. Just list the emojis separated by space."
        response = await get_gemini_response(prompt, ctx.author.id, username=ctx.author.name, guild_id=ctx.guild.id if ctx.guild else None)
        await ctx.send(f"😊 **Emojis for '{text}'**: {response[:100]}")
    except Exception as e:
        await ctx.send(f"❌ Error: {str(e)}")

@bot.command(name="calculate")
async def calculate_command(ctx, *, expression: str = None):
    """Do quick math. Usage: !calculate 50+25*2"""
    if not expression:
        await ctx.send("Usage: !calculate <math expression>\nExample: !calculate 100+50/2")
        return
    
    try:
        expression = expression.replace('^', '**')
        result = eval(expression, {"__builtins__": {}}, {})
        await ctx.send(f"🧮 **Result**: {expression} = **{result}**")
    except:
        await ctx.send("❌ Invalid math expression. Use: +, -, *, /, ^(power), %, etc")

@bot.command(name="weather")
async def weather_command(ctx, *, location: str = None):
    """Get weather for any location. Usage: !weather New York"""
    if not location:
        await ctx.send("Usage: !weather <city name>\nExample: !weather London")
        return
    
    try:
        url = f"https://wttr.in/{location}?format=3"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            await ctx.send(f"🌤️ **Weather in {location}**: {response.text}")
        else:
            await ctx.send(f"❌ Couldn't find weather for '{location}'")
    except:
        await ctx.send("❌ Weather service unavailable. Try again later!")

@bot.command(name="profile", aliases=["p"])
async def profile_command(ctx, member: discord.Member = None):
    """Show the user's Prime Portfolio or Profile."""
    member = member or ctx.author
    
    # Logic: If author hasn't set their link, ask for it interactively
    work_link = user_portfolios.get(member.id)
    if not work_link and member == ctx.author:
        prompt_msg = await ctx.send(
            f"👋 {ctx.author.mention}, you haven't linked your portfolio yet!\n"
            f"Please send your YouTube/Portfolio link below now to complete your card (or type `cancel`)."
        )
        
        def check(m):
            return m.author == ctx.author and m.channel == ctx.channel
            
        try:
            msg = await bot.wait_for('message', check=check, timeout=60.0)
            if msg.content.lower() == 'cancel':
                await ctx.send("Cancelled.")
                return
            if msg.content.startswith(("http://", "https://")):
                user_portfolios[ctx.author.id] = msg.content
                save_portfolios(user_portfolios)
                work_link = msg.content
                await ctx.send("✅ Link saved! Generating your card...", delete_after=5)
            else:
                await ctx.send("❌ That doesn't look like a link. I'll show your basic card for now.")
        except asyncio.TimeoutError:
            await ctx.send("⌛ Timed out. Showing basic card.", delete_after=5)

    async with ctx.typing():
        level_data = user_levels.get(ctx.guild.id, {}).get(member.id, {"level": 0, "xp": 0})
        
        # Generate the card image
        try:
            image_bytes = await generate_portfolio_card(member, level_data, work_link)
            file = discord.File(image_bytes, filename=f"portfolio_{member.id}.png")
            
            embed = discord.Embed(
                title=f"💎 PRIME PORTFOLIO | {member.display_name}", 
                description=f"View {member.mention}'s professional editing profile.",
                color=0x00FFB4
            )
            embed.set_image(url=f"attachment://portfolio_{member.id}.png")
            
            if work_link:
                embed.add_field(name="🔗 FEATURED WORK", value=f"[Click to Watch]({work_link})")
            else:
                embed.add_field(name="🔗 WORK", value="*Not yet linked*")
                
            embed.set_footer(text="Prime Leveling • Exclusive Edition", icon_url=bot.user.display_avatar.url)
            
            await ctx.send(file=file, embed=embed)
        except Exception as e:
            logger.error(f"Error generating portfolio card: {e}")
            await ctx.send(f"❌ Error: {str(e)}")

@bot.group(name="portfolio", invoke_without_command=True)
async def portfolio_group(ctx):
    """Manage your portfolio work link. Usage: !portfolio add [link] or !portfolio remove"""
    await ctx.send("📋 **Portfolio Commands:**\n`!portfolio add [link]` - Link your work\n`!portfolio remove` - Remove your link\n`!profile` - View your portfolio card")

@portfolio_group.command(name="add")
async def portfolio_add(ctx, link: str = None):
    """Add or update your portfolio link."""
    if not link:
        await ctx.send("❌ Please provide a link! Usage: `!portfolio add [link]`")
        return
    
    if not link.startswith(("http://", "https://")):
        await ctx.send("❌ That's not a valid link! Make sure it starts with `https://`.")
        return

    # Assuming user_portfolios is a global dict or we need to update DB
    user_portfolios[ctx.author.id] = link
    save_portfolios(user_portfolios)
    
    await ctx.send(f"✅ Portfolio link updated to: <{link}>\nRun `!profile` to see your updated card!")


@portfolio_group.command(name="remove")
async def portfolio_remove(ctx):
    """Remove your portfolio link."""
    if ctx.author.id in user_portfolios:
        del user_portfolios[ctx.author.id]
        save_portfolios(user_portfolios)
        await ctx.send("🗑️ **Portfolio link removed.**")
    else:
        await ctx.send("❌ You don't have a portfolio link set.")

@bot.command(name="serverinfo")
async def serverinfo_command(ctx):
    """Display server information and statistics"""
    guild = ctx.guild
    if not guild:
        await ctx.send("❌ This command only works in servers!")
        return
    
    embed = discord.Embed(title=f"Server Info - {guild.name}", color=0x5865F2)
    embed.add_field(name="Members", value=guild.member_count, inline=True)
    embed.add_field(name="Channels", value=len(guild.channels), inline=True)
    embed.add_field(name="Roles", value=len(guild.roles), inline=True)
    embed.add_field(name="Owner", value=guild.owner.mention if guild.owner else "Unknown", inline=False)
    embed.add_field(name="Created", value=guild.created_at.strftime("%B %d, %Y"), inline=False)
    embed.add_field(name="Verification Level", value=str(guild.verification_level).title(), inline=False)
    embed.set_thumbnail(url=guild.icon.url if guild.icon else None)
    
    await ctx.send(embed=embed)

@bot.command(name="invite")
async def invite_command(ctx):
    """Get the official invite link for PRIME"""
    invite_url = f"https://discord.com/oauth2/authorize?client_id={bot.user.id}&permissions=8&scope=bot"
    
    embed = discord.Embed(
        title="🚀 ADD PRIME TO YOUR SERVER",
        description=(
            "Bring the ultimate AI power to your community.\n\n"
            f"**[Click Here to Invite PRIME]({invite_url})**\n\n"
            "**Official Website:** [prime.helperbot.fun](https://prime.helperbot.fun)"
        ),
        color=0x00FFB4
    )
    embed.set_footer(text="Prime Collective | Powered by BMR")
    embed.set_thumbnail(url=bot.user.display_avatar.url)
    
    await ctx.send(embed=embed)

@bot.command(name="vote")
async def vote_command(ctx):
    """Vote for PRIME on bot lists and support growth!"""
    embed = discord.Embed(
        title="💎 SUPPORT PRIME GROWTH",
        description=(
            "Voting helps PRIME reach more creators and communities.\n\n"
            "🗳️ **[Vote on Top.gg](https://top.gg/bot/1466029119275012139/vote)**\n"
            "🗳️ **[Vote on DiscordBotList.com](https://discordbotlist.com/bots/1466029119275012139/vote)**\n\n"
            "*Your support keeps the updates rolling. Thank you!*"
        ),
        color=0x00FFB4
    )
    embed.set_footer(text="Prime Collective | Powered by BMR")
    embed.set_thumbnail(url=bot.user.display_avatar.url)
    
    await ctx.send(embed=embed)

# ============================================================================
# CREATIVE TOOLS COMMANDS
# ============================================================================

@bot.command(name="creative")
async def creative_command(ctx, *, topic: str = None):
    """Generate creative ideas & prompts"""
    if not topic:
        await ctx.send("Usage: !creative [topic/idea]\nExample: !creative sci-fi story")
        return
    
    try:
        prompt = f"Generate 5 creative and unique ideas, prompts, or concepts for: {topic}. Be imaginative and innovative."
        response = await get_gemini_response(prompt, ctx.author.id, username=ctx.author.name, guild_id=ctx.guild.id if ctx.guild else None)
        await ctx.send(f"💡 **Creative Ideas for '{topic}'**:\n{response[:1900]}")
    except Exception as e:
        await ctx.send(f"❌ Error: {str(e)}")

@bot.command(name="story")
async def story_command(ctx, *, prompt: str = None):
    """Create short stories instantly"""
    if not prompt:
        await ctx.send("Usage: !story [story prompt]\nExample: !story a mysterious door")
        return
    
    try:
        gemini_prompt = f"Write a creative short story (3-4 paragraphs) based on: {prompt}"
        response = await get_gemini_response(gemini_prompt, ctx.author.id, username=ctx.author.name, guild_id=ctx.guild.id if ctx.guild else None)
        await ctx.send(f"📖 **Story**: {response[:1900]}")
    except Exception as e:
        await ctx.send(f"❌ Error: {str(e)}")

@bot.command(name="quote")
async def quote_command(ctx, style: str = None):
    """Produce inspirational or funny quotes"""
    if not style:
        await ctx.send("Usage: !quote [inspirational/funny/random]\nExample: !quote inspirational")
        return
    
    try:
        prompt = f"Generate an original {style} quote that is meaningful and memorable."
        response = await get_gemini_response(prompt, ctx.author.id, username=ctx.author.name, guild_id=ctx.guild.id if ctx.guild else None)
        await ctx.send(f"✨ **Quote**: {response[:500]}")
    except Exception as e:
        await ctx.send(f"❌ Error: {str(e)}")

@bot.command(name="brainstorm")
async def brainstorm_command(ctx, *, topic: str = None):
    """Brainstorm ideas with AI"""
    if not topic:
        await ctx.send("Usage: !brainstorm [topic]\nExample: !brainstorm content ideas for youtube")
        return
    
    try:
        prompt = f"Brainstorm 8 creative and practical ideas for: {topic}. List them clearly with brief explanations."
        response = await get_gemini_response(prompt, ctx.author.id, username=ctx.author.name, guild_id=ctx.guild.id if ctx.guild else None)
        await ctx.send(f"🧠 **Brainstorm Results**:\n{response[:1900]}")
    except Exception as e:
        await ctx.send(f"❌ Error: {str(e)}")

@bot.command(name="design")
async def design_command(ctx, *, project: str = None):
    """Suggest design themes & creative layouts"""
    if not project:
        await ctx.send("Usage: !design [project description]\nExample: !design website for tech startup")
        return
    
    try:
        prompt = f"Suggest 5 design themes, color schemes, and layout ideas for: {project}. Be specific and modern."
        response = await get_gemini_response(prompt, ctx.author.id, username=ctx.author.name, guild_id=ctx.guild.id if ctx.guild else None)
        await ctx.send(f"🎨 **Design Suggestions**:\n{response[:1900]}")
    except Exception as e:
        await ctx.send(f"❌ Error: {str(e)}")

@bot.command(name="name")
async def name_command(ctx, category: str = None):
    """Generate usernames, bot names, brand names"""
    if not category:
        await ctx.send("Usage: !name [username/brand/bot]\nExample: !name gaming_username")
        return
    
    try:
        prompt = f"Generate 10 creative, catchy, and memorable {category} names. They should be unique and cool."
        response = await get_gemini_response(prompt, ctx.author.id, username=ctx.author.name, guild_id=ctx.guild.id if ctx.guild else None)
        await ctx.send(f"✍️ **Name Ideas**:\n{response[:1900]}")
    except Exception as e:
        await ctx.send(f"❌ Error: {str(e)}")

@bot.command(name="aesthetic")
async def aesthetic_command(ctx, style: str = None):
    """Suggest color palettes + aesthetic styles"""
    if not style:
        await ctx.send("Usage: !aesthetic [aesthetic style]\nExample: !aesthetic cyberpunk")
        return
    
    try:
        prompt = f"Suggest a complete {style} aesthetic with: color palette (hex codes), typography, mood, and design elements."
        response = await get_gemini_response(prompt, ctx.author.id, username=ctx.author.name, guild_id=ctx.guild.id if ctx.guild else None)
        await ctx.send(f"🎭 **{style.title()} Aesthetic**:\n{response[:1900]}")
    except Exception as e:
        await ctx.send(f"❌ Error: {str(e)}")

@bot.command(name="get")
async def get_asset(ctx, category=None, *, query=None):
    """Smart Asset Finder for editors. Usage: !get [sfx/overlay/font/stock] [description]"""
    if not category or not query:
        embed = discord.Embed(
            title="🔍 PRIME ASSET FINDER",
            description="Find high-quality assets for your edits instantly!",
            color=0x00FFB4
        )
        embed.add_field(name="🚀 Usage", value="`!get [category] [description]`")
        embed.add_field(name="📁 Categories", value="`sfx`, `overlay`, `font`, `stock`, `pfp`, `banner`")
        embed.add_field(name="💡 Example", value="`!get sfx cinematic impact` or `!get font futuristic cyberpunk`", inline=False)
        embed.set_footer(text="Prime creative tools • Developed by BMR")
        await ctx.send(embed=embed)
        return

    async with ctx.typing():
        try:
            # Use Gemini to find/suggest the best links
            asset_prompt = f"""
            You are a professional creative asset scout for video editors.
            A user is looking for assets in the category: '{category}' with description: '{query}'.
            
            Find the TOP 3-5 high-quality, free, or royalty-free websites where they can find EXACTLY this.
            Return a JSON object with:
            1. 'recommendation': A short (15-20 words) professional advice on what specifically to look for.
            2. 'sites': A dictionary where keys are site names and values are the DIRECT search URLs (if possible) or homepages.
            
            Example search URL patterns:
            - Mixkit SFX: https://mixkit.co/free-sound-effects/[query]/
            - DaFont: https://www.dafont.com/search.php?q=[query]
            - Pixabay: https://pixabay.com/search/?q=[query]
            
            Return ONLY the JSON.
            """
            
            response = gemini_client.models.generate_content(
                model=PRIMARY_MODEL,
                contents=asset_prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            
            if response.text:
                data = json.loads(response.text)
                reco = data.get('recommendation', 'Explore these resources to find the perfect asset for your project.')
                sites = data.get('sites', {})
                
                embed = discord.Embed(
                    title=f"📦 ASSET RESULTS: {category.upper()}",
                    description=f"**Search Query:** `{query}`\n\n💡 {reco}",
                    color=0x00FFB4,
                    timestamp=datetime.now(timezone.utc)
                )
                
                if not sites:
                    await ctx.send("❌ I couldn't find any specific sites for this. Try a different description!")
                    return

                view = AssetFinderView(sites)
                await ctx.send(embed=embed, view=view)
            else:
                await ctx.send("❌ Error contacting the asset scout. Please try again.")

        except Exception as e:
            logger.error(f"Asset Finder error: {e}")
            await ctx.send("❌ An error occurred while searching for assets. Please try again later.")

@bot.command(name="topics")
async def topics_command(ctx, context: str = None):
    """Give conversation or content topics"""
    if not context:
        await ctx.send("Usage: !topics [context]\nExample: !topics social media content")
        return
    
    try:
        prompt = f"Generate 10 interesting and engaging topics for: {context}. Make them relevant and trending."
        response = await get_gemini_response(prompt, ctx.author.id, username=ctx.author.name, guild_id=ctx.guild.id if ctx.guild else None)
        await ctx.send(f"📋 **Topic Ideas**:\n{response[:1900]}")
    except Exception as e:
        await ctx.send(f"❌ Error: {str(e)}")

@bot.command(name="motivate")
async def motivate_command(ctx):
    """Send motivational messages"""
    try:
        prompt = "Generate a short, powerful motivational message that will inspire someone to take action today."
        response = await get_gemini_response(prompt, ctx.author.id, username=ctx.author.name, guild_id=ctx.guild.id if ctx.guild else None)
        await ctx.send(f"💪 **Motivation**: {response[:500]}")
    except Exception as e:
        await ctx.send(f"❌ Error: {str(e)}")

# ============================================================================
# SUPREME AI FEATURES (NEW)
# ============================================================================

@bot.command(name="sync")
async def sync_beat_map(ctx):
    """AI Beat Mapping. Upload an audio file for a frame-map."""
    if not ctx.message.attachments:
        await ctx.send("❌ **Error**: Please upload an audio file (MP3/WAV) to map its beat structure.")
        return

    attachment = ctx.message.attachments[0]
    if not any(attachment.filename.lower().endswith(ext) for ext in ['.mp3', '.wav', '.m4a']):
        await ctx.send("❌ **Unsupported Format**: Only MP3, WAV, and M4A are supported for beat mapping.")
        return

    async with ctx.typing():
        try:
            audio_bytes = await attachment.read()
            
            prompt = """
            Analyze this audio file as a high-end music video editor. 
            Identify the major drops, snares, and high-impact moments.
            Provide a Frame-Sync Map for a 60fps composition.
            
            Return format:
            1. **Beat Map**: A list of key frames for impacts (e.g., "F-120: Main Drop").
            2. **Velocity Logic**: Suggest where to ramp speed (e.g., "F-60 to F-120: Slow ramp 100%->30%").
            3. **Style Analysis**: Recommended CC and look.
            
            Be technical and precise.
            """
            
            # Using the same bytes-sending logic as video/images but for audio
            response = gemini_client.models.generate_content(
                model=PRIMARY_MODEL,
                contents=[
                    types.Part.from_bytes(data=audio_bytes, mime_type=attachment.content_type or "audio/mpeg"),
                    prompt
                ]
            )
            
            embed = discord.Embed(
                title="🎼 BEAT SYNC MAP",
                description=response.text[:4000],
                color=0x00FFB4,
                timestamp=datetime.now(timezone.utc)
            )
            embed.set_footer(text="Prime | Beat Mapping")
            await ctx.send(embed=embed)
            
        except Exception as e:
            logger.error(f"Sync command error: {e}")
            await ctx.send(BOT_ERROR_MSG)

@bot.command(name="stack")
async def style_helper(ctx, *, description: str = None):
    """Style Helper. Describe a look to get the plugin stack."""
    if not description:
        await ctx.send("❌ **Query Required**: Describe the style (e.g., '!stack high contrast grainy anime glow').")
        return

    async with ctx.typing():
        try:
            prompt = f"""
            You are a Grandmaster VFX Architect. The user wants to achieve this look: '{description}'.
            Provide an EXACT plugin stack for After Effects. 
            Include:
            1. Plugins needed (Sapphire, BCC, Universe, Deep Glow, etc.)
            2. EXACT slider values (e.g., 'Glow Radius: 320, Intensity: 1.2')
            3. Layering order.
            4. Blend modes.
            
            Be extremely technical and professional.
            """
            response = await get_gemini_response(prompt, ctx.author.id, username=ctx.author.name, guild_id=ctx.guild.id if ctx.guild else None)
            
            embed = discord.Embed(
                title="🦾 VFX PLUGIN STACK",
                description=f"**Style Target:** `{description}`\n\n{response[:3800]}",
                color=0x00FFB4
            )
            embed.set_footer(text="Prime | Creative Stack")
            await ctx.send(embed=embed)
        except Exception as e:
            await ctx.send(BOT_ERROR_MSG)

@bot.command(name="server")
@commands.has_permissions(administrator=True)
async def server_stats_command(ctx):
    """Server Analytics. Administrator Access Only."""
    async with ctx.typing():
        try:
            # Gather server data points
            total_members = ctx.guild.member_count
            active_today = len([m for m in ctx.guild.members if m.status != discord.Status.offline])
            
            # Use Gemini to generate an "Analytics Report" based on current server status
            prompt = f"""
            Generate a creative report for the server '{ctx.guild.name}'.
            Data: Total Members: {total_members}, Currently Active: {active_today}.
            
            Format as:
            1. **Vibe Check**: Friendly analysis of how the chat is feeling.
            2. **Future Growth**: Where do you see this community going?
            3. **Chat Health**: How's the balance between projects and general talk?
            4. **Pro Advice**: One solid tip to make the server even better.
            
            Speak naturally and avoid robotic terms.
            """
            response = await get_gemini_response(prompt, ctx.author.id, username=ctx.author.name, guild_id=ctx.guild.id if ctx.guild else None)
            
            embed = discord.Embed(
                title="📊 SERVER REPORT",
                description=response[:4000],
                color=0x00FFB4,
                timestamp=datetime.now(timezone.utc)
            )
            embed.set_footer(text="Prime | Server Analysis")
            await ctx.send(embed=embed)
        except Exception as e:
            await ctx.send(BOT_ERROR_MSG)

@bot.command(name="pulse")
async def prime_pulse(ctx):
    """Summarize recent server activity and vibes."""
    async with ctx.typing():
        try:
            messages = []
            async for msg in ctx.channel.history(limit=50):
                if not msg.author.bot:
                    messages.append(f"{msg.author.name}: {msg.content}")
            
            if not messages:
                await ctx.send("❌ **Nothing found**: No recent activity to analyze.")
                return
            
            chat_blob = "\n".join(messages[::-1])
            prompt = f"""
            Analyze this recent chat activity and provide a summary.
            Who was active? What's the current 'vibe' of the channel?
            
            CHAT LOGS:
            {chat_blob}
            
            Be sharp and concise. Talk like a human member of the crew.
            """
            
            summary = await get_gemini_response(prompt, ctx.author.id, username=ctx.author.name, mode="briefing", guild_id=ctx.guild.id if ctx.guild else None)
            
            embed = discord.Embed(
                title="📡 RECENT CHAT RECAP",
                description=summary[:4000],
                color=0x00FFB4,
                timestamp=datetime.now(timezone.utc)
            )
            embed.set_footer(text="Prime | Chat Recap")
            await ctx.send(embed=embed)
        except Exception as e:
            await ctx.send(BOT_ERROR_MSG)

@bot.command(name="think")
async def think_command(ctx, *, query: str):
    """Thinking Mode: Use advanced reasoning for complex problems."""
    msg = await ctx.send("🧠 **PRIME THOUGHT ENGINE**\n*Analyzing context and mapping logical vectors...*")
    async with ctx.typing():
        try:
            # We add a bit of drama
            await asyncio.sleep(1.5)
            await msg.edit(content="🧠 **PRIME THOUGHT ENGINE**\n*Simulating outcomes and identifying optimal paths...*")
            
            response = await get_gemini_response(query, ctx.author.id, username=ctx.author.name, use_thought=True, guild_id=ctx.guild.id if ctx.guild else None)
            
            embed = discord.Embed(
                title="🧠 DEEP THOUGHT ANALYSIS",
                description=response[:4000],
                color=0x00FFB4
            )
            embed.set_footer(text=f"Requested by {ctx.author.name} • Logic Tier: Elite")
            await msg.edit(content=None, embed=embed)
        except Exception as e:
            await msg.edit(content=f"❌ **Thread Interrupted**: {str(e)}")

@bot.command(name="briefing")
async def executive_briefing(ctx):
    """Get an elite executive briefing on current server and global status."""
    async with ctx.typing():
        try:
            # Get internal status
            active_members = len([m for m in ctx.guild.members if m.status != discord.Status.offline])
            
            # Get external trends via search
            from brain import search_and_summarize
            trends = await search_and_summarize("latest creative technology and editing trends 2026")
            
            prompt = f"Server: {ctx.guild.name}. Active Members: {active_members}. Global Trends: {trends}. Give an elite briefing."
            response = await get_gemini_response(prompt, ctx.author.id, username=ctx.author.name, mode="briefing", guild_id=ctx.guild.id if ctx.guild else None)
            
            embed = discord.Embed(
                title="💼 EXECUTIVE BRIEFING",
                description=response[:4000],
                color=0x00FFB4,
                timestamp=datetime.now(timezone.utc)
            )
            embed.set_footer(text="Prime | Executive Intel")
            await ctx.send(embed=embed)
        except Exception as e:
            await ctx.send(BOT_ERROR_MSG)

    await ctx.send(embed=embed)

@bot.command(name="overlay")
@commands.has_permissions(administrator=True)
async def aesthetic_overlay_command(ctx, aesthetic: str = None):
    """Set the server's aesthetic overlay (mood/personality). Usage: !overlay [cyberpunk/minimalist/chaos]"""
    if not aesthetic:
        embed = discord.Embed(
            title="🎭 AESTHETIC OVERLAYS",
            description="Transform Prime's personality for this server.",
            color=0x00FFB4
        )
        embed.add_field(name="Available Styles", value="`cyberpunk`, `minimalist`, `chaos`, `lofi`, `aggressive`, `professional`")
        embed.add_field(name="Current", value=f"`{db_manager.get_guild_setting(ctx.guild.id, 'aesthetic_overlay', 'Standard')}`")
        await ctx.send(embed=embed)
        return

    valid_styles = ['cyberpunk', 'minimalist', 'chaos', 'lofi', 'aggressive', 'professional', 'reset']
    if aesthetic.lower() not in valid_styles:
        await ctx.send(f"❌ **Invalid Style**: Choose from {', '.join(valid_styles)}")
        return

    if aesthetic.lower() == 'reset':
        db_manager.save_guild_setting(ctx.guild.id, "aesthetic_overlay", None)
        await ctx.send("🌌 **System Reset**: Prime has returned to standard personality.")
    else:
        db_manager.save_guild_setting(ctx.guild.id, "aesthetic_overlay", aesthetic.lower())
        await ctx.send(f"🎭 **Overlay Applied**: Prime is now in **{aesthetic.upper()}** mode for this server.")

@bot.command(name="setup_updates")
async def setup_updates(ctx, channel: discord.TextChannel = None):
    """Set the channel for bot updates. Usage: !setup_updates #channel"""
    if not await is_server_admin(ctx.author, ctx.guild):
        await ctx.reply("💬 Only server admins can configure bot updates.")
        return
        
    if not channel:
        await ctx.reply("❓ Please mention a channel: `!setup_updates #updates`")
        return
        
    db_manager.save_guild_setting(ctx.guild.id, "update_channel_id", channel.id)
    await ctx.reply(f"✅ **Success!** Bot updates will now be sent to {channel.mention}.")
    logger.info(f"Update channel set to {channel.id} for guild {ctx.guild.id}")

@bot.command(name="appeal_link")
async def appeal_link(ctx, member: discord.Member = None):
    """Send an appeal button to a member. Usage: !appeal_link @user"""
    if not await is_server_admin(ctx.author, ctx.guild):
        return
        
    if not member:
        await ctx.reply("❓ Mention the user you want to send the appeal link to.")
        return
        
    try:
        view = AppealButtonView(ctx.guild.id)
        embed = discord.Embed(
            title="⚖️ Appeal Opportunity",
            description=f"You have been given the chance to appeal your recent moderation in **{ctx.guild.name}**. Click the button below to explain your case.",
            color=0x5865F2
        )
        await member.send(embed=embed, view=view)
        await ctx.reply(f"✅ Sent an appeal link to **{member.display_name}**.")
    except Exception as e:
        await ctx.reply(f"❌ Could not DM user: they might have DMs closed.")

@bot.command(name="broadcast_update")
async def broadcast_update(ctx, *, message: str):
    """[OWNER ONLY] Send an update to all servers that have an update channel configured."""
    # Check if user is owner or BMR
    if not await bot.is_owner(ctx.author) and "bmr" not in ctx.author.name.lower():
        return
        
    await ctx.send("📡 **Starting global broadcast...**")
    count = 0
    failed = 0
    
    embed = discord.Embed(
        title="📢 SYSTEM UPDATE",
        description=message,
        color=0x00FFB4,
        timestamp=datetime.now(timezone.utc)
    )
    embed.set_footer(text="Prime Collective | Technical Intelligence")
    
    for guild in bot.guilds:
        channel_id = db_manager.get_guild_setting(guild.id, "update_channel_id")
        if channel_id:
            channel = guild.get_channel(channel_id)
            if channel:
                try:
                    await channel.send(embed=embed)
                    count += 1
                except:
                    failed += 1
                    
    await ctx.send(f"✅ **Broadcast Complete!** Sent to `{count}` servers. (Failed: `{failed}`)")

@bot.command(name="strategize")
async def strategize_command(ctx, *, query: str):
    """Decision Architect: Map out a strategic roadmap for your project."""
    async with ctx.typing():
        try:
            response = await get_gemini_response(query, ctx.author.id, username=ctx.author.name, mode="architect", guild_id=ctx.guild.id if ctx.guild else None)
            embed = discord.Embed(
                title="🏗️ STRATEGIC ROADMAP",
                description=response[:4000],
                color=0x00FFB4,
                timestamp=datetime.now(timezone.utc)
            )
            embed.set_footer(text="Prime | Decision Architect")
            await ctx.send(embed=embed)
        except Exception as e:
            await ctx.send(BOT_ERROR_MSG)

@bot.command(name="pulse_check")
async def manual_pulse(ctx, *, query: str = "latest video editing trends"):
    """Search and summarize current trends or news."""
    async with ctx.typing():
        try:
            from brain import search_and_summarize
            summary = await search_and_summarize(query)
            await ctx.send(f"🛰️ **Pulse Analysis: {query}**\n\n{summary}")
        except Exception as e:
            await ctx.send(BOT_ERROR_MSG)

@bot.command(name="aura")
async def show_aura(ctx, member: discord.Member = None):
    """Analyze a user's vibe and profile."""
    member = member or ctx.author
    async with ctx.typing():
        user_memory = db_manager.get_user_memory(member.id)
        if not user_memory:
            await ctx.send(f"❌ **No data**: {member.display_name} hasn't talked enough for an analysis yet.")
            return

        summary = user_memory.get('profile_summary', 'No summary available.')
        vibe = user_memory.get('vibe', 'Neutral')
        
        embed = discord.Embed(
            title=f"🔮 USER PROFILE: {member.display_name}",
            description=f"**Current Vibe:** `{vibe.upper()}`\n\n**AI Profile Summary:**\n{summary}",
            color=0x00FFB4,
            timestamp=datetime.now(timezone.utc)
        )
        embed.set_thumbnail(url=member.display_avatar.url if member.display_avatar else None)
        embed.set_footer(text="Prime | Identity Analysis")
        await ctx.send(embed=embed)

@bot.command(name="clout")
async def clout_analysis(ctx):
    """Analyze viral potential for clips and images."""
    if not ctx.message.attachments:
        await ctx.send("❌ **Upload something**: Please upload an edit (Video/Image) for viral potential analysis.")
        return

    attachment = ctx.message.attachments[0]
    is_video = any(attachment.filename.lower().endswith(ext) for ext in ['.mp4', '.avi', '.mkv', '.webm'])
    
    async with ctx.typing():
        try:
            file_bytes = await attachment.read()
            prompt = """
            Analyze this creative work for 'Viral Clout Potential'. 
            As an elite social media strategist, provide:
            1. **Clout Score (1-100)**: Based on quality and 'vibe'.
            2. **Retention Hook**: Analysis of the first 3 seconds.
            3. **Engagement Analysis**: Likely comments/shares and target audience.
            4. **Optimization**: One specific edit to make it go viral.
            
            Be sharp and honest.
            """
            
            if is_video:
                response = await analyze_video(file_bytes, attachment.filename, ctx.author.id)
            else:
                # Get the response text
                response = await get_gemini_response(prompt, ctx.author.id, username=ctx.author.name, image_bytes=file_bytes, guild_id=ctx.guild.id if ctx.guild else None)
            
            # Since analyze_video returns long text, we use Gemini specifically for Clout if it's broad
            # For simplicity, we just use the response if it's descriptive enough
            
            embed = discord.Embed(
                title="📈 VIRAL ANALYSIS",
                description=response[:4000],
                color=0x00FFB4,
                timestamp=datetime.now(timezone.utc)
            )
            embed.set_footer(text="Prime | Engagement Analysis")
            await ctx.send(embed=embed)
        except Exception as e:
            await ctx.send(BOT_ERROR_MSG)

@bot.command(name="mimic")
async def mimic_mode(ctx, target: discord.Member, *, prompt: str = "Hello everyone!"):
    """Mimic a user's vibe and slang."""
    async with ctx.typing():
        try:
            # Fetch target's recent messages from DB or channel
            history = []
            async for m in ctx.channel.history(limit=100):
                if m.author.id == target.id and not m.content.startswith('!'):
                    history.append(m.content)
                if len(history) >= 20: break
            
            if not history:
                await ctx.send(f"❌ **Prime Gap**: Not enough data to mimic {target.display_name}.")
                return

            vibe_data = "\n".join(history)
            mimic_prompt = f"""
            SYSTEM: MIMIC MODE ACTIVE.
            TARGET USER: {target.display_name}
            PAST DATA:
            {vibe_data}
            
            TASK: Respond to the following prompt strictly as if you were this user. 
            Match their slang, typos, capitalization, and emoji usage. Do NOT break character.
            PROMPT: {prompt}
            """
            
            response = await get_gemini_response(mimic_prompt, ctx.author.id, username=ctx.author.name, guild_id=ctx.guild.id if ctx.guild else None)
            await ctx.send(f"👤 **{target.display_name} (AI Mimic):** {response}")
        except Exception as e:
            await ctx.send(BOT_ERROR_MSG)

@bot.command(name="bio")
async def identity_report(ctx, member: discord.Member = None):
    """Generate a professional creator bio."""
    member = member or ctx.author
    async with ctx.typing():
        user_memory = db_manager.get_user_memory(member.id)
        if not user_memory:
            await ctx.send(f"❌ **No data**: {member.display_name} has no activity to create a bio from.")
            return

        summary = user_memory.get('profile_summary', 'No summary.')
        prompt = f"Convert this raw server interaction summary into a professional, highly impressive 'Creative Bio': {summary}"
        bio = await get_gemini_response(prompt, ctx.author.id, username=ctx.author.name, guild_id=ctx.guild.id if ctx.guild else None)
        
        # Report as file
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode='w', encoding='utf-8') as tf:
            tf.write(f"PRIME IDENTITY REPORT\n{'='*30}\nSUBJECT: {member.name}\nDesignation: {user_memory.get('vibe', 'Unknown').upper()}\n\n{bio}")
            temp_path = tf.name
            
        embed = discord.Embed(
            title="📑 CREATIVE BIO EXPORT",
            description=f"Generated a professional bio for **{member.name}**.",
            color=0x00FFB4
        )
        embed.set_footer(text="Prime | Bio Export")
        await ctx.send(embed=embed, file=discord.File(temp_path, filename=f"bio_{member.name.lower()}.txt"))
        
        try: os.remove(temp_path)
        except: pass

@bot.command(name="snipe")
async def chat_snipe(ctx):
    """Retrieve the last deleted message."""
    chan_id = ctx.channel.id
    if chan_id not in deleted_messages or not deleted_messages[chan_id]:
        await ctx.send("❌ **Clear**: No recently deleted messages found here.")
        return

    msg = deleted_messages[chan_id].pop()
    async with ctx.typing():
        # Roast or analyze the deletion
        prompt = f"The user '{msg['author']}' just deleted this message: '{msg['content']}'. Give a sharp, technical, and slightly roast-heavy analysis of why they might have deleted it or what it says about their vibe."
        roast = await get_gemini_response(prompt, ctx.author.id, username=ctx.author.name, guild_id=ctx.guild.id if ctx.guild else None)
        
        embed = discord.Embed(
            title="🎯 SNIPED MESSAGE",
            description=f"**Author:** {msg['author']}\n**Content:** {msg['content']}\n\n**AI Analysis:**\n{roast}",
            color=0xFF0055,
            timestamp=msg['time']
        )
        embed.set_footer(text="Prime | Sniper")
        await ctx.send(embed=embed)

@bot.command(name="duel")
async def creative_duel(ctx, opponent: discord.Member):
    """Creative Face-off. AI challenges two users to a technical duel."""
    if opponent == ctx.author:
        await ctx.send("❌ **Error**: You cannot duel yourself.")
        return

    async with ctx.typing():
        prompt = f"Generate a technical 'Creative Duel' prompt for two video editors: {ctx.author.name} and {opponent.name}. The prompt should be a specific, complex editing scenario they have to solve with words. Make it tuff."
        duel_prompt = await get_gemini_response(prompt, ctx.author.id, username=ctx.author.name, guild_id=ctx.guild.id if ctx.guild else None)
        
        embed = discord.Embed(
            title="⚔️ CREATIVE DUEL",
            description=f"**Competitors:** {ctx.author.mention} vs {opponent.mention}\n\n**The Challenge:**\n{duel_prompt}\n\n*Response phase active. Prime will judge your answers.*",
            color=0xFFAA00
        )
        embed.set_footer(text="Prime | Duel")
        await ctx.send(embed=embed)

@bot.command(name="match")
async def prime_match(ctx):
    """Find a user with a similar vibe for collaboration."""
    async with ctx.typing():
        current_memory = db_manager.get_user_memory(ctx.author.id)
        if not current_memory:
            await ctx.send("❌ **Not enough info**: You need to speak more for the AI to get your vibe first.")
            return
        
        my_vibe = current_memory.get('vibe', 'Neutral')
        
        # This is a sample logic - in a real bot, we'd query the DB for similar vibes
        # For now, we'll simulate a scan of the local area
        prompt = f"Based on a user with a '{my_vibe}' vibe, describe what kind of 'Creative Twin' they should look for in this server. Be technical and aesthetic-focused."
        advice = await get_gemini_response(prompt, ctx.author.id, username=ctx.author.name, guild_id=ctx.guild.id if ctx.guild else None)
        
        embed = discord.Embed(
            title="🧬 CREATIVE MATCHMAKER",
            description=f"**Your Vibe:** `{my_vibe.upper()}`\n\n**Suggestion:**\n{advice}",
            color=0x00FFB4
        )
        embed.set_footer(text="Prime | Matchmaker")
        await ctx.send(embed=embed)

@bot.command(name="vision")
async def technical_vision(ctx):
    """Analyze lighting, color, and composition of an image."""
    if not ctx.message.attachments:
        await ctx.send("❌ **Upload something**: Upload an image for aesthetic analysis.")
        return
    
    attachment = ctx.message.attachments[0]
    if not any(attachment.filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.webp']):
        await ctx.send("❌ **Not supported**: Only image formats are compatible with Vision analysis.")
        return

    async with ctx.typing():
        try:
            image_bytes = await attachment.read()
            prompt = """
            Analyze this image as a Master Cinematographer and Creative Director.
            Provide:
            1. **Lighting Analysis**: Source, quality (soft/hard), and temperature.
            2. **Color Profile**: Dominant hues and psychological impact.
            3. **Composition**: Rule of thirds, leading lines, and depth.
            4. **Editing Suggestion**: One tuff way to grade this for a 'PRIME' look.
            
            Be extremely technical and professional.
            """
            response = await get_gemini_response(prompt, ctx.author.id, username=ctx.author.name, image_bytes=image_bytes, guild_id=ctx.guild.id if ctx.guild else None)
            
            embed = discord.Embed(
                title="👁️ VISION ANALYSIS",
                description=response[:4000],
                color=0x00FFB4,
                timestamp=datetime.now(timezone.utc)
            )
            embed.set_footer(text="Prime | Vision")
            await ctx.send(embed=embed)
        except Exception as e:
            await ctx.send(BOT_ERROR_MSG)

@bot.command(name="palette")
async def color_palette_extraction(ctx):
    """Generate a hex color palette from an image."""
    if not ctx.message.attachments:
        await ctx.send("❌ **Upload something**: Upload an image to extract its color palette.")
        return

    attachment = ctx.message.attachments[0]
    async with ctx.typing():
        try:
            image_bytes = await attachment.read()
            prompt = "Extract the 5 most dominant and aesthetically pleasing hex codes from this image. Provide them with brief names. Return ONLY the palette in a clean format."
            response = await get_gemini_response(prompt, ctx.author.id, username=ctx.author.name, image_bytes=image_bytes, guild_id=ctx.guild.id if ctx.guild else None)
            
            embed = discord.Embed(
                title="🎨 COLOR PALETTE",
                description=f"**Extracted Palette:**\n\n{response}",
                color=0x00FFB4
            )
            embed.set_footer(text="Prime | Palette Extraction")
            await ctx.send(embed=embed)
        except Exception as e:
            await ctx.send(BOT_ERROR_MSG)

@bot.command(name="structure")
async def project_structure_Report(ctx, *, type: str = "Video Edit"):
    """Project Organization. Generate a professional project structure."""
    async with ctx.typing():
        try:
            prompt = f"Generate a clean and professional folder structure for a '{type}' project. Include specific folders for Assets, Pre-comps, Audio, SFX, and Renders. Format it as a clean directory tree."
            response = await get_gemini_response(prompt, ctx.author.id, username=ctx.author.name, guild_id=ctx.guild.id if ctx.guild else None)
            
            # Report as file for the "tuff" factor
            with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode='w', encoding='utf-8') as tf:
                tf.write(f"PRIME PROJECT STRUCTURE\n{'='*40}\nTYPE: {type.upper()}\n\n{response}")
                temp_path = tf.name
                
            embed = discord.Embed(
                title="📁 PROJECT STRUCTURE",
                description=f"Generated a professional structure for **{type}**.",
                color=0x00FFB4
            )
            embed.set_footer(text="Prime | Organization")
            await ctx.send(embed=embed, file=discord.File(temp_path, filename=f"structure_{type.lower().replace(' ', '_')}.txt"))
            
            try: os.remove(temp_path)
            except: pass
        except Exception as e:
            await ctx.send(BOT_ERROR_MSG)

@bot.command(name="blueprint")
async def technical_blueprint(ctx, *, query: str = None):
    """Generate AE expressions and technical guides."""
    if not query:
        await ctx.send("❌ **Need a description**: Tell me what you need (e.g., '!blueprint pulsing glow synced to bass').")
        return
    
    async with ctx.typing():
        try:
            prompt = f"""
            You are a Senior Motion Designer. The user needs an After Effects expression or logic guide for: '{query}'.
            Provide:
            1. **The Expression**: Clean, optimized code block.
            2. **Guide**: Step-by-step setup.
            3. **Pro Tip**: One tip for better performance.
            
            Be precise and helpful.
            """
            response = await get_gemini_response(prompt, ctx.author.id, username=ctx.author.name, guild_id=ctx.guild.id if ctx.guild else None)
            
            # Use Report logic
            with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode='w', encoding='utf-8') as tf:
                tf.write(f"PRIME EXPRESSION GUIDE\n{'='*40}\nQUERY: {query.upper()}\n\n{response}")
                temp_path = tf.name
 
            embed = discord.Embed(
                title="⚙️ EXPRESSION GUIDE",
                description=f"Generated a setup guide for: `{query}`\n\n*(Full guide attached as file)*",
                color=0x00FFB4
            )
            embed.set_footer(text="Prime | Logic & Technical")
            await ctx.send(embed=embed, file=discord.File(temp_path, filename=f"guide_{os.urandom(2).hex()}.txt"))
            
            try: os.remove(temp_path)
            except: pass
        except Exception as e:
            await ctx.send(BOT_ERROR_MSG)

@bot.command(name="scout")
async def talent_scout(ctx, member: discord.Member = None):
    """Analyze a user for creative teams."""
    member = member or ctx.author
    async with ctx.typing():
        try:
            user_memory = db_manager.get_user_memory(member.id)
            history = []
            async for m in ctx.channel.history(limit=50):
                if m.author.id == member.id: history.append(m.content)
            
            chat_context = "\n".join(history[:10])
            memory_summary = user_memory.get('profile_summary', 'Unknown') if user_memory else 'Unknown'
            
            prompt = f"""
            Analyze this creator for a creative team.
            Subject: {member.name}
            Summary: {memory_summary}
            Recent Messages: {chat_context}
            
            Generate a 'Creator Report' covering:
            1. **Skill Level**: Estimated skill in creative tasks.
            2. **Style**: The 'vibe' of their work.
            3. **Team Fit**: How they'd work in a team.
            4. **Overall Verdict**: Solid advice for team leads.
            
            Be direct and honest.
            """
            response = await get_gemini_response(prompt, ctx.author.id, username=ctx.author.name, guild_id=ctx.guild.id if ctx.guild else None)
            
            embed = discord.Embed(
                title="📑 CREATOR ANALYSIS",
                description=response[:4000],
                color=0x00FFB4,
                timestamp=datetime.now(timezone.utc)
            )
            embed.set_footer(text="Prime | Talent Analysis")
            await ctx.send(embed=embed)
        except Exception as e:
            await ctx.send(BOT_ERROR_MSG)

@bot.command(name="phantom")
async def spectral_phantom(ctx):
    """Analyze the channel's vibe and dynamics."""
    async with ctx.typing():
        try:
            messages = []
            async for msg in ctx.channel.history(limit=100):
                if not msg.author.bot:
                    messages.append(f"{msg.author.name}: {msg.content}")
            
            chat_blob = "\n".join(messages[::-1])
            prompt = f"""
            Analyze the 'vibe' of this channel.
            Don't just summarize; find the hidden dynamics, the mood, and who is currently leading the conversation.
            
            CHAT LOGS:
            {chat_blob}
            
            Be observant and sharp. Don't use robotic language.
            """
            response = await get_gemini_response(prompt, ctx.author.id, username=ctx.author.name, guild_id=ctx.guild.id if ctx.guild else None)
            
            embed = discord.Embed(
                title="👻 VIBE REPORT",
                description=response[:4000],
                color=0x6600FF,
                timestamp=datetime.now(timezone.utc)
            )
            embed.set_footer(text="Prime | Channel Analysis")
            await ctx.send(embed=embed)
        except Exception as e:
            await ctx.send(BOT_ERROR_MSG)

@bot.command(name="glitch")
async def text_glitch(ctx, *, text: str = None):
    """Generate a stylized version of text."""
    if not text:
        await ctx.send("❌ **Error**: Provide text to glitch (e.g., '!glitch PRIME').")
        return
    
    async with ctx.typing():
        try:
            prompt = f"Convert the following text into a 'tuff', stylized version for a discord bio: '{text}'. Use cool symbols naturally."
            response = await get_gemini_response(prompt, ctx.author.id, username=ctx.author.name, guild_id=ctx.guild.id if ctx.guild else None)
            await ctx.send(f"⚡ **STYLED TEXT**:\n```\n{response[:1900]}\n```")
        except Exception as e:
            await ctx.send(BOT_ERROR_MSG)

@bot.command(name="uplift")
async def aesthetic_uplift(ctx):
    """Suggest a technical way to uplift the channel's creative quality."""
    async with ctx.typing():
        try:
            messages = []
            async for msg in ctx.channel.history(limit=20):
                if not msg.author.bot:
                    messages.append(f"{msg.author.name}: {msg.content}")
            
            chat_context = "\n".join(messages[::-1])
            prompt = f"Analyze this recent creative discussion: '{chat_context}'. Suggest 3 solid, technical ways to improve the quality of what's being discussed (e.g., suggesting specific plugins, techniques, or optimizations). Be helpful and direct."
            response = await get_gemini_response(prompt, ctx.author.id, username=ctx.author.name, guild_id=ctx.guild.id if ctx.guild else None)
            
            embed = discord.Embed(
                title="✨ QUALITY UPLIFT",
                description=response[:4000],
                color=0x00FFB4,
                timestamp=datetime.now(timezone.utc)
            )
            embed.set_footer(text="Prime | Quality Control")
            await ctx.send(embed=embed)
        except Exception as e:
            await ctx.send(BOT_ERROR_MSG)

@bot.command(name="sandbox")
async def expression_sandbox(ctx, *, expression: str = None):
    """Analyze and optimize After Effects expressions."""
    if not expression:
        await ctx.send("❌ **Query Required**: Provide an AE expression to analyze.")
        return
    
    async with ctx.typing():
        try:
            prompt = f"Analyze this After Effects expression: '{expression}'. 1. Detect any errors. 2. Provide an optimized version. 3. Explain the logic briefly."
            response = await get_gemini_response(prompt, ctx.author.id, username=ctx.author.name, guild_id=ctx.guild.id if ctx.guild else None)
            
            embed = discord.Embed(
                title="🧪 EXPRESSION ANALYSIS",
                description=response[:4000],
                color=0x00FFB4
            )
            embed.set_footer(text="Prime | AE Sandbox")
            await ctx.send(embed=embed)
        except Exception as e:
            await ctx.send(BOT_ERROR_MSG)

@bot.command(name="override")
async def secret_override(ctx):
    """Secret developer command list for BMR."""
    # Check if user is BMR
    if 'bmr' not in ctx.author.name.lower():
        await ctx.send("❌ **Unauthorized**: Access to Black-Ops Override is restricted to the Creator (BMR).")
        return
    
    embed = discord.Embed(
        title="🕵️ PRIME OVERRIDE: DEVELOPER TOOLS",
        description="Management interface for BMR.",
        color=0xFF0000
    )
    
    embed.add_field(name="🔒 System Control", value="`!debug_memory`: View raw user memory\n`!purge_cache`: Clear temporary buffers\n`!force_rotate`: Manually cycle API keys\n`!server_scan`: Security check", inline=False)
    embed.add_field(name="🧬 Profile Tuning", value="`!tweak_vibe @user [vibe]`: Force-update user's vibe\n`!wipe_history @user`: Purge user's history", inline=False)
    
    embed.set_footer(text="DEVELOPER ACCESS • PRIME")
    await ctx.author.send(embed=embed)
    await ctx.send("🛡️ **Override Active**: Check your DMs, BMR.", delete_after=5)

@bot.command(name="force_rotate")
async def manual_rotate(ctx):
    """Creator only: Manually cycle API keys."""
    if 'bmr' not in ctx.author.name.lower(): return
    rotate_gemini_key()
    await ctx.send(f"🔄 **Manual Rotation**: Switched to key position {current_key_index + 1}.")

@bot.command(name="setrules")
@commands.has_permissions(administrator=True)
async def set_rules_command(ctx):
    """Post and pin the professional server rules. Administrator only."""
    try:
        # Delete the command message to keep it clean
        try:
            await ctx.message.delete()
        except:
            pass

        # Create the main rules embed
        embed = discord.Embed(
            title="🛡️ SERVER RULES",
            description=(
                "Welcome to the community. To ensure a high-tier creative environment for everyone, "
                "all members are required to adhere to the following protocols. Failure to comply "
                "may result in administrative action."
            ),
            color=0x00FFB4, # Prime Green
            timestamp=datetime.now(timezone.utc)
        )

        embed.add_field(
            name="🚫 1. ZERO SPAM TOLERANCE",
            value=(
                "• No excessive mentions, character spam, or wall-of-text messages.\n"
                "• Refrain from image or sticker spamming.\n"
                "• No obnoxious noises or music-bot abuse in voice channels."
            ),
            inline=False
        )

        embed.add_field(
            name="🔞 2. CONTENT GUIDELINES",
            value=(
                "• Strictly no NSFW, gory, or disturbing content. This includes 'meme' gore.\n"
                "• All media must be safe for work and follow Discord's Terms of Service.\n"
                "• Nicknames, avatars, and statuses must remain professional and clean."
            ),
            inline=False
        )

        embed.add_field(
            name="🤝 3. PROFESSIONAL CONDUCT",
            value=(
                "• No harassment, hate speech, or targeted toxicity.\n"
                "• Respect all members and their creative work.\n"
                "• Mild swearing is permitted if not directed as an insult towards others."
            ),
            inline=False
        )

        embed.add_field(
            name="📢 4. COMMUNICATIONS",
            value=(
                "• Mass mentions (@everyone / @here) are strictly forbidden.\n"
                "• Keep discussions in their appropriate channels.\n"
                "• No self-promotion or advertising in general lanes."
            ),
            inline=False
        )

        embed.add_field(
            name="🛡️ 5. ADMINISTRATIVE AUTHORITY",
            value=(
                "• Moderators and Admins reserve the right to edit/delete any content.\n"
                "• Staff decisions are final. If you have an issue, use the proper appeal channels."
            ),
            inline=False
        )

        footer_text = "PRIME | THIS SERVER IS PROTECTED BY AI"
        if ctx.guild and ctx.guild.icon:
            embed.set_footer(text=footer_text, icon_url=ctx.guild.icon.url)
        else:
            embed.set_footer(text=footer_text)
        
        # Send the message
        rules_msg = await ctx.send(embed=embed)
        
        # Pin the message
        try:
            await rules_msg.pin()
            await ctx.send("✅ **Success**: Rules have been posted and pinned.", delete_after=5)
        except Exception as e:
            await ctx.send(f"⚠️ **Warning**: Rules posted, but I couldn't pin them (check my permissions).", delete_after=10)
            logger.error(f"Failed to pin rules: {e}")

    except Exception as e:
        logger.error(f"Error in setrules command: {e}")
        await ctx.send(f"❌ **System Error**: Failed to post rules. {str(e)}", delete_after=10)

@bot.command(name="editingsoftwares")
async def editing_softwares_command(ctx):
    """List all available creative software."""
    embed = discord.Embed(
        title="🎬 CREATIVE SOFTWARE HUB",
        description=(
            "Welcome to the **Prime Software Hub**. Here are all the essential versions for your creative setup.\n\n"
            "**Available Software:**\n"
            "• `!ae` - After Effects\n"
            "• `!pr` - Premiere Pro\n"
            "• `!me` - Media Encoder\n"
            "• `!ps` - Photoshop\n"
            "• `!topaz` - Topaz AI\n"
            "• `!animate` - Adobe Animate\n"
            "• `!illustrator` - Adobe Illustrator\n"
            "• `!indesign` - Adobe InDesign\n"
            "• `!lightroom` - Lightroom Classic\n"
            "• `!audition` - Adobe Audition\n"
            "• `!incopy` - Adobe InCopy\n\n"
            "*Type a command from above to see the versions and download links.*"
        ),
        color=0x00FFB4,
        timestamp=datetime.now(timezone.utc)
    )
    embed.set_footer(text="Prime | Software Hub")
    if ctx.guild and ctx.guild.icon:
        embed.set_author(name=ctx.guild.name, icon_url=ctx.guild.icon.url)
    
    await ctx.send(embed=embed)

@bot.command(name="ae")
async def ae_versions_command(ctx):
    """Versions for After Effects."""
    versions = {
        "After Effects 2020": "https://pixeldrain.com/u/RQyayo7K",
        "After Effects 2021": "https://pixeldrain.com/u/LPrfhXHv",
        "After Effects 2023": "https://pixeldrain.com/u/iViNM8uc",
        "After Effects 2024": "https://pixeldrain.com/u/Gt6XmdR9",
        "After Effects 2025": "https://pixeldrain.com/u/AJvwJ4tG",
        "After Effects 2026": "https://pixeldrain.com/u/AJvwJ4tG"
    }
    
    embed = discord.Embed(
        title="🎬 AFTER EFFECTS | VERSIONS",
        description="Pick the version you need. All links are tested and fast.\n\n🔑 **Password**: `star`",
        color=0x9999FF,
        timestamp=datetime.now(timezone.utc)
    )
    
    for version, link in versions.items():
        embed.add_field(name=version, value=f"🔗 [Download]({link})", inline=True)
    
    embed.set_footer(text="Prime | VFX Suite")
    await ctx.send(embed=embed)

@bot.command(name="pr")
async def pr_versions_command(ctx):
    """Distributions for Premiere Pro."""
    versions = {
        "Premiere Pro 2024": "https://pixeldrain.com/u/BXs2eBhN",
        "Premiere Pro 2025": "https://pixeldrain.com/u/rAi8ZUDG",
        "Premiere Pro 2026": "https://pixeldrain.com/u/h55RGoHf"
    }
    
    embed = discord.Embed(
        title="🎞️ PREMIERE PRO | VERSIONS",
        description="Fast download versions for your editing setup.\n\n🔑 **Password**: `star`",
        color=0x000055,
        timestamp=datetime.now(timezone.utc)
    )
    
    for version, link in versions.items():
        embed.add_field(name=version, value=f"🔗 [Download]({link})", inline=True)
    
    embed.set_footer(text="Prime | Edit Suite")
    await ctx.send(embed=embed)

@bot.command(name="me")
async def me_versions_command(ctx):
    """Distributions for Media Encoder."""
    versions = {
        "Media Encoder 2020": "https://pixeldrain.com/u/Yy8cD7Qi",
        "Media Encoder 2021": "https://pixeldrain.com/u/mdJGkfhD",
        "Media Encoder 2023": "https://pixeldrain.com/u/fS5jYvhw",
        "Media Encoder 2024": "https://pixeldrain.com/u/r1V9ZfM1",
        "Media Encoder 2025": "https://pixeldrain.com/u/GfgUa7eA",
        "Media Encoder 2026": "https://pixeldrain.com/u/GfiPWYdR"
    }
    
    embed = discord.Embed(
        title="📦 MEDIA ENCODER | VERSIONS",
        description="Tools for rendering and encoding your projects.\n\n🔑 **Password**: `star`",
        color=0xCCFF00,
        timestamp=datetime.now(timezone.utc)
    )
    
    for version, link in versions.items():
        embed.add_field(name=version, value=f"🔗 [Download]({link})", inline=True)
    
    embed.set_footer(text="Prime | Render Suite")
    await ctx.send(embed=embed)

@bot.command(name="ps")
async def ps_versions_command(ctx):
    """Distributions for Photoshop."""
    versions = {
        "Photoshop 2024": "https://pixeldrain.com/u/N3A7wg8E",
        "Photoshop 2025": "https://pixeldrain.com/u/ZxAswi4P",
        "Photoshop 2026": "https://pixeldrain.com/u/4Q2b6pEF"
    }
    
    embed = discord.Embed(
        title="🎨 PHOTOSHOP | VERSIONS",
        description="The best versions for design and texture work.\n\n🔑 **Password**: `star`",
        color=0x31A8FF,
        timestamp=datetime.now(timezone.utc)
    )
    
    for version, link in versions.items():
        embed.add_field(name=version, value=f"🔗 [Download]({link})", inline=True)
    
    embed.set_footer(text="Prime | Design Suite")
    await ctx.send(embed=embed)

@bot.command(name="topaz")
async def topaz_versions_command(ctx):
    """Distributions for Topaz AI."""
    versions = {
        "Topaz Video AI": "https://pixeldrain.com/u/jCZq3ABi",
        "Topaz Photo AI": "https://pixeldrain.com/u/iE9KwJ58",
        "Topaz Gigapixel AI": "https://pixeldrain.com/u/YQtHLn9Q"
    }
    
    embed = discord.Embed(
        title="💎 TOPAZ AI | VERSIONS",
        description="AI upscaling and tools for your media.\n\n🔑 **Password**: `star`",
        color=0x005AF0,
        timestamp=datetime.now(timezone.utc)
    )
    
    for version, link in versions.items():
        embed.add_field(name=version, value=f"🔗 [Download]({link})", inline=True)
    
    embed.set_footer(text="Prime | AI Suite")
    await ctx.send(embed=embed)

@bot.command(name="animate")
async def animate_command(ctx):
    """Adobe Animate."""
    versions = {
        "Adobe Animate 2024": "https://pixeldrain.com/u/xHwmBZtc"
    }
    embed = discord.Embed(
        title="🎬 ADOBE ANIMATE | VERSIONS",
        description="Pick the version you need. All links are tested and fast.\n\n🔑 **Password**: `star`",
        color=0xFF6600,
        timestamp=datetime.now(timezone.utc)
    )
    for name, link in versions.items():
        embed.add_field(name=name, value=f"🔗 [Download]({link})", inline=True)
    embed.set_footer(text="Prime | Animation Setup")
    await ctx.send(embed=embed)

@bot.command(name="illustrator")
async def illustrator_command(ctx):
    """Adobe Illustrator."""
    versions = {
        "Adobe Illustrator 2026": "https://pixeldrain.com/u/PQyatwYo"
    }
    embed = discord.Embed(
        title="🎨 ADOBE ILLUSTRATOR | VERSIONS",
        description="Essential tools for vector design.\n\n🔑 **Password**: `star`",
        color=0xFF9900,
        timestamp=datetime.now(timezone.utc)
    )
    for name, link in versions.items():
        embed.add_field(name=name, value=f"🔗 [Download]({link})", inline=True)
    embed.set_footer(text="Prime | Design Setup")
    await ctx.send(embed=embed)

@bot.command(name="indesign")
async def indesign_command(ctx):
    """Adobe InDesign."""
    versions = {
        "Adobe InDesign 2026": "https://pixeldrain.com/u/Q5UT44JN"
    }
    embed = discord.Embed(
        title="📖 ADOBE INDESIGN | VERSIONS",
        description="Pick the version you need for layout and design.\n\n🔑 **Password**: `star`",
        color=0xFF3366,
        timestamp=datetime.now(timezone.utc)
    )
    for name, link in versions.items():
        embed.add_field(name=name, value=f"🔗 [Download]({link})", inline=True)
    embed.set_footer(text="Prime | Layout Setup")
    await ctx.send(embed=embed)

@bot.command(name="lightroom")
async def lightroom_command(ctx):
    """Adobe Lightroom Classic."""
    versions = {
        "Lightroom Classic": "https://pixeldrain.com/u/12CcF1oZ"
    }
    embed = discord.Embed(
        title="📸 ADOBE LIGHTROOM | VERSIONS",
        description="Professional photo editing and organization.\n\n🔑 **Password**: `star`",
        color=0x31A8FF,
        timestamp=datetime.now(timezone.utc)
    )
    for name, link in versions.items():
        embed.add_field(name=name, value=f"🔗 [Download]({link})", inline=True)
    embed.set_footer(text="Prime | Photo Setup")
    await ctx.send(embed=embed)

@bot.command(name="audition")
async def audition_command(ctx):
    """Adobe Audition."""
    versions = {
        "Adobe Audition 2025": "https://pixeldrain.com/u/9vJstMbs"
    }
    embed = discord.Embed(
        title="🎙️ ADOBE AUDITION | VERSIONS",
        description="Professional audio workstation software.\n\n🔑 **Password**: `star`",
        color=0x00FFCC,
        timestamp=datetime.now(timezone.utc)
    )
    for name, link in versions.items():
        embed.add_field(name=name, value=f"🔗 [Download]({link})", inline=True)
    embed.set_footer(text="Prime | Audio Setup")
    await ctx.send(embed=embed)

@bot.command(name="incopy")
async def incopy_command(ctx):
    """Adobe InCopy."""
    versions = {
        "Adobe InCopy 2026": "https://pixeldrain.com/u/NyBb6p44"
    }
    embed = discord.Embed(
        title="🖋️ ADOBE INCOPY | VERSIONS",
        description="Pick the version you need for collaborative writing.\n\n🔑 **Password**: `star`",
        color=0xFF33CC,
        timestamp=datetime.now(timezone.utc)
    )
    for name, link in versions.items():
        embed.add_field(name=name, value=f"🔗 [Download]({link})", inline=True)
    embed.set_footer(text="Prime | Writing Setup")
    await ctx.send(embed=embed)

@bot.command(name="plugins")
async def plugins_command(ctx):
    """The main hub for all VFX and editing plugins."""
    embed = discord.Embed(
        title="🔌 PLUGIN HUB",
        description=(
            "Welcome to the **Prime Plugin Hub**. Get all the essential tools to supercharge your workflow.\n\n"
            "**Vendor Commands:**\n"
            "• `!borisfx` - Continuum, Sapphire, Mocha\n"
            "• `!maxon` - Red Giant Suite & Universe\n"
            "• `!revisionfx` - Twixtor & RSMB\n"
            "• `!videocopilot` - Element 3D, Optical Flares, Saber\n"
            "• `!plugineverything` - Deep Glow, AutoFill, Shadow Studio\n"
            "• `!autokroma` - AfterCodecs & BRAW Studio\n"
            "• `!zaebects` - JPEG Glitch, Modulation, Physarum\n"
            "• `!elementsupply` - Glitchify & Polytrace\n"
            "• `!pixelsorter` - Pixel Sorter Bundle & BallPoint\n"
            "• `!filmconvert` - FilmConvert Bundle\n\n"
            "*Type a command from above to see the versions and download links.*"
        ),
        color=0xFF0055,
        timestamp=datetime.now(timezone.utc)
    )
    embed.set_footer(text="Prime | Plugin Hub")
    await ctx.send(embed=embed)

@bot.command(name="borisfx")
async def borisfx_command(ctx):
    """BorisFX Plugins."""
    plugins = {
        "Continuum (BCC) 2026.0": "https://pixeldrain.com/u/9GedTDgJ",
        "Sapphire 2026.0": "https://pixeldrain.com/u/3jA6Amu4",
        "Mocha Pro 2026.0": "https://pixeldrain.com/u/QsFujGNU",
        "Silhouette 2025.5": "https://pixeldrain.com/u/rEJTBd73"
    }
    embed = discord.Embed(
        title="🔥 BORISFX SUITE",
        description="Elite VFX and post-production tools.\n\n🔑 **Password**: `star`",
        color=0xFFCC00,
        timestamp=datetime.now(timezone.utc)
    )
    for name, link in plugins.items():
        embed.add_field(name=name, value=f"🔗 [Download]({link})", inline=True)
    await ctx.send(embed=embed)

@bot.command(name="maxon")
async def maxon_command(ctx):
    """Maxon/Red Giant Plugins."""
    plugins = {
        "Red Giant 2026.2.1 Bundle": "https://pixeldrain.com/u/MsMuPXWk"
    }
    embed = discord.Embed(
        title="🔴 MAXON | RED GIANT SUITE",
        description="Includes Magic Bullet, Trapcode, VFX Suite, and Universe.\n\n🔑 **Password**: `star`",
        color=0xED1C24,
        timestamp=datetime.now(timezone.utc)
    )
    for name, link in plugins.items():
        embed.add_field(name=name, value=f"🔗 [Download]({link})", inline=True)
    await ctx.send(embed=embed)

@bot.command(name="revisionfx")
async def revisionfx_command(ctx):
    """RE:VisionFX Plugins."""
    plugins = {
        "Twixtor 8.1.0": "https://pixeldrain.com/u/fKrtcmNJ",
        "RE:VisionFX Bundle 25.08": "https://pixeldrain.com/u/CcEnVQRm",
        "RE:VisionFX Bundle 23.08": "https://pixeldrain.com/u/tWyU6upA"
    }
    embed = discord.Embed(
        title="🌀 RE:VISIONFX SUITE",
        description="Professional motion and time-remapping tools.\n\n🔑 **Password**: `star`",
        color=0x00AEEF,
        timestamp=datetime.now(timezone.utc)
    )
    for name, link in plugins.items():
        embed.add_field(name=name, value=f"🔗 [Download]({link})", inline=True)
    await ctx.send(embed=embed)

@bot.command(name="videocopilot")
async def videocopilot_command(ctx):
    """Video Copilot Plugins."""
    plugins = {
        "Element 3D 2.2.3": "https://pixeldrain.com/u/DfFjPrir",
        "Heat Distortion 1.0.31": "https://pixeldrain.com/u/d5Fygtp2",
        "Pro Shader 1": "https://pixeldrain.com/u/CoMCdagj",
        "Pro Shader 2": "https://pixeldrain.com/u/XqJrEHd3",
        "Optical Flares 1.3.8": "https://pixeldrain.com/u/Xo18xgRM",
        "Saber": "https://www.videocopilot.net/blog/2016/03/new-plug-in-saber-now-available-100-free/",
        "Twitch": "https://pixeldrain.com/u/7TRPRqD8",
        "VC Color Vibrance": "https://www.videocopilot.net/blog/2014/05/new-plug-in-color-vibrance/"
    }
    embed = discord.Embed(
        title="📽️ VIDEO COPILOT SUITE",
        description="Advanced 3D and motion design tools.\n\n🔑 **Password**: `star`",
        color=0x0055AA,
        timestamp=datetime.now(timezone.utc)
    )
    for name, link in plugins.items():
        embed.add_field(name=name, value=f"🔗 [Download]({link})", inline=True)
    await ctx.send(embed=embed)

@bot.command(name="autokroma")
async def autokroma_command(ctx):
    """AutoKroma Plugins."""
    plugins = {
        "AfterCodecs 1.12.1 (2026 Ready)": "https://example.com/autokroma/aftercodecs",
        "AutoKroma 2025.7 Bundle": "https://example.com/autokroma/bundle"
    }
    embed = discord.Embed(
        title="🌈 AUTOKROMA SUITE",
        description="Essential encoding and BRAW workflow tools.\n\n🔑 **Password**: `star`",
        color=0x33CC33,
        timestamp=datetime.now(timezone.utc)
    )
    for name, link in plugins.items():
        embed.add_field(name=name, value=f"🔗 [Download]({link})", inline=True)
    await ctx.send(embed=embed)

@bot.command(name="zaebects")
async def zaebects_command(ctx):
    """Zaebects Plugins."""
    plugins = {
        "JPEG Glitch 1.0.4": "https://example.com/zaebects/jpeg",
        "Modulation 2.1": "https://example.com/zaebects/modulation",
        "Physarum v1.3": "https://example.com/zaebects/physarum",
        "Signal 1.2.3": "https://example.com/zaebects/signal"
    }
    embed = discord.Embed(
        title="⚡ ZAEBECTS SUITE",
        description="Unique glitch and simulation effects.\n\n🔑 **Password**: `star`",
        color=0xFF6600,
        timestamp=datetime.now(timezone.utc)
    )
    for name, link in plugins.items():
        embed.add_field(name=name, value=f"🔗 [Download]({link})", inline=True)
    await ctx.send(embed=embed)

@bot.command(name="plugineverything")
async def plugineverything_command(ctx):
    """Plugin Everything Suite."""
    plugins = {
        "Deep Glow v2": "https://pixeldrain.com/u/fZvB2tFZ",
        "Deep Glow v1": "https://pixeldrain.com/u/iML27rzS",
        "Shadow Studio": "https://pixeldrain.com/u/nutT5H9T"
    }
    embed = discord.Embed(
        title="✨ PLUGIN EVERYTHING SUITE",
        description="Modern tools for every After Effects project.\n\n🔑 **Password**: `star`",
        color=0xFF33CC,
        timestamp=datetime.now(timezone.utc)
    )
    for name, link in plugins.items():
        embed.add_field(name=name, value=f"🔗 [Download]({link})", inline=True)
    await ctx.send(embed=embed)

@bot.command(name="elementsupply")
async def elementsupply_command(ctx):
    """ElementSupply Co Plugins."""
    plugins = {
        "Glitchify": "https://pixeldrain.com/u/zsEQ3aad",
        "Polytrace": "https://pixeldrain.com/u/obW9g2ba"
    }
    embed = discord.Embed(
        title="📦 ELEMENTSUPPLY CO",
        description="Creative elements and tracing plugins.\n\n🔑 **Password**: `star`",
        color=0x999999,
        timestamp=datetime.now(timezone.utc)
    )
    for name, link in plugins.items():
        embed.add_field(name=name, value=f"🔗 [Download]({link})", inline=True)
    await ctx.send(embed=embed)

@bot.command(name="pixelsorter")
async def pixelsorter_command(ctx):
    """Pixel Sorter Studio Plugins."""
    plugins = {
        "Pixel Sorter Studio Bundle": "https://pixeldrain.com/u/g6bMMUCF"
    }
    embed = discord.Embed(
        title="🎞️ PIXEL SORTER STUDIO",
        description="Advanced pixel sorting and motion textures.\n\n🔑 **Password**: `star`",
        color=0xCC99FF,
        timestamp=datetime.now(timezone.utc)
    )
    for name, link in plugins.items():
        embed.add_field(name=name, value=f"🔗 [Download]({link})", inline=True)
    await ctx.send(embed=embed)

@bot.command(name="filmconvert")
async def filmconvert_command(ctx):
    """FilmConvert Plugins."""
    plugins = {
        "FilmConvert Bundle": "https://pixeldrain.com/u/fs4GCNa2"
    }
    embed = discord.Embed(
        title="🎞️ FILMCONVERT SUITE",
        description="Cinematic film grading and grain tools.\n\n🔑 **Password**: `star`",
        color=0x663300,
        timestamp=datetime.now(timezone.utc)
    )
    for name, link in plugins.items():
        embed.add_field(name=name, value=f"🔗 [Download]({link})", inline=True)
    await ctx.send(embed=embed)

@bot.command(name="extensions")
async def extensions_command(ctx):
    """Essential extensions for your creative setup."""
    items = {
        "Flow Extension": "https://pixeldrain.com/u/eJKBowtz",
        "CRT Emulator": "https://pixeldrain.com/u/zS3s8syM",
        "FX Console": "https://www.videocopilot.net/blog/2018/05/fx-console-updated-to-v1-0-3/"
    }
    
    embed = discord.Embed(
        title="🧩 CREATIVE EXTENSIONS",
        description="Select the extensions you need to boost your efficiency.\n\n🔑 **Password**: `star`",
        color=0x00FFB4,
        timestamp=datetime.now(timezone.utc)
    )
    
    for name, link in items.items():
        embed.add_field(name=name, value=f"🔗 [Download]({link})", inline=True)
    
    embed.set_footer(text="Prime | Extensions")
    await ctx.send(embed=embed)

@bot.command(name="access")
@commands.has_permissions(administrator=True)
async def access_instructions_command(ctx):
    """Post and pin instructions on how to use the software and plugin commands."""
    embed = discord.Embed(
        title="📥 HOW TO ACCESS CONTENT",
        description=(
            "Welcome to the hub. You can use the following commands to get everything you need for your creative setup.\n\n"
            "**Main Commands:**\n"
            "• `!editingsoftwares` - Full video and photo editing apps.\n"
            "• `!plugins` - All VFX and editing plugin suites.\n"
            "• `!extensions` - Essential After Effects extensions.\n\n"
            "**How to use:**\n"
            "1. You can type these commands in any channel in the server.\n"
            "2. If you want to keep it private, you can just **DM the bot** directly with any of these commands.\n"
            "3. For specific tools (like Sapphire or Element 3D), use the individual commands found inside the main menus above."
        ),
        color=0x00FFB4,
        timestamp=datetime.now(timezone.utc)
    )
    embed.set_footer(text="Prime | Access Hub")
    
    # Send the embed
    msg = await ctx.send(embed=embed)
    
    # Pin the message
    try:
        await msg.pin()
        await ctx.send("✅ **Success**: Access instructions have been pinned.", delete_after=5)
    except Exception as e:
        logger.warning(f"Failed to pin access instructions: {e}")
        await ctx.send("⚠️ **Warning**: Instructions posted, but I couldn't pin them (check my permissions).", delete_after=10)

@bot.command(name="setup_verification", aliases=["fixsetupverification"])
@commands.has_permissions(administrator=True)
async def setup_verification_cmd(ctx):
    """Interactive role setup for the verification system."""
    guild_id = ctx.guild.id
    settings = db_manager.get_guild_setting(guild_id, "all_settings", {})
    
    # If setup is already done and user used !setup_verification, redirect them to fix
    if ctx.invoked_with == "setup_verification" and settings.get("verified_role") and settings.get("unverified_role"):
        await ctx.reply("<:Muted:1476140288497160285> **Server roles are already saved!** Please use the `!fixsetupverification` command if you wish to modify them.")
        return

    view = VerificationSetupView(guild_id, ctx.author.id)
    embed = view.create_embed()
    await ctx.send(embed=embed, view=view)

def run_bot():
    """Function to start the bot with the token from environment variables."""
    # Load configuration
    config = load_config()

    # Get token from environment variable
    token = config.get('DISCORD_TOKEN')

    if not token:
        logger.error("No Discord token found. Please set the DISCORD_TOKEN environment variable.")
        return

    # Run the bot
    logger.info("Starting bot...")
    
    # DB Status Log
    db_status = "PostgreSQL (Persistent)" if db_manager.is_postgres else "SQLite (Local - Data will reset on Railway redeploy!)"
    logger.info(f"💾 DATABASE STATUS: {db_status}")
    
    bot.run(os.getenv("DISCORD_TOKEN"))

if __name__ == "__main__":
    run_bot()
