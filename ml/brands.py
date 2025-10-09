# ml/brands.py
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Pattern, List, Tuple

@dataclass(frozen=True)
class BrandRule:
    name: str
    category: str
    pattern: Pattern[str]

# Helper: compile a forgiving regex from common aliases
def rx(aliases: List[str]) -> Pattern[str]:
    # Join aliases by |, allow optional spaces/dots and case-insensitive match.
    # We also escape + in inputs where needed and keep \b word boundaries where safe.
    alt = "|".join(aliases)
    return re.compile(alt, re.IGNORECASE)

# Each entry: (canonical name, category, [regex alternatives])
# Keep names lowercase for consistency in your pipeline display if you prefer.
RAW_RULES: List[Tuple[str, str, List[str]]] = [
    # ------------ Entertainment / Streaming ------------
    ("netflix", "entertainment", [r"\bnetflix(\.com)?\b"]),
    ("disney plus", "entertainment", [r"\bdisney\+\b", r"\bdisney\s*plus\b"]),
    ("max", "entertainment", [r"\bhbo\s*max\b", r"\bmax\b"]),
    ("hulu", "entertainment", [r"\bhulu\b"]),
    ("prime video", "entertainment", [r"\bamazon\s*prime\b", r"\bprime\s*video\b"]),
    ("apple tv+", "entertainment", [r"\bapple\s*tv\+\b", r"\bapple\s*tv\s*\+\b"]),
    ("paramount+", "entertainment", [r"\bparamount\+\b", r"\bparamount\s*plus\b"]),
    ("peacock", "entertainment", [r"\bpeacock\b"]),
    ("crunchyroll", "entertainment", [r"\bcrunchyroll\b"]),
    ("espn+", "entertainment", [r"\bespn\+\b", r"\bespn\s*plus\b"]),
    ("dazn", "entertainment", [r"\bdazn\b"]),

    # ------------ Music / Audio ------------
    ("spotify", "entertainment", [r"\bspotify\b"]),
    ("apple music", "entertainment", [r"\bapple\s*music\b"]),
    ("youtube premium", "entertainment", [r"\byoutube\s+premium\b", r"\byt\s+premium\b"]),
    ("youtube music", "entertainment", [r"\byoutube\s+music\b"]),
    ("tidal", "entertainment", [r"\btidal\b"]),
    ("audible", "entertainment", [r"\baudible\b"]),
    ("siriusxm", "entertainment", [r"\bsirius\s*xm\b", r"\bsiriusxm\b"]),

    # ------------ News / Magazines ------------
    ("nyt", "news", [r"\bnew\s*york\s*times\b", r"\bnytimes\b", r"\bnyt\b"]),
    ("wsj", "news", [r"\bwall\s*street\s*journal\b", r"\bwsj\b"]),
    ("washington post", "news", [r"\bwashington\s*post\b", r"\bwapo\b"]),
    ("the economist", "news", [r"\beconomist\b", r"\bthe\s*economist\b"]),
    ("bloomberg", "news", [r"\bbloomberg\b"]),
    ("financial times", "news", [r"\bfinancial\s*times\b", r"\bft\.com\b", r"\bft\b"]),
    ("medium", "news", [r"\bmedium\b"]),

    # ------------ Gaming ------------
    ("playstation plus", "gaming", [r"\bplaystation\s*plus\b", r"\bps\s*plus\b"]),
    ("xbox game pass", "gaming", [r"\bxbox\s*game\s*pass\b", r"\bgame\s*pass\b"]),
    ("nintendo switch online", "gaming", [r"\bnintendo\s*(switch)?\s*online\b"]),
    ("twitch turbo", "gaming", [r"\btwitch\s*turbo\b"]),

    # ------------ Productivity / Software ------------
    ("adobe creative cloud", "productivity", [r"\badobe(\s*creative\s*cloud|[\s-]*cc)\b"]),
    ("adobe photoshop", "productivity", [r"\bphotoshop\b"]),
    ("adobe illustrator", "productivity", [r"\billustrator\b"]),
    ("adobe acrobat", "productivity", [r"\bacrobat\b"]),
    ("microsoft 365", "productivity", [r"\bmicrosoft\s*365\b", r"\boffice\s*365\b", r"\bo365\b"]),
    ("notion", "productivity", [r"\bnotion\b"]),
    ("slack", "productivity", [r"\bslack\b"]),
    ("zoom", "productivity", [r"\bzoom\b"]),
    ("canva", "productivity", [r"\bcanva\b"]),
    ("figma", "productivity", [r"\bfigma\b"]),
    ("asana", "productivity", [r"\basana\b"]),
    ("monday.com", "productivity", [r"\bmonday(\.com)?\b"]),
    ("evernote", "productivity", [r"\bevernote\b"]),
    ("grammarly", "productivity", [r"\bgrammarly\b"]),
    ("dropbox", "productivity", [r"\bdropbox\b"]),
    ("box", "productivity", [r"\bbox(\.com)?\b"]),
    ("one drive", "productivity", [r"\bone\s*drive\b", r"\bonedrive\b"]),

    # ------------ Cloud / Dev / Hosting ------------
    ("github", "developer", [r"\bgithub\b"]),
    ("gitlab", "developer", [r"\bgitlab\b"]),
    ("bitbucket", "developer", [r"\bbitbucket\b"]),
    ("digitalocean", "developer", [r"\bdigital\s*ocean\b", r"\bdigitalocean\b"]),
    ("linode", "developer", [r"\blinode\b"]),
    ("heroku", "developer", [r"\bheroku\b"]),
    ("vercel", "developer", [r"\bvercel\b"]),
    ("netlify", "developer", [r"\bnetlify\b"]),
    ("render", "developer", [r"\brender\b"]),
    ("cloudflare", "developer", [r"\bcloudflare\b"]),
    ("aws", "developer", [r"\bamazon\s*web\s*services\b", r"\baws\b"]),
    ("gcp", "developer", [r"\bgoogle\s*cloud\b", r"\bgcp\b"]),
    ("azure", "developer", [r"\bazure\b"]),

    # ------------ Storage / Backup ------------
    ("google one", "cloud_storage", [r"\bgoogle\s*(one|storage)\b"]),
    ("apple icloud", "cloud_storage", [r"\bapple\s*i\s*cloud\b", r"\bi\s*cloud\b", r"\bicloud\b"]),
    ("backblaze", "cloud_storage", [r"\bbackblaze\b"]),
    ("idrive", "cloud_storage", [r"\bidrive\b"]),
    ("mega", "cloud_storage", [r"\bmega(\.nz)?\b"]),

    # ------------ Security / VPN / Passwords ------------
    ("1password", "security", [r"\b1\s*password\b", r"\b1password\b"]),
    ("lastpass", "security", [r"\blast\s*pass\b", r"\blastpass\b"]),
    ("dashlane", "security", [r"\bdashlane\b"]),
    ("malwarebytes", "security", [r"\bmalwarebytes\b"]),
    ("nordvpn", "security", [r"\bnord\s*vpn\b", r"\bnordvpn\b"]),
    ("expressvpn", "security", [r"\bexpress\s*vpn\b", r"\bexpressvpn\b"]),
    ("surfshark", "security", [r"\bsurfshark\b"]),
    ("proton vpn", "security", [r"\bproton\s*vpn\b"]),

    # ------------ Education / Learning ------------
    ("coursera", "education", [r"\bcoursera\b"]),
    ("udemy", "education", [r"\budemy\b"]),
    ("skillshare", "education", [r"\bskill\s*share\b", r"\bskillshare\b"]),
    ("linkedin learning", "education", [r"\blinked(in)?\s*learning\b"]),
    ("duolingo plus", "education", [r"\bduolingo\s*plus\b"]),
    ("babbel", "education", [r"\bbabbel\b"]),
    ("brilliant", "education", [r"\bbrilliant\b"]),
    ("chegg", "education", [r"\bchegg\b"]),

    # ------------ Fitness / Health / Wellness ------------
    ("peloton", "fitness", [r"\bpeloton\b"]),
    ("fitbit premium", "fitness", [r"\bfitbit\s*premium\b"]),
    ("strava", "fitness", [r"\bstrava\b"]),
    ("myfitnesspal", "fitness", [r"\bmy\s*fitness\s*pal\b", r"\bmyfitnesspal\b"]),
    ("headspace", "wellness", [r"\bheadspace\b"]),
    ("calm", "wellness", [r"\bcalm\b"]),

    # ------------ Finance / Budgeting / Bills ------------
    ("ynab", "finance", [r"\byou\s*need\s*a\s*budget\b", r"\bynab\b"]),
    ("quickbooks", "finance", [r"\bquick\s*books\b", r"\bquickbooks\b"]),
    ("xero", "finance", [r"\bxero\b"]),
    ("mint", "finance", [r"\bmint\b"]),  # legacy but might appear
    ("rocket money", "finance", [r"\brocket\s*money\b", r"\btruebill\b"]),

    # ------------ Shopping / Memberships ------------
    ("amazon prime", "shopping", [r"\bamazon\s*prime\b", r"\bprime\s*membership\b"]),
    ("walmart+", "shopping", [r"\bwalmart\+\b", r"\bwalmart\s*plus\b"]),
    ("costco membership", "shopping", [r"\bcostco\b", r"\bcostco\s*membership\b"]),
    ("sam's club", "shopping", [r"\bsam'?s\s*club\b"]),

    # ------------ Mobility / Transport ------------
    ("uber one", "mobility", [r"\buber\s*one\b"]),
    ("lyft pink", "mobility", [r"\blyft\s*pink\b"]),

    # ------------ Communication / Email ------------
    ("google workspace", "communication", [r"\bgoogle\s*workspace\b", r"\bg\s*suite\b"]),
    ("proton mail", "communication", [r"\bproton\s*mail\b"]),
    ("fastmail", "communication", [r"\bfastmail\b"]),
    ("zoom pro", "communication", [r"\bzoom\s*pro\b"]),

    # ------------ AI / Creative ------------
    ("chatgpt plus", "ai", [r"\bchatgpt\s*plus\b", r"\bopenai\b"]),
    ("claude pro", "ai", [r"\bclaude\s*pro\b", r"\banthropic\b"]),
    ("midjourney", "ai", [r"\bmid\s*journey\b", r"\bmidjourney\b"]),
    ("github copilot", "ai", [r"\bgithub\s*copilot\b"]),
    ("jasper ai", "ai", [r"\bjasper(\s*ai)?\b"]),
    ("microsoft copilot", "ai", [r"\bmicrosoft\s*copilot\b"]),
]

# Compile into BrandRule list
BRAND_RULES: List[BrandRule] = [
    BrandRule(name, category, rx(aliases))
    for (name, category, aliases) in RAW_RULES
]
