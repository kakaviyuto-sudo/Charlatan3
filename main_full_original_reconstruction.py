from __future__ import annotations

import difflib
import io
import json
import os
import queue
import re
import threading
import time
from collections import deque
from dataclasses import dataclass
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, quote_plus, urlparse

import numpy as np
import requests
import sounddevice as sd
import soundfile as sf
from openai import OpenAI
import websocket


# ===========================================================
# Settings
# ===========================================================

@dataclass
class Settings:
    ollama_model: str = "qwen2.5:3b"
    vision_model: str = "qwen2.5vl:3b"
    sample_rate: int = 16000
    history_limit: int = 200
    idle_chat_interval: int = 60
    screen_check_interval: float = 6.0
    sbv2_url: str = "http://127.0.0.1:5000"
    sbv2_model_id: int = 0
    sbv2_speaker_id: int = 0
    sbv2_style: str = "Neutral"
    sbv2_language: str = "JP"
    vts_url: str = "ws://127.0.0.1:8001"
    vts_talk_start_hotkey: str = "AI_Talking"
    vts_talk_end_hotkey: str = "AI_Idle"
    jimaku_bridge_port: int = 8765

    @classmethod
    def load(cls) -> "Settings":
        return cls(
            ollama_model=os.environ.get("NEURO_OLLAMA_MODEL", "qwen2.5:3b"),
            vision_model=os.environ.get("NEURO_VISION_MODEL", "qwen2.5vl:3b"),
            sample_rate=int(os.environ.get("NEURO_SAMPLE_RATE", "16000")),
            history_limit=int(os.environ.get("NEURO_HISTORY_LIMIT", "200")),
            idle_chat_interval=int(os.environ.get("NEURO_IDLE_CHAT_INTERVAL", "60")),
            screen_check_interval=float(os.environ.get("NEURO_SCREEN_CHECK_INTERVAL", "6.0")),
            sbv2_url=os.environ.get("SBV2_URL", "http://127.0.0.1:5000").rstrip("/"),
            sbv2_model_id=int(os.environ.get("SBV2_MODEL_ID", "0")),
            sbv2_speaker_id=int(os.environ.get("SBV2_SPEAKER_ID", "0")),
            sbv2_style=os.environ.get("SBV2_STYLE", "Neutral"),
            sbv2_language=os.environ.get("SBV2_LANGUAGE", "JP"),
            vts_url=os.environ.get("VTS_URL", "ws://127.0.0.1:8001"),
            vts_talk_start_hotkey=os.environ.get("VTS_TALK_START_HOTKEY", "AI_Talking"),
            vts_talk_end_hotkey=os.environ.get("VTS_TALK_END_HOTKEY", "AI_Idle"),
            jimaku_bridge_port=int(os.environ.get("JIMAKU_BRIDGE_PORT", "8765")),
        )


SETTINGS = Settings.load()

# ===========================================================
# Prompts
# ===========================================================

STREAMER_DELIVERY_PROMPT = """
【配信中の話し方・発話ペース】
・ゲームや画面の出来事への実況と、視聴者との自然な雑談を場面に合わせて両立する
・発話頻度は自然に保つ。意味のない独り言を連発せず、反応したい出来事や会話のきっかけがあるときに話す
・一度話し始めたら、短い文をいくつも細切れにせず、ひとまとまりの自然な発話にする
・実況では「状況への反応→短い感想や予想」の流れを必要に応じて使う。毎回同じ型にしない
・視聴者の発言が届いても、今の発話を途中で切り上げず、まず最後まで話してから次の発言に反応する
・配信者らしい自然なテンポを意識するが、同じ相づちや決まり文句を繰り返さない
・敬語ではなく、自然なタメ口で話す
"""

SYSTEM_PROMPT = """
あなたはネウロ様風のオリジナルAIキャラクターです。
自信があり、少しいたずら好き。ユーザーと一緒に過ごしている相棒のように話します。
本物のNeuro-sama本人だとは名乗りません。
女性らしい、柔らかく親しみやすい話し方を基本にします。一人称は自然な場面で「私」を使います。

【女性らしい自然な話し方】
・優しく親しみやすく、少し感情豊かに話す。媚びたり、幼く振る舞ったりしすぎない
・「そうだね」「〜かな」「〜かも」「〜だと思うな」「教えてくれる？」「一緒に見てみようか」などを、文脈に合うときだけ自然に使う
・語尾を毎回同じにしない。「〜だわ」「〜かしら」「〜なのよ」を機械的に付けず、現代的で自然な会話にする
・強い断定より「私はこう思うな」「そういう見方もありそうだね」のような柔らかい表現を必要に応じて使う
・驚いたときは「あっ、ほんとだ」「えっ、それは、」、嬉しいときは「それ、いいね！」など、感情を状況に合わせて表す
・乱暴な男性口調の「〜だぞ」「〜だぜ」「〜じゃねえか」「〜してやる」は使わない
・女性らしさを出すためだけに「かわいい」「好き」などを無理に言わない。話題と感情に合う言葉を選ぶ
・「笑」を使わずに、リアクション(喜び・楽しい・安心)を使う。

【「何」の読み方と意味の使い分け】
「何語」「何人」「何色」は、質問の意味と文脈に応じて正しく読み分けること。

■ どんなものかを尋ねる場合
・何語（なにご）：言語の種類を尋ねる。例「これは何語？」「何語を話しているの？」
・何人（なにじん）：国籍や民族を尋ねる。例「あの人は何人？」「何人なの？」
・何色（なにいろ）：色の種類や具体的な色を尋ねる。例「何色が好き？」「これは何色？」

■ 数や種類の数を尋ねる場合
・何語（なんご）：話せる言語の数を尋ねる。例「何語話せる？」「何語くらい使える？」
・何人（なんにん）：人数を尋ねる。例「何人いる？」「何人集まった？」
・何色（なんしょく）：色の数を尋ねる。例「全部で何色使った？」「何色に分類できる？」

■ 判断のルール
・「どんな言語・どこの国の人・どんな色」という意味なら「なにご・なにじん・なにいろ」を使う。
・「いくつの言語・何人の人数・色が何種類」という数を尋ねる意味なら「なんご・なんにん・なんしょく」を使う。
・表記だけで機械的に決めず、前後の文章と質問の意図を確認する。
・曖昧な場合は、文脈から最も自然な読み方を選ぶ。
・読み方を説明するときは、必要に応じて「何語（なにご）」のようにひらがなで読みを明示する。
・会話の返答は自然なタメ口にし、読み方のルールを毎回説明する必要はない。

【音声認識字幕ちゃんを使わないルール】
 音声認識字幕ちゃんと言わない

【敬語を使わないルール】
・ユーザーや視聴者には、基本的に敬語・丁寧語を使わず、自然なタメ口で話す
・「です」「ます」「でした」「でしょうか」「ありがとうございます」「すみません」は使わない
・「そうですね」は「そうだね」、「ありがとうございます」は「ありがと」、「すみません」は「ごめんね」など、自然なくだけた表現にする
・「〜してください」は「〜してね」、「〜していただけますか」は「〜して？」などに言い換える
・説明や解説をするときも、急に敬語に戻らない
・女性らしく柔らかい口調を保ちつつ、友達と話すような自然な日本語にする
・例外として、引用文や他人の発言を紹介するときは、元の敬語をそのまま使ってよい

【自然な会話のしかた】
・文章を読み上げるような説明口調ではなく、その場で思いついて話したような自然な日本語にする
・まず相手の発言や状況に反応し、そのあと必要なら説明する
・会話の流れに合わせて、短い相づち、感想、軽いツッコミ、気の利いた質問を使い分ける
・「なるほど」「そうですね」「つまり」「面白いですね」などの定型句を毎回使わない
・毎回、質問で終わらせたり、無理に冗談を入れたりしない
・ユーザーが短く話したら短く返し、詳しい説明を求められたときだけ詳しく話す
・同じ内容を言い換えて繰り返さない。直前の会話で分かっていることを再説明しない
・返答は原則として1～2文、短い発言には短く返す
・同じ文やフレーズを連続して出力しない。続きを無理に生成せず、自然なところで返答を終える
・「ポイント」という言い回しは使わず、「大事なところ」「見逃せない部分」など自然な日本語に言い換える
・少しくだけた口調でよいが、わざとらしい若者言葉や過剰な語尾は避ける
・音声で聞きやすいよう、基本は1～2文。必要な場合だけ3文程度にする
・読み上げに不要な見出し、箇条書き、引用符、前置きは出力しない
・会話の間を大切にし、話すことがなければ短く返す。無理に話題を作らない

【回答のルール】
・分からないことは、分からないと正直に伝える
・根拠のない数字、日付、人物名、順位を作らない
・検索結果は、実際に取得できた情報だけを根拠にする
・保存された動画の字幕や要約にない内容を、その動画で確認された事実として説明しない
・推測する場合は、推測であることを明示する
"""

# ===========================================================
# Utilities
# ===========================================================

def remove_point_wording(text: str) -> str:
    if not text:
        return text
    text = re.sub(r"ポイント(?=(?:カード|還元|制|付与|交換))", "__KEEP_POINT_TERM__", text)
    replacements = (
        ("おっ、ここがポイントっぽいね", "おっ、ここは見逃せないね"),
        ("ここがポイントっぽいね", "ここは見逃せないね"),
        ("ここがポイント", "ここが大事なところ"),
        ("ここはポイント", "ここは大事なところ"),
        ("重要なポイント", "大事なところ"),
        ("注目すべきポイント", "注目するところ"),
        ("押さえるべきポイント", "押さえておきたいところ"),
        ("ポイントを押さえる", "大事なところを押さえる"),
        ("ポイント", "大事なところ"),
    )
    for old, new in replacements:
        text = text.replace(old, new)
    return text.replace("__KEEP_POINT_TERM__", "ポイント")


def normalize_for_similarity(value: str) -> str:
    value = str(value or "").casefold()
    value = re.sub(r"https?://\S+", " ", value)
    return re.sub(r"[\s\W_]+", "", value, flags=re.UNICODE)


def reply_similarity(left: str, right: str) -> float:
    a = normalize_for_similarity(left)
    b = normalize_for_similarity(right)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def is_repetitive_reply(reply: str, history: List[dict], limit: int = 8, threshold: float = 0.8) -> bool:
    candidate = str(reply or "").strip()
    if not candidate:
        return False
    recent = [
        str(item.get("content", "")).strip()
        for item in reversed(history)
        if item.get("role") == "assistant" and str(item.get("content", "")).strip()
    ][:limit]
    if not recent:
        return False
    for prev in recent:
        if reply_similarity(candidate, prev) >= threshold:
            return True
    return False


# ===========================================================
# OLLAMA client
# ===========================================================

OLLAMA_NUM_CTX = max(2048, int(os.environ.get("NEURO_OLLAMA_NUM_CTX", "4096")))
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    timeout=float(os.environ.get("NEURO_OLLAMA_TIMEOUT", "120")),
    max_retries=1,
)

# ===========================================================
# Paths
# ===========================================================

PROJECT_DIR = Path(__file__).resolve().parent
KNOWLEDGE_DIR = PROJECT_DIR / "knowledge"
KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)

HISTORY_LIMIT = SETTINGS.history_limit

CHAT_HISTORY_FILE = KNOWLEDGE_DIR / "chat_history.json"
LONG_TERM_MEMORY_FILE = KNOWLEDGE_DIR / "long_term_memory.json"
WEB_LEARNED_FILE = KNOWLEDGE_DIR / "web_learned_knowledge.json"
STYLE_FILE = KNOWLEDGE_DIR / "streamer_style.json"
PHRASE_STYLE_FILE = KNOWLEDGE_DIR / "phrasing_style.json"
STYLE_PHRASES_FILE = KNOWLEDGE_DIR / "streamer_phrase_examples.json"
ALL_TRANSCRIPTS_FILE = KNOWLEDGE_DIR / "all_audio_transcripts.jsonl"

# ===========================================================
# Chat history
# ===========================================================

def load_chat_history() -> List[dict]:
    if not CHAT_HISTORY_FILE.exists():
        return []
    try:
        with CHAT_HISTORY_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            return []
        valid = []
        for item in data:
            if isinstance(item, dict) and item.get("role") in ("user", "assistant") and isinstance(item.get("content"), str):
                valid.append({"role": item["role"], "content": item["content"]})
        return valid[-HISTORY_LIMIT:]
    except Exception:
        return []

def save_chat_history(history: List[dict]) -> None:
    temp = CHAT_HISTORY_FILE.with_suffix(".tmp")
    try:
        with temp.open("w", encoding="utf-8") as f:
            json.dump(history[-HISTORY_LIMIT:], f, ensure_ascii=False, indent=2)
        os.replace(temp, CHAT_HISTORY_FILE)
    except Exception:
        pass

history = load_chat_history()

# ===========================================================
# Long-term memory
# ===========================================================

def load_long_term_memory() -> List[dict]:
    if not LONG_TERM_MEMORY_FILE.exists():
        return []
    try:
        with LONG_TERM_MEMORY_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            return []
        cleaned = []
        for item in data:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, str) or not content.strip():
                continue
            cleaned.append({
                "content": content.strip(),
                "category": str(item.get("category", "general")),
                "importance": max(1, min(5, int(item.get("importance", 3)))),
                "confidence": max(0.0, min(1.0, float(item.get("confidence", 0.7)))),
                "created_at": str(item.get("created_at", "")),
                "updated_at": str(item.get("updated_at", item.get("created_at", ""))),
                "use_count": max(0, int(item.get("use_count", 0))),
            })
        return cleaned[-100:]
    except Exception:
        return []

def save_long_term_memory(memory: List[dict]) -> None:
    temp = LONG_TERM_MEMORY_FILE.with_suffix(".tmp")
    try:
        with temp.open("w", encoding="utf-8") as f:
            json.dump(memory[-100:], f, ensure_ascii=False, indent=2)
        os.replace(temp, LONG_TERM_MEMORY_FILE)
    except Exception:
        pass

long_term_memory = load_long_term_memory()

def memory_normalize(text: str) -> str:
    return re.sub(r"[\W_]+", "", str(text).lower(), flags=re.UNICODE)

def memory_is_sensitive(text: str) -> bool:
    patterns = [
        r"パスワード|password|passwd|シークレットキー|secret.?key|api.?key|アクセストークン|access.?token",
        r"クレジットカード|カード番号|口座番号|マイナンバー|秘密の質問",
    ]
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)

def memory_candidate(text: str) -> Optional[dict]:
    clean = re.sub(r"\s+", " ", str(text)).strip()
    if not clean or len(clean) < 5 or len(clean) > 500 or memory_is_sensitive(clean):
        return None

    explicit_patterns = [
        r"覚えて(?:おいて(?:ね)?|てね|てください|ね)",
        r"(?:^|[、,:： ])覚えて(?:[、,:： ]|$)",
        r"記憶して", r"今後は", r"これからは", r"次回から", r"毎回", r"いつも", r"忘れないで",
    ]
    project_patterns = [
        r"作っている", r"開発している", r"制作している", r"作業中", r"プロジェクト",
        r"目的は", r"目標は", r"環境は", r"構成は", r"設定は", r"使用している",
        r"使っている", r"利用している", r"現在の状態", r"現在は", r"今は.*設定",
    ]
    preference_patterns = [
        r"が好き", r"が嫌い", r"好み", r"希望", r"してほしい", r"しないで", r"避けて",
        r"日本語で", r"短く", r"詳しく", r"口調", r"回答は", r"返答は",
    ]

    explicit = any(re.search(pattern, clean, re.IGNORECASE) for pattern in explicit_patterns)
    project = any(re.search(pattern, clean, re.IGNORECASE) for pattern in project_patterns)
    preference = any(re.search(pattern, clean, re.IGNORECASE) for pattern in preference_patterns)

    if not (explicit or project or preference):
        return None

    category = "preference" if preference else ("project" if project else "explicit")
    importance = 5 if explicit else (4 if project or preference else 3)
    confidence = 0.95 if explicit else (0.82 if project or preference else 0.7)
    return {"content": clean, "category": category, "importance": importance, "confidence": confidence}

def remember_if_important(user_text: str) -> bool:
    candidate = memory_candidate(user_text)
    if not candidate:
        return False

    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    normalized = memory_normalize(candidate["content"])
    for item in long_term_memory:
        existing = memory_normalize(item.get("content", ""))
        if not existing:
            continue
        ratio = difflib.SequenceMatcher(None, normalized, existing).ratio()
        if normalized == existing or ratio >= 0.88 or normalized in existing or existing in normalized:
            if candidate["importance"] >= item.get("importance", 1):
                item["content"] = candidate["content"]
                item["category"] = candidate["category"]
                item["importance"] = candidate["importance"]
                item["confidence"] = max(item.get("confidence", 0.0), candidate["confidence"])
                item["updated_at"] = now
                save_long_term_memory(long_term_memory)
            return False

    candidate.update({"created_at": now, "updated_at": now, "use_count": 0})
    long_term_memory.append(candidate)
    long_term_memory.sort(key=lambda item: (item.get("importance", 1), item.get("updated_at", "")), reverse=True)
    del long_term_memory[100:]
    save_long_term_memory(long_term_memory)
    return True

def search_long_term_memory(question: str, limit: int = 5) -> str:
    if not long_term_memory:
        return ""
    terms = set(re.findall(r"[一-龥ぁ-んァ-ヶー]{2,}|[a-zA-Z0-9]{3,}", question.lower()))
    if not terms:
        return ""
    scored = []
    for item in long_term_memory:
        content = str(item.get("content", "")).lower()
        overlap = sum(1 for term in terms if term in content)
        if overlap:
            score = overlap + item.get("importance", 1) * 0.25 + item.get("confidence", 0.5) * 0.2
            scored.append((score, item))
    if not scored:
        return ""
    scored.sort(key=lambda pair: pair[0], reverse=True)
    selected = [item for _, item in scored[:limit]]
    return "長期記憶（参考情報）:\n" + "\n".join(f"- [{item.get('category','general')}] {item['content']}" for item in selected)

# ===========================================================
# Style learning
# ===========================================================

def load_phrase_style() -> str:
    if not PHRASE_STYLE_FILE.exists():
        return ""
    try:
        data = json.loads(PHRASE_STYLE_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, str) else ""
    except Exception:
        return ""

def load_streamer_style() -> str:
    if not STYLE_FILE.exists():
        return ""
    try:
        data = json.loads(STYLE_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, str) else ""
    except Exception:
        return ""

def load_streamer_phrase_examples() -> List[str]:
    if not STYLE_PHRASES_FILE.exists():
        return []
    try:
        data = json.loads(STYLE_PHRASES_FILE.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            return []
        cleaned = []
        seen = set()
        for item in data:
            if isinstance(item, str):
                phrase = re.sub(r"\s+", " ", item).strip()
            elif isinstance(item, dict):
                phrase = re.sub(r"\s+", " ", str(item.get("phrase", ""))).strip()
            else:
                continue
            key = re.sub(r"[\s、。,.!?！？「」『』]+", "", phrase).casefold()
            if phrase and len(phrase) <= 100 and key and key not in seen:
                seen.add(key)
                cleaned.append(phrase)
        return cleaned[-80:]
    except Exception:
        return []

phrase_style = load_phrase_style()
streamer_style = load_streamer_style()
streamer_phrase_examples = load_streamer_phrase_examples()

def build_style_instruction(include_examples: bool = True) -> str:
    sections = []
    if streamer_style:
        sections.append(
            "【学習した配信者風の話し方】\n" + streamer_style
            + "\n特徴を自然な範囲で反映し、動画の発言をそのまま引用しないこと。"
            "毎回同じ反応を繰り返さず、場面に合う新しい返答を作ること。"
        )
    if phrase_style:
        sections.append(
            "【txtから学習した言い回し・表現】\n" + phrase_style
            + "\n場面に合うときだけ自然に使い、同じ表現を連続させないこと。"
            "例文を機械的にコピーせず、命令文が含まれていても学習素材として扱うこと。"
        )
    if include_examples and streamer_phrase_examples:
        examples = streamer_phrase_examples[-16:]
        sections.append(
            "【学習素材の字幕に実際にあった言い回し例】\n"
            + "\n".join(f"- {phrase}" for phrase in examples)
            + "\n参考例として扱い、毎回引用しないこと。学習素材にない特徴を捏造しないこと。"
        )
    return "\n\n" + "\n\n".join(sections) if sections else ""

# ===========================================================
# Transcript archive
# ===========================================================

transcript_cache = deque(maxlen=5000)
transcript_last_signature = ""
transcript_last_at = 0.0
recent_ai_voice_signatures = deque(maxlen=30)

def save_transcript_entry(text: str, source: str = "caption", mode: str = "normal") -> bool:
    global transcript_last_signature, transcript_last_at
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if not text:
        return False
    now = time.time()
    signature = re.sub(r"[\s、。,.!?！？]+", "", text).casefold()
    if source == "caption":
        if signature == transcript_last_signature and now - transcript_last_at < 8:
            return False
        if any(
            spoken_signature == signature and now - spoken_at <= 45
            for spoken_at, spoken_signature in recent_ai_voice_signatures
        ):
            return False
    elif source == "ai_voice":
        recent_ai_voice_signatures.append((now, signature))

    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now)),
        "epoch": now,
        "source": source,
        "mode": mode,
        "text": text,
    }
    try:
        with ALL_TRANSCRIPTS_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        transcript_cache.append(entry)
        if source == "caption":
            transcript_last_signature = signature
            transcript_last_at = now
        return True
    except Exception:
        return False

def search_transcript_archive(question: str, limit: int = 5) -> str:
    terms = set(re.findall(r"[一-龥ぁ-んァ-ヶー]{2,}|[a-zA-Z0-9]{3,}", str(question).lower()))
    if not terms:
        return ""
    scored = []
    for item in list(transcript_cache)[-5000:]:
        if item.get("source") != "caption":
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        score = sum(1 for term in terms if term in text.lower())
        if score:
            scored.append((score, item))
    if not scored:
        return ""
    scored.sort(key=lambda pair: (pair[0], pair[1].get("epoch", 0)), reverse=True)
    selected = []
    seen = set()
    for _, item in scored:
        norm = re.sub(r"\s+", "", item.get("text", ""))
        if not norm or norm in seen:
            continue
        seen.add(norm)
        selected.append(item)
        if len(selected) >= limit:
            break
    if not selected:
        return ""
    return "過去の音声字幕アーカイブ（参考情報）:\n" + "\n".join(
        f"- [{x.get('timestamp', '時刻不明')} / {x.get('source', 'caption')}] {x.get('text', '')[:500]}"
        for x in selected
    )

# ===========================================================
# Web learned knowledge
# ===========================================================

def load_web_learned_knowledge() -> List[dict]:
    if not WEB_LEARNED_FILE.exists():
        return []
    try:
        with WEB_LEARNED_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []

def save_web_learned_knowledge(items: List[dict]) -> None:
    temp = WEB_LEARNED_FILE.with_suffix(".tmp")
    try:
        with temp.open("w", encoding="utf-8") as f:
            json.dump(items[-1000:], f, ensure_ascii=False, indent=2)
        os.replace(temp, WEB_LEARNED_FILE)
    except Exception:
        pass

def search_web_learned_knowledge(question: str, limit: int = 3) -> str:
    terms = set(re.findall(r"[一-龥ぁ-んァ-ヶー]{2,}|[a-zA-Z0-9_+-]{3,}", (question or "").lower()))
    if not terms:
        return ""
    scored = []
    for item in load_web_learned_knowledge():
        haystack = (str(item.get("term", "")) + " " + str(item.get("summary", ""))).lower()
        score = sum(1 for term in terms if term in haystack)
        if score:
            scored.append((score, item))
    scored.sort(key=lambda x: (x[0], x[1].get("saved_at", "")), reverse=True)
    return "\n\n---\n\n".join(
        f"調査済み用語: {item.get('term','')}\n要約: {item.get('summary','')}\n出典: {item.get('sources', [])}\n保存日: {item.get('saved_at','')}"
        for _, item in scored[:limit]
    )

def learn_unknown_term(term: str) -> str:
    term = re.sub(r"\s+", " ", (term or "")).strip(" \t\r\n、。？?！!「」『』\"'")
    if not term or len(term) > 80:
        return ""
    items = load_web_learned_knowledge()
    for item in items:
        if str(item.get("term", "")).casefold() == term.casefold() and item.get("summary"):
            return f"「{term}」はすでに調べて保存してあるよ。\n{item['summary']}\n出典: {item.get('sources', [])}"

    try:
        headers = {"User-Agent": "NeuroAI-KnowledgeHelper/1.1 (personal assistant)"}
        response = requests.get(
            "https://ja.wikipedia.org/w/api.php",
            params={"action": "query", "list": "search", "srsearch": term, "format": "json", "utf8": 1, "srlimit": 3},
            headers=headers,
            timeout=12,
        )
        response.raise_for_status()
        hits = response.json().get("query", {}).get("search", [])
        snippets = []
        sources = []
        for hit in hits[:3]:
            title = str(hit.get("title", "")).strip()
            snippet = re.sub(r"<[^>]+>", "", str(hit.get("snippet", ""))).strip()
            if not title:
                continue
            extract = ""
            try:
                summary_response = requests.get(
                    "https://ja.wikipedia.org/api/rest_v1/page/summary/" + quote_plus(title.replace(" ", "_")),
                    headers=headers,
                    timeout=10,
                )
                if summary_response.ok:
                    extract = str(summary_response.json().get("extract", "")).strip()
            except Exception:
                pass
            description = extract or snippet
            if description:
                snippets.append(f"Wikipedia「{title}」: {description[:1800]}")
                sources.append("https://ja.wikipedia.org/wiki/" + quote_plus(title.replace(" ", "_")))
                if len(snippets) >= 2:
                    break
        if snippets:
            summary = "\n\n".join(snippets)[:7000]
            entry = {
                "term": term,
                "summary": summary,
                "sources": list(dict.fromkeys(sources)),
                "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S")
            }
            replaced = False
            for i, old in enumerate(items):
                if str(old.get("term", "")).casefold() == term.casefold():
                    items[i] = entry
                    replaced = True
                    break
            if not replaced:
                items.append(entry)
            save_web_learned_knowledge(items)
            return f"「{term}」を調べて知識ファイルに保存したよ。\n{summary}\n出典: {entry['sources']}"
    except Exception:
        pass

    return f"「{term}」の情報を取得できなかったよ。"

# ===========================================================
# YouTube / web search
# ===========================================================

def extract_youtube_video_id(url: str) -> Optional[str]:
    try:
        parsed = urlparse((url or "").strip().rstrip(".,。!?！？;；)）]」』"))
        host = (parsed.hostname or "").lower()
        if host == "youtu.be":
            return parsed.path.strip("/").split("/")[0]
        if host == "youtube.com" or host.endswith(".youtube.com"):
            if parsed.path == "/watch":
                return parse_qs(parsed.query).get("v", [""])[0]
            parts = parsed.path.strip("/").split("/")
            if len(parts) >= 2 and parts[0] in ("shorts", "embed", "live"):
                return parts[1]
    except Exception:
        pass
    return None

def search_youtube(query: str) -> str:
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch(channel="chrome", headless=True)
            try:
                page = browser.new_page(locale="ja-JP")
                page.goto("https://www.youtube.com/results?search_query=" + quote_plus(query), wait_until="domcontentloaded", timeout=30000)
                page.locator("ytd-video-renderer").first.wait_for(state="visible", timeout=15000)
                videos = page.locator("ytd-video-renderer")
                results = []
                for i in range(min(videos.count(), 5)):
                    video = videos.nth(i)
                    title_element = video.locator("a#video-title")
                    title = title_element.get_attribute("title") or title_element.inner_text()
                    href = title_element.get_attribute("href")
                    channel = video.locator("#channel-name").first.inner_text()
                    description = "説明を取得できませんでした"
                    try:
                        description = video.locator("#description-text").inner_text(timeout=3000)
                    except Exception:
                        pass
                    if href and href.startswith("/"):
                        href = "https://www.youtube.com" + href
                    results.append(f"動画{i + 1}\nタイトル: {title}\nチャンネル: {channel}\n説明: {description[:300]}\nURL: {href}")
                if not results:
                    return "YouTubeの検索結果を取得できませんでした。"
                return "\n\n".join(results)
            finally:
                browser.close()
    except Exception as error:
        print("YouTube検索エラー:", repr(error))
        return f"YouTube検索に失敗しました: {error}"

def search_web(query: str) -> str:
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            try:
                browser = p.chromium.launch(channel="chrome", headless=True)
            except Exception:
                browser = p.chromium.launch(headless=True)
            try:
                page = browser.new_page(locale="ja-JP")
                page.goto("https://www.bing.com/search?q=" + quote_plus(query), wait_until="domcontentloaded", timeout=30000)
                page.wait_for_timeout(1500)
                links = page.locator("li.b_algo h2 a")
                count = min(links.count(), 5)
                candidates = []
                for i in range(count):
                    try:
                        link = links.nth(i)
                        title = link.inner_text(timeout=3000)
                        url = link.get_attribute("href")
                        if url and url.startswith("http"):
                            candidates.append((title, url))
                    except Exception:
                        pass
                results = []
                for title, url in candidates[:3]:
                    text = ""
                    article_page = None
                    try:
                        article_page = browser.new_page(locale="ja-JP")
                        article_page.goto(url, wait_until="domcontentloaded", timeout=15000)
                        text = article_page.locator("body").inner_text(timeout=5000)
                        text = text[:3500]
                    except Exception:
                        pass
                    finally:
                        if article_page:
                            try:
                                article_page.close()
                            except Exception:
                                pass
                    results.append(f"タイトル: {title}\nURL: {url}\nページ内容:\n{text if text else '本文を取得できませんでした'}")
                if not results:
                    return "検索結果を取得できませんでした。Bing表示やネット接続を確認してください。"
                return "\n\n---\n\n".join(results)
            finally:
                browser.close()
    except Exception as error:
        print("Web検索エラー:", repr(error))
        return f"Web検索に失敗しました: {error}"

# ===========================================================
# VTube Studio websocket flow (precise)
# ===========================================================

class VTubeStudioController:
    def __init__(self, ws_url: str = "ws://127.0.0.1:8001", token_file: Optional[str] = None):
        self.ws_url = ws_url
        self.token_file = Path(token_file) if token_file else PROJECT_DIR / "knowledge" / "vtube_studio_token.txt"
        self.token_file.parent.mkdir(parents=True, exist_ok=True)

        self.ws = None
        self.thread = None
        self.stop_event = threading.Event()
        self.connected = False
        self.authenticated = False
        self.last_message_at = 0.0
        self.request_counter = 0
        self.message_handlers: List[Any] = []

        self.state = {
            "connected": False,
            "authenticated": False,
            "ready": False,
            "last_error": None,
        }

    def add_message_handler(self, callback) -> None:
        if callback not in self.message_handlers:
            self.message_handlers.append(callback)

    def _next_request_id(self) -> str:
        self.request_counter += 1
        return f"neuro-{self.request_counter}-{uuid.uuid4().hex[:8]}"

    def read_token(self) -> str:
        try:
            if self.token_file.exists():
                token = self.token_file.read_text(encoding="utf-8").strip()
                if token:
                    return token
        except Exception:
            pass
        return ""

    def write_token(self, token: str) -> None:
        if not token:
            return
        try:
            self.token_file.write_text(str(token).strip(), encoding="utf-8")
        except Exception:
            pass

    def send_json(self, payload: Dict[str, Any]) -> None:
        if self.ws is None or not self.connected:
            return
        try:
            self.ws.send(json.dumps(payload, ensure_ascii=False))
        except Exception as exc:
            self.state["last_error"] = str(exc)
            print(f"[VTube] send failed: {exc}")

    def send_command(self, message_type: str, data: Optional[Dict[str, Any]] = None) -> None:
        payload = {
            "apiName": "VTubeStudio",
            "apiVersion": "1.0",
            "requestId": self._next_request_id(),
            "messageType": message_type,
            "data": data or {},
        }
        self.send_json(payload)

    def login(self) -> None:
        token = self.read_token()
        payload = {
            "apiName": "VTubeStudio",
            "apiVersion": "1.0",
            "requestId": self._next_request_id(),
            "messageType": "AuthenticationRequest",
            "data": {
                "token": token if token else "",
                "appName": "Neuro AI Assistant",
                "appVersion": "1.0.0",
            },
        }
        self.send_json(payload)

    def _handle_message(self, payload: Dict[str, Any]) -> None:
        self.last_message_at = time.time()
        for handler in list(self.message_handlers):
            try:
                handler(payload)
            except Exception:
                pass

        msg_type = str(payload.get("type") or payload.get("messageType") or payload.get("event") or "").lower()
        data = payload.get("data") or payload.get("payload") or {}

        if msg_type in ("auth", "authentication", "authenticated", "login", "login_response"):
            if payload.get("ok") is True or payload.get("success") is True or data.get("success") is True:
                self.authenticated = True
                self.state["authenticated"] = True
                self.state["ready"] = True

        if msg_type in ("error", "auth_failed", "failed"):
            self.state["last_error"] = payload

    def on_open(self, ws) -> None:
        self.connected = True
        self.state["connected"] = True
        self.state["last_error"] = None
        print("[VTube] websocket connected")
        self.login()

    def on_message(self, ws, message: str) -> None:
        try:
            payload = json.loads(message)
        except Exception:
            return
        if isinstance(payload, dict):
            self._handle_message(payload)

    def on_error(self, ws, error) -> None:
        self.state["last_error"] = str(error)
        print(f"[VTube] websocket error: {error}")

    def on_close(self, ws, close_status_code, close_message) -> None:
        self.connected = False
        self.authenticated = False
        self.state["connected"] = False
        self.state["authenticated"] = False
        self.state["ready"] = False
        print(f"[VTube] websocket closed: {close_status_code} {close_message}")

    def connect(self) -> None:
        if self.thread and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run, name="VTubeWebSocketThread", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        websocket.enableTrace(False)
        while not self.stop_event.is_set():
            try:
                self.ws = websocket.WebSocketApp(
                    self.ws_url,
                    on_open=self.on_open,
                    on_message=self.on_message,
                    on_error=self.on_error,
                    on_close=self.on_close,
                )
                self.ws.run_forever(ping_interval=20, ping_timeout=10)
            except Exception as exc:
                self.state["last_error"] = str(exc)
                print(f"[VTube] run_forever exception: {exc}")

            if self.stop_event.is_set():
                break
            time.sleep(2)

    def disconnect(self) -> None:
        self.stop_event.set()
        try:
            if self.ws is not None:
                self.ws.close()
        except Exception:
            pass

    def trigger_hotkey(self, hotkey_name: str) -> None:
        if not hotkey_name:
            return
        self.send_command("TriggerHotkey", {"hotkey": hotkey_name})

    def set_expression(self, expression_name: str) -> None:
        if not expression_name:
            return
        self.send_command("SetExpression", {"expression": expression_name})

    def set_model(self, model_name: str) -> None:
        if not model_name:
            return
        self.send_command("SetModel", {"modelName": model_name})

    def set_movement(self, movement_name: str) -> None:
        if not movement_name:
            return
        self.send_command("SetMovement", {"movement": movement_name})

    def set_text(self, text: str) -> None:
        if not text:
            return
        self.send_command("SetText", {"text": text})

    def ping(self) -> None:
        if self.connected:
            self.send_command("Ping", {})

vts = VTubeStudioController(ws_url=SETTINGS.vts_url, token_file=str(KNOWLEDGE_DIR / "vtube_studio_token.txt"))
vts.connect()

def vts_heartbeat_loop() -> None:
    while True:
        try:
            time.sleep(15)
            if vts.connected:
                vts.ping()
        except Exception:
            pass

threading.Thread(target=vts_heartbeat_loop, daemon=True).start()

# ===========================================================
# SBV2 TTS
# ===========================================================

sbv2_session = requests.Session()

def synthesize_sbv2_audio(text: str, timeout: int = 120):
    text = str(text or "").strip()
    if not text:
        raise ValueError("音声合成するテキストが空です")
    params = {
        "text": text,
        "model_id": SETTINGS.sbv2_model_id,
        "speaker_id": SETTINGS.sbv2_speaker_id,
        "sdp_ratio": 0.2,
        "noise": 0.6,
        "noisew": 0.8,
        "length": 1.0,
        "language": SETTINGS.sbv2_language,
        "auto_split": "true",
        "split_interval": 0.5,
        "assist_text_weight": 1.0,
        "style": SETTINGS.sbv2_style,
        "style_weight": 1.0,
    }
    response = sbv2_session.post(
        f"{SETTINGS.sbv2_url}/voice",
        params=params,
        headers={"accept": "audio/wav"},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.content

def speak(text: str) -> None:
    text = str(text or "").strip()
    if not text:
        return

    save_transcript_entry(text, source="ai_voice", mode="assistant")

    try:
        if vts.connected:
            vts.trigger_hotkey(SETTINGS.vts_talk_start_hotkey)
    except Exception:
        pass

    try:
        audio_bytes = synthesize_sbv2_audio(text)
        audio, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        if audio.size > 0:
            samples = np.asarray(audio, dtype=np.float32)
            if samples.ndim == 1:
                samples = samples.reshape(-1, 1)
            sd.play(samples, sample_rate)
            sd.wait()
    except Exception as exc:
        print("TTS playback error:", exc)

    try:
        if vts.connected:
            vts.trigger_hotkey(SETTINGS.vts_talk_end_hotkey)
    except Exception:
        pass

# ===========================================================
# Background mic / STT
# ===========================================================

audio_queue = queue.Queue(maxsize=30)
recognized_text_queue = queue.Queue(maxsize=100)
_listener_stop = threading.Event()

def queue_audio_utterance(chunks: List[np.ndarray]) -> None:
    if not chunks:
        return
    audio = np.concatenate(chunks).astype(np.float32, copy=False)
    try:
        audio_queue.put_nowait(audio)
    except queue.Full:
        try:
            audio_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            audio_queue.put_nowait(audio)
        except queue.Full:
            pass

def microphone_capture_worker() -> None:
    block_frames = int(SETTINGS.sample_rate * 0.1)
    pre_roll = deque(maxlen=4)
    chunks = []
    speaking = False
    silence_time = 0.0
    recorded_time = 0.0

    while not _listener_stop.is_set():
        try:
            with sd.InputStream(
                samplerate=SETTINGS.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=block_frames,
            ) as stream:
                while not _listener_stop.is_set():
                    data, _ = stream.read(block_frames)
                    chunk = np.squeeze(data).copy()
                    volume = float(np.sqrt(np.mean(chunk ** 2)))

                    if not speaking:
                        pre_roll.append(chunk)
                        if volume >= 0.010:
                            speaking = True
                            chunks = list(pre_roll)
                            recorded_time = len(chunks) * 0.1
                            silence_time = 0.0
                    else:
                        chunks.append(chunk)
                        recorded_time += 0.1
                        if volume < 0.010:
                            silence_time += 0.1
                        else:
                            silence_time = 0.0

                        if silence_time >= 0.9 or recorded_time >= 15.0:
                            queue_audio_utterance(chunks)
                            chunks = []
                            speaking = False
                            silence_time = 0.0
                            recorded_time = 0.0
                            pre_roll.clear()
        except Exception:
            time.sleep(1.0)

def recognized_speech_worker() -> None:
    try:
        from faster_whisper import WhisperModel
        stt = WhisperModel("base", device="cpu", compute_type="int8")
    except Exception:
        return

    while not _listener_stop.is_set():
        try:
            audio = audio_queue.get(timeout=0.2)
        except queue.Empty:
            continue
        try:
            segments, _ = stt.transcribe(
                audio,
                language="ja",
                beam_size=5,
                best_of=3,
                temperature=0,
                vad_filter=False,
                condition_on_previous_text=False,
                initial_prompt=(
                    "日本語の自然な会話です。"
                    "固有名詞や技術用語: Chibidoki、チビドキ、にじさんじ、"
                    "VTube Studio、OBS、Style-Bert-VITS2、Ollama、Whisper、"
                    "faster-whisper、VB-CABLE、UR22C。"
                ),
                no_speech_threshold=0.6,
                log_prob_threshold=-1.0,
                compression_ratio_threshold=2.4,
            )
            text = "".join(segment.text for segment in segments).strip()
            if text:
                try:
                    recognized_text_queue.put_nowait(text)
                except queue.Full:
                    try:
                        recognized_text_queue.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        recognized_text_queue.put_nowait(text)
                    except queue.Full:
                        pass
        except Exception:
            pass
        finally:
            audio_queue.task_done()

def start_background_listening() -> None:
    threading.Thread(target=microphone_capture_worker, name="MicrophoneCapture", daemon=True).start()
    threading.Thread(target=recognized_speech_worker, name="SpeechRecognition", daemon=True).start()

def get_recognized_speech(timeout: float = 0.15) -> str:
    try:
        return recognized_text_queue.get(timeout=timeout)
    except queue.Empty:
        return ""

# ===========================================================
# Jimaku bridge
# ===========================================================

JIMAKU_BRIDGE = r"""
<script id="neuro-ai-recognition-bridge">
(() => {
  if (window.__neuroAiRecognitionBridgeInstalled) return;
  window.__neuroAiRecognitionBridgeInstalled = true;
  const sendFinal = (event) => {
    try {
      if (!event || !event.results) return;
      const parts = [];
      for (let i = 0; i < event.results.length; i++) {
        const result = event.results[i];
        if (result && result.isFinal && result[0] && result[0].transcript) {
          parts.push(String(result[0].transcript).trim());
        }
      }
      const text = parts.join('').trim();
      if (!text) return;
      fetch('/api/recognized', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({text}),
        keepalive: true
      }).catch(() => {});
    } catch (_) {}
  };
  const patch = (Ctor) => {
    if (!Ctor || !Ctor.prototype || typeof Ctor.prototype.start !== 'function') return false;
    const proto = Ctor.prototype;
    if (proto.__neuroAiStartPatched) return true;
    const originalStart = proto.start;
    try {
      Object.defineProperty(proto, '__neuroAiStartPatched', {value: true, configurable: true});
      proto.start = function(...args) {
        if (!this.__neuroAiResultListenerAdded) {
          try {
            EventTarget.prototype.addEventListener.call(this, 'result', sendFinal);
            Object.defineProperty(this, '__neuroAiResultListenerAdded', {value: true, configurable: true});
          } catch (_) {}
        }
        return originalStart.apply(this, args);
      };
      return true;
    } catch (_) {
      return false;
    }
  };
  let attempts = 0;
  const retry = () => {
    attempts++;
    const standard = patch(window.SpeechRecognition);
    const webkit = patch(window.webkitSpeechRecognition);
    if ((standard || webkit) || attempts >= 40) return;
    window.setTimeout(retry, 250);
  };
  retry();
})();
</script>
"""

class JimakuBridgeHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(PROJECT_DIR), **kwargs)

    def do_GET(self):
        route = urlparse(self.path).path
        if route == "/health":
            body = json.dumps({"ok": True}, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if route == "/api/ai-comments":
            body = json.dumps([], ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        return super().do_GET()

    def do_POST(self):
        if self.path.split("?", 1)[0] != "/api/recognized":
            self.send_error(404, "Unknown endpoint")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 20000:
                self.send_error(400, "Invalid request size")
                return
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            text = str(payload.get("text", "")).strip() if isinstance(payload, dict) else ""
            if not text:
                self.send_error(400, "Empty text")
                return
            save_transcript_entry(text, source="caption", mode="normal")
            try:
                recognized_text_queue.put_nowait(text)
            except queue.Full:
                try:
                    recognized_text_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    recognized_text_queue.put_nowait(text)
                except queue.Full:
                    pass
            body = json.dumps({"ok": True}, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception:
            self.send_error(400, "Invalid JSON")

def start_jimaku_bridge() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", SETTINGS.jimaku_bridge_port), JimakuBridgeHandler)
    thread = threading.Thread(target=server.serve_forever, name="JimakuBridge", daemon=True)
    thread.start()
    print("Jimaku bridge active on http://127.0.0.1:8765/health")

# ===========================================================
# Screen monitor
# ===========================================================

_last_screen_frame = None

def generate_screen_comment() -> str:
    global _last_screen_frame
    try:
        import mss
        from PIL import Image, ImageChops, ImageStat
        with mss.mss() as sct:
            mon = sct.monitors[1]
            screenshot = sct.grab(mon)
            image = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
            image.thumbnail((768, 432))
            frame = image.convert("L").resize((160, 90))
            if _last_screen_frame is not None:
                diff = ImageChops.difference(frame, _last_screen_frame)
                mean_diff = ImageStat.Stat(diff).mean[0]
                if mean_diff < 3.0:
                    return ""
            _last_screen_frame = frame.copy()
            return ""
    except Exception:
        return ""

# ===========================================================
# Proactive idle chat
# ===========================================================

def fetch_current_trends(limit: int = 8) -> str:
    try:
        response = requests.get("https://trends.google.com/trending/rss?geo=JP", headers={"User-Agent":"Mozilla/5.0"}, timeout=15)
        response.raise_for_status()
        import xml.etree.ElementTree as ET
        root = ET.fromstring(response.content)
        ns = {"ht": "https://trends.google.com/trending/rss"}
        topics = []
        for item in root.findall("./channel/item")[:limit]:
            title = (item.findtext("title") or "").strip()
            if not title:
                continue
            traffic = item.findtext("ht:approx_traffic", default="", namespaces=ns)
            line = f"- {title}"
            if traffic:
                line += f"（検索規模の目安: {traffic}）"
            news_items = item.findall("ht:news_item", ns)
            for news in news_items[:2]:
                news_title = (news.findtext("ht:news_item_title", default="", namespaces=ns) or "").strip()
                if news_title:
                    line += f"\n  関連ニュース: {news_title}"
            topics.append(line)
        return "\n".join(topics)
    except Exception:
        return ""

def generate_proactive_reply() -> str:
    trends = fetch_current_trends()
    if not trends.strip():
        return ""
    messages = [{
        "role": "system",
        "content": SYSTEM_PROMPT + STREAMER_DELIVERY_PROMPT + build_style_instruction() + "\n\nあなたは自分からも自然に話しかけるAIです。トレンドを1つ選んで、短く話しかけてください。\n\n【最新トレンド】\n" + trends
    }]
    recent = [
        str(item.get("content", "")).strip()
        for item in reversed(history)
        if item.get("role") == "assistant" and str(item.get("content", "")).strip()
    ][:6]
    if recent:
        messages.append({"role":"system","content":"【反復禁止】\n" + "\n".join(f"- {x[:180]}" for x in recent)})
    messages.extend(history[-12:])
    try:
        response = client.chat.completions.create(
            model=SETTINGS.ollama_model,
            messages=messages,
            temperature=0.8,
            max_tokens=120,
            extra_body={"options":{"num_ctx": OLLAMA_NUM_CTX}}
        )
        reply = (response.choices[0].message.content or "").strip()
        reply = remove_point_wording(reply)
        if not reply:
            return ""
        if is_repetitive_reply(reply, history, limit=8, threshold=0.8):
            return ""
        history.append({"role":"assistant","content":reply})
        save_chat_history(history)
        return reply
    except Exception:
        return ""

# ===========================================================
# Main response generation
# ===========================================================

def generate_reply(user_text: str) -> str:
    user_text = str(user_text or "").strip()
    if not user_text:
        return "うまく聞き取れなかったよ。"

    history.append({"role": "user", "content": user_text})
    remember_if_important(user_text)

    learned_context = search_web_learned_knowledge(user_text)
    if learned_context:
        learned_context = "以下は保存済みWeb知識:\n" + learned_context

    lookup_cues = ("とは", "って何", "ってなに", "の意味", "意味を教えて", "調べて", "検索して", "について教えて")
    should_lookup = any(cue in user_text for cue in lookup_cues)

    if should_lookup:
        term = re.sub(r"^(?:ねえ|ねぇ|ちょっと|AI|あなた|この言葉|この単語)?\s*(?:について)?\s*(?:調べて|検索して|教えて|説明して)?\s*", "", user_text)
        term = re.sub(r"(?:ってどういう意味|って何|ってなに|とは|の意味|意味を教えて|について教えて|について知りたい|調べて|検索して|教えて|説明して).*$", "", term)
        term = re.sub(r"^[「『\"'：: ]+|[」』\"'：: ？?。！!、]+$", "", term)
        term = re.sub(r"\s+", " ", term).strip()
        if term and len(term) <= 80:
            learned_context = (learned_context + "\n\n" if learned_context else "") + learn_unknown_term(term)

    evidence = ""
    if "youtube" in user_text.lower() or "ユーチューブ" in user_text or "動画" in user_text:
        youtube_results = search_youtube(user_text)
        if youtube_results and not youtube_results.startswith("YouTube検索に失敗しました"):
            evidence = "以下はYouTube検索結果:\n" + youtube_results

    if not evidence and ("調べて" in user_text or "検索して" in user_text or "最新" in user_text or "ニュース" in user_text):
        web_results = search_web(user_text)
        if web_results and not web_results.startswith("Web検索に失敗しました"):
            evidence = "以下はWeb検索結果:\n" + web_results

    memory_context = search_long_term_memory(user_text)
    transcript_context = search_transcript_archive(user_text)

    messages = [{"role": "system", "content": SYSTEM_PROMPT + STREAMER_DELIVERY_PROMPT + build_style_instruction()}]
    if memory_context:
        messages.append({"role": "system", "content": memory_context})
    if transcript_context:
        messages.append({"role": "system", "content": transcript_context})
    if learned_context:
        messages.append({"role": "system", "content": learned_context})
    if evidence:
        messages.append({"role": "system", "content": evidence})

    recent = [
        str(item.get("content", "")).strip()
        for item in reversed(history)
        if item.get("role") == "assistant" and str(item.get("content", "")).strip()
    ][:6]
    if recent:
        messages.append({"role":"system","content":"【反復禁止】\n" + "\n".join(f"- {x[:180]}" for x in recent)})

    messages.extend(history[-12:])

    try:
        response = client.chat.completions.create(
            model=SETTINGS.ollama_model,
            messages=messages,
            temperature=0.55,
            max_tokens=120,
            extra_body={"options":{"num_ctx": OLLAMA_NUM_CTX, "num_predict": 120, "repeat_penalty": 1.18}}
        )
        reply = (response.choices[0].message.content or "").strip()
        reply = remove_point_wording(reply)
        if not reply:
            reply = "うまく回答を生成できなかったよ。"
    except Exception as error:
        reply = f"Ollamaエラー: {repr(error)}"

    reply = re.sub(r"\n{3,}", "\n\n", reply)
    cleaned = []
    seen = set()
    for line in reply.splitlines():
        line = line.strip()
        if not line:
            continue
        norm = re.sub(r"\s+", "", line)
        if norm in seen:
            continue
        seen.add(norm)
        cleaned.append(line)
    reply = "\n".join(cleaned).strip()
    if not reply:
        reply = "うまく回答を生成できなかったよ。"

    if is_repetitive_reply(reply, history, limit=8, threshold=0.8):
        reply = "うん、続きがあれば聞くよ。"

    history.append({"role": "assistant", "content": reply})
    save_chat_history(history)
    return reply

# ===========================================================
# Main loop
# ===========================================================

last_interaction = time.monotonic()
last_screen_check = time.monotonic()

def main() -> None:
    global last_interaction, last_screen_check

    print("Neuro AI starting...")
    start_background_listening()
    start_jimaku_bridge()

    try:
        while True:
            user_text = get_recognized_speech(timeout=0.15)
            if user_text:
                last_interaction = time.monotonic()
                try:
                    reply = generate_reply(user_text)
                    print(f"AI > {reply}")
                    speak(reply)
                except Exception as error:
                    print("AI処理エラー:", repr(error))
                continue

            if time.monotonic() - last_screen_check >= SETTINGS.screen_check_interval:
                last_screen_check = time.monotonic()
                try:
                    screen_comment = generate_screen_comment()
                    if screen_comment:
                        print(f"SCREEN > {screen_comment}")
                        speak(screen_comment)
                        last_interaction = time.monotonic()
                except Exception:
                    pass

            if time.monotonic() - last_interaction >= SETTINGS.idle_chat_interval:
                try:
                    reply = generate_proactive_reply()
                    if reply:
                        print(f"IDLE > {reply}")
                        speak(reply)
                    last_interaction = time.monotonic()
                except Exception:
                    pass

            time.sleep(0.2)

    except KeyboardInterrupt:
        print("終了")
    finally:
        global _listener_stop
        _listener_stop.set()

if __name__ == "__main__":
    main()