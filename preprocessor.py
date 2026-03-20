import re
import os
import pandas as pd
import emoji
import nltk
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

nltk.download("stopwords", quiet=True)

# ── Stopwords ─────────────────────────────────────────────────────────────────
ENGLISH_STOPS = set(stopwords.words("english"))
URDU_STOPS = {
    "ہے","ہیں","کا","کی","کے","میں","سے","پر","اور","یہ","وہ","ایک",
    "نہیں","بھی","تو","ہو","تھا","تھی","تھے","جو","کو","نے","لیے",
    "آپ","ہم","مجھے","اس","اب","یہاں","وہاں","کیا","کیوں","کیسے",
    "ہوں","ہوگا","ہوگی","تھا","رہا","رہی","رہے","گیا","گئی","گئے",
}

# ── Slang map ─────────────────────────────────────────────────────────────────
SLANG = {
    # english slang
    "lol":"laugh","lmao":"laugh","rofl":"laugh","wtf":"angry","omg":"surprised",
    "brb":"okay","idk":"unsure","tbh":"honest","ngl":"honest","imo":"opinion",
    "goat":"greatest","lit":"excellent","fire":"excellent","mid":"average",
    "cap":"lie","no cap":"honest","bussin":"excellent","slay":"excellent",
    # urdu/roman slang
    "mashallah":"praise","subhanallah":"praise","alhamdulillah":"praise",
    "inshallah":"hope","jazakallah":"thanks","masha":"praise",
    "zabardast":"excellent","behtareen":"excellent","kamaal":"excellent",
    "bekar":"useless","bakwas":"nonsense","faltu":"useless",
    "acha":"good","accha":"good","acha ha":"good","bura":"bad",
    "ganda":"dirty","sahi":"correct","galat":"wrong",
    "yaar":"friend","bhai":"brother","sis":"sister","jaan":"love",
    "bohot":"very","bohat":"very","bhot":"very",
    "rona":"cry","rula":"cry","roya":"cry",
    "hans":"laugh","haha":"laugh","hehe":"laugh",
    "dil":"heart","pyar":"love","mohabbat":"love","ishq":"love",
    "dard":"pain","takleef":"pain","mushkil":"difficult",
    "mast":"great","shandar":"excellent","lajawaab":"excellent",
    "khubsoorat":"beautiful","haseen":"beautiful","sunder":"beautiful",
    "waah":"praise","wah":"praise","waah waah":"praise",
    "osm":"awesome","awsm":"awesome","awesome":"excellent",
}

# ── Negation words ────────────────────────────────────────────────────────────
NEGATIONS = {
    "nahi","nahin","na","not","no","never","mat","نہیں","nah","nope",
    "bilkul nahi","kabhi nahi","kabi nahi",
}

# ── Roman Urdu positive lexicon ───────────────────────────────────────────────
ROMAN_POS = {
    # general positive
    "acha","accha","zabardast","behtareen","mast","sahi","wah","waah",
    "shandar","kamaal","lajawaab","osm","awesome","amazing","great",
    "best","nice","love","pasand","khubsoorat","haseen","pyara","pyari",
    "praise","excellent","good","laugh","perfect","superb","beautiful",
    "sunder","jazbati","dil","khushi","happy","khush","mubarak",
    # religious praise (common in Pakistani comments)
    "mashallah","alhamdulillah","subhanallah","jazakallah","inshallah",
    # drama/song specific
    "outstanding","brilliant","masterpiece","classic","iconic","legend",
    "emotional","touching","heart","dard","mohabbat","ishq","pyar",
    "ost","song","music","drama","acting","performance","scene",
    "rula diya","dil ko","cha gaya","pasand aaya","bohat acha",
    "bohot acha","bhot acha","too good","very good","so good",
    "favourite","favorite","fav","obsessed","blessed","grateful",
    "talented","gifted","voice","beautiful voice","crying happy",
    "goosebumps","chills","speechless","wow","woww","wowww",
}

# ── Roman Urdu negative lexicon ───────────────────────────────────────────────
ROMAN_NEG = {
    # general negative
    "bura","bekar","bakwas","ganda","ghatiya","nafrat","angry","worst",
    "hate","bored","sad","terrible","pathetic","disgusting","useless",
    "nonsense","problem","issue","wrong","bad","horrible","awful","dirty",
    "faltu","wahiyat","sharam","sharmnak","besharam","bewaqoof",
    # drama/song specific negative
    "boring","waste","skip","dislike","overrated","copied","plagiarism",
    "fake","forced","cringe","overdone","loud","screaming","annoying",
    "disappointed","disappointing","ruined","destroyed","flat","dull",
    "repetitive","dragged","cliche","predictable",
    # emotional pain
    "rula","dard","takleef","roya","broken","hurt","pain","suffering",
    "unfair","injustice","zulm","zalim","cruel","evil","heartbreak",
}

# ── Urdu char normalization ───────────────────────────────────────────────────
URDU_NORM = str.maketrans("يكى", "یکی")

analyzer = SentimentIntensityAnalyzer()


def normalize_urdu(text: str) -> str:
    return text.translate(URDU_NORM)


def handle_emojis(text: str) -> str:
    return emoji.replace_emoji(text, replace=lambda ch, _: (
        "love "      if ch in "😍🥰❤️💕💖💗💓💞💝💘" else
        "angry "     if ch in "😡🤬😠👿💢" else
        "sad "       if ch in "😢😭😞😔💔😿" else
        "happy "     if ch in "😊😀😁🥳🎉😄😃🤩" else
        "funny "     if ch in "😂🤣😆😝" else
        "surprised " if ch in "😮😯😲🤯" else
        "praise "    if ch in "🙏👏🤲" else
        "fire "      if ch in "🔥" else
        "heart "     if ch in "💙💚💛🧡🖤🤍🤎💜" else
        ""
    ))


def handle_negation(tokens: list[str]) -> list[str]:
    result, negate = [], False
    for tok in tokens:
        if tok in NEGATIONS:
            negate = True
            result.append(tok)
        elif negate and tok not in NEGATIONS:
            result.append(f"NEG_{tok}")
            negate = False
        else:
            result.append(tok)
    return result


def clean(text: str) -> str:
    text = str(text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = handle_emojis(text)
    text = normalize_urdu(text)
    text = text.lower()
    text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_and_filter(text: str) -> list[str]:
    tokens = text.split()
    tokens = [SLANG.get(t, t) for t in tokens]
    tokens = [t for t in tokens
              if t not in ENGLISH_STOPS
              and t not in URDU_STOPS
              and len(t) > 1]
    tokens = handle_negation(tokens)
    return tokens


def auto_label(text: str, tokens: list[str]) -> str:
    token_set = set(tokens)
    pos_hits  = len(token_set & ROMAN_POS)
    neg_hits  = len(token_set & ROMAN_NEG)

    # negated positives → flip to negative
    neg_pos = sum(1 for t in tokens if t.startswith("NEG_") and t[4:] in ROMAN_POS)
    neg_neg = sum(1 for t in tokens if t.startswith("NEG_") and t[4:] in ROMAN_NEG)
    pos_hits += neg_neg
    neg_hits += neg_pos

    if pos_hits > neg_hits and pos_hits > 0:
        return "Positive"
    if neg_hits > pos_hits and neg_hits > 0:
        return "Negative"

    # fallback: VADER
    score = analyzer.polarity_scores(text)["compound"]
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    return "Neutral"


def detect_language(text: str) -> str:
    urdu_chars = len(re.findall(r"[\u0600-\u06FF]", text))
    total = len(text.replace(" ", ""))
    if total == 0:
        return "Unknown"
    if urdu_chars / total > 0.4:
        return "Urdu"
    
    words = text.lower().split()
    
    # only count actual Roman Urdu words — not general English
    roman_only = {
        "acha","accha","yaar","bhai","nahi","nahin","bohot","bohat","bhot",
        "tha","thi","hain","hai","mujhe","tumhe","apna","apni","koi","sab",
        "kya","kyun","kaisa","kaisi","wah","waah","zabardast","behtareen",
        "mashallah","alhamdulillah","subhanallah","jazakallah","kamaal",
        "shandar","lajawaab","bakwas","bekar","ganda","bura","galat",
        "pyar","mohabbat","ishq","dil","dard","khushi","rona","hasna",
        "acchi","bhai","yaar","ary","geo","hum","khubsoorat","shukriya",
    }
    
    roman_hits = sum(1 for w in words if w in roman_only)
    total_words = len(words)
    
    # need at least 2 Roman Urdu words OR 20% of words to be Roman Urdu
    if roman_hits >= 2 or (total_words > 0 and roman_hits / total_words >= 0.2):
        return "Roman Urdu"
    
    return "English"

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    print(f"[preprocessor] Processing {len(df)} comments...")
    df = df.copy()
    df["cleaned"]   = df["text"].apply(clean)
    df["tokens"]    = df["cleaned"].apply(tokenize_and_filter)
    df["processed"] = df["tokens"].apply(lambda t: " ".join(t))
    df["label"]     = df.apply(lambda r: auto_label(r["cleaned"], r["tokens"]), axis=1)
    df["language"]  = df["text"].apply(detect_language)
    df = df[df["processed"].str.strip().str.len() > 2]
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/cleaned_labeled.csv", index=False)
    print(f"[preprocessor] Saved {len(df)} rows → data/cleaned_labeled.csv")
    print(df["label"].value_counts())
    return df


if __name__ == "__main__":
    raw = pd.read_csv("data/raw_comments.csv")
    preprocess(raw)