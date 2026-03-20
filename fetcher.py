from googleapiclient.discovery import build
from dotenv import load_dotenv
import pandas as pd
import os

load_dotenv()

KEYWORDS = [
    # tech & education
    'pakistan tech', 'pakistan education', 'urdu lecture', 'pakistan documentary',

    # news & politics
    'pakistan politics', 'pakistan news', 'pakistan economy', 'pakistan interview',

    # cricket & sports
    'pakistan cricket', 'pakistan army',

    # entertainment & lifestyle
    'pakistan vlog', 'pakistan food', 'pakistan fashion', 'pakistan gaming',
    'pakistan motivation', 'pakistan reaction',

    # music & songs
    'pakistani songs', 'pakistani new song', 'coke studio pakistan',
    'nescafe basement', 'pakistani pop songs', 'pakistani sad songs',
    'pakistani romantic songs', 'pakistani lo-fi', 'pakistani classical music',
    'pakistani folk songs', 'pakistani wedding songs', 'pakistani qawwali',
    'pakistani ost', 'pakistani album',

    # dramas
    'pakistani drama', 'pakistani drama ost', 'hum tv drama',
    'geo tv drama', 'ary digital drama', 'top pakistani drama',
    'best pakistani drama', 'pakistani drama scene',
    'pakistani drama episode', 'pakistani drama comedy',
    'pakistani drama romance', 'pakistani drama sad scene',

    # urdu content
    'urdu funny', 'urdu poetry', 'urdu storytelling', 'urdu motivational',
]


def get_client():
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        raise ValueError("YOUTUBE_API_KEY not found in .env file")
    return build("youtube", "v3", developerKey=api_key)


def search_videos(keyword: str, max_results: int = 5) -> list[dict]:
    yt = get_client()
    res = yt.search().list(
        q=keyword, part="snippet",
        type="video", maxResults=max_results
    ).execute()
    return [
        {
            "video_id":  item["id"]["videoId"],
            "title":     item["snippet"]["title"],
            "channel":   item["snippet"]["channelTitle"],
            "published": item["snippet"]["publishedAt"][:10],
            "thumbnail": item["snippet"]["thumbnails"]["medium"]["url"],
        }
        for item in res.get("items", [])
    ]


def fetch_comments(video_id: str, max_comments: int = 300) -> pd.DataFrame:
    yt = get_client()
    comments, next_page = [], None

    while len(comments) < max_comments:
        kwargs = dict(
            part="snippet", videoId=video_id,
            maxResults=min(100, max_comments - len(comments)),
            textFormat="plainText", order="relevance"
        )
        if next_page:
            kwargs["pageToken"] = next_page
        try:
            res = yt.commentThreads().list(**kwargs).execute()
        except Exception as e:
            print(f"[fetcher] Skipped {video_id}: {e}")
            break

        for item in res.get("items", []):
            s = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "video_id":  video_id,
                "text":      s["textDisplay"],
                "likes":     s["likeCount"],
                "published": s["publishedAt"][:10],
            })
        next_page = res.get("nextPageToken")
        if not next_page:
            break

    return pd.DataFrame(comments)


def collect(
    keyword: str = None,
    max_videos: int = 5,
    max_comments: int = 300,
    keywords: list[str] = None
) -> pd.DataFrame:

    kw_list = keywords if keywords else ([keyword] if keyword else KEYWORDS)
    all_dfs = []

    for kw in kw_list:
        print(f"[fetcher] Fetching: {kw}")
        try:
            videos = search_videos(kw, max_videos)
        except Exception as e:
            print(f"[fetcher] Search failed for '{kw}': {e}")
            continue

        for v in videos:
            df = fetch_comments(v["video_id"], max_comments)
            if not df.empty:
                df["video_title"] = v["title"]
                df["thumbnail"]   = v["thumbnail"]
                all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()

    result = pd.concat(all_dfs, ignore_index=True)
    result = result.drop_duplicates("text")
    result = result[result["text"].str.strip().str.len() > 5]
    os.makedirs("data", exist_ok=True)
    result.to_csv("data/raw_comments.csv", index=False)
    print(f"[fetcher] Saved {len(result)} comments → data/raw_comments.csv")
    return result


if __name__ == "__main__":
    df = collect()
    print(f"\nTotal unique comments: {len(df)}")
    print(df["video_title"].value_counts().head(10))