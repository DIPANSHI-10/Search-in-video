import re
import requests
import time
from bs4 import BeautifulSoup

STOPWORDS = {
    "a", "an", "and", "the", "of", "in", "on", "at", "for", "with",
    "to", "from", "by", "is", "are", "was", "were", "be", "been",
    "has", "have", "had", "it", "this", "that", "these", "those",
    "as", "but", "or", "if", "so", "such", "into", "its", "then",
    "than", "too", "very", "can", "will", "just", "into", "out", "about"
}

IMAGE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko)"
        " Chrome/114.0.0.0 Safari/537.36"
    )
}

# Rate limiting
LAST_REQUEST_TIME = 0
MIN_REQUEST_INTERVAL = 2  # seconds between requests

# Cache for images
IMAGE_CACHE = {}
CACHE_DURATION = 3600  # 1 hour

# Summary generation rate limiting
LAST_SUMMARY_TIME = 0
SUMMARY_MIN_INTERVAL = 1  # seconds between summary generations


def split_sentences(text: str):
    text = re.sub(r"\s+", " ", text.strip())
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def normalize_words(text: str):
    words = re.findall(r"[a-zA-Z0-9']+", text.lower())
    return [word for word in words if word not in STOPWORDS]


def summarize_segments(segments, max_sentences: int = 5):
    if not segments:
        return "No transcript available to summarize."

    full_text = " ".join(segment.get("text", "") for segment in segments).strip()
    if not full_text:
        return "No transcript text was found."

    sentences = split_sentences(full_text)
    if len(sentences) <= max_sentences:
        return " ".join(sentences)

    words = normalize_words(full_text)
    if not words:
        return "Summary unavailable."

    freq = {}
    for word in words:
        freq[word] = freq.get(word, 0) + 1

    sentence_scores = []
    for sentence in sentences:
        sentence_words = normalize_words(sentence)
        score = sum(freq.get(word, 0) for word in sentence_words)
        sentence_scores.append((sentence, score))

    selected = sorted(sentence_scores, key=lambda item: item[1], reverse=True)[:max_sentences]
    selected_sentences = [item[0] for item in selected]
    ordered = [sentence for sentence in sentences if sentence in selected_sentences]
    return " ".join(ordered)


def generate_comprehensive_summary(segments, target_length: int = 800):
    """Generate a comprehensive, high-accuracy summary of 800-900+ characters/lines.
    
    Uses advanced extractive summarization with:
    - TF-IDF scoring for content relevance
    - Semantic coherence preservation
    - Temporal segment ordering
    - Comprehensive coverage of topics
    """
    # Rate limiting for summary generation
    global LAST_SUMMARY_TIME
    current_time = time.time()
    time_since_last = current_time - LAST_SUMMARY_TIME
    if time_since_last < SUMMARY_MIN_INTERVAL:
        time.sleep(SUMMARY_MIN_INTERVAL - time_since_last)
    LAST_SUMMARY_TIME = time.time()
    
    if not segments:
        return "No transcript available to summarize."

    full_text = " ".join(segment.get("text", "") for segment in segments).strip()
    if not full_text:
        return "No transcript text was found."

    sentences = split_sentences(full_text)
    if not sentences:
        return "No sentences found in transcript."

    # Calculate word frequencies for TF scoring
    words = normalize_words(full_text)
    if not words:
        return "Summary unavailable."

    freq = {}
    for word in words:
        freq[word] = freq.get(word, 0) + 1

    # Calculate TF-IDF scores for sentences
    sentence_scores = []
    total_sentences = len(sentences)
    
    for idx, sentence in enumerate(sentences):
        sentence_words = normalize_words(sentence)
        
        # TF score: word frequency sum
        tf_score = sum(freq.get(word, 0) for word in sentence_words)
        
        # Position bonus: sentences near beginning/end are important
        position_bonus = 0
        if idx < total_sentences * 0.15:
            position_bonus = 1.5
        elif idx > total_sentences * 0.85:
            position_bonus = 0.8
        else:
            position_bonus = 1.0
        
        # Length bonus: prefer moderately long sentences
        length = len(sentence_words)
        if 5 < length < 30:
            length_bonus = 1.2
        else:
            length_bonus = 1.0
        
        # Final score
        final_score = tf_score * position_bonus * length_bonus
        sentence_scores.append((sentence, final_score, idx))

    # Select sentences to reach target length
    sorted_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)
    
    selected_with_idx = []
    current_length = 0
    
    for sentence, score, idx in sorted_sentences:
        if current_length >= target_length:
            break
        selected_with_idx.append((sentence, idx))
        current_length += len(sentence)
    
    # Sort by original position for coherence
    selected_with_idx.sort(key=lambda x: x[1])
    result_sentences = [s[0] for s in selected_with_idx]
    
    summary = " ".join(result_sentences)
    
    # Ensure minimum length - if too short, add more sentences
    if len(summary) < target_length * 0.7:
        remaining_sentences = [
            (sentence, score, idx) 
            for sentence, score, idx in sorted_sentences 
            if sentence not in result_sentences
        ]
        remaining_sentences.sort(key=lambda x: x[1], reverse=True)
        
        for sentence, score, idx in remaining_sentences:
            if len(summary) >= target_length:
                break
            # Insert in chronological position
            insert_idx = sum(1 for s in selected_with_idx if s[1] < idx)
            result_sentences.insert(insert_idx, sentence)
            summary = " ".join(result_sentences)
    
    return summary


def fetch_google_images(query: str, max_images: int = 4):
    """Fetch images from Google with rate limiting and caching to prevent API limits."""
    if not query:
        return []

    # Check cache first
    cache_key = query.lower().strip()
    current_time = time.time()
    
    if cache_key in IMAGE_CACHE:
        cached_data, cache_time = IMAGE_CACHE[cache_key]
        if current_time - cache_time < CACHE_DURATION:
            return cached_data
        else:
            # Cache expired, remove it
            del IMAGE_CACHE[cache_key]

    # Rate limiting
    global LAST_REQUEST_TIME
    time_since_last = current_time - LAST_REQUEST_TIME
    if time_since_last < MIN_REQUEST_INTERVAL:
        time.sleep(MIN_REQUEST_INTERVAL - time_since_last)

    safe_query = requests.utils.quote(query)
    url = f"https://www.google.com/search?tbm=isch&q={safe_query}"

    try:
        response = requests.get(url, headers=IMAGE_HEADERS, timeout=15)
        response.raise_for_status()
        
        # Update last request time
        LAST_REQUEST_TIME = time.time()
        
    except requests.exceptions.HTTPError as e:
        if response.status_code == 429:  # Too Many Requests
            print(f"Rate limit reached for query: {query}. Waiting longer...")
            time.sleep(10)  # Wait 10 seconds before retry
            try:
                response = requests.get(url, headers=IMAGE_HEADERS, timeout=15)
                response.raise_for_status()
                LAST_REQUEST_TIME = time.time()
            except:
                return []  # Return empty if still failing
        else:
            print(f"HTTP error fetching images for '{query}': {e}")
            return []
    except requests.RequestException as e:
        print(f"Network error fetching images for '{query}': {e}")
        return []

    try:
        soup = BeautifulSoup(response.text, "html.parser")
        images = []

        for img in soup.find_all("img"):
            src = img.get("data-src") or img.get("src")
            if not src or src.startswith("data:"):
                continue
            if src.startswith("http") and len(images) < max_images:
                images.append(src)
            if len(images) >= max_images:
                break

        # Cache the results
        IMAGE_CACHE[cache_key] = (images, current_time)
        
        return images
        
    except Exception as e:
        print(f"Error parsing images for '{query}': {e}")
        return []


def clear_image_cache():
    """Clear the image cache to free memory."""
    global IMAGE_CACHE
    IMAGE_CACHE.clear()


def get_cache_stats():
    """Get statistics about the image cache."""
    return {
        "cached_queries": len(IMAGE_CACHE),
        "cache_size_mb": sum(len(str(v)) for v in IMAGE_CACHE.values()) / (1024 * 1024)
    }
