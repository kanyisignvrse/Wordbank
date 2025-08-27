from pathlib import Path
import csv
import json
import pandas as pd


def load_ngsl_words() -> set[str]:
    csv_path = Path("./NGSL_1.2_stats_no_articles.csv")
    fallback_path = Path("./NGSL_1.2_stats.csv")
    articles = {"a", "an", "the"}

    if not csv_path.exists():
        csv_path = fallback_path

    unique_words: set[str] = set()

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = row.get("Lemma")
            if not raw:
                continue
            word = str(raw).strip().lower()
            if csv_path.name == fallback_path.name and word in articles:
                # If using the original file, exclude articles
                continue
            if word:
                unique_words.add(word)

    return unique_words


def load_oxford_words() -> set[str]:
    json_path = Path("./oxford_5000.json")
    with json_path.open(encoding="utf-8") as f:
        data = json.load(f)
    words = data.get("words", [])
    return {str(w).strip().lower() for w in words if str(w).strip()}

def load_city_words()-> set[str]:
    json_path = Path('./cities.json')
    with json_path.open(encoding='utf-8') as f:
        data = json.load(f)
    words = data.get('cities', [])
    words.extend(data.get('landmarks', []))
    return {str(w).strip().lower() for w in words if str(w).strip()}

def load_general_words()-> set[str]:
    json_path = Path('./general.json')
    with json_path.open(encoding='utf-8') as f:
        data = json.load(f)
    words = data.get('professions', [])
    words.extend(data.get('animals', []))
    words.extend(data.get('nature', []))
    words.extend(data.get('health', []))
    words.extend(data.get('general_knowledge', []))
    words.extend(data.get('sports', []))
    words.extend(data.get('education', []))
    words.extend(data.get('politics', []))
    words.extend(data.get('english_ksl_word_list', []))
    
    return {str(w).strip().lower() for w in words if str(w).strip()}

def main() -> None:
    ngsl = load_ngsl_words()
    oxford = load_oxford_words()
    cities = load_city_words()
    general = load_general_words()
    combined = ngsl | oxford | cities | general
    print(len(combined))

    # Filter out compound words (those containing spaces)
    single_words = {word for word in combined if " " not in word}
    print(f"After removing compound words: {len(single_words)}")

    # Create a DataFrame of the combined unique words and export to CSV
    words_sorted = sorted(single_words)
    df = pd.DataFrame({"word": words_sorted})
    df.to_csv(Path("./combined_unique_words.csv"), index=False)


if __name__ == "__main__":
    main()



