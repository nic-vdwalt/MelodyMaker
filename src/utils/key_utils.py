# utils/key_utils.py

from sklearn.preprocessing import LabelEncoder

ALL_KEYS = [
    "C major","G major","D major","A major","E major","B major","F# major","C# major",
    "F major","Bb major","Eb major","Ab major","Db major","Gb major","Cb major",
    "A minor","E minor","B minor","F# minor","C# minor","G# minor","D# minor","A# minor",
    "D minor","G minor","C minor","F minor","Bb minor","Eb minor","Ab minor"
]

# train a simple encoder over all key names
key_enc = LabelEncoder().fit(ALL_KEYS)

def determine_key_from_phrase(phrase: str) -> int:
    """
    Naively map the first letter of your phrase to a key.
    Replace with any NLP or heuristic you like.
    Returns the integer index into key_enc.classes_.
    """
    first = phrase.strip()[0].upper() if phrase else "C"
    mapping = {
        "A": "A minor", "B": "B minor",
        "C": "C major", "D": "D minor",
        "E": "E minor", "F": "F major",
        "G": "G major"
    }
    key_name = mapping.get(first, "C major")
    return int(key_enc.transform([key_name])[0])
