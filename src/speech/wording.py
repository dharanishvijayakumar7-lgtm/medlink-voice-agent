"""Text checks for speech, independent of a model's claimed language code.

Script and sentence checks catch obvious defects, not translation or clinical
accuracy. Never repair a cut-off medical sentence by guessing its missing words.
"""

import re
import unicodedata

SCRIPT_RANGES = {
    "hi-IN": (0x0900, 0x097F),
    "ta-IN": (0x0B80, 0x0BFF),
    "te-IN": (0x0C00, 0x0C7F),
    "kn-IN": (0x0C80, 0x0CFF),
    "ml-IN": (0x0D00, 0x0D7F),
}


def validate_spoken_text(text: str, language: str, *, complete: bool = False) -> None:
    if "\ufffd" in text or re.search(r"[\u0000-\u0008\u000b\u000c\u000e-\u001f]", text):
        raise ValueError("spoken_text: Regenerate corrupt speech text")
    ending = text.rstrip().rstrip('"\u201d\u2019')
    if complete and (
        not re.search(r"[.!?।॥]$", ending) or ending.endswith(("...", "…"))
    ):
        raise ValueError(
            "spoken_text: Finish every spoken assessment field with a complete sentence"
        )
    letters = [c for c in text if unicodedata.category(c).startswith("L")]
    if language == "en-IN":
        matches = sum("LATIN" in unicodedata.name(c, "") for c in letters)
    else:
        low, high = SCRIPT_RANGES[language]
        matches = sum(low <= ord(c) <= high for c in letters)
        # A native-script sentence can still contain an English diagnosis.
        # Permit short uppercase acronyms, but require ordinary words in script.
        latin_words = re.findall(r"[A-Za-z]+", text)
        if any(not (w.isupper() and 2 <= len(w) <= 5) for w in latin_words):
            raise ValueError(
                "spoken_language: Use native script for every spoken word, including the possible cause"
            )
    if not letters or matches / len(letters) < 0.9:
        raise ValueError(
            "spoken_language: Each spoken field must use the selected language's script"
        )


def normalize_for_speech(text: str) -> str:
    # The installed provider sentence tokenizer recognizes .?! but not danda.
    # Full stops also give short clauses a clearer boundary than semicolons.
    return text.translate(str.maketrans({";": ".", "।": ".", "॥": "."}))


LANGUAGE_STYLE = {
    "en-IN": "Use everyday spoken English. Say 'may be related to', not a confirmed diagnosis.",
    "hi-IN": "सहज बोलचाल की हिंदी लिखें। वजह के लिए 'हो सकता है' कहें। अंग्रेज़ी में बीमारी का नाम या कठिन अनुवाद न लिखें।",
    "ta-IN": "இயல்பான பேச்சுத் தமிழில் பதில் சொல்லுங்கள். காரணம் உறுதி இல்லையெனில் 'இருக்கலாம்' என்று சொல்லுங்கள். ஆங்கில நோய்ப் பெயர்களையும் கடினமான சொற்களையும் தவிருங்கள்.",
    "te-IN": "సహజంగా మాట్లాడే తెలుగులో చిన్న వాక్యాలు రాయండి. కారణం ఖచ్చితంగా తెలియకపోతే 'కావచ్చు' లేదా 'ఉండవచ్చు' అనండి. ఆంగ్ల వ్యాధి పేర్లు, కఠినమైన వైద్య పదాలు వాడకండి. ఉదాహరణకు 'పరీక్ష లేకుండా కారణాన్ని ఖచ్చితంగా చెప్పలేను' అనండి.",
    "kn-IN": "ಸಹಜವಾಗಿ ಮಾತನಾಡುವ ಕನ್ನಡದಲ್ಲಿ ಚಿಕ್ಕ ವಾಕ್ಯಗಳನ್ನು ಬಳಸಿ. ಕಾರಣ ಖಚಿತವಿಲ್ಲದಾಗ 'ಇರಬಹುದು' ಎಂದು ಹೇಳಿ. ಇಂಗ್ಲಿಷ್ ರೋಗದ ಹೆಸರುಗಳು ಮತ್ತು ಕಠಿಣ ವೈದ್ಯಕೀಯ ಪದಗಳನ್ನು ತಪ್ಪಿಸಿ.",
    "ml-IN": "സ്വാഭാവികമായി സംസാരിക്കുന്ന മലയാളത്തിൽ ചെറിയ വാക്യങ്ങൾ എഴുതുക. കാരണം ഉറപ്പില്ലെങ്കിൽ 'ആകാം' എന്ന് പറയുക. ഇംഗ്ലീഷിലുള്ള രോഗപ്പേരുകളും ബുദ്ധിമുട്ടുള്ള വൈദ്യപദങ്ങളും ഒഴിവാക്കുക.",
}
