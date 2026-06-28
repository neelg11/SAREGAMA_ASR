"""
transliterate.py
================

Unicode-based positional transliteration from Devanagari into nine other
Indic scripts.

The model (Whisper) always returns Devanagari text. This module maps each
Devanagari code point onto the equivalent slot of a target script using a
positional lookup table that mirrors the layout of the original JavaScript
``SCRIPTS`` registry. Characters that have no defined target slot (for
example, a nukta that does not exist in Bengali) fall back to the original
Devanagari character so that no information is silently dropped. Whitespace,
Latin text, and punctuation pass through unchanged.

NOTE: This is a *positional* Unicode mapping, not a phonetic or orthographic
transliterator. It is faithful to the source registry and intentionally simple;
script-specific correction rules can be layered on later.
"""

from __future__ import annotations

from typing import Dict, List

# ---------------------------------------------------------------------------
# Script registry
#
# Every per-category list is positionally aligned with Devanagari, i.e. index
# ``i`` of ``consonants`` in any script is the counterpart of index ``i`` of
# Devanagari ``consonants``. Empty strings mark slots a script does not have.
# ---------------------------------------------------------------------------

SCRIPTS: Dict[str, Dict[str, object]] = {
    "devanagari": {
        "label": "Devanagari",
        "hint": "हिन्दी / संस्कृत",
        "vowels_indep": "अ आ इ ई उ ऊ ऋ ॠ ऌ ॡ ए ऐ ओ औ".split(" "),
        "consonants": "क ख ग घ ङ च छ ज झ ञ ट ठ ड ढ ण त थ द ध न प फ ब भ म य र ल व श ष स ह".split(" "),
        "matras": ["ा", "ि", "ी", "ु", "ू", "ृ", "ॄ", "ॢ", "ॣ", "े", "ै", "ो", "ौ"],
        "signs": ["ँ", "ं", "ः", "ऽ"],
        "digits": list("०१२३४५६७८९"),
        "virama": "्", "nukta": "़", "avagraha": "ऽ",
    },
    "bengali": {
        "label": "Bengali",
        "hint": "বাংলা",
        "vowels_indep": "অ আ ই ঈ উ ঊ ঋ ৠ ঌ ৡ এ ঐ ও ঔ".split(" "),
        "consonants": "ক খ গ ঘ ঙ চ ছ জ ঝ ঞ ট ঠ ড ঢ ণ ত থ দ ধ ন প ফ ব ভ ম য র ল ব শ ষ স হ".split(" "),
        "matras": ["া", "ি", "ী", "ু", "ূ", "ৃ", "", "", "", "ে", "ৈ", "ো", "ৌ"],
        "signs": ["ঁ", "ং", "ঃ", "ঽ"],
        "digits": list("০১২৩৪৫৬৭৮৯"),
        "virama": "্", "nukta": "", "avagraha": "ঽ",
    },
    "gurmukhi": {
        "label": "Gurmukhi",
        "hint": "ਪੰਜਾਬੀ",
        "vowels_indep": ["ਅ", "ਆ", "ਇ", "ਈ", "ਉ", "ਊ", "ੲ", "੠", "੢", "੣", "ਏ", "ਐ", "ਓ", "ਔ"],
        "consonants": ["ਕ", "ਖ", "ਗ", "ਘ", "ਙ", "ਚ", "ਛ", "ਜ", "ਝ", "ਞ", "ਟ", "ਠ", "ਡ", "ਢ", "ਣ", "ਤ", "ਥ", "ਦ", "ਧ", "ਨ", "ਪ", "ਫ", "ਬ", "ਭ", "ਮ", "ਯ", "ਰ", "ਲ", "ਵ", "ਸ਼", "ਸ਼", "ਸ", "ਹ"],
        "matras": ["ਾ", "ਿ", "ੀ", "ੁ", "ੂ", "੍ਰ", "", "", "", "ੇ", "ੈ", "ੋ", "ੌ"],
        "signs": ["ਂ", "ੰ", "ਃ", ""],
        "digits": list("੦੧੨੩੪੫੬੭੮੯"),
        "virama": "੍", "nukta": "਼", "avagraha": "",
    },
    "gujarati": {
        "label": "Gujarati",
        "hint": "ગુજરાતી",
        "vowels_indep": "અ આ ઇ ઈ ઉ ઊ ઋ ૠ ઌ ૡ એ ઐ ઓ ઔ".split(" "),
        "consonants": "ક ખ ગ ઘ ઙ ચ છ જ ઝ ઞ ટ ઠ ડ ઢ ણ ત થ દ ધ ન પ ફ બ ભ મ ય ર લ વ શ ષ સ હ".split(" "),
        "matras": "ા િ ી ુ ૂ ૃ ૄ ૢ ૣ ે ૈ ો ૌ".split(" "),
        "signs": ["ઁ", "ં", "ઃ", "ઽ"],
        "digits": list("૦૧૨૩૪૫૬૭૮૯"),
        "virama": "્", "nukta": "઼", "avagraha": "ઽ",
    },
    "odia": {
        "label": "Odia",
        "hint": "ଓଡ଼ିଆ",
        "vowels_indep": "ଅ ଆ ଇ ଈ ଉ ଊ ଋ ୠ ଌ ୡ ଏ ଐ ଓ ଔ".split(" "),
        "consonants": "କ ଖ ଗ ଘ ଙ ଚ ଛ ଜ ଝ ଞ ଟ ଠ ଡ ଢ ଣ ତ ଥ ଦ ଧ ନ ପ ଫ ବ ଭ ମ ଯ ର ଲ ଵ ଶ ଷ ସ ହ".split(" "),
        "matras": "ା ି ୀ ୁ ୂ ୃ ୄ ୢ ୣ େ ୈ ୋ ୌ".split(" "),
        "signs": ["", "ଂ", "ଃ", "ଽ"],
        "digits": list("୦୧୨୩୪୫୬୭୮୯"),
        "virama": "୍", "nukta": "", "avagraha": "ଽ",
    },
    "tamil": {
        "label": "Tamil",
        "hint": "தமிழ்",
        "vowels_indep": "அ ஆ இ ஈ உ ஊ ஋ ௠ ஌ ௡ ஏ ஐ ஓ ஔ".split(" "),
        "consonants": ["க", "க", "க", "க", "ங", "ச", "ச", "ஜ", "ஜ", "ஞ", "ட", "ட", "ட", "ட", "ண", "த", "த", "த", "த", "ந", "ப", "ப", "ப", "ப", "ம", "ய", "ர", "ல", "வ", "ஷ", "ஷ", "ஸ", "ஹ"],
        "matras": ["ா", "ி", "ீ", "ு", "ூ", "௃", "௄", "௢", "௣", "ே", "ை", "ோ", "ௌ"],
        "signs": ["", "ஂ", "ஃ", ""],
        "digits": list("௦௧௨௩௪௫௬௭௮௯"),
        "virama": "்", "nukta": "", "avagraha": "",
    },
    "telugu": {
        "label": "Telugu",
        "hint": "తెలుగు",
        "vowels_indep": "అ ఆ ఇ ఈ ఉ ఊ ఋ ౠ ఌ ౡ ఏ ఐ ఓ ఔ".split(" "),
        "consonants": "క ఖ గ ఘ ఙ చ ఛ జ ఝ ఞ ట ఠ డ ఢ ణ త థ ద ధ న ప ఫ బ భ మ య ర ల వ శ ష స హ".split(" "),
        "matras": "ా ి ీ ు ూ ృ ౄ ౢ ౣ ే ై ో ౌ".split(" "),
        "signs": ["ఁ", "ం", "ః", "ఽ"],
        "digits": list("౦౧౨౩౪౫౬౭౮౯"),
        "virama": "్", "nukta": "", "avagraha": "ఽ",
    },
    "kannada": {
        "label": "Kannada",
        "hint": "ಕನ್ನಡ",
        "vowels_indep": "ಅ ಆ ಇ ಈ ಉ ಊ ಋ ೠ ಌ ೡ ಏ ಐ ಓ ಔ".split(" "),
        "consonants": "ಕ ಖ ಗ ಘ ಙ ಚ ಛ ಜ ಝ ಞ ಟ ಠ ಡ ಢ ಣ ತ ಥ ದ ಧ ನ ಪ ಫ ಬ ಭ ಮ ಯ ರ ಲ ವ ಶ ಷ ಸ ಹ".split(" "),
        "matras": ["ಾ", "ಿ", "ೀ", "ು", "ೂ", "ೃ", "ೄ", "ೢ", "ೣ", "ೆ", "ೈ", "ೊ", "ೋ"],
        "signs": ["", "ಂ", "ಃ", "ಽ"],
        "digits": list("೦೧೨೩೪೫೬೭೮೯"),
        "virama": "್", "nukta": "", "avagraha": "ಽ",
    },
    "malayalam": {
        "label": "Malayalam",
        "hint": "മലയാളം",
        "vowels_indep": "അ ആ ഇ ഈ ഉ ഊ ഋ ൠ ഌ ൡ ഏ ഐ ഓ ഔ".split(" "),
        "consonants": "ക ഖ ഗ ഘ ങ ച ഛ ജ ഝ ഞ ട ഠ ഡ ഢ ണ ത ഥ ദ ധ ന പ ഫ ബ ഭ മ യ ര ല വ ശ ഷ സ ഹ".split(" "),
        "matras": ["ാ", "ി", "ീ", "ു", "ൂ", "ൃ", "ൄ", "ൢ", "ൣ", "േ", "ൈ", "ോ", "ൌ"],
        "signs": ["", "ം", "ഃ", "ഽ"],
        "digits": list("൦൧൨൩൪൫൬൭൮൯"),
        "virama": "്", "nukta": "", "avagraha": "ഽ",
    },
}

# Categories whose lists are positionally aligned across scripts.
_LIST_CATEGORIES = ("vowels_indep", "consonants", "matras", "signs", "digits")
# Categories that are single scalar characters.
_SCALAR_CATEGORIES = ("virama", "nukta", "avagraha")

SUPPORTED_SCRIPTS: List[str] = list(SCRIPTS.keys())


def _build_devanagari_index() -> Dict[str, tuple]:
    """Map each Devanagari character to its (category, index) slot.

    Returns a dict like ``{"क": ("consonants", 0), ...}``. Scalar categories
    use index ``-1`` as a sentinel since they are single characters.
    """
    dev = SCRIPTS["devanagari"]
    index: Dict[str, tuple] = {}

    for category in _LIST_CATEGORIES:
        for i, ch in enumerate(dev[category]):  # type: ignore[arg-type]
            if ch and ch not in index:
                index[ch] = (category, i)

    for category in _SCALAR_CATEGORIES:
        ch = dev[category]  # type: ignore[assignment]
        if ch and ch not in index:
            index[ch] = (category, -1)

    return index


# Precompute once at import time; the table is static.
_DEVANAGARI_INDEX: Dict[str, tuple] = _build_devanagari_index()


def transliterate(text: str, target: str) -> str:
    """Transliterate Devanagari ``text`` into the ``target`` script.

    Args:
        text: Source string in Devanagari (the model output).
        target: Key into :data:`SCRIPTS`, e.g. ``"telugu"``.

    Returns:
        The transliterated string. If ``target`` is ``"devanagari"`` or
        unknown, the input is returned unchanged. Characters with no target
        slot fall back to the original Devanagari character.
    """
    if not text:
        return text
    if target == "devanagari":
        return text

    script = SCRIPTS.get(target)
    if script is None:
        # Unknown target: fail open, return source untouched.
        return text

    out: List[str] = []
    for ch in text:
        slot = _DEVANAGARI_INDEX.get(ch)
        if slot is None:
            # Whitespace, Latin, punctuation, ZWJ, etc. — pass through.
            out.append(ch)
            continue

        category, idx = slot
        if idx == -1:
            # Scalar category (virama / nukta / avagraha).
            mapped = script.get(category, "")  # type: ignore[assignment]
        else:
            seq = script.get(category, [])  # type: ignore[assignment]
            mapped = seq[idx] if idx < len(seq) else ""  # type: ignore[index]

        # Fall back to the source glyph when the target slot is empty.
        out.append(mapped if mapped else ch)

    return "".join(out)


def list_scripts() -> List[Dict[str, str]]:
    """Return UI-friendly metadata for every supported script."""
    return [
        {"id": key, "label": str(val["label"]), "hint": str(val["hint"])}
        for key, val in SCRIPTS.items()
    ]


if __name__ == "__main__":
    # Tiny smoke test.
    sample = "नमस्ते दुनिया १२३"
    print(f"{'source':<12}: {sample}")
    for sid in SUPPORTED_SCRIPTS:
        print(f"{sid:<12}: {transliterate(sample, sid)}")
