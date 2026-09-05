"""Portable editing settings shared by the command line and saved jobs."""
import math

DEFAULTS = {"duration": "auto", "style": "hype", "look": "natural", "strength": 0.0,
            "quality": "auto", "audio_level": .4, "codec": "hevc", "music": None,
            "music_level": .75, "music_offset": 0.0, "music_fade": 1.5, "music_end": "fade",
            "beat_sync": True, "social_formats": [], "framing": "blur", "focus_x": .5,
            "edit_order": "story", "recovery": 2.5, "recognition": "auto"}


def resolve_settings(args, saved):
    settings = {key: getattr(args, key, None) if getattr(args, key, None) is not None
                else saved.get(key, value) for key, value in DEFAULTS.items()}
    if getattr(args, "no_music", False):
        settings["music"] = None
    for key, minimum, maximum in (("strength", 0, 1), ("audio_level", 0, 1),
                                  ("music_level", 0, 1), ("music_offset", 0, 86400),
                                  ("music_fade", 0, 15), ("focus_x", 0, 1), ("recovery", .5, 8)):
        value = settings[key]
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not minimum <= value <= maximum:
            raise ValueError(f"{key.replace('_', ' ').title()} must be between {minimum} and {maximum}")
    choices = {"duration": ["auto", "15", "30", "60", "90", "120", "180"],
               "style": ["hype", "cinematic", "freestyle", "flow"],
               "look": ["natural", "punch", "cinematic"], "quality": ["auto", "lanczos", "ai"],
               "codec": ["hevc", "h264"], "music_end": ["fade", "loop"],
               "framing": ["blur", "fit", "fill"], "edit_order": ["story", "chronological"],
               "recognition": ["auto", "off", "thorough"]}
    for key, allowed in choices.items():
        if settings[key] not in allowed:
            raise ValueError(f"Invalid saved {key}: choose {', '.join(allowed)}")
    formats = settings["social_formats"]
    if isinstance(formats, str):
        formats = [] if formats in ("none", "") else formats.split(",")
    if not isinstance(formats, list) or any(code not in ("vertical", "square", "portrait") for code in formats):
        raise ValueError("Social formats must be vertical, square, portrait, or none")
    settings["social_formats"] = list(dict.fromkeys(formats))
    if not isinstance(settings["beat_sync"], bool):
        raise ValueError("Beat sync must be on or off")
    return settings
