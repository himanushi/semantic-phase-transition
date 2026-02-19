"""Ambiguous prompt definitions for phase transition experiments."""

from typing import TypedDict


class PromptSet(TypedDict):
    neutral: str
    weak_A: str
    weak_B: str
    strong_A: str
    strong_B: str


class AmbiguousWord(TypedDict):
    word: str
    interpretation_A: str
    interpretation_B: str
    prompt_A: str  # 方向ベクトル ê_A 用の明確な文
    prompt_B: str  # 方向ベクトル ê_B 用の明確な文
    prompts: PromptSet


AMBIGUOUS_WORDS: dict[str, AmbiguousWord] = {
    "bank": {
        "word": "bank",
        "interpretation_A": "finance",
        "interpretation_B": "river",
        "prompt_A": "I deposited money at the bank",
        "prompt_B": "I sat by the river bank",
        "prompts": {
            "neutral": "The bank",
            "weak_A": "She went to the bank to",
            "weak_B": "He walked along the bank near",
            "strong_A": "She deposited her savings at the bank",
            "strong_B": "The fish swam near the muddy bank of the river",
        },
    },
    "bat": {
        "word": "bat",
        "interpretation_A": "animal",
        "interpretation_B": "sports",
        "prompt_A": "The bat hung upside down in the dark cave",
        "prompt_B": "The baseball player picked up his wooden bat",
        "prompts": {
            "neutral": "The bat",
            "weak_A": "The bat flew through",
            "weak_B": "He swung the bat at",
            "strong_A": "The bat hung upside down in the dark cave at night",
            "strong_B": "He hit the baseball hard with his bat",
        },
    },
    "crane": {
        "word": "crane",
        "interpretation_A": "bird",
        "interpretation_B": "machine",
        "prompt_A": "The white crane flew gracefully over the wetlands",
        "prompt_B": "The construction crane lifted steel beams to the roof",
        "prompts": {
            "neutral": "The crane",
            "weak_A": "The crane stood in the shallow",
            "weak_B": "The crane lifted the heavy",
            "strong_A": "The white crane flew gracefully over the wetlands",
            "strong_B": "The construction crane lifted steel beams to the roof",
        },
    },
}
