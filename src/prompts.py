"""Ambiguous prompt definitions for phase transition experiments (v2).

All prompts place disambiguating context BEFORE the target word,
respecting the causal attention mask of decoder-only models.
"""


# 方向ベクトル用プロンプト: 各解釈について3-5個の明確な文
# 最終層残差ストリームを平均して方向ベクトルを作る
DIRECTION_PROMPTS: dict[str, dict] = {
    "bank": {
        "word": "bank",
        "interpretation_A": "finance",
        "interpretation_B": "river",
        "prompts_A": [
            "I deposited money at the bank",
            "She cashed her paycheck at the bank",
            "The teller at the savings bank",
            "He applied for a loan at the bank",
            "After checking her account she left the bank",
        ],
        "prompts_B": [
            "He sat by the river bank",
            "The muddy water eroded the bank",
            "Fish gathered near the steep bank",
            "She walked along the grassy bank",
            "Trees grew along the creek bank",
        ],
    },
    "bat": {
        "word": "bat",
        "interpretation_A": "animal",
        "interpretation_B": "sports",
        "prompts_A": [
            "In the dark cave hung a sleeping bat",
            "At night from the attic emerged a bat",
            "Using echolocation the flying bat",
            "With its leathery wings the nocturnal bat",
            "Hanging upside down was a vampire bat",
        ],
        "prompts_B": [
            "The baseball player swung his wooden bat",
            "He hit a home run with the aluminum bat",
            "The slugger gripped his Louisville bat",
            "She picked up the heavy softball bat",
            "In the dugout he grabbed his favorite bat",
        ],
    },
    "crane": {
        "word": "crane",
        "interpretation_A": "bird",
        "interpretation_B": "machine",
        "prompts_A": [
            "Flying over the wetlands was a white crane",
            "Standing in the shallow marsh the tall crane",
            "During migration the whooping crane",
            "With its long legs the elegant crane",
            "Nesting by the lake the sandhill crane",
        ],
        "prompts_B": [
            "At the construction site the tower crane",
            "Lifting heavy steel beams the industrial crane",
            "The operator drove the massive crane",
            "To build the skyscraper they used a crane",
            "On the dock the cargo loading crane",
        ],
    },
    "spring": {
        "word": "spring",
        "interpretation_A": "season",
        "interpretation_B": "coil",
        "prompts_A": [
            "Flowers bloom every spring",
            "After the cold winter came the warm spring",
            "Birds returned in the early spring",
            "Cherry blossoms appeared in spring",
            "The garden flourished in the late spring",
        ],
        "prompts_B": [
            "The mattress had a broken coil spring",
            "He compressed the metal spring",
            "The mechanism relied on a tiny spring",
            "Inside the clock was a wound spring",
            "The garage door had a torsion spring",
        ],
    },
    "rock": {
        "word": "rock",
        "interpretation_A": "stone",
        "interpretation_B": "music",
        "prompts_A": [
            "He threw the heavy rock",
            "At the base of the cliff lay a massive rock",
            "She climbed over the slippery rock",
            "The geologist examined the volcanic rock",
            "Sitting on the smooth flat rock",
        ],
        "prompts_B": [
            "The band played loud rock",
            "He listened to classic rock",
            "The guitar solo defined rock",
            "She loved punk rock",
            "At the stadium they performed rock",
        ],
    },
    "match": {
        "word": "match",
        "interpretation_A": "fire",
        "interpretation_B": "competition",
        "prompts_A": [
            "He lit the candle with a match",
            "She struck the sulfur match",
            "In the dark he found a match",
            "The campfire started with one match",
            "He carefully held the burning match",
        ],
        "prompts_B": [
            "The tennis players started the match",
            "They won the championship match",
            "The crowd cheered during the boxing match",
            "After a long soccer match",
            "The referee started the wrestling match",
        ],
    },
    "light": {
        "word": "light",
        "interpretation_A": "illumination",
        "interpretation_B": "weight",
        "prompts_A": [
            "She turned on the bright light",
            "The room was filled with warm light",
            "Sunlight cast a golden light",
            "The lamp gave off a dim light",
            "Through the window streamed the morning light",
        ],
        "prompts_B": [
            "The suitcase was surprisingly light",
            "This aluminum frame is very light",
            "Compared to steel the carbon fiber is light",
            "She preferred the light",
            "The feather was extremely light",
        ],
    },
    "pitcher": {
        "word": "pitcher",
        "interpretation_A": "container",
        "interpretation_B": "baseball",
        "prompts_A": [
            "She poured water from the ceramic pitcher",
            "The glass pitcher",
            "On the table sat a full pitcher",
            "He filled the iced tea pitcher",
            "The bartender grabbed the beer pitcher",
        ],
        "prompts_B": [
            "The starting pitcher",
            "On the mound stood the relief pitcher",
            "He was drafted as a pitcher",
            "The left-handed pitcher",
            "After nine innings the exhausted pitcher",
        ],
    },
    "bass": {
        "word": "bass",
        "interpretation_A": "fish",
        "interpretation_B": "music",
        "prompts_A": [
            "He caught a largemouth bass",
            "The lake was full of bass",
            "Using a lure he hooked a striped bass",
            "The fisherman landed a big bass",
            "In the river they found smallmouth bass",
        ],
        "prompts_B": [
            "She played the electric bass",
            "The bass",
            "He plucked the upright bass",
            "The band needed a bass",
            "With deep low notes the bass",
        ],
    },
}


# 実験用プロンプト: 全て対象語の「前」に文脈を配置
EXPERIMENT_PROMPTS: dict[str, dict[str, str]] = {
    "bank": {
        "neutral": "She arrived at the bank",
        "weak_A": "With her paycheck she went to the bank",
        "weak_B": "Walking along the water he reached the bank",
        "strong_A": "After checking her savings account balance she left the bank",
        "strong_B": "Sitting by the muddy river with fishing gear he watched the bank",
    },
    "bat": {
        "neutral": "He saw the bat",
        "weak_A": "In the dark cave he spotted the bat",
        "weak_B": "In the dugout he grabbed the bat",
        "strong_A": "Using echolocation in the dark cave at night he heard the bat",
        "strong_B": "The baseball player in the dugout picked up his wooden bat",
    },
    "crane": {
        "neutral": "He looked at the crane",
        "weak_A": "Near the wetlands he watched the crane",
        "weak_B": "At the construction site he operated the crane",
        "strong_A": "Flying over the marsh with its long legs the graceful crane",
        "strong_B": "Lifting steel beams at the construction site the massive crane",
    },
    "spring": {
        "neutral": "He thought about the spring",
        "weak_A": "After the cold winter came the spring",
        "weak_B": "Inside the mechanism was a spring",
        "strong_A": "Cherry blossoms and warm weather marked the arrival of spring",
        "strong_B": "The engineer replaced the broken coil spring",
    },
    "rock": {
        "neutral": "She enjoyed the rock",
        "weak_A": "At the cliff he picked up the rock",
        "weak_B": "At the concert she heard the rock",
        "strong_A": "The geologist examined the volcanic rock",
        "strong_B": "With electric guitars and drums they played rock",
    },
    "match": {
        "neutral": "He watched the match",
        "weak_A": "To light the candle he used a match",
        "weak_B": "At the stadium he watched the match",
        "strong_A": "He carefully struck the sulfur match",
        "strong_B": "The crowd cheered during the championship tennis match",
    },
    "light": {
        "neutral": "She noticed the light",
        "weak_A": "In the dark room she turned on the light",
        "weak_B": "She picked up the surprisingly light",
        "strong_A": "Through the window streamed the bright morning light",
        "strong_B": "Compared to the heavy iron the aluminum was very light",
    },
    "pitcher": {
        "neutral": "She looked at the pitcher",
        "weak_A": "She poured water from the pitcher",
        "weak_B": "On the mound stood the pitcher",
        "strong_A": "She filled the glass iced tea pitcher",
        "strong_B": "The left-handed relief pitcher",
    },
    "bass": {
        "neutral": "He liked the bass",
        "weak_A": "With a lure he caught the bass",
        "weak_B": "In the band she played the bass",
        "strong_A": "Using a fishing rod he caught a largemouth bass",
        "strong_B": "With deep low notes she plucked the electric bass",
    },
}
