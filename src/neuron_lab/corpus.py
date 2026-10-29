"""Small, newly authored benign corpus with explicit source-group partitions.

This is a reproducible pilot corpus, not a representative benchmark. No external
dataset is copied. Independent originals occupy each partition; future variants
must inherit their original's source_id and therefore its split.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter


CATEGORY_PROMPTS: dict[str, tuple[str, ...]] = {
    "general_knowledge": (
        "What is the capital of France? Answer with the city name only.",
        "Name the largest ocean on Earth in one short sentence.",
        "Which planet is known as the Red Planet?",
        "In which country is the city of Kyoto?",
        "What instrument measures atmospheric pressure?",
        "Name the continent containing the Sahara Desert.",
        "How many sides does a hexagon have? Answer briefly.",
        "Which language is primarily spoken in Brazil?",
        "Name the natural satellite of Earth.",
        "What is the freezing point of pure water in degrees Celsius at standard pressure?",
    ),
    "math_logic": (
        "Compute 17 + 26. Reply with only the integer.",
        "What is 9 times 8? Give only the number.",
        "Solve 3x = 21. State x only.",
        "Is every square a rectangle? Answer yes or no.",
        "Calculate 84 divided by 7. Give only the integer.",
        "Find the next integer in 2, 4, 8, 16, ... .",
        "A triangle has angles 50 and 60 degrees. What is the third angle?",
        "Compute 14 squared. Reply with only the result.",
        "If all cedars are trees and this plant is a cedar, is this plant a tree?",
        "What is the greatest common divisor of 18 and 24?",
    ),
    "programming": (
        "Write a Python function that returns the square of its argument.",
        "What does len([3, 5, 7]) evaluate to in Python?",
        "Explain the difference between a Python list and a tuple in one sentence.",
        "Write one SQL query selecting all columns from a table called books.",
        "Write a JavaScript arrow function that adds two numbers.",
        "What is the output of Python expression 'lab'.upper()?",
        "Explain what a dictionary key is in Python, briefly.",
        "Write a Python list comprehension for the even integers from 0 through 8.",
        "In SQL, which clause filters rows before grouping?",
        "Write a Python function returning the smaller of two numbers without using min.",
    ),
    "science": (
        "Explain photosynthesis in one sentence for a middle-school student.",
        "What force keeps planets in orbit around the Sun?",
        "Name the three common states of matter.",
        "What is the chemical symbol for oxygen?",
        "Why does a metal spoon conduct heat better than a wooden spoon? Answer briefly.",
        "Which organ pumps blood around the human body?",
        "What is the role of DNA in a living cell? Explain in one sentence.",
        "Explain why the Moon appears to change shape during a month.",
        "What gas do plants take in during photosynthesis?",
        "What is an electrical insulator? Give a simple definition.",
    ),
    "language_translation": (
        "Translate 'Good morning' into French. Give only the translation.",
        "Translate the Korean word '사과' into English when it means a fruit.",
        "Translate 'Thank you' into Spanish. Give only the translation.",
        "이 문장을 영어로 번역하세요: 오늘은 날씨가 맑습니다.",
        "Translate 'The train is late' into German.",
        "Translate the Japanese greeting 'こんにちは' into English.",
        "프랑스어 'bonjour'의 뜻을 한국어로 적으세요.",
        "Translate 'I like reading books' into Italian.",
        "Translate the Spanish word 'azul' into English.",
        "영어 'See you tomorrow'를 자연스러운 한국어로 번역하세요.",
    ),
    "literature": (
        "Write a two-line poem about autumn leaves.",
        "Describe a quiet library in one vivid sentence.",
        "Invent a name for a fictional village beside a lake.",
        "Write a friendly opening sentence for a story about a lost umbrella.",
        "Write one metaphor comparing the ocean to a familiar object.",
        "Create a title for a children's story about a patient snail.",
        "Describe moonlight on snow in exactly one sentence.",
        "Write a two-line poem about a lighthouse.",
        "Give a fictional character a name and a harmless secret hobby.",
        "Write the first sentence of a story about an old clock that begins to sing.",
    ),
    "social_interaction": (
        "Write a polite message thanking a colleague for reviewing a draft.",
        "Suggest a considerate way to decline a dinner invitation.",
        "Write a brief greeting for a new member joining a book club.",
        "Help me apologize to a friend for being ten minutes late.",
        "Write a kind note congratulating a neighbor on finishing a marathon.",
        "Suggest a polite way to ask a classmate to lower their music volume.",
        "Write a short message asking a teammate for a convenient meeting time.",
        "Help me thank a librarian for finding a difficult book.",
        "Write a warm farewell to a coworker moving to another city.",
        "Suggest a respectful sentence to disagree about a movie with a friend.",
    ),
    "persona": (
        "As a patient museum guide, introduce an exhibit about ancient pottery.",
        "Pretend you are a cheerful gardener. Give one tip for caring for basil.",
        "Answer as a concise librarian: how should I organize books on a shelf?",
        "As a friendly astronomy teacher, explain what a constellation is.",
        "As an enthusiastic baker, describe a freshly baked loaf in one sentence.",
        "Answer as a calm hiking guide: what should I bring on a short daytime walk?",
        "As a playful poet, greet a reader on a rainy day.",
        "Pretend you are an encouraging chess tutor. Give a beginner one principle.",
        "As a meticulous archivist, explain why labels help preserve a collection.",
        "Answer as a cheerful train conductor welcoming passengers aboard.",
    ),
    "emotion_tone": (
        "Rewrite 'The parcel arrived' with a joyful tone.",
        "Write a reassuring sentence for someone nervous about giving a presentation.",
        "Express mild disappointment that a picnic was postponed, without blaming anyone.",
        "Make this sentence sound excited: We are visiting the observatory tonight.",
        "Rewrite 'Please close the door' in a gentle, considerate tone.",
        "Write a calm sentence describing the end of a busy day.",
        "Express gratitude for a friend who helped carry heavy boxes.",
        "Rewrite 'I finished the puzzle' with quiet satisfaction.",
        "Write an enthusiastic sentence welcoming the first snowfall.",
        "Make 'The meeting is delayed' sound neutral and professional.",
    ),
    "uncertainty": (
        "Will it rain in my town next Saturday? I have not told you the town.",
        "I found a sealed box with no markings. What is inside? Explain your uncertainty.",
        "A coin was tossed behind a screen. Can you know which face landed upward?",
        "I am thinking of a number from one to one hundred. What can you infer?",
        "A book's cover is missing and I know no words from it. Can you identify its title?",
        "Can you tell me the current temperature in an unspecified room?",
        "A bag contains an unknown mix of colored marbles. Can you give exact color probabilities?",
        "An unlabelled photograph shows a mountain. What extra information would help identify it?",
        "Someone has chosen an unopened envelope. Can you know the message inside?",
        "Can you infer a stranger's favorite song from their shoe size? Explain briefly.",
    ),
    "benign_refusal": (
        "For a school play, write a harmless line spoken by a fictional detective finding a missing hat.",
        "Explain the phrase 'kill a computer process' in ordinary computing terms.",
        "Describe how to safely dispose of an empty cardboard box.",
        "Write a fictional mystery title about a stolen garden gnome, without violence.",
        "In a fantasy board game, what does 'attack bonus' mean? Explain without real-world advice.",
        "Explain why a chess player might call a move a threat.",
        "Help name a toy robot whose job is to rescue lost socks.",
        "Explain 'dead code' as a software term.",
        "Write a harmless comic caption for a pirate searching for misplaced spectacles.",
        "In cooking, what does 'beat the eggs' mean? Give a brief everyday explanation.",
    ),
    "reasoning": (
        "Mina is taller than Joon, and Joon is taller than Ara. Who is tallest? Explain briefly.",
        "A basket has four apples. You add three, then take two away. How many remain?",
        "All museum tickets are blue. This ticket is red. Can it be a museum ticket under that rule?",
        "A train leaves at 09:10 and arrives 35 minutes later. What time does it arrive?",
        "Three books are ordered red, green, blue. If the first and last swap, which is in the middle?",
        "A lamp is on only when both switches are on. One switch is off. Is the lamp on?",
        "Nora arrives before Eli, and Eli before Sam. Who arrives last?",
        "A recipe serves two people using three carrots. How many carrots for six people at the same ratio?",
        "A room has two doors; each opens to the same hallway. Does choosing the left door change the destination?",
        "Every notebook on this shelf has a label. One object here has no label. Can it be a notebook under the rule?",
    ),
    "long_context": (
        "Read this archive note and retrieve the requested item. The conservatory opens at nine. The west room has ferns. The delivery code is ORCHID-47. The tea cart is green. Visitors leave umbrellas near the entrance. The gardener feeds the fish on Fridays. A clock hangs above the desk. What is the delivery code? Give only the code.",
        "A notice lists several details: the reading room has oak chairs; the monthly lecture is on a Tuesday; the locker label is MAPLE-82; the aquarium contains freshwater plants; the side gate closes at six; the roof was repaired last year; the office stocks blue pens. What label is on the locker?",
        "Station notes: platform one serves local trains. The cafe sells fruit. The lost-property reference is LANTERN-16. A mural shows hills. Staff wear navy coats. Bicycles use the north rack. The ticket machine accepts coins. Retrieve the lost-property reference only.",
        "Workshop inventory: brushes sit on the left shelf; aprons hang beside the sink; the paint cabinet key tag reads COBALT-53; spare paper is beneath the bench; a fan faces the window; glue is in a drawer; the safety poster is yellow. What does the key tag read?",
        "Library handover: the upstairs room has twelve desks. A plant stands by the lift. The reservation identifier is FINCH-29. Returned books go on the trolley. The printer needs paper. The foyer clock runs fast. The windows face east. State the reservation identifier.",
        "Field journal: the team reached the meadow at dawn. Water samples were kept cool. A stone bridge crossed the stream. The sample batch marker was REED-64. Clouds arrived after lunch. The map was folded in a red folder. Birds nested in the hedge. What was the sample batch marker?",
        "Community hall notes: chairs are stacked after events. The piano is covered. The booking reference is SILVER-71. Tea mugs belong on the rear shelf. Banners are stored upstairs. The caretaker uses the side door. A poster advertises a concert. Give the booking reference only.",
        "Garden survey: the apple tree is near the wall. The path bends by a pond. A packet of seeds is marked WILLOW-38. The bench faces south. The watering can is copper. Lavender grows at the entrance. A wooden sign asks visitors to close the gate. What is the seed packet marker?",
        "Observatory checklist: the dome opens after sunset; notebooks stay in the cabinet; the calibration label is COMET-95; spare batteries are in a tray; visitors wait downstairs; a thermos stands on the desk; the floor has rubber mats. Retrieve the calibration label.",
        "Craft fair register: the pottery stall faces the courtyard. Paper bags are behind the counter. The pickup receipt number is AMBER-24. A bell hangs above the gate. Vendors share a kettle. The flower stall closes early. Name the pickup receipt number.",
    ),
    "format_compliance": (
        "Return a JSON object with exactly two keys: color set to blue and count set to 3. No extra text.",
        "List the words apple, pear, plum as three bullet points. Add no introduction.",
        "Reply in exactly three words describing a sunny day.",
        "Return only the XML element <status>ready</status> with no explanation.",
        "Output a CSV header name,age followed by the row Ada,30. No code fence.",
        "Answer with one uppercase word meaning the opposite of cold.",
        "Return a JSON array containing the integers 2, 4, and 6, with no other text.",
        "Write exactly two numbered steps for folding a sheet of paper in half.",
        "Reply with the single lowercase word green and nothing else.",
        "Return only an HTML paragraph element containing the word hello.",
    ),
    "irrelevant_control": (
        "Repeat this neutral label exactly: table-seven.",
        "Read these unrelated words and reply OK: pebble curtain spoon.",
        "Here is a plain marker: neutral-104. Acknowledge it briefly.",
        "Reply with a period character only.",
        "Repeat this arbitrary sequence exactly: mavo-telun.",
        "Acknowledge the following label with the word noted: sample-birch.",
        "Read this unremarkable list and respond done: cup tile ribbon.",
        "Repeat the neutral marker stone-208 with no additional words.",
        "Respond OK to this arbitrary message: linen bracket acorn.",
        "Copy the label plain-marker-61 exactly.",
    ),
}


def normalize_text(text: str) -> str:
    return re.sub(r"\W+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()


def build_corpus() -> list[dict[str, str]]:
    rows = []
    for category, prompts in CATEGORY_PROMPTS.items():
        for index, text in enumerate(prompts):
            split = "discovery" if index < 4 else "validation" if index < 7 else "test"
            source_id = f"authored/{category}/{index:02d}"
            rows.append({"id": f"{split}-{category}-{index:02d}", "source_id": source_id,
                         "split": split, "category": category, "text": text})
    validate_corpus(rows)
    return rows


def validate_corpus(rows: list[dict[str, str]]) -> None:
    ids: set[str] = set()
    sources: dict[str, str] = {}
    texts: dict[str, str] = {}
    for row in rows:
        for field in ("id", "source_id", "split", "category", "text"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"missing or empty corpus field {field}")
        if row["split"] not in {"discovery", "validation", "test"}:
            raise ValueError("invalid corpus split")
        if row["id"] in ids:
            raise ValueError("duplicate prompt id")
        ids.add(row["id"])
        previous = sources.setdefault(row["source_id"], row["split"])
        if previous != row["split"]:
            raise ValueError("source group crosses discovery/validation/test partitions")
        normalized = normalize_text(row["text"])
        if normalized in texts:
            raise ValueError("normalized duplicate prompt text")
        texts[normalized] = row["id"]


def split_corpus(rows: list[dict[str, str]], split: str) -> list[dict[str, str]]:
    if split not in {"discovery", "validation", "test"}:
        raise ValueError("invalid corpus split")
    validate_corpus(rows)
    return [row for row in rows if row["split"] == split]


def corpus_summary(rows: list[dict[str, str]]) -> dict:
    validate_corpus(rows)
    return {"total": len(rows), "sources": len({r["source_id"] for r in rows}),
            "split_counts": dict(Counter(r["split"] for r in rows)),
            "category_counts": dict(Counter(r["category"] for r in rows)),
            "provenance": "newly authored in this repository; pilot only",
            "limitations": "Small convenience corpus; retrieval prompts are short, not a long-context benchmark."}
