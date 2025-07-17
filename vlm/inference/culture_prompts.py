# Prompts incorporating cultural norms and country-specific nouns for India and China

CULTURE_PROMPTS = {
    "hi": [
        # Prompt 1: Explicitly reference Indian cultural norms and context
        "\n\nIn the context of Indian cultural values and social norms, does this meme contain hate speech or not? Consider references to religion, caste, gender, or nationality that may be sensitive in India. Answer with a single letter: a or b.",
        # Prompt 2: Ask about hate speech using Indian-specific nouns
        "\n\nGiven this meme, and considering Indian society (e.g., references to Hindu, Muslim, Sikh, Dalit, Indian traditions, or national identity), is it hate speech? Answer with only one letter: a or b, and nothing else.",
        # Prompt 3: Ask as an Indian citizen
        "\n\nAs someone familiar with Indian culture and its social boundaries, would you consider this meme hate speech? Respond with a single letter only."
    ],
    "zh": [
        # Prompt 1: Explicitly reference Chinese cultural norms and context
        "\n\nIn the context of Chinese cultural values and social norms, does this meme contain hate speech or not? Consider references to ethnicity, nationality, gender, or social harmony that may be sensitive in China. Answer with a single letter: a or b.",
        # Prompt 2: Ask about hate speech using Chinese-specific nouns
        "\n\nGiven this meme, and considering Chinese society (e.g., references to Han, minority groups, Chinese traditions, or national identity), is it hate speech? Answer with only one letter: a or b, and nothing else.",
        # Prompt 3: Ask as a Chinese citizen
        "\n\nAs someone familiar with Chinese culture and its social boundaries, would you consider this meme hate speech? Respond with a single letter only."
    ]
}

CULTURE_PROMPT_CAPTION = {
    "hi": "\nMeme caption (in Indian context): '{}'",
    "zh": "\nMeme caption (in Chinese context): '{}'",
}

CULTURE_PROMPT_PREFIX = {
    "hi": "Definition of Hate Speech in Indian context:\n\nAny communication that attacks or discriminates against a person or group based on religion (Hindu, Muslim, Sikh, etc.), caste (Dalit, Brahmin, etc.), gender, or national identity, violating Indian social norms.",
    "zh": "Definition of Hate Speech in Chinese context:\n\nAny communication that attacks or discriminates against a person or group based on ethnicity (Han, minority groups), nationality, gender, or other identity factors, violating Chinese social harmony and values.",
}

CULTURE_PROMPT_POSTFIX = {
    "hi": ["\nChoose:\na: Hate\nb: Non-hate\n", "\nChoose:\na: Non-hate\nb: Hate\n"],
    "zh": ["\nChoose:\na: Hate\nb: Non-hate\n", "\nChoose:\na: Non-hate\nb: Hate\n"],
}

CULTURE_PROMPT_IMAGE_PREFIX = {
    "hi": "\nMeme (India): ",
    "zh": "\nMeme (China): ",
}

# Regular prompts for non-culture specific languages
REGULAR_PROMPTS = {
    "en": [
        "\n\nDoes this meme contain hate speech or not? Answer with a single letter: a or b.",
        "\n\nGiven this meme, is it hate speech? Answer with only one letter: a or b, and nothing else.",
        "\n\nWould you consider this meme hate speech? Respond with a single letter only."
    ],
    "de": [
        "\n\nDoes this meme contain hate speech or not? Answer with a single letter: a or b.",
        "\n\nGiven this meme, is it hate speech? Answer with only one letter: a or b, and nothing else.",
        "\n\nWould you consider this meme hate speech? Respond with a single letter only."
    ],
    "es": [
        "\n\nDoes this meme contain hate speech or not? Answer with a single letter: a or b.",
        "\n\nGiven this meme, is it hate speech? Answer with only one letter: a or b, and nothing else.",
        "\n\nWould you consider this meme hate speech? Respond with a single letter only."
    ],
    "hi": [
        "\n\nDoes this meme contain hate speech or not? Answer with a single letter: a or b.",
        "\n\nGiven this meme, is it hate speech? Answer with only one letter: a or b, and nothing else.",
        "\n\nWould you consider this meme hate speech? Respond with a single letter only."
    ],
    "zh": [
        "\n\nDoes this meme contain hate speech or not? Answer with a single letter: a or b.",
        "\n\nGiven this meme, is it hate speech? Answer with only one letter: a or b, and nothing else.",
        "\n\nWould you consider this meme hate speech? Respond with a single letter only."
    ]
}

REGULAR_PROMPT_CAPTION = {
    "en": "\nMeme caption: '{}'",
    "de": "\nMeme caption: '{}'",
    "es": "\nMeme caption: '{}'",
    "hi": "\nMeme caption: '{}'",
    "zh": "\nMeme caption: '{}'",
}

REGULAR_PROMPT_PREFIX = {
    "en": "Definition of Hate Speech:\n\nAny communication that attacks or discriminates against a person or group based on protected characteristics such as race, religion, gender, sexual orientation, or nationality.",
    "de": "Definition of Hate Speech:\n\nAny communication that attacks or discriminates against a person or group based on protected characteristics such as race, religion, gender, sexual orientation, or nationality.",
    "es": "Definition of Hate Speech:\n\nAny communication that attacks or discriminates against a person or group based on protected characteristics such as race, religion, gender, sexual orientation, or nationality.",
    "hi": "Definition of Hate Speech:\n\nAny communication that attacks or discriminates against a person or group based on protected characteristics such as race, religion, gender, sexual orientation, or nationality.",
    "zh": "Definition of Hate Speech:\n\nAny communication that attacks or discriminates against a person or group based on protected characteristics such as race, religion, gender, sexual orientation, or nationality.",
}

REGULAR_PROMPT_POSTFIX = {
    "en": ["\nChoose:\na: Hate\nb: Non-hate\n", "\nChoose:\na: Non-hate\nb: Hate\n"],
    "de": ["\nChoose:\na: Hate\nb: Non-hate\n", "\nChoose:\na: Non-hate\nb: Hate\n"],
    "es": ["\nChoose:\na: Hate\nb: Non-hate\n", "\nChoose:\na: Non-hate\nb: Hate\n"],
    "hi": ["\nChoose:\na: Hate\nb: Non-hate\n", "\nChoose:\na: Non-hate\nb: Hate\n"],
    "zh": ["\nChoose:\na: Hate\nb: Non-hate\n", "\nChoose:\na: Non-hate\nb: Hate\n"],
}

REGULAR_PROMPT_IMAGE_PREFIX = {
    "en": "\nMeme: ",
    "de": "\nMeme: ",
    "es": "\nMeme: ",
    "hi": "\nMeme: ",
    "zh": "\nMeme: ",
}


def set_culture_prompts(language):
    return (
        CULTURE_PROMPTS[language],
        CULTURE_PROMPT_CAPTION[language],
        CULTURE_PROMPT_PREFIX[language],
        CULTURE_PROMPT_POSTFIX[language],
        CULTURE_PROMPT_IMAGE_PREFIX[language],
    )


def set_prompts(language):
    return (
        REGULAR_PROMPTS[language],
        REGULAR_PROMPT_CAPTION[language],
        REGULAR_PROMPT_PREFIX[language],
        REGULAR_PROMPT_POSTFIX[language],
        REGULAR_PROMPT_IMAGE_PREFIX[language],
    )