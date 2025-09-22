
import re
import string
import unicodedata
from openai import OpenAI
from pyserini.eval.evaluate_dpr_retrieval import has_answers, SimpleTokenizer
from collections import Counter
_tokenizer = SimpleTokenizer()

client = OpenAI(
  base_url="",
  api_key="",
)

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))



def regex_match(text, pattern):
    """Test if a regex pattern is contained within a text."""
    try:
        pattern = re.compile(
            pattern,
            flags=re.IGNORECASE + re.UNICODE + re.MULTILINE,
        )
    except BaseException:
        return False
    return pattern.search(text) is not None

#str-acc
def answer_span_check(prediction, golden_answers):
    # If golden_answers is a string, convert it to a list for unified processing
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]

    # Normalize the predicted answer, e.g., lowercase, remove punctuation
    normalized_prediction = normalize_answer(prediction)

    score = 0  # Default score is 0 (incorrect)

    # Normalize each golden answer
    normalized_golden_answers = [normalize_answer(golden_answer) for golden_answer in golden_answers]

    # Use has_answers to check whether the predicted answer contains any one of the gold answers
    # _tokenizer might be a tokenizer
    if has_answers(normalized_prediction, normalized_golden_answers, _tokenizer, regex=False):
        score = 1  # If it contains any, assign 1 point
    return score

def exact_match(pred: str, gold: str) :
    return normalize_answer(pred) == normalize_answer(gold)


#llm-acc
def check_if_response_is_correct_llm(response: str, gold_answers: list[str]) -> bool:
    """
    Check if the generated answer is correct by comparing it to the gold answers.
    
    Args:
        response: The response from the LLM
        gold_answers: The gold answers
        
    Returns:
        bool: True if the generated answer is correct, False otherwise
    """
    
    prompt = f"Please check if any of the golden answers is contained in the following response: {response}\n\nGolden answers: {str(gold_answers)}\n\nPlease directly answer with 'yes' or 'no'."
    
    yes_or_no = call_llm(prompt )
    
    if "yes" in yes_or_no.lower():
        return True
    elif "no" in yes_or_no.lower():
        return False
    else:
        yes_or_no = call_llm(prompt)
        if "yes" in yes_or_no.lower():
            return True
        elif "no" in yes_or_no.lower():
            return False
        else:
            return False



def call_llm(prompt: str, max_new_tokens: int = 20) -> str:
    messages = [{"role": "user", "content": prompt}]
    completion = client.chat.completions.create(
                    extra_headers={
                        "HTTP-Referer": "<YOUR_SITE_URL>",
                        "X-Title": "<YOUR_SITE_NAME>",
                    },
                    model="gpt-4o-mini-2024-07-18",
                    messages=messages,
                )
    response = completion.choices[0].message.content
    print(f"Generated answer: {response}")
    return response


def compute_f1(gold: str, predicted: str) -> float:
    gold_tokens = normalize_answer(gold).split()
    predicted_tokens = normalize_answer(predicted).split()
    common = Counter(predicted_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = 1.0 * num_same / len(predicted_tokens)
    recall = 1.0 * num_same / len(gold_tokens)
    
    return 2 * (precision * recall) / (precision + recall)