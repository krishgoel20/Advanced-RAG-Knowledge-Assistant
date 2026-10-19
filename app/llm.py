import re

from groq import Groq

from app.config import GROQ_API_KEY, GROQ_MODEL

_client = None

def get_client():
    global _client
    if _client is None:
        _client = Groq(api_key=GROQ_API_KEY)
    return _client

def build_prompt(query, chunks):
    context = "\n\n".join(
        f"[{i+1}] (Source: {c['source']}, Page: {c['page']})\n{c['text']}"
        for i, c in enumerate(chunks)
    )
    return f"""You are a helpful assistant answering questions based on the provided document excerpts.
Use ONLY the information in the numbered excerpts below to answer. If the excerpts don't contain
the answer, say so clearly — do not make up information.

Every factual claim in your answer MUST end with a citation marker like [1] or [2] referencing
the excerpt number it came from. If a claim draws on multiple excerpts, cite all of them, e.g. [1][3].
Do not cite an excerpt number that doesn't exist in the list below.
Cite only the excerpts that directly support a claim, not every excerpt by default.
Write citations with plain square brackets only, like [1], never fullwidth or other bracket styles.

Excerpts:
{context}

Question: {query}

Answer (with inline [n] citations after every factual claim):"""

def generate_answer(query, chunks):
    client = get_client()
    prompt = build_prompt(query, chunks)

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    return normalize_answer(response.choices[0].message.content)

def normalize_answer(text):
    """Unify citation brackets and tidy odd unicode characters in model output."""
    # fullwidth brackets -> [n]
    text = re.sub(r"\u3010\s*(\d+)\s*\u3011", r"[\1]", text)

    # [1, 2] or [1,2,3] -> [1][2][3]
    def split_group(m):
        return "".join(f"[{n}]" for n in re.findall(r"\d+", m.group(0)))
    text = re.sub(r"\[\s*\d+(?:\s*,\s*\d+)+\s*\]", split_group, text)

    # narrow / no-break spaces and non-breaking hyphens -> plain ASCII
    text = text.replace("\u202f", " ").replace("\u00a0", " ").replace("\u2011", "-")
    return text

def validate_citations(answer, num_chunks):
    """Returns a list of warning strings (empty if citations look clean)."""
    warnings = []
    cited_numbers = set(int(n) for n in re.findall(r"\[(\d+)\]", answer))

    invalid = [n for n in cited_numbers if n < 1 or n > num_chunks]
    if invalid:
        warnings.append(f"Answer cites non-existent source(s): {invalid}")

    if not cited_numbers:
        warnings.append("Answer contains no citations at all")

    return warnings

def extract_cited_ids(answer):
    """Returns the sorted set of [n] citation numbers found in the answer."""
    return sorted(set(int(n) for n in re.findall(r"\[(\d+)\]", answer)))

if __name__ == "__main__":
    from app.retrieval import retrieve

    query = input("Enter a test query: ")
    chunks = retrieve(query)
    answer = generate_answer(query, chunks)
    print("\n--- Answer ---")
    print(answer)