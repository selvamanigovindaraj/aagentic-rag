import re


def valid_citation_numbers(answer: str, source_count: int) -> bool:
    citations = [int(value) for value in re.findall(r"\[(\d+)]", answer)]
    return all(1 <= value <= source_count for value in citations)
