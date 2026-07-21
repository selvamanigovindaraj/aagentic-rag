from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    cache_hit_tokens: int = 0


def estimated_cost(usage: TokenUsage, *, pro: bool) -> float:
    input_rate, output_rate = (0.435, 0.87) if pro else (0.14, 0.28)
    billable_input = max(usage.prompt_tokens - usage.cache_hit_tokens, 0)
    return (billable_input * input_rate + usage.completion_tokens * output_rate) / 1_000_000
