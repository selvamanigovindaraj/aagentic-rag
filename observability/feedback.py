from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FeedbackSignal:
    grounded: bool
    useful: bool


def quality_score(signal: FeedbackSignal) -> float:
    return (float(signal.grounded) + float(signal.useful)) / 2
