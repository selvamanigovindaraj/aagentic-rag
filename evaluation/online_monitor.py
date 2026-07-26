from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class QualityWindow:
    grounded_claims: int
    generated_claims: int
    valid_citation_references: int
    citation_references: int

    @property
    def grounded_claim_rate(self) -> float:
        return self.grounded_claims / self.generated_claims if self.generated_claims else 0.0

    @property
    def citation_validity(self) -> float:
        if not self.citation_references:
            return 0.0
        return self.valid_citation_references / self.citation_references
