"""Prespecified discarded windows for fixed-kernel settling assessment.

Two consecutive nonoverlapping windows and their pooled window must satisfy
numerical and rank R-hat checks before production starts. Production is
assessed separately using all retained draws and the specified ESS criteria."""
from dataclasses import dataclass


@dataclass(frozen=True)
class SettlingPolicy:
    burnin: int = 4000
    window: int = 4000
    confirmations: int = 2
    maximum: int = 20000

    def __post_init__(self):
        if (self.burnin < 0 or self.window < 500 or self.confirmations < 2
                or self.maximum < self.burnin + self.window*self.confirmations
                or (self.maximum-self.burnin) % self.window):
            raise ValueError('Invalid prespecified settling policy')

    def is_assessment(self, completed):
        return (self.burnin < completed <= self.maximum
                and (completed-self.burnin) % self.window == 0)

    def bounds(self, completed):
        if not self.is_assessment(completed):
            raise ValueError('Not a scheduled assessment endpoint')
        return completed-self.window, completed

    def candidate_ready(self, previous, current):
        if not current['passed']:
            return False
        tail = list(previous[-(self.confirmations-1):])
        if len(tail) != self.confirmations-1:
            return False
        if not all(row.get('window_passed', False) for row in tail):
            return False
        endpoints = [row['completed'] for row in tail] + [current['completed']]
        return all(b-a == self.window for a,b in zip(endpoints[:-1],endpoints[1:]))

    def metadata(self):
        return dict(
            schema='hibo-case5-prespecified-settling-v1',
            frozen_burnin_discarded_per_chain=self.burnin,
            nonoverlapping_assessment_draws_per_chain=self.window,
            consecutive_windows_required=self.confirmations,
            pooled_confirmation_rhat_also_required=True,
            maximum_discarded_frozen_draws_per_chain=self.maximum,
            assessment_endpoints=list(range(self.burnin+self.window,
                                            self.maximum+1,self.window)),
            assessment_ess_descriptive_only=True,
            production_draws_never_trimmed=True,
            passing_is_not_proof_of_equilibration=True)
