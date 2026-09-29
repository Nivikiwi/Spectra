"""
Round-Robin baseline.

Visits the 238 candidate RF bands sequentially.
No learning is performed.
"""

from receiver_ml_config import N_BANDS


class RoundRobinScheduler:

    def __init__(self):

        self.next_band = 0

    def select(self, belief=None):

        band = self.next_band

        self.next_band = (
            self.next_band + 1
        ) % N_BANDS

        return band

    def update(
        self,
        band,
        hit,
    ):
        # Round Robin does not learn.
        pass