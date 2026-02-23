"""
Cinematic Amplifier - Radial Emotional Amplification + Cinematic Snap

Pipeline:

1) Amplify emotional magnitude around neutral (0.5) using nonlinear radial gain
2) Optionally snap toward nearest cinematic exemplar

Two knobs:

passion (0.0–3.0+) → emotional magnitude amplification
drama   (0.0–1.0)  → distance to cinematic anchor
"""

import numpy as np
from datasets import load_dataset
from typing import List, Tuple


class CinematicAmplifier:
    """
    Amplify emotional magnitude first, then optionally snap toward cinema.

    passion:
        0.0 = no emotional amplification
        1.0 = moderate intensity
        2.0 = strong intensity
        3.0+ = extreme (will clip toward 0/1)

    drama:
        0.0 = no cinematic snapping (pure amplified emotion)
        0.5 = halfway to nearest cinematic anchor
        1.0 = full snap to nearest cinematic anchor
    """

    # ---- DEFAULT KNOBS ----
    DEFAULT_DRAMA = 0.65
    MAX_PASSION = 3.5

    def __init__(
        self,
        dataset_name: str = "danielritchie/cinematic-mood-palette",
        verbose: bool = False
    ):

        if verbose:
            print(f"Loading cinematic exemplars from {dataset_name}...")

        dataset = load_dataset(dataset_name)

        self.examples = dataset["train"]

        self.cinematic_space = np.array([ex["input"] for ex in self.examples])
        self.example_names = [ex["name"] for ex in self.examples]
        self.example_sources = [ex["source"] for ex in self.examples]

        # Emotional importance weighting
        self.dimension_weights = np.array([
            3.0,  # Valence
            3.0,  # Arousal
            1.2,  # Dominance
            0.3,  # Complexity
            0.3   # Coherence
        ])

        if verbose:
            print(f"Loaded {len(self.examples)} cinematic states")
            print(f"Dimension weights: {self.dimension_weights}")

    # -------------------------------------------------------------
    # Public API (Stepwise Control)
    # -------------------------------------------------------------
    def amplify_passion(
        self,
        values: List[float],
        passion: float
    ) -> np.ndarray:
        """
        Apply only radial emotional amplification.
        """
        passion = max(0.0, min(self.MAX_PASSION, passion))
        values = np.array(values)
        amplified = self._radial_amplify(values, passion)
        return np.clip(amplified, 0.0, 1.0)

    def snap_drama(
        self,
        values: List[float],
        drama: float,
        k: int = 1
    ) -> np.ndarray:
        """
        Snap toward nearest cinematic exemplar.
        """
        drama = max(0.0, min(1.0, drama))
        values = np.array(values)

        if drama <= 0:
            return values

        target = self._nearest_exemplar(values, k=k)
        snapped = values + drama * (target - values)

        return np.clip(snapped, 0.0, 1.0)

    # -------------------------------------------------------------
    # Combined Convenience Method
    # -------------------------------------------------------------
    def amplify(
        self,
        natural_vad_cc: List[float],
        passion: float = 1.0,
        drama: float = None,
        k: int = 1,
        verbose: bool = False
    ) -> np.ndarray:

        if drama is None:
            drama = self.DEFAULT_DRAMA

        natural = np.array(natural_vad_cc)

        post_passion = self.amplify_passion(natural, passion)
        post_drama = self.snap_drama(post_passion, drama, k=k)

        if verbose:
            print("\n" + "=" * 60)
            print(f"Natural:   {natural}")
            print(f"Passion:   {passion}")
            print(f"After AMP: {post_passion}")
            print(f"Drama:     {drama}")
            print(f"Final:     {post_drama}")
            print("=" * 60)

        return post_drama

    # -------------------------------------------------------------
    # Radial Emotional Amplification (Private)
    # -------------------------------------------------------------
    def _radial_amplify(self, values: np.ndarray, passion: float) -> np.ndarray:
        """
        Amplifies V, A, D around neutral (0.5).

        Neutral stays fixed.
        Further from 0.5 = amplified more aggressively.
        """

        result = values.copy()

        for i in range(3):  # Only V, A, D
            delta = values[i] - 0.5
            magnitude = abs(delta)

            # Nonlinear radial gain
            gain = 1 + passion * magnitude

            result[i] = 0.5 + delta * gain

        return result

    # -------------------------------------------------------------
    # Find Nearest Cinematic Exemplar (Private)
    # -------------------------------------------------------------
    def _nearest_exemplar(self, vad_cc: np.ndarray, k: int = 1) -> np.ndarray:

        weighted_diff = (self.cinematic_space - vad_cc) * self.dimension_weights
        distances = np.linalg.norm(weighted_diff, axis=1)

        nearest_idx = np.argsort(distances)[:k]

        if k == 1:
            return self.cinematic_space[nearest_idx[0]]

        weights = 1.0 / (distances[nearest_idx] + 0.01)
        weights /= weights.sum()

        return np.average(
            self.cinematic_space[nearest_idx],
            axis=0,
            weights=weights
        )

    # -------------------------------------------------------------
    # Debug Helpers
    # -------------------------------------------------------------
    def get_nearest_exemplar(
        self,
        vad_cc: List[float],
        k: int = 1
    ) -> List[Tuple[str, str, float]]:

        weighted_diff = (self.cinematic_space - np.array(vad_cc)) * self.dimension_weights
        distances = np.linalg.norm(weighted_diff, axis=1)
        nearest_idx = np.argsort(distances)[:k]

        return [
            (
                self.example_names[idx],
                self.example_sources[idx],
                float(distances[idx])
            )
            for idx in nearest_idx
        ]

    def get_exemplar_stats(self) -> dict:
        return {
            "mean": self.cinematic_space.mean(axis=0).tolist(),
            "std": self.cinematic_space.std(axis=0).tolist(),
            "min": self.cinematic_space.min(axis=0).tolist(),
            "max": self.cinematic_space.max(axis=0).tolist(),
            "count": len(self.examples),
            "dimension_weights": self.dimension_weights.tolist(),
        }


# ------------------------------------------------------------------
# Quick Manual Test
# ------------------------------------------------------------------
if __name__ == "__main__":

    amplifier = CinematicAmplifier(verbose=True)

    test = [0.42, 0.65, 0.55, 0.6, 0.7]

    print("\nTesting Passion 2.0 (stepwise)")
    post_passion = amplifier.amplify_passion(test, passion=2.0)
    print("After Passion:", post_passion)

    print("\nApplying Drama 0.7")
    post_drama = amplifier.snap_drama(post_passion, drama=0.7)
    print("Final Result:", post_drama)

