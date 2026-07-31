from abc import ABC, abstractmethod
from typing import Callable, List

import numpy as np

from src.types import PointValue


class VortexLumpingStrategy(ABC):
    """
    Parent class/interface for vortex lumping strategies.
    """

    @abstractmethod
    def apply_lumping(
        self,
        trailing_edge: np.ndarray,
        wake: List[PointValue],
    ) -> List[PointValue]:
        """
        Apply a lumping strategy to the current wake. Returns the updated (post-lumping) wake to be used for the next time step solution.

        Args:
            trailing_edge: Trailing-edge position in the same coordinate frame as
                the wake-vortex positions.
            wake: Current wake, ordered from nearest/newest vortex at index 0 to
                farthest/oldest vortex at the end of the list.

        Returns:
            A new list containing the wake after lumping.
        """
        pass


class WCStrategy(VortexLumpingStrategy):
    """
    Weighted-centroid vortex lumping strategy.

    Eligible same-signed vortices are processed from the far end of the wake
    toward the airfoil. Each group is replaced by one vortex whose circulation
    is the sum of the grouped circulations and whose position is their
    circulation-weighted centroid.
    """

    def __init__(
        self,
        min_lumping_distance_from_af: float,
        max_vortices_to_lump: int,
        max_lumping_distance: float,
    ):
        """
        Args:
            min_lumping_distance_from_af: Minimum Euclidean distance from the
                trailing edge at which a vortex may be considered for lumping.
            max_vortices_to_lump: Maximum number of current wake vortices that
                may be combined into one vortex at a time.
            max_lumping_distance: Maximum Euclidean distance permitted between
                any two vortices in the same lump.
        """
        if (
            not np.isfinite(min_lumping_distance_from_af)
            or min_lumping_distance_from_af < 0.0
        ):
            raise ValueError(
                "min_lumping_distance_from_af must be finite and non-negative"
            )
        if isinstance(max_vortices_to_lump, bool) or not isinstance(
            max_vortices_to_lump, (int, np.integer)
        ):
            raise TypeError("max_vortices_to_lump must be an integer")
        if max_vortices_to_lump < 2:
            raise ValueError("max_vortices_to_lump must be at least 2")
        if max_lumping_distance <= 0.0:
            raise ValueError("max_lumping_distance must be positive")

        self.min_lumping_distance_from_af = float(min_lumping_distance_from_af)
        self.max_vortices_to_lump = int(max_vortices_to_lump)
        self.max_lumping_distance = float(max_lumping_distance)

    @staticmethod
    def _weighted_centroid(vortices: List[PointValue]) -> PointValue:
        """Combine a non-empty, same-signed group into one PointValue."""
        circulation_sum = float(sum(vortex.value for vortex in vortices))

        # A same-signed group cannot have a zero sum unless invalid/non-finite
        # values were supplied.
        if not np.isfinite(circulation_sum) or circulation_sum == 0.0:
            raise ValueError(
                "Cannot calculate a weighted centroid for a zero or non-finite "
                "circulation sum"
            )

        weighted_position_sum = np.zeros(2, dtype=float)
        for vortex in vortices:
            if not np.all(np.isfinite(vortex.point)) or not np.isfinite(vortex.value):
                raise ValueError("Wake vortices must contain finite points and values")
            weighted_position_sum += vortex.value * vortex.point

        centroid = weighted_position_sum / circulation_sum
        return PointValue(centroid, circulation_sum)

    def _lump_one_sign(
        self,
        wake: List[PointValue],
        trailing_edge: np.ndarray,
        sign_predicate: Callable[[float], bool],
    ) -> List[PointValue]:
        """Perform one far-to-near pass for either positive or negative gamma."""
        candidate_indices = []

        # The wake is stored nearest-to-farthest, so reverse traversal gives the
        # requested far-to-near lumping priority.
        for index in range(len(wake) - 1, -1, -1):
            vortex = wake[index]
            distance_from_te = float(np.linalg.norm(vortex.point - trailing_edge))

            # early exit to avoid reiterating the entire wake
            if distance_from_te < self.min_lumping_distance_from_af:
                break

            if (
                distance_from_te >= self.min_lumping_distance_from_af
                and sign_predicate(vortex.value)
            ):
                candidate_indices.append(index)

        replacements: dict[int, PointValue] = {}
        removed_indices: set[int] = set()

        # Build each group greedily from the far side of the wake. A group ends
        # when the next candidate is too far away or the maximum group size
        # has been reached.
        group_indices = []
        for candidate_index in [*candidate_indices, None]:
            can_add = (
                candidate_index is not None
                and len(group_indices) < self.max_vortices_to_lump
                and all(
                    np.linalg.norm(
                        wake[candidate_index].point - wake[group_index].point
                    )
                    <= self.max_lumping_distance
                    for group_index in group_indices
                )
            )

            if can_add:
                group_indices.append(candidate_index)
                continue

            if len(group_indices) >= 2:
                grouped_vortices = [wake[index] for index in group_indices]
                lumped_vortex = self._weighted_centroid(grouped_vortices)

                # Insert the replacement at the nearest original index in the group.
                replacement_index = min(group_indices)
                replacements[replacement_index] = lumped_vortex
                removed_indices.update(group_indices)

            group_indices = [] if candidate_index is None else [candidate_index]

        lumped_wake: List[PointValue] = []
        for index, vortex in enumerate(wake):
            if index in replacements:
                lumped_wake.append(replacements[index])
            elif index not in removed_indices:
                lumped_wake.append(vortex)

        return lumped_wake

    def apply_lumping(
        self,
        trailing_edge: np.ndarray,
        wake: List[PointValue],
    ) -> List[PointValue]:
        """
        Apply weighted-centroid lumping in two passes: positive circulation first,
        followed by negative circulation.

        Zero-circulation vortices and vortices closer to the trailing edge than
        ``min_lumping_distance_from_af`` are retained unchanged. Candidate groups
        end when the next vortex exceeds ``max_lumping_distance``.
        """
        trailing_edge = np.asarray(trailing_edge, dtype=float)
        if trailing_edge.shape != (2,):
            raise ValueError("trailing_edge must be a 2D point with shape (2,)")
        if not np.all(np.isfinite(trailing_edge)):
            raise ValueError("trailing_edge must contain finite coordinates")

        wake_after_positive_pass = self._lump_one_sign(
            list(wake),
            trailing_edge,
            lambda gamma: gamma > 0.0,
        )
        wake_after_negative_pass = self._lump_one_sign(
            wake_after_positive_pass,
            trailing_edge,
            lambda gamma: gamma < 0.0,
        )

        return wake_after_negative_pass
