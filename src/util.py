import numpy as np

class SolverUtils:
    """
    Utility class containing universal solver helper methods.
    """

    @classmethod
    def unit_vector_normal_to(cls, v: np.ndarray) -> np.ndarray:
        """
        Returns a unit vector normal to a given 2D vector.
        
        :param v: A 2-dimensional vector.
        :return: A 2-dimensional normalized normal vector.
        """
        v = np.asarray(v, dtype=float)
        if v.shape != (2,):
            raise ValueError("Input vector must be 2-dimensional.")
        
        normal = np.array([-v[1], v[0]])
        norm = np.linalg.norm(normal)
        if norm == 0:
            return normal
        return normal / norm
