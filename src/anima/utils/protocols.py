"""Core protocols for anima extensibility."""

from typing import Protocol, List

# Type alias for emotional state vectors
Vector = List[float]


class Extractor(Protocol):
    """Protocol for extracting emotional state from text.
    
    Extractors create vectors from text input.
    """
    
    def extract(self, text: str) -> Vector:
        """Extract emotional state vector from text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Emotional state vector (typically 5D: V, A, D, C, C)
        """
        ...


class Transform(Protocol):
    """Protocol for transforming emotional state vectors.
    
    Transforms operate on vectors to modify emotional expression.
    """
    
    def transform(self, **vectors: Vector) -> Vector:
        """Transform one or more input vectors into output vector.
        
        Args:
            **vectors: Named input vectors
            
        Returns:
            Transformed emotional state vector
        """
        ...
