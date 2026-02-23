"""Central monotonic clock for deterministic timing."""

import time


class TimeSource:
    """Monotonic clock providing single source of truth for time.
    
    Uses time.monotonic() to avoid wall-clock issues (DST, NTP, etc).
    Tracks both elapsed time and tick count for flexibility.
    """
    
    def __init__(self):
        self._start = time.monotonic()
        self._current_tick = 0
    
    def now(self) -> float:
        """Current time in seconds since time_source creation.
        
        Returns:
            Elapsed seconds (monotonic)
        """
        return time.monotonic() - self._start
    
    def tick(self) -> int:
        """Increment and return current tick count.
        
        Returns:
            Current tick number
        """
        self._current_tick += 1
        return self._current_tick
    
    def elapsed_since(self, timestamp: float) -> float:
        """Calculate elapsed time since a previous timestamp.
        
        Args:
            timestamp: Previous timestamp from now()
            
        Returns:
            Seconds elapsed
        """
        return self.now() - timestamp
    
    def ticks_to_seconds(self, ticks: int, tick_rate_hz: float) -> float:
        """Convert tick count to seconds.
        
        Args:
            ticks: Number of ticks
            tick_rate_hz: Tick rate in Hz
            
        Returns:
            Equivalent duration in seconds
        """
        return ticks / tick_rate_hz
