"""
3D Print Farm Management System

This module implements a scheduling system for managing multiple 3D printers and print jobs.
It uses the Strategy pattern for scheduling algorithms and composition for organizing
printers and jobs within a scheduler.

Architecture:
- Domain objects: FilamentSpool, PrintJob, Printer classes
- Strategy pattern: ISchedulingStrategy and implementations
- Main orchestrator: Scheduler class that coordinates everything
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
import itertools


# ---------- Domain Objects ----------------------------------------------------

class ValidationError(Exception):
    """Custom exception raised when domain validation rules are violated."""
    pass


class FilamentSpool:
    """
    Represents a spool of 3D printing filament with material type, color, and quantity.
    
    Tracks both total capacity and remaining amount, allowing consumption during printing.
    Implements resource management with validation to prevent over-consumption.
    """
    def __init__(self, material: str, color: str, total_g: int):
        """
        Initialize a new filament spool.
        
        Args:
            material: Type of filament (e.g., 'PLA', 'ABS', 'PETG')
            color: Color description of the filament
            total_g: Total weight capacity in grams (must be positive)
            
        Raises:
            ValidationError: If total_g is not positive
        """
        if total_g <= 0:
            raise ValidationError("total_g doit être > 0")
        self._material = material.upper()  # Normalize material to uppercase for consistency
        self._color = color
        self._total_g = total_g
        self._remaining_g = total_g  # Initially full

    @property
    def material(self) -> str:
        """Get the filament material type (normalized to uppercase)."""
        return self._material

    @property
    def color(self) -> str:
        """Get the filament color."""
        return self._color

    @property
    def remaining_g(self) -> int:
        """Get remaining filament quantity in grams (read-only to prevent direct modification)."""
        return self._remaining_g

    def consume(self, grams: int) -> None:
        """
        Consume a specified amount of filament from the spool.
        
        Args:
            grams: Amount to consume in grams (must be non-negative)
            
        Raises:
            ValidationError: If grams is negative or exceeds remaining amount
        """
        if grams < 0:
            raise ValidationError("grams doit être >= 0")
        if grams > self._remaining_g:
            raise ValidationError(f"Pas assez de filament ({self._remaining_g}g restants).")
        self._remaining_g -= grams

    def __repr__(self) -> str:
        """String representation showing material, color, and remaining/total capacity."""
        return f"FilamentSpool({self._material}, {self._color}, {self._remaining_g}/{self._total_g}g)"

    def __del__(self):
        """Destructor with trace for educational purposes (garbage collection monitoring)."""
        print(f"[GC] Destruction de la bobine: {self!r}")


class IPrinter(ABC):
    """
    Abstract interface defining the contract for 3D printers.
    
    Implements the Strategy pattern - different printer types can have different
    implementations while maintaining a common interface for the scheduler.
    """
    @property
    @abstractmethod
    def name(self) -> str: 
        """Get the printer's unique name/identifier."""
        ...

    @abstractmethod
    def can_print(self, job: "PrintJob") -> bool: 
        """
        Check if this printer can execute the given job.
        
        Considers filament compatibility and availability.
        """
        ...

    @abstractmethod
    def start_job(self, job: "PrintJob") -> None: 
        """
        Begin executing a print job.
        
        Should consume required filament and set printer to busy state.
        """
        ...

    @abstractmethod
    def is_busy(self) -> bool: 
        """Check if printer is currently executing a job."""
        ...

    @abstractmethod
    def tick(self, delta_min: int) -> None: 
        """
        Advance printer's internal time by specified minutes.
        
        Used for simulation - updates job progress and completion.
        """
        ...
    
    @abstractmethod
    def status(self) -> str: 
        """Get human-readable status description."""
        ...

    @abstractmethod
    def load_filament(self, spool: FilamentSpool) -> None: 
        """Load a filament spool into the printer."""
        ...


class BasePrinter(IPrinter):
    """
    Base implementation providing common functionality for all printer types.
    
    Implements the Template Method pattern - defines the algorithm structure
    while allowing subclasses to customize specific aspects.
    """
    def __init__(self, name: str, nozzle_mm: float):
        """
        Initialize base printer functionality.
        
        Args:
            name: Unique identifier for this printer
            nozzle_mm: Nozzle diameter in millimeters (affects print capabilities)
        """
        self._name = name
        self.nozzle_mm = nozzle_mm
        self._current_spool: Optional[FilamentSpool] = None  # Currently loaded filament
        self._current_job: Optional[PrintJob] = None         # Job being executed
        self._mins_left: int = 0                             # Time remaining for current job
        self._printed_jobs: List[PrintJob] = []              # History of completed jobs

    @property
    def name(self) -> str:
        """Get printer name."""
        return self._name

    def load_filament(self, spool: FilamentSpool) -> None:
        """Load a filament spool into the printer (simple assignment for this simulation)."""
        self._current_spool = spool

    def can_print(self, job: "PrintJob") -> bool:
        """
        Determine if printer can execute the job based on filament compatibility.
        
        Checks:
        1. Filament is loaded
        2. Material types match (case-insensitive)
        3. Sufficient filament quantity available
        """
        if self._current_spool is None:
            return False
        return (
            self._current_spool.material == job.material.upper() and
            self._current_spool.remaining_g >= job.required_g
        )

    def start_job(self, job: "PrintJob") -> None:
        """
        Begin executing a print job.
        
        Validates printer availability and job compatibility before starting.
        Consumes required filament immediately (simplified model).
        
        Raises:
            RuntimeError: If printer is already busy
            ValidationError: If job cannot be printed (incompatible filament)
        """
        if self.is_busy():
            raise RuntimeError(f"{self._name} est occupée.")
        if not self.can_print(job):
            raise ValidationError(f"{self._name} ne peut pas imprimer le job {job.job_id} (filament ou matière incompatible).")
        
        # Consume filament at assignment time (simplification - real printers consume gradually)
        assert self._current_spool
        self._current_spool.consume(job.required_g)
        self._current_job = job
        self._mins_left = job.estimated_minutes

    def is_busy(self) -> bool:
        """Check if printer is currently executing a job."""
        return self._current_job is not None

    def tick(self, delta_min: int) -> None:
        """
        Simulate passage of time on the printer.
        
        Decrements remaining time for current job and handles completion.
        
        Args:
            delta_min: Minutes to advance the simulation
        """
        if not self._current_job:
            return
        
        self._mins_left = max(0, self._mins_left - delta_min)
        
        # Check if job completed
        if self._mins_left == 0:
            self._printed_jobs.append(self._current_job)  # Add to history
            self._current_job = None                      # Mark as idle

    def status(self) -> str:
        """Get human-readable status with current job info or idle state."""
        if self._current_job:
            return f"{self._name}: imprime {self._current_job.job_id} ({self._mins_left} min restantes)"
        return f"{self._name}: idle"

    def __del__(self):
        """Destructor with trace for educational purposes."""
        print(f"[GC] Destruction de l'imprimante {self._name}")


class BambuPrinter(BasePrinter):
    """
    Bambu brand 3D printer implementation.
    
    Currently identical to base implementation but allows for future
    brand-specific customizations (different nozzle sizes, capabilities, etc.).
    """
    def __init__(self, name: str):
        super().__init__(name, nozzle_mm=0.4)  # Standard 0.4mm nozzle


class PrusaPrinter(BasePrinter):
    """
    Prusa brand 3D printer implementation.
    
    Currently identical to base implementation but structured for future
    brand-specific features and behaviors.
    """
    def __init__(self, name: str):
        super().__init__(name, nozzle_mm=0.4)  # Standard 0.4mm nozzle


@dataclass(frozen=True)
class PrintJob:
    """
    Immutable representation of a 3D print job.
    
    Uses dataclass for automatic __init__, __eq__, __hash__ generation.
    Frozen=True ensures immutability for safe use in collections and scheduling.
    """
    job_id: str              # Unique identifier for the job
    material: str           # Required filament material (e.g., 'PLA', 'ABS')
    required_g: int         # Amount of filament needed in grams
    estimated_minutes: int  # Expected print duration
    _priority: int = 3      # Priority level: 1=highest, 5=lowest, 3=normal

    def __post_init__(self):
        """
        Validate field values after dataclass initialization.
        
        Called automatically by dataclass after __init__.
        
        Raises:
            ValidationError: If any validation rule is violated
        """
        if self.required_g <= 0:
            raise ValidationError("required_g doit être > 0")
        if self.estimated_minutes <= 0:
            raise ValidationError("estimated_minutes doit être > 0")
        if not (1 <= self._priority <= 5):
            raise ValidationError("priority doit être entre 1 et 5")

    @property
    def priority(self) -> int:
        """
        Get job priority (1=highest, 5=lowest).
        
        Implemented as property to maintain encapsulation despite dataclass.
        """
        return self._priority


# ---------- Scheduling --------------------------------------------------------

class ISchedulingStrategy(ABC):
    """
    Strategy pattern interface for job scheduling algorithms.
    
    Allows different scheduling policies to be plugged into the scheduler
    without changing the core scheduling logic.
    """
    @abstractmethod
    def pick_next(self, queue: List[PrintJob]) -> PrintJob: 
        """
        Select the next job to execute from the available queue.
        
        Args:
            queue: List of jobs that can be executed (already filtered for compatibility)
            
        Returns:
            The selected job to execute next
        """
        ...


class ShortestJobFirst(ISchedulingStrategy):
    """
    Scheduling strategy that prioritizes jobs with shortest estimated duration.
    
    Minimizes average wait time but may cause starvation of longer jobs.
    Good for maximizing throughput when job sizes vary significantly.
    """
    def pick_next(self, queue: List[PrintJob]) -> PrintJob:
        """Select job with minimum estimated duration."""
        return min(queue, key=lambda j: j.estimated_minutes)


class PriorityFirst(ISchedulingStrategy):
    """
    Scheduling strategy that respects job priorities with duration as tiebreaker.
    
    Ensures high-priority jobs are processed first while using shortest-job-first
    as secondary criteria for jobs of equal priority.
    """
    def pick_next(self, queue: List[PrintJob]) -> PrintJob:
        """
        Select job with highest priority (lowest number), breaking ties by duration.
        
        Uses tuple sorting: (priority, duration) - both ascending order.
        """
        return min(queue, key=lambda j: (j.priority, j.estimated_minutes))


class Scheduler:
    """
    Main orchestrator that manages printers and job queue using composition.
    
    Coordinates the assignment of jobs to available printers based on:
    - Printer availability (idle/busy state)  
    - Filament compatibility between printer and job
    - Scheduling strategy for job selection
    
    Implements a discrete event simulation for time advancement.
    """
    def __init__(self, strategy: ISchedulingStrategy):
        """
        Initialize scheduler with a specific scheduling strategy.
        
        Args:
            strategy: Algorithm for selecting which job to assign next
        """
        self._printers: List[IPrinter] = []    # Managed printers (composition)
        self._queue: List[PrintJob] = []       # Jobs waiting to be assigned
        self._time_elapsed: int = 0            # Total simulation time in minutes
        self._events: List[str] = []           # Event log for analysis/debugging
        self._strategy = strategy              # Scheduling algorithm (strategy pattern)

    def add_printer(self, printer: IPrinter) -> None:
        """Add a printer to the managed fleet."""
        self._printers.append(printer)

    def submit(self, job: PrintJob) -> None:
        """Submit a new job to the scheduling queue."""
        self._queue.append(job)

    def is_done(self) -> bool:
        """
        Check if all work is complete.
        
        Returns True when both:
        1. No jobs remain in queue
        2. All printers are idle (no active jobs)
        """
        return (not self._queue) and all(not p.is_busy() for p in self._printers)

    def _assign(self) -> None:
        """
        Attempt to assign jobs to available printers.
        
        Core assignment algorithm:
        1. Find all idle printers
        2. For each idle printer, find compatible jobs
        3. Use strategy to select best job from compatible options
        4. Assign job and remove from queue
        
        This method implements the main scheduling logic.
        """
        idle_printers = [p for p in self._printers if not p.is_busy()]
        
        # Try to assign a job to each idle printer
        for p in idle_printers:
            # Filter jobs that this printer can actually execute
            candidates = [j for j in self._queue if p.can_print(j)]
            if not candidates:
                continue  # No compatible jobs for this printer
            
            # Let strategy choose among compatible jobs only
            pick = self._strategy.pick_next(candidates)
            p.start_job(pick)
            self._queue.remove(pick)
            
            # Log the assignment event
            self._events.append(f"T+{self._time_elapsed:04d} min: {p.name} démarre {pick.job_id}")

    def _advance_time(self) -> int:
        """
        Advance simulation time to next significant event.
        
        Uses discrete event simulation: advance to the next moment when
        at least one printer will finish its current job.
        
        Returns:
            Number of minutes advanced
        """
        busy = [p for p in self._printers if p.is_busy()]
        
        if not busy:
            # No active jobs - advance by minimum unit
            advance = 1
        else:
            # Find minimum remaining time across all busy printers
            remaining = []
            for p in busy:
                # Parse status string to extract remaining time (simplified approach)
                s = p.status()
                # Status format: 'Name: imprime J1 (12 min restantes)'
                idx = s.find("(")
                if idx != -1 and "min" in s:
                    try:
                        part = s[idx:].split(" ")[0].lstrip("(")
                        rem = int(part)
                        remaining.append(rem)
                    except Exception:
                        remaining.append(1)  # Fallback if parsing fails
                else:
                    remaining.append(1)
            
            # Advance to next completion event
            advance = max(1, min(remaining))
        
        # Apply time advancement to all printers
        for p in self._printers:
            p.tick(advance)
        
        self._time_elapsed += advance
        return advance

    def run(self) -> Tuple[int, List[str]]:
        """
        Execute the main scheduling loop until all jobs complete.
        
        Algorithm:
        1. Try initial job assignments
        2. While work remains:
           a. Advance time to next event
           b. Try new assignments (freed printers may now accept jobs)
        3. Return total time and event log
        
        Returns:
            Tuple of (total_minutes_elapsed, event_log)
        """
        # Initial assignment attempt
        self._assign()
        
        # Main simulation loop
        while not self.is_done():
            advanced = self._advance_time()
            # After time advancement, some printers may have finished jobs
            # so attempt assignments again
            self._assign()
        
        # Log completion
        self._events.append(f"T+{self._time_elapsed:04d} min: tous les jobs terminés")
        return self._time_elapsed, self._events

    # Properties for external inspection and monitoring
    @property
    def queue(self) -> List[PrintJob]:
        """Get copy of current job queue (defensive copy to prevent external modification)."""
        return list(self._queue)

    @property
    def time_elapsed(self) -> int:
        """Get total simulation time elapsed."""
        return self._time_elapsed

"""
Entry point would be here for demonstration/testing.
Currently commented out to allow module import without execution.

if __name__ == "__main__":
    main()
"""