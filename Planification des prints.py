from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
import itertools


# ---------- Domain Objects ----------------------------------------------------

class ValidationError(Exception):
    pass


class FilamentSpool:
    """
    Bobine de filament.
    """
    def __init__(self, material: str, color: str, total_g: int):
        if total_g <= 0:
            raise ValidationError("total_g doit être > 0")
        self._material = material.upper()
        self._color = color
        self._total_g = total_g
        self._remaining_g = total_g

    @property
    def material(self) -> str:
        return self._material

    @property
    def color(self) -> str:
        return self._color

    @property
    def remaining_g(self) -> int:
        """Quantité restante en grammes (read-only)."""
        return self._remaining_g

    def consume(self, grams: int) -> None:
        if grams < 0:
            raise ValidationError("grams doit être >= 0")
        if grams > self._remaining_g:
            raise ValidationError(f"Pas assez de filament ({self._remaining_g}g restants).")
        self._remaining_g -= grams

    def __repr__(self) -> str:
        return f"FilamentSpool({self._material}, {self._color}, {self._remaining_g}/{self._total_g}g)"

    def __del__(self):
        # NB: En Python, __del__ n'est pas garanti immédiatement.
        # On l'utilise ici à but pédagogique (trace de nettoyage).
        # Dans la vraie vie, on préférera un contexte ou close()/with.
        print(f"[GC] Destruction de la bobine: {self!r}")


class IPrinter(ABC):
    """Interface d'imprimante 3D (contrat minimal)."""
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def can_print(self, job: "PrintJob") -> bool: ...

    @abstractmethod
    def start_job(self, job: "PrintJob") -> None: ...

    @abstractmethod
    def is_busy(self) -> bool: ...

    @abstractmethod
    def tick(self, delta_min: int) -> None: ...
    
    @abstractmethod
    def status(self) -> str: ...

    @abstractmethod
    def load_filament(self, spool: FilamentSpool) -> None: ...


class BasePrinter(IPrinter):
    """Implémentation de base commune aux imprimantes."""
    def __init__(self, name: str, nozzle_mm: float):
        self._name = name
        self.nozzle_mm = nozzle_mm
        self._current_spool: Optional[FilamentSpool] = None
        self._current_job: Optional[PrintJob] = None
        self._mins_left: int = 0
        self._printed_jobs: List[PrintJob] = []

    @property
    def name(self) -> str:
        return self._name

    def load_filament(self, spool: FilamentSpool) -> None:
        self._current_spool = spool

    def can_print(self, job: "PrintJob") -> bool:
        if self._current_spool is None:
            return False
        return (
            self._current_spool.material == job.material.upper() and
            self._current_spool.remaining_g >= job.required_g
        )

    def start_job(self, job: "PrintJob") -> None:
        if self.is_busy():
            raise RuntimeError(f"{self._name} est occupée.")
        if not self.can_print(job):
            raise ValidationError(f"{self._name} ne peut pas imprimer le job {job.job_id} (filament ou matière incompatible).")
        # Consommation filament à l'assignation (simplification)
        assert self._current_spool
        self._current_spool.consume(job.required_g)
        self._current_job = job
        self._mins_left = job.estimated_minutes

    def is_busy(self) -> bool:
        return self._current_job is not None

    def tick(self, delta_min: int) -> None:
        """Simule l'écoulement du temps sur l'imprimante."""
        if not self._current_job:
            return
        self._mins_left = max(0, self._mins_left - delta_min)
        if self._mins_left == 0:
            self._printed_jobs.append(self._current_job)
            self._current_job = None

    def status(self) -> str:
        if self._current_job:
            return f"{self._name}: imprime {self._current_job.job_id} ({self._mins_left} min restantes)"
        return f"{self._name}: idle"

    def __del__(self):
        # Trace pédagogique
        print(f"[GC] Destruction de l'imprimante {self._name}")


class BambuPrinter(BasePrinter):
    def __init__(self, name: str):
        super().__init__(name, nozzle_mm=0.4)


class PrusaPrinter(BasePrinter):
    def __init__(self, name: str):
        super().__init__(name, nozzle_mm=0.4)


@dataclass(frozen=True)
class PrintJob:
    job_id: str
    material: str
    required_g: int
    estimated_minutes: int
    _priority: int = 3  # 1=haut, 5=bas

    def __post_init__(self):
        if self.required_g <= 0:
            raise ValidationError("required_g doit être > 0")
        if self.estimated_minutes <= 0:
            raise ValidationError("estimated_minutes doit être > 0")
        if not (1 <= self._priority <= 5):
            raise ValidationError("priority doit être entre 1 et 5")

    # Propriété avec validation (via un proxy car dataclass frozen)
    @property
    def priority(self) -> int:
        return self._priority


# ---------- Scheduling --------------------------------------------------------

class ISchedulingStrategy(ABC):
    """Interface stratégie d'ordonnancement."""
    @abstractmethod
    def pick_next(self, queue: List[PrintJob]) -> PrintJob: ...


class ShortestJobFirst(ISchedulingStrategy):
    def pick_next(self, queue: List[PrintJob]) -> PrintJob:
        return min(queue, key=lambda j: j.estimated_minutes)


class PriorityFirst(ISchedulingStrategy):
    def pick_next(self, queue: List[PrintJob]) -> PrintJob:
        # priorité 1 = la plus haute -> tri ascendant
        return min(queue, key=lambda j: (j.priority, j.estimated_minutes))


class Scheduler:
    """
    Le Scheduler détient des imprimantes et une file de jobs (composition).
    Il orchestre l'assignation des jobs aux imprimantes disponibles.
    """
    def __init__(self, strategy: ISchedulingStrategy):
        self._printers: List[IPrinter] = []
        self._queue: List[PrintJob] = []
        self._time_elapsed: int = 0  # minutes
        self._events: List[str] = []
        self._strategy = strategy

    def add_printer(self, printer: IPrinter) -> None:
        self._printers.append(printer)

    def submit(self, job: PrintJob) -> None:
        self._queue.append(job)

    def is_done(self) -> bool:
        return (not self._queue) and all(not p.is_busy() for p in self._printers)

    def _assign(self) -> None:
        idle_printers = [p for p in self._printers if not p.is_busy()]
        # Pour chaque imprimante idle, tenter d'assigner un job compatible
        for p in idle_printers:
            # filtrer les jobs que p peut imprimer
            candidates = [j for j in self._queue if p.can_print(j)]
            if not candidates:
                continue
            # choisir un job selon la stratégie parmi les candidats uniquement
            pick = self._strategy.pick_next(candidates)
            p.start_job(pick)
            self._queue.remove(pick)
            self._events.append(f"T+{self._time_elapsed:04d} min: {p.name} démarre {pick.job_id}")

    def _advance_time(self) -> int:
        """Fait avancer le temps jusqu'au prochain évènement (fin d'au moins une imprimante)."""
        busy = [p for p in self._printers if p.is_busy()]
        if not busy:
            # rien à faire -> on avance arbitrairement de 1 minute
            advance = 1
        else:
            # avancer jusqu'au plus petit temps restant
            remaining = []
            for p in busy:
                # introspection de status pour récupérer le temps restant
                s = p.status()
                # ... status comme 'Name: imprime J1 (12 min restantes)'
                idx = s.find("(")
                if idx != -1 and "min" in s:
                    try:
                        part = s[idx:].split(" ")[0].lstrip("(")
                        rem = int(part)
                        remaining.append(rem)
                    except Exception:
                        remaining.append(1)
                else:
                    remaining.append(1)
            advance = max(1, min(remaining))
        for p in self._printers:
            p.tick(advance)
        self._time_elapsed += advance
        return advance

    def run(self) -> Tuple[int, List[str]]:
        """Boucle principale: assigne et avance jusqu'à tout terminer."""
        # Tenter une première assignation
        self._assign()
        while not self.is_done():
            advanced = self._advance_time()
            # après chaque avancée, tenter d'assigner à nouveau
            self._assign()
        self._events.append(f"T+{self._time_elapsed:04d} min: tous les jobs terminés")
        return self._time_elapsed, self._events

    # Pour inspection externe
    @property
    def queue(self) -> List[PrintJob]:
        return list(self._queue)

    @property
    def time_elapsed(self) -> int:
        return self._time_elapsed

"""
if __name__ == "__main__":
    main()
"""