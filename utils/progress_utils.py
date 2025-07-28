"""Progress tracking utilities for CLI operations"""

import time
import threading
from typing import Dict, Any, List, Callable
from datetime import timedelta
from dataclasses import dataclass
from contextlib import contextmanager

from utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class ProgressStats:
    """Progress statistics for operations"""
    current: int = 0
    total: int = 0
    start_time: float = 0.0
    elapsed_time: float = 0.0
    estimated_remaining: float = 0.0
    items_per_second: float = 0.0
    current_operation: str = ""
    phase: str = ""
    
    @property
    def percentage(self) -> float:
        """Calculate completion percentage"""
        return (self.current / self.total * 100) if self.total > 0 else 0.0
    
    @property
    def eta_formatted(self) -> str:
        """Format estimated time remaining"""
        if self.estimated_remaining <= 0:
            return "Unknown"
        
        eta = timedelta(seconds=int(self.estimated_remaining))
        return str(eta)


class ProgressIndicator:
    """Enhanced progress indicator with detailed metrics"""
    
    def __init__(self, total: int, description: str = "", show_rate: bool = True,
                 show_eta: bool = True, bar_length: int = 40):
        """Initialize progress indicator"""
        self.total = total
        self.description = description
        self.show_rate = show_rate
        self.show_eta = show_eta
        self.bar_length = bar_length
        
        self.stats = ProgressStats(total=total, start_time=time.time())
        self.last_update = 0.0
        self.update_interval = 0.1  # Update every 100ms
        
        # Thread safety
        self._lock = threading.Lock()
        
        logger.debug(f"Progress indicator initialized: {description} (total: {total})")
    
    def update(self, current: int, operation: str = "", phase: str = "") -> None:
        """Update progress with current count"""
        
        current_time = time.time()
        
        # Rate limiting updates
        if current_time - self.last_update < self.update_interval and current < self.total:
            return
        
        with self._lock:
            self.stats.current = current
            self.stats.current_operation = operation
            self.stats.phase = phase
            self.stats.elapsed_time = current_time - self.stats.start_time
            
            # Calculate rate and ETA
            if self.stats.elapsed_time > 0:
                self.stats.items_per_second = current / self.stats.elapsed_time
                
                if self.stats.items_per_second > 0:
                    remaining_items = self.total - current
                    self.stats.estimated_remaining = remaining_items / self.stats.items_per_second
            
            self.last_update = current_time
            
            # Display progress
            self._display_progress()
    
    def _display_progress(self) -> None:
        """Display progress bar and statistics"""
        
        # Create progress bar
        filled_length = int(self.bar_length * self.stats.current // self.total) if self.total > 0 else 0
        bar = '█' * filled_length + '░' * (self.bar_length - filled_length)
        
        # Build progress line
        progress_parts = [
            f"\r🔄 {self.description}" if self.description else "\r🔄 Progress",
            f"|{bar}|",
            f"{self.stats.percentage:.1f}%",
            f"({self.stats.current}/{self.stats.total})"
        ]
        
        # Add rate information
        if self.show_rate and self.stats.items_per_second > 0:
            if self.stats.items_per_second >= 1:
                progress_parts.append(f"{self.stats.items_per_second:.1f} items/s")
            else:
                progress_parts.append(f"{1/self.stats.items_per_second:.1f}s/item")
        
        # Add ETA
        if self.show_eta and self.stats.estimated_remaining > 0:
            progress_parts.append(f"ETA: {self.stats.eta_formatted}")
        
        # Add current operation
        if self.stats.current_operation:
            progress_parts.append(f"- {self.stats.current_operation}")
        
        # Add phase
        if self.stats.phase:
            progress_parts.append(f"[{self.stats.phase}]")
        
        progress_line = " ".join(progress_parts)
        
        # Truncate if too long
        max_width = 120
        if len(progress_line) > max_width:
            progress_line = progress_line[:max_width-3] + "..."
        
        print(progress_line, end='', flush=True)
    
    def finish(self, message: str = "Complete") -> None:
        """Finish progress tracking"""
        
        with self._lock:
            self.stats.current = self.total
            self.stats.elapsed_time = time.time() - self.stats.start_time
            
            print(f"\r✅ {message} - {self.stats.total} items in {self.stats.elapsed_time:.2f}s")
            
            if self.stats.items_per_second > 0:
                print(f"   Average rate: {self.stats.items_per_second:.2f} items/s")
    
    def get_stats(self) -> ProgressStats:
        """Get current progress statistics"""
        with self._lock:
            return ProgressStats(
                current=self.stats.current,
                total=self.stats.total,
                start_time=self.stats.start_time,
                elapsed_time=self.stats.elapsed_time,
                estimated_remaining=self.stats.estimated_remaining,
                items_per_second=self.stats.items_per_second,
                current_operation=self.stats.current_operation,
                phase=self.stats.phase
            )


class DetailedProgressTracker:
    """Multi-phase progress tracker with detailed metrics and learning statistics"""
    
    def __init__(self, phases: List[str], verbose: bool = False):
        """Initialize detailed progress tracker"""
        self.phases = phases
        self.verbose = verbose
        self.current_phase_index = 0
        self.phase_progress = {}
        self.phase_timings = {}
        self.learning_stats = {}
        
        # Overall tracking
        self.start_time = time.time()
        self.total_items_processed = 0
        
        # Thread safety
        self._lock = threading.Lock()
        
        logger.info(f"Detailed progress tracker initialized with {len(phases)} phases")
        
        if self.verbose:
            print(f"🚀 Starting multi-phase operation with {len(phases)} phases:")
            for i, phase in enumerate(phases, 1):
                print(f"   {i}. {phase}")
            print()
    
    def start_phase(self, phase_name: str, total_items: int = 0) -> ProgressIndicator:
        """Start a new phase"""
        
        with self._lock:
            if phase_name in self.phases:
                self.current_phase_index = self.phases.index(phase_name)
            
            phase_start_time = time.time()
            self.phase_timings[phase_name] = {'start': phase_start_time}
            
            if self.verbose:
                print(f"\n📋 Phase {self.current_phase_index + 1}/{len(self.phases)}: {phase_name}")
                if total_items > 0:
                    print(f"   Processing {total_items} items...")
            
            # Create progress indicator for this phase
            progress = ProgressIndicator(
                total=total_items,
                description=f"Phase {self.current_phase_index + 1}: {phase_name}",
                show_rate=True,
                show_eta=True
            )
            
            self.phase_progress[phase_name] = progress
            
            logger.info(f"Started phase: {phase_name} (items: {total_items})")
            
            return progress
    
    def finish_phase(self, phase_name: str, items_processed: int = 0, 
                    learning_data: Dict[str, Any] = None) -> None:
        """Finish current phase"""
        
        with self._lock:
            phase_end_time = time.time()
            
            if phase_name in self.phase_timings:
                self.phase_timings[phase_name]['end'] = phase_end_time
                duration = phase_end_time - self.phase_timings[phase_name]['start']
                self.phase_timings[phase_name]['duration'] = duration
            
            self.total_items_processed += items_processed
            
            # Store learning data
            if learning_data:
                self.learning_stats[phase_name] = learning_data
            
            if self.verbose:
                duration = self.phase_timings.get(phase_name, {}).get('duration', 0)
                print(f"\n✅ Phase completed: {phase_name}")
                print(f"   Duration: {duration:.2f}s")
                print(f"   Items processed: {items_processed}")
                
                if learning_data:
                    print("   Learning metrics:")
                    for key, value in learning_data.items():
                        print(f"     • {key}: {value}")
            
            logger.info(f"Finished phase: {phase_name} (duration: {duration:.2f}s, items: {items_processed})")
    
    def update_learning_stats(self, phase_name: str, stats: Dict[str, Any]) -> None:
        """Update learning statistics for current phase"""
        
        with self._lock:
            if phase_name not in self.learning_stats:
                self.learning_stats[phase_name] = {}
            
            self.learning_stats[phase_name].update(stats)
            
            if self.verbose:
                print(f"\n🧠 Learning Update - {phase_name}:")
                for key, value in stats.items():
                    print(f"   • {key}: {value}")
    
    def show_overall_progress(self) -> None:
        """Show overall progress across all phases"""
        
        with self._lock:
            elapsed_time = time.time() - self.start_time
            completed_phases = len([p for p in self.phase_timings.values() if 'end' in p])
            
            print("\n📊 Overall Progress:")
            print(f"   Phases completed: {completed_phases}/{len(self.phases)}")
            print(f"   Total elapsed time: {elapsed_time:.2f}s")
            print(f"   Total items processed: {self.total_items_processed}")
            
            if self.total_items_processed > 0 and elapsed_time > 0:
                overall_rate = self.total_items_processed / elapsed_time
                print(f"   Overall processing rate: {overall_rate:.2f} items/s")
    
    def show_learning_summary(self) -> None:
        """Show comprehensive learning statistics summary"""
        
        if not self.learning_stats:
            return
        
        print("\n🧠 Learning Statistics Summary:")
        print("=" * 50)
        
        total_patterns_learned = 0
        total_suggestions_made = 0
        total_feedback_received = 0
        
        for phase_name, stats in self.learning_stats.items():
            print(f"\n📋 {phase_name}:")
            
            for key, value in stats.items():
                print(f"   • {key}: {value}")
                
                # Aggregate totals
                if 'patterns' in key.lower():
                    total_patterns_learned += value if isinstance(value, (int, float)) else 0
                elif 'suggestions' in key.lower():
                    total_suggestions_made += value if isinstance(value, (int, float)) else 0
                elif 'feedback' in key.lower():
                    total_feedback_received += value if isinstance(value, (int, float)) else 0
        
        print("\n📈 Learning Totals:")
        print(f"   • Total patterns learned: {total_patterns_learned}")
        print(f"   • Total suggestions made: {total_suggestions_made}")
        print(f"   • Total feedback received: {total_feedback_received}")
        
        if total_suggestions_made > 0:
            feedback_rate = (total_feedback_received / total_suggestions_made) * 100
            print(f"   • Feedback rate: {feedback_rate:.1f}%")
    
    def get_phase_summary(self) -> Dict[str, Any]:
        """Get summary of all phases"""
        
        with self._lock:
            total_duration = time.time() - self.start_time
            
            summary = {
                'total_phases': len(self.phases),
                'completed_phases': len([p for p in self.phase_timings.values() if 'end' in p]),
                'total_duration': total_duration,
                'total_items_processed': self.total_items_processed,
                'phase_details': {},
                'learning_summary': self.learning_stats.copy()
            }
            
            for phase_name, timing in self.phase_timings.items():
                summary['phase_details'][phase_name] = {
                    'duration': timing.get('duration', 0),
                    'completed': 'end' in timing
                }
            
            return summary
    
    def finish(self) -> Dict[str, Any]:
        """Finish tracking and return summary"""
        
        total_duration = time.time() - self.start_time
        
        if self.verbose:
            print("\n🎉 All phases completed!")
            print(f"   Total duration: {total_duration:.2f}s")
            print(f"   Total items processed: {self.total_items_processed}")
            
            if self.total_items_processed > 0:
                overall_rate = self.total_items_processed / total_duration
                print(f"   Overall rate: {overall_rate:.2f} items/s")
            
            # Show learning summary
            self.show_learning_summary()
        
        logger.info(f"Detailed progress tracking completed (duration: {total_duration:.2f}s)")
        
        return self.get_phase_summary()


@contextmanager
def progress_context(total: int, description: str = "", verbose: bool = False):
    """Context manager for progress tracking"""
    
    progress = ProgressIndicator(total, description)
    
    try:
        if verbose:
            print(f"🚀 Starting: {description}")
        
        yield progress
        
        progress.finish()
        
        if verbose:
            stats = progress.get_stats()
            print(f"✅ Completed: {description}")
            print(f"   Duration: {stats.elapsed_time:.2f}s")
            print(f"   Rate: {stats.items_per_second:.2f} items/s")
    
    except Exception as e:
        print(f"\n❌ Progress interrupted: {e}")
        raise


def create_learning_progress_callback(tracker: DetailedProgressTracker, 
                                    phase_name: str) -> Callable[[Dict[str, Any]], None]:
    """Create callback for learning progress updates"""
    
    def callback(learning_data: Dict[str, Any]) -> None:
        """Update learning progress"""
        tracker.update_learning_stats(phase_name, learning_data)
    
    return callback