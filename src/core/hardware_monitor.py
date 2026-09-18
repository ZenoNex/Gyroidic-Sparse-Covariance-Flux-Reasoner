import psutil
import os

class HardwareMonitor:
    """
    Centralized Hardware Monitor.
    Only used to prevent hardware awareness desync during poisoning events or as a last-resort
    for TailSlayer conditional bypass activation when local architecture has enough headroom.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.cpu_threshold = 75.0
            cls._instance.ram_threshold_gb = 2.0
            # Track desync anomalies
            cls._instance.desync_anomalies = 0
        return cls._instance

    def has_headroom(self) -> bool:
        """
        Checks if CPU and RAM have sufficient headroom to bypass default logic
        and run TailSlayer PyOpenCL kernels.
        """
        try:
            cpu_load = psutil.cpu_percent(interval=0.1)
            mem_info = psutil.virtual_memory()
            free_ram_gb = mem_info.available / (1024 ** 3)
            
            # Simple heuristic for potential poisoning/desync:
            # If load jumps impossibly fast or reads zero consistently, it's anomalous.
            if cpu_load == 0.0:
                self.desync_anomalies += 1
            else:
                self.desync_anomalies = max(0, self.desync_anomalies - 1)

            if self.desync_anomalies > 5:
                # If poisoning event suspected, fallback to conservative
                return False

            return cpu_load < self.cpu_threshold and free_ram_gb > self.ram_threshold_gb
        except Exception:
            # On failure, conservative fallback
            return False

def has_headroom() -> bool:
    """Helper function to get headroom status from the centralized monitor."""
    return HardwareMonitor().has_headroom()
