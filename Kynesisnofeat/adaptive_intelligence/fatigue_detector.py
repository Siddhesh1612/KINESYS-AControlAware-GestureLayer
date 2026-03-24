import time

class FatigueDetector:
    def __init__(self, warning_interval=1800):
        # Default warn every 30 minutes
        self.start_time = time.time()
        self.warning_interval = warning_interval
        
    def check_fatigue(self):
        elapsed = time.time() - self.start_time
        if elapsed > self.warning_interval:
            self.start_time = time.time()
            return True
        return False
