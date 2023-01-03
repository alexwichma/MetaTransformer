from __future__ import absolute_import, division, print_function

import time

class Timer:
    """Allows to track and parse elapsed time"""

    def __init__(self, elapsed_time=0) -> None:
        self.elapsed_time = elapsed_time
        self.last_time_checkpoint = None

    def start(self):
        self.last_time_checkpoint = time.time()

    def set_elapsed_time(self, elapsed_time):
        self.elapsed_time = elapsed_time

    def update_elapsed_time(self):
        current_time = time.time()
        self.elapsed_time = self.elapsed_time + (current_time - self.last_time_checkpoint)
        self.last_time_checkpoint = current_time

    def get_elapsed_time(self):
        return self.elapsed_time

    def get_elapsed_time_in_hours(self):
        return self.elapsed_time / 60.0 / 60.0
