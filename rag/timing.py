import time


class Timer:
    def __init__(self):
        self.times = {}

    def start(self, name: str):
        self.times[name] = {"start": time.perf_counter()}

    def stop(self, name: str):
        if name not in self.times or "start" not in self.times[name]:
            return
        end = time.perf_counter()
        start = self.times[name]["start"]

        self.times[name] = round((end - start) * 1000, 2)

    def get(self):
        return self.times
