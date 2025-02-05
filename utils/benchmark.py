#! coding: utf-8

from utils.timer import Timer


class Benchmark:
    def __init__(self, description: str = 'Done'):
        self.description = description

    def __enter__(self):
        self.timer = Timer()
        return self

    def __exit__(self, *args):
        print(f'{self.description}: {self.timer.stop():.4f} seconds.')
