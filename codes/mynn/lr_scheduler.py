from abc import ABC, abstractmethod

class Scheduler(ABC):
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.step_count = 0

    @abstractmethod
    def step(self):
        pass


class StepLR(Scheduler):
    def __init__(self, optimizer, step_size, gamma):
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma

    def step(self):
        self.step_count += 1
        if self.step_count % self.step_size == 0:
            self.optimizer.lr *= self.gamma


class MultiStepLR(Scheduler):
    def __init__(self, optimizer, milestones, gamma):
        super().__init__(optimizer)
        self.milestones = sorted(milestones)
        self.gamma = gamma
    def step(self):
        self.step_count += 1
        if self.step_count in self.milestones:
            self.optimizer.lr *= self.gamma


class ExponentialLR(Scheduler):
    def __init__(self, optimizer, gamma):
        super().__init__(optimizer)
        self.gamma = gamma

    def step(self):
        self.step_count += 1
        self.optimizer.lr *= self.gamma
