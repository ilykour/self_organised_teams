from typing import Any, Dict
import Worker


class Hackathon:
    """The hackathon that workers have to finish

    workers: Workers that have registered this hackathon or are working on this hackathon.
    worker_number: Number of workers in this hackathon.
    id: (DEPRECATED) Id of this hackathon.
    time: (DEPRECATED) Time that this hackathon takes.
    """

    id: int
    worker_number: int
    time: float
    workers: Dict[int, Any]

    def __init__(self, _id: int, x: int, risk: float) -> None:
        self.id = _id
        self.worker_number = x
        self.time = 50.0
        self.workers = dict()
        for i in range(x):
            self.workers[i] = Worker.Worker(i, risk)

    def __repr__(self) -> str:
        out = "Task: \n"
        for key, value in self.__dict__.items():
            out += "\t" + key + ": " + str(value) + "\n"
        return out

    def should_start(self) -> bool:
        """DEPRECATED
        Whether this hackathon has hired enough workers.
        """
        if len(self.workers) >= self.worker_number:
            return True
        else:
            return False
