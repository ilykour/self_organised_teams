from numpy.core.fromnumeric import mean
import Hackathon
from typing import Any, Dict, List
from random import choice, choices
from numpy import random

knowledge_domains = {
    1: "Software developer",
    2: "Designer",
    3: "Marketer",
    4: "Sound engineer"
}

nationalities = {
    1: "USA",
    2: "India",
    3: "Other"
}

educational_levels = {
    1: "High school",
    2: "Bachelor",
    3: "Master or above"
}

ages = {
    1: "0 - 20",
    2: "21 - 30",
    3: "31 - 40",
    4: "41 - 50",
    5: "51 - 60",
    6: "Above 61"
}
# AMT data
age_prob = [0.0152, 0.4091, 0.3636, 0.0758, 0.0909, 0.0454]

personalities = {
    1: "Dominant",
    2: "Inspiring",
    3: "Supportive",
    4: "Cautious"
}
# Data from paper "Personality Matters: Balancing for Personality Types Leads to Better Outcomes for Crowd Teams"
personality_prob = [0.5, 0.1, 0.2, 0.2]


def limit_value(value: float):
    return max(min(value, 1), 0)


class Attribute:
    """
    All attributes in an worker
    """
    knowledge_domain: str
    nationality: str
    education: str
    age: str
    personality: str

    risk_appetite: float  # To what extent is a worker willing to stay instead of leaving current team.
    diversity_preference: float  # DEPRECATED
    expertise: float  # How skilled is a worker.

    def __init__(self, risk: float) -> None:

        # These (categorical) attributes are generated based on uniform distribution or certain weights.
        self.knowledge_domain = choice(list(knowledge_domains.values()))
        self.nationality = choice(list(nationalities.values()))
        self.education = choice(list(educational_levels.values()))
        self.age = choices(list(ages.values()), weights=age_prob, k=1)[0]
        self.personality = choices(list(personalities.values()), weights=personality_prob, k=1)[0]

        # These (numerical) attributes are generated based on beta distribution.
        # If risk > 0, then return the normal beta distribution value.
        if risk > 0:
            self.risk_appetite = limit_value(random.beta(a=2, b=risk))
        # If risk < 0, return 1 - the beta distribution with the negative of current beta value.
        elif risk < 0:
            self.risk_appetite = limit_value(1 - random.beta(a=2, b=-risk))
        self.diversity_preference = limit_value(random.beta(a=5, b=1))
        self.expertise = limit_value(random.beta(a=2, b=2))

    def __repr__(self) -> str:
        out = "\tAttributes: \n"
        for key, value in self.__dict__.items():
            out += "\t\t\t" + key + ": " + str(value) + "\n"
        return out


class Worker:
    """
    The worker model

    attributes: all attributes needed to model an agent
    example: {rating, nationality, background, expertise, age, etc.}

    preference dict: (DEPRECATED) a dict containing the agent's preference of each attribute
    example: {rating: [100%, 80%, 50%], nationality: [USA, China]}

    collaboration_history: a dict containing the history of all the workers it has worked with by their id
    """
    id: int
    attributes: Attribute
    preference_dict: Dict[str, List[Any]]  # DEPRECATED
    current_hackathon: Hackathon
    reward_history: Dict[int, float]  # round-reward pair
    average_reward: float
    teammate_history: Dict[int, List[int]]  # round-teammate id list pair

    def __init__(self, _id: int, risk: float) -> None:
        self.id = _id
        self.attributes = Attribute(risk)
        self.preference_dict = {}
        self.current_hackathon = None
        self.reward_history = {}
        self.average_reward = 0
        self.teammate_history = {}

    def __repr__(self) -> str:
        out = "Worker: \n"
        for key, value in self.__dict__.items():
            out += "\t\t" + key + ": " + str(value) + "\n"
        return out

    def append_reward(self, _round: int, reward: float) -> None:
        """Record reward from this round."""
        self.reward_history[_round] = reward
        self.average_reward = mean(list(self.reward_history.values()))

    def record_teammate(self, _round: int, teammate_id: int) -> None:
        """Record teammate ids from this round."""
        if _round in self.teammate_history.keys():
            self.teammate_history[_round].append(teammate_id)
        else:
            self.teammate_history[_round] = [teammate_id]

    def should_stay(self, _round: int) -> bool:
        """Worker's decision if this worker should stay in the same team based their risk appetite and reward from last round."""
        return self.reward_history[_round] >= self.attributes.risk_appetite

    def get_one_similarity(self, other_worker) -> int:
        """Return the similarity between this worker and another worker."""
        similarity = 0
        if self.attributes.knowledge_domain == other_worker.attributes.knowledge_domain:
            similarity += 1
        if self.attributes.nationality == other_worker.attributes.nationality:
            similarity += 1
        if self.attributes.education == other_worker.attributes.education:
            similarity += 1
        if self.attributes.age == other_worker.attributes.age:
            similarity += 1
        return similarity

    def get_avg_similarity(self, *args) -> float:
        """Get the average similarity between current worker and other workers.

        Args:
            *args: One or multiple workers.
        """
        avg_similarity = 0.0
        for worker in args:
            avg_similarity += self.get_one_similarity(worker)
        return avg_similarity / len(args)
