from typing import Counter, Tuple

from Worker import *


class Team:
    """The team class is used to represent each team during the hackathon.

    Class attributes:
        team_members: A list of workers in this team.
    """
    team_members: List[Worker]

    # In Developer's domain, Worker 1 with expertise 0.9, Worker 2 with expertise 0.9
    # The second one will only count 0.9 * diminishing value = 0.81
    diminshing_value = 0.9

    # If a worker has NEVER worked with ANYONE in this team, his/her expertise will count less due to learning effect.
    learning_effect_value = 0.9

    min_team_size = 3
    max_team_size = 5

    def __init__(self) -> None:
        self.team_members = []

    def __repr__(self) -> str:
        out = "\tTeam: \n"
        for key, value in self.__dict__.items():
            out += "\t\t" + key + ": " + str(value) + "\n"
        return out

    def add_team_member(self, *args) -> None:
        """Add new team members to the team.

        Args:
            args: One or more workers to be added to the team.
        """
        for worker in args:
            self.team_members.append(worker)

    def remove_team_member(self, _id) -> Worker:
        """Remove the worker with this id from the team.

        Args:
            _id: The id of the worker to be removed.
        """
        for _i in range(len(self.team_members.copy())):
            if self.team_members[_i].id == _id:
                return self.team_members.pop(_i)

    def get_outcome(self) -> float:
        """Get the outcome of this team. """
        outcome = 0.0
        # Add up all outcomes time their weights.
        # Check the last part of this class.
        for func, weight in self.outcome_weights.items():
            outcome += func(self) * weight
        return outcome

    def get_skill_outcome(self) -> float:
        """Check the workers' domain and corresponding expertise, then calculate the skill outcome. """
        skills = {}
        for worker in self.team_members:
            domain = worker.attributes.knowledge_domain
            if domain not in skills.keys():
                skills[domain] = []
            if self.has_collaborated(worker=worker):
                # Check collaboration history.
                learning_effect = 1
            else:
                learning_effect = self.learning_effect_value
            skills[domain].append(worker.attributes.expertise * learning_effect)

        # Get the skill outcome.
        # We have more experts for one domain, then there is a diminishing utility function applied
        skill_outcome = 0.0
        for domain, expertises in skills.items():
            factor = 1.0
            for expertise in sorted(expertises, reverse=True):
                # Apply the diminishing effect.
                skill_outcome += expertise * factor
                factor *= self.diminshing_value
        # Example:
        # Developer: Worker 1 with expertise 1, Worker 3 with expertise 0.8
        # Marketer: Worker 2 with expertise 0.9
        # Designer: Worker 4 with expertise 0.8, Worker 5 with expertise 0.7
        # Skill outcome = 1/5 * ((1 + 0.8 * 0.9) + (0.9) + (0.8 + 0.7 * 0.9))
        return skill_outcome / len(self.team_members)

    def has_collaborated(self, worker: Worker) -> bool:
        """Check if this worker has worked with any other workers in the team.

        Args:
            worker: The worker to be checked.

        Returns:
            True if the worker has worked with any of them, False otherwise.
        """
        for _id in self.get_worker_ids():
            for teammate_ids in worker.teammate_history.values():
                for teammate_id in teammate_ids:
                    if _id == teammate_id:
                        return True
        return False

    def get_compatibility_outcome(self) -> float:
        """Calculate the compatibility outcome.
        Penalize outcome if they do not cover all four personality types, or if there are more than two D's
        """
        compatibility_outcome = 0.0
        # Example
        # ["D": 1, "I": 2, "C": 2]
        # Unique personality size: 3
        team_personalities = Counter([worker.attributes.personality for worker in self.team_members])
        # 0.4 (= 0.2 * (3 - 1))
        compatibility_outcome += 0.2 * (len(list(team_personalities.keys())) - 1)
        # 0.8 (= 0.4 + 0.4)
        if team_personalities["Dominant"] < 2:
            compatibility_outcome += 0.4
        return compatibility_outcome

    def get_team_size_outcome(self) -> float:
        """Calculate the team size outcome."""
        team_size = len(self.team_members)
        if team_size < self.min_team_size:
            return 1 - 0.1 * (self.min_team_size - team_size)
        elif team_size > self.max_team_size:
            return max(0.0, 1 - 0.1 * (team_size - self.max_team_size))
        else:
            return 1.0

    def get_new_worker_similarity(self, new_worker: Worker) -> float:
        """Get the average similarity of the new worker between current workers."""
        similarity = 0.0
        for worker in self.team_members:
            similarity += worker.get_one_similarity(new_worker)
        return similarity / len(self.team_members)

    def get_worker_ids(self) -> Tuple:
        return tuple([worker.id for worker in self.team_members])

    outcome_weights = {
        get_skill_outcome: 0.5,
        get_compatibility_outcome: 0.3,
        get_team_size_outcome: 0.2
    }