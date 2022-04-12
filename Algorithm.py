import copy
from Team import Team
from abc import abstractmethod, ABC
from typing import Dict, List, Set, Tuple
import random
import math
import heapq as hq


class Algorithm(ABC):
    """Abstract class of all algorithms."""

    @abstractmethod
    def set_param(self, param_dict) -> None:
        pass

    @abstractmethod
    def get_teams(self, simulation) -> Set[Team]:
        pass

    @abstractmethod
    def get_outcomes(self, simulation) -> List[Tuple]:
        pass

    @abstractmethod
    def record_results(self, simulation, all_outcome) -> None:
        pass


class HIVEAlgorithm(Algorithm):
    """Hive network rotation algorithm.

    Class attributes:
        k: k value used in the logistic function.
        lam: Dampening factor to decrease tie strength.
        alpha: Weight used in objective function to trade off network efficiency and tie strength.
        epsilon: Probability used to stop stochastic search for network rotation.
        graph: Tie strength between every two nodes. weight = graph[id_1][id_2].
    """

    k: float
    lam: float
    alpha: float
    epsilon: float
    graph: Dict[int, Dict[int, float]]
    network_graph: Dict[int, Set[int]]

    def __init__(self) -> None:
        self.graph = {}
        self.network_graph = {}

    def set_param(self, param_dict) -> None:
        """Set all parameters used in Hive algorithm."""
        self.k = param_dict["k"]
        self.lam = param_dict["lam"]
        self.alpha = param_dict["alpha"]
        self.epsilon = param_dict["epsilon"]

    def get_teams(self, simulation) -> Set[Team]:
        """Get a set of teams in this round.

        Args:
            simulation: The simulation system.

        Returns:
            A set of teams.
        """
        if simulation.current_round == 1:
            # In first round, we build the graph and call first round function.
            self.build_graph(simulation)
            return self.get_first_round_team(simulation)
        else:
            return self.get_next_round_team(simulation)

    def build_graph(self, simulation) -> None:
        """Build the initial graph between every workers."""
        for id_1 in simulation.hackathon.workers.copy().keys():
            if id_1 not in self.network_graph.keys():
                self.network_graph[id_1] = set()
            # Declare the sub-dictionary if it doesn't exist.
            if id_1 not in self.graph.keys():
                self.graph[id_1] = {}
            for id_2 in simulation.hackathon.workers.copy().keys():
                if id_1 != id_2:
                    # Exclude self-loop of course. Initial weight is 0.
                    self.graph[id_1][id_2] = 0

    def get_first_round_team(self, simulation) -> Set[Team]:
        """Get the team leaders and assign team members to each leader.

        Args:
            simulation: The simulation system.

        Returns:
            A set of teams.
        """
        # Get the number of teams that we want to form. Desired team size: minimal team size + 1.
        num_of_teams = math.ceil(
            simulation.hackathon.worker_number / (Team.min_team_size + 1)
        )

        # Get all dominant person.
        team_leader = [
            _id
            for _id in simulation.hackathon.workers.keys()
            if simulation.hackathon.workers[_id].attributes.personality == "Dominant"
        ]
        # Sample a subset if there are too many dominant person.
        if num_of_teams < len(team_leader):
            team_leader = random.sample(team_leader, num_of_teams)

        # Get all workers that are not team leader.
        free_ids = [
            _id for _id in simulation.hackathon.workers.keys() if _id not in team_leader
        ]

        for _id in team_leader:
            # For each team: add team leader and record team leader's id.
            team = Team()
            team.add_team_member(simulation.hackathon.workers[_id])
            team.leader_id = team.team_members[0].id

            # If this the last team leader, it should accept all worker left.
            if len(simulation.teams) == len(team_leader) - 1:
                for free_id in free_ids:
                    team.add_team_member(simulation.hackathon.workers[free_id])
            else:
                for _ in range(Team.min_team_size):
                    # Add the number of minimal team size of workers into this team.
                    if free_ids:
                        team.add_team_member(
                            simulation.hackathon.workers[free_ids.pop()]
                        )
                    else:
                        break
            # In the end, add this new team into the simulation.
            simulation.add_new_team(team)
        return simulation.teams

    def get_next_round_team(self, simulation) -> Set[Team]:
        """Get the solution and update the team members.

        Args:
            simulation: The simulation system.

        Returns:
            A set of teams.
        """
        # Find the solution using stochastic search.
        # key: The id of worker that should leave current team.
        # val: The id of team leader of target team.
        solution = self.stochastic_search(simulation)
        # print("Solution in this round: ")
        # print(solution)
        for _id, target in solution.items():
            for team in simulation.teams:
                if _id in team.get_worker_ids():
                    # First remove this worker from its current team.
                    worker = team.remove_team_member(_id)
                    break
            for team in simulation.teams:
                if target in team.get_worker_ids():
                    # Then add this worker to its target team.
                    team.add_team_member(worker)
                    break
        return simulation.teams

    def stochastic_search(self, simulation) -> Dict[int, int]:
        """Stochastic search to find the rotation that increases object.

        solution: The best solution so far.
        bad_moves: Pairs of the ids of workers and the ids of team leaders that we don't want to add the workers to.
        candidate: The possible candidate solution by adding moves to best solution.
        _id, target: The id of a worker and the id of team leader that we JUST added to candidate.

        Args:
            simulation: The simulation system.

        Returns:
            A dictionary containing leaving worker id and team leader id of target team.
        """
        bad_moves = {}
        solution = {}
        while True:
            try:
                candidate, _id, target = self.add_valid_move(
                    solution, simulation, bad_moves
                )
            except TypeError:
                # If we couldn't find any valid move, return current solution.
                # print("Search space empty, returning solution. ")
                return solution

            if candidate == solution:
                # If we can't add new move to solution, return it.
                return solution

            # Transform current teams into new teams.
            current_teams = self.transform(solution, simulation)
            new_teams = self.transform(candidate, simulation)
            # Calculate objective function value.
            old_f = self.get_f(
                current_teams, self.add_path_to_network(solution, self.network_graph)
            )
            new_f = self.get_f(
                new_teams, self.add_path_to_network(candidate, self.network_graph)
            )
            """
            print(solution)
            print(candidate)
            print(old_f)
            print(new_f)
            print(bad_moves)
            for node in self.graph.items():
                print(node)
            print()
            """
            if new_f > old_f:
                # If objective value of the new team is larger than current team setting, update the solution.
                solution = candidate
            else:
                # Record this move as bad cause its objective is lower than current solution.
                if _id not in bad_moves.keys():
                    bad_moves[_id] = []
                bad_moves[_id].append(target)

            # Prevent getting stuck in local maxima.
            if random.random() < self.epsilon:
                print("Returning solution cause epsilon reached. ")
                return solution

    def add_valid_move(
        self, solution: Dict, simulation, bad_moves: Dict
    ) -> Tuple[Dict, int, int]:
        """Search for a valid move.

        Args:
            solution: The current solution that we want to add a new move to.
            simulation: The simulation system.
            bad_moves: Pairs of the ids of workers and the ids of team leaders that we don't want to add the workers to.

        Returns:
            A tuple consists of a valid candidate we found, the new worker id, and the target team leader id.
        """
        # Get all team leader ids.
        leader_ids = [team.leader_id for team in simulation.teams]
        # Transform current teams into the teams using given solution.
        current_teams = self.transform(solution, simulation)

        # Shuffle them to maintain randomness.
        random.shuffle(current_teams)
        random.shuffle(leader_ids)
        for i in range(len(current_teams)):
            for _id in current_teams[i]:
                # For all workers.
                if _id not in leader_ids:
                    # Leader cannot move
                    for leader_id in leader_ids:
                        for team in current_teams:
                            # Get the target team ids
                            if leader_id in team:
                                target_team = team
                                break
                        # Rules:
                        # Must be a new team
                        # Must be within team size range
                        # Must not be a bad move that we already found
                        # If it is in the solution, then new team must be different from current solution
                        if (
                            leader_id not in current_teams[i]
                            and len(current_teams[i]) > Team.min_team_size
                            and len(target_team) < Team.max_team_size
                            and (
                                _id not in bad_moves.keys()
                                or leader_id not in bad_moves[_id]
                            )
                            and (
                                _id not in solution.keys() or leader_id != solution[_id]
                            )
                        ):
                            candidate = copy.deepcopy(solution)
                            candidate[_id] = leader_id
                            return candidate, _id, leader_id

    def add_path_to_network(self, candidate, network_graph) -> Dict[int, Set[int]]:
        """Add all moves in candidate solution to network graph, i.e. build the edges between the moves.

        Args:
            candidate: The candidate solution that we found.
            network_graph: The network graph containing all edges between nodes.

        Returns:
            A new network graph.
        """
        graph = copy.deepcopy(network_graph)
        for _id, target in candidate.items():
            graph[_id].add(target)
            graph[target].add(_id)
        return graph

    def transform(self, candidate, simulation) -> List[List]:
        """Transform the teams to a new list of teams using candidate rotation.

        Args:
            candidate: The candidate solution that we found.
            simulation: The simulation system.

        Returns:
            A list of new teams containing their new team members.
        """
        team_ids = [list(team) for team in simulation.get_team_ids()]
        for _id, target in candidate.items():
            for i in range(len(team_ids)):
                if _id in team_ids[i]:
                    # Remove it from old team.
                    team_ids[i].remove(_id)
                if target in team_ids[i]:
                    # Add it to the new team.
                    team_ids[i].append(_id)
        return team_ids

    def get_outcomes(self, simulation) -> List[Tuple]:
        """Get outcome for each team and sort them in descending order.

        Args:
             simulation: The simulation system.

        Returns:
            All outcome with their corresponding teams, sorted in descending order.
        """
        all_outcome = []
        for team in simulation.teams:
            outcome = team.get_outcome()
            all_outcome.append((outcome, team))
        return sorted(all_outcome, reverse=True, key=lambda pair: pair[0])

    def record_results(self, simulation, all_outcome) -> None:
        """Record results for each worker in this round. Update tie strengths for each pair of workers in this round.

        Args:
            simulation: The simulation system.
            all_outcome: All outcome with their corresponding teams, sorted in descending order.
        """
        for team in simulation.teams:
            ids = team.get_worker_ids()
            for worker in team.team_members:
                for _id in ids:
                    if _id != worker.id:
                        worker.record_teammate(simulation.current_round, _id)
                        self.network_graph[worker.id].add(_id)
                        # Logistic function to increase tie strength
                        self.graph[worker.id][_id] = self.sigmoid(
                            self.graph[worker.id][_id]
                        )
                for _id in self.graph[worker.id].keys():
                    if _id not in ids:
                        # Dampening factor to decrease tie strength
                        self.graph[worker.id][_id] = (
                            self.lam * self.graph[worker.id][_id]
                        )

    def sigmoid(self, x) -> float:
        """Sigmoid (logistic) function used to increase tie strength between workers.

        Args:
            x: Old tie strength between workers.

        Returns:
            New tie strength after applied sigmoid function.
        """
        return 1 / (1 + math.exp(-self.k * (x - 0.2)))

    def get_network_efficiency(self, graph) -> float:
        """Get the network efficiency of this graph.

        Args:
            graph: The graph representation of the tie strength between workers.

        Returns:
            The total network efficiency of this graph.
        """
        # For all node in the graph, find the shortest path to every other nodes using Dijkstra algorithm,
        # then add the inverse of the path to network efficiency.
        # In the end, divide it by the total number of paths in the graph, i.e. n * (n - 1).
        # Note we add each path twice, so n * (n - 1) here.
        network_efficiency = 0
        for start in graph.keys():
            for target, path in self.dijkstra(start, graph).items():
                if start != target:
                    network_efficiency += 1 / path
        return network_efficiency / (len(graph.keys()) * (len(graph.keys()) - 1))

    def dijkstra(self, start, graph) -> Dict[int, float]:
        """Dijkstra algorithm to find the minimum path from start node.

        Args:
            start: Start node.
            graph: The Dict[int, Dict[int, float]] representation of a graph.

        Returns:
            A dict containing the shortest paths from start node to every nodes.
        """
        visited = {}
        weights = {}
        for key in graph.keys():
            visited[key] = False
            weights[key] = math.inf
        queue = []
        weights[start] = 0
        hq.heappush(queue, (0, start))
        while len(queue) > 0:
            g, u = hq.heappop(queue)
            visited[u] = True
            for v in graph[u]:
                if not visited[v]:
                    f = g + 1
                    if f < weights[v]:
                        weights[v] = f
                        hq.heappush(queue, (f, v))
        return weights

    def get_team_tie_strength(self, ids: List[int]) -> float:
        """Get tie strength within a team.

        Args:
            ids: A list of workers' ids in the team.

        Returns:
            The average tie strength of this team.
        """
        if len(ids) == 1:
            return 0
        tie_strength = 0.0
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                tie_strength += self.graph[ids[i]][ids[j]]
        return tie_strength / (len(ids) * (len(ids) - 1) / 2)

    def get_f(self, teams, graph) -> float:
        """Get the objective function.

        Args:
            teams: Lists of ids of the team members.
            graph: The Dict[int, Dict[int, float]] representation of a graph.

        Returns:
            The objective value based on tie strength and network efficiency.
        """
        tie_strength = 0.0
        for team in teams:
            tie_strength += self.get_team_tie_strength(team)
        # print("NE: " + str(self.get_network_efficiency(graph)))
        # print("TS: " + str(tie_strength))
        return (1 - self.alpha) * self.get_network_efficiency(
            graph
        ) + self.alpha * tie_strength * 0.005


class SOTAlgorithm(Algorithm):
    """Self-Organizing Teams algorithm.

    Class attributes:
        homophily_threshold: The threshold to determine if a worker want to join a team or form a team with others.
        variation: Benchmark variation number. 1 for teams choose first, and 2 for workers choose first.
    """

    homophily_threshold: float
    variation: bool

    def __init__(self):
        pass

    def set_param(self, param_dict) -> None:
        """Set all parameters used in Hive algorithm."""
        self.homophily_threshold = param_dict["homophily_threshold"]
        self.variation = param_dict["team_priority"]

    def get_teams(self, simulation) -> Set[Team]:
        if simulation.current_round == 1:
            return self.get_random_team(simulation)
        else:
            return self.variation_function[self.variation](self, simulation)

    def get_random_team(self, simulation) -> Set[Team]:
        """Get randomly formed teams. The attributes of the worker have already been randomly generated,
        so we just have to append the worker one by one.

        Args:
            simulation: The simulation system.

        Returns:
            A set of teams.
        """
        teams = set()
        team = Team()
        for worker in simulation.hackathon.workers.values():
            team.add_team_member(worker)
            if len(team.team_members) == Team.min_team_size + 1:
                teams.add(team)
                team = Team()
        if team.team_members:
            teams.add(team)
        return teams

    def get_team_variation_1(self, simulation) -> Set[Team]:
        """SOT benchmark variation 1 (teams choose first, prioritizes existing teams).

        Args:
            simulation: The simulation system.

        Returns:
            A set of teams.
        """
        # Get all workers that want to leave their team.
        free_workers = self.get_all_free_workers(simulation)

        # Let current teams choose free workers firstly
        for team in simulation.teams.copy():
            while free_workers:
                if not self.find_new_worker(free_workers=free_workers, team=team):
                    break

        # Build graph for rest workers
        worker_graph = self.build_graph(free_workers)

        while free_workers:
            # Sort current graph based on edge weight (similarity)
            worker_graph = sorted(worker_graph, key=lambda edge: edge[0], reverse=True)

            new_team = Team()

            if not worker_graph and free_workers.keys():
                # If there is only one worker left, put rest workers in the free worker set.
                # They are not in any team in this round.
                for _id in free_workers.copy().keys():
                    simulation.free_workers[_id] = free_workers.pop(_id)
                break
            else:
                # Add the first three workers with largest similarity among all possible combinations.
                new_team.add_team_member(
                    free_workers.pop(worker_graph[0][1]),
                    free_workers.pop(worker_graph[0][2]),
                    free_workers.pop(worker_graph[0][3]),
                )
                while free_workers:
                    # Try to find all other workers that can join this new team.
                    if not self.find_new_worker(free_workers, new_team):
                        break
            simulation.add_new_team(new_team)
            for _id in new_team.get_worker_ids():
                # Remove the edges connected to the nodes in this new team.
                worker_graph = self.remove_edges(worker_graph=worker_graph, _id=_id)

        return simulation.teams

    def get_team_variation_2(self, simulation) -> Set[Team]:
        """SOT benchmark variation 2 (workers choose first, does not prioritize existing teams).

        Args:
            simulation: The simulation system.

        Returns:
            A set of teams.
        """
        # Get all free workers firstly.
        free_workers = self.get_all_free_workers(simulation)

        # Build graph for rest workers
        worker_graph = self.build_graph(free_workers)

        while free_workers:
            # Sort current graph based on edge weight (similarity)
            worker_graph = sorted(worker_graph, key=lambda edge: edge[0], reverse=True)

            max_similarity_with_team = 0.0
            for worker in free_workers.values():
                for team in simulation.teams:
                    # Find the maximal similarity among all workers with all teams.
                    current_similarity = team.get_new_worker_similarity(worker)
                    if current_similarity > max_similarity_with_team:
                        max_similarity_with_team = current_similarity
                        max_team = team
                        max_worker = worker
            if not worker_graph or max_similarity_with_team > worker_graph[0][0]:
                # Add the worker to the team if their similarity is the largest one.
                max_team.add_team_member(free_workers.pop(max_worker.id))
                worker_graph = self.remove_edges(worker_graph, max_worker.id)
            else:
                # Else make new team for the three workers with largest similarity.
                new_team = Team()
                # If there is only one worker left.
                if not worker_graph and free_workers.keys():
                    for _id in free_workers.copy().keys():
                        simulation.free_workers[_id] = free_workers.pop(_id)
                else:
                    new_team.add_team_member(
                        free_workers.pop(worker_graph[0][1]),
                        free_workers.pop(worker_graph[0][2]),
                        free_workers.pop(worker_graph[0][3]),
                    )
                    simulation.add_new_team(new_team)
                    for _id in new_team.get_worker_ids():
                        worker_graph = self.remove_edges(
                            worker_graph=worker_graph, _id=_id
                        )

        return simulation.teams

    def remove_edges(self, worker_graph: List, _id: int) -> List[Tuple]:
        """Remove all edges connected to the workers in given list.

        Args:
            worker_graph: The graph containing similarities and worker ids.
            _id: The ids of workers that we want to remove.

        Returns:
            A new graph with certain edges removed.
        """
        return [
            edge
            for edge in worker_graph
            if edge[1] != _id and edge[2] != _id and edge[3] != _id
        ]

    def find_new_worker(self, free_workers: Dict, team: Team) -> bool:
        """Find a free worker with highest similarity for current team.

        Args:
            free_workers: The potential workers.
            team: The current team that want to have new worker.

        Returns:
            True if this team found new worker. False otherwise.
        """
        # Get similarities between all free workers and current team
        similarity_dict = {}
        max_similarity = 0
        for free_worker in free_workers.copy().values():
            # Find the worker with maximal similarity.
            similarity = team.get_new_worker_similarity(free_worker)
            max_similarity = max(max_similarity, similarity)
            if similarity not in similarity_dict:
                similarity_dict[similarity] = []
            similarity_dict[similarity].append(free_worker)
        # Return false if none of the similarities reach the threshold
        if max_similarity < self.homophily_threshold:
            return False
        else:
            # Get the worker with highest similarity and highest reward
            new_worker = sorted(
                similarity_dict[max_similarity],
                key=lambda worker: worker.average_reward,
                reverse=True,
            )[0]
            team.add_team_member(free_workers.pop(new_worker.id))
            return True

    def get_all_free_workers(self, simulation) -> Dict:
        """Get all workers that want to leave and remove them from their current team.

        Args:
            simulation: The simulation system.

        Returns:
            A dict containing workers and their ids that want to leave.
        """
        free_workers = simulation.free_workers
        simulation.free_workers = {}

        for team in simulation.teams.copy():
            # Get all workers that want to leave
            left_index = [
                i
                for i in range(len(team.team_members))
                if not team.team_members[i].should_stay(simulation.current_round - 1)
            ]
            # If only one worker wants to stay -> team dissolves as well
            if len(left_index) == len(team.team_members) - 1:
                left_index = [i for i in range(len(team.team_members))]
            # Remove all workers that want to leave
            left_index.reverse()
            for index in left_index:
                worker = team.team_members.pop(index)
                free_workers[worker.id] = worker
            # Remove this team if no worker in it
            if not team.team_members:
                simulation.teams.remove(team)

        return free_workers

    def build_graph(self, workers) -> List[Tuple]:
        """Build the graph whose nodes are workers, edges are the similarities between each two workers.

        Args:
            workers: Dict containing workers and their ids.

        Returns:
            A list of tuple of: Similarity, id1, id2, id3.
        """
        return [
            (worker_1.get_avg_similarity(worker_2, worker_3), id_1, id_2, id_3)
            for id_1, worker_1 in workers.items()
            for id_2, worker_2 in workers.items()
            for id_3, worker_3 in workers.items()
            if id_1 != id_2 and id_1 != id_3 and id_2 != id_3
        ]

    def get_outcomes(self, simulation) -> List[Tuple]:
        """Get outcome for each team and sort them in descending order.

        Args:
             simulation: The simulation system.

        Returns:
            All outcome with their corresponding teams, sorted in descending order.
        """
        all_outcome = []
        for team in simulation.teams:
            outcome = team.get_outcome()
            all_outcome.append((outcome, team))
        return sorted(all_outcome, reverse=True, key=lambda pair: pair[0])

    def record_results(self, simulation, all_outcome) -> None:
        """Record reward (normalized to [0, 1]) based on their ranking.
        Record teammate ids for each worker in this round.

        Args:
            simulation: The simulation system.
            all_outcome: All outcome with their corresponding teams, sorted in descending order.
        """
        length = len(all_outcome)
        for i in range(length):
            if length == 1:
                reward = 1
            else:
                reward = (float(length) - 1 - i) / (float(length) - 1)
            for worker in all_outcome[i][1].team_members:
                simulation.hackathon.workers[worker.id].append_reward(
                    simulation.current_round, reward
                )

        for team in simulation.teams:
            ids = team.get_worker_ids()
            for worker in team.team_members:
                for _id in ids:
                    if _id != worker.id:
                        worker.record_teammate(simulation.current_round, _id)

    variation_function = {True: get_team_variation_1, False: get_team_variation_2}


class HybridHive(Algorithm):
    """HybridHive network rotation algorithm.

    Class attributes:
        k: k value used in the logistic function.
        lam: Dampening factor to decrease tie strength.
        alpha: Weight used in objective function to trade off network efficiency and tie strength.
        epsilon: Probability used to stop stochastic search for network rotation.
        graph: Tie strength betweent every two nodes. weight = graph[id_1][id_2].
    """

    k: float
    lam: float
    alpha: float
    epsilon: float
    homophily_threshold: float
    graph: Dict[int, Dict[int, float]]
    network_graph: Dict[int, Set[int]]

    def __init__(self) -> None:
        self.graph = {}
        self.network_graph = {}

    def set_param(self, param_dict) -> None:
        """Set all parameters used in Hive algorithm."""
        self.k = param_dict["k"]
        self.lam = param_dict["lam"]
        self.alpha = param_dict["alpha"]
        self.epsilon = param_dict["epsilon"]
        self.homophily_threshold = param_dict["homophily_threshold"]

    def get_teams(self, simulation) -> Set[Team]:
        """Get a set of teams in this round.

        Args:
            simulation: The simulation system.

        Returns:
            A set of teams.
        """
        if simulation.current_round == 1:
            # In first round, we build the graph and call first round function.
            self.build_graph(simulation)
            return self.get_first_round_team(simulation)
        else:
            return self.get_next_round_team(simulation)

    def build_graph(self, simulation) -> None:
        """Build the initial graph between every workers."""
        for id_1 in simulation.hackathon.workers.copy().keys():
            if id_1 not in self.network_graph.keys():
                self.network_graph[id_1] = set()
            # Declare the sub-dictionary if it doesn't exist.
            if id_1 not in self.graph.keys():
                self.graph[id_1] = {}
            for id_2 in simulation.hackathon.workers.copy().keys():
                if id_1 != id_2:
                    # Exclude self-loop of course. Initial weight is 0.
                    self.graph[id_1][id_2] = 0

    def get_first_round_team(self, simulation) -> Set[Team]:
        """Get the team leaders and assign team members to each leader.

        Args:
            simulation: The simulation system.

        Returns:
            A set of teams.
        """
        # Get the number of teams that we want to form. Desired team size: minimal team size + 1.
        num_of_teams = math.ceil(
            simulation.hackathon.worker_number / (Team.min_team_size + 1)
        )

        # Get all dominant person.
        team_leader = [
            _id
            for _id in simulation.hackathon.workers.keys()
            if simulation.hackathon.workers[_id].attributes.personality == "Dominant"
        ]
        # Sample a subset if there are too many dominant person.
        if num_of_teams < len(team_leader):
            team_leader = random.sample(team_leader, num_of_teams)

        # Get all workers that are not team leader.
        free_ids = [
            _id for _id in simulation.hackathon.workers.keys() if _id not in team_leader
        ]

        for _id in team_leader:
            # For each team: add team leader and record team leader's id.
            team = Team()
            team.add_team_member(simulation.hackathon.workers[_id])
            team.leader_id = team.team_members[0].id

            # If this the last team leader, it should accept all worker left.
            if len(simulation.teams) == len(team_leader) - 1:
                for free_id in free_ids:
                    team.add_team_member(simulation.hackathon.workers[free_id])
            else:
                for _ in range(Team.min_team_size):
                    # Add the number of minimal team size of workers into this team.
                    if free_ids:
                        team.add_team_member(
                            simulation.hackathon.workers[free_ids.pop()]
                        )
                    else:
                        break
            # In the end, add this new team into the simulation.
            simulation.add_new_team(team)
        return simulation.teams

    def get_next_round_team(self, simulation) -> Set[Team]:
        """Get the solution and update the team members.

        Args:
            simulation: The simulation system.

        Returns:
            A set of teams.
        """
        # Find the solution using stochastic search.
        # key: The id of worker that should leave current team.
        # val: The id of team leader of target team.
        solution = self.stochastic_search(simulation)
        # print("Solution in this round: ")
        # print(solution)
        for _id, target in solution.items():
            for team in simulation.teams:
                if _id in team.get_worker_ids():
                    # First remove this worker from its current team.
                    worker = team.remove_team_member(_id)
                    break
            for team in simulation.teams:
                if target in team.get_worker_ids():
                    # Then add this worker to its target team.
                    team.add_team_member(worker)
                    break
        return simulation.teams

    def stochastic_search(self, simulation) -> Dict[int, int]:
        """Stochastic search to find the rotation that increases object.

        solution: The best solution so far.
        bad_moves: Pairs of the ids of workers and the ids of team leaders that we don't want to add the workers to.
        candidate: The possible candidate solution by adding moves to best solution.
        _id, target: The id of a worker and the id of team leader that we JUST added to candidate.

        Args:
            simulation: The simulation system.

        Returns:
            A dictionary containing leaving worker id and team leader id of target team.
        """
        bad_moves = {}
        solution = {}
        while True:
            try:
                candidate, _id, target = self.add_valid_move(
                    solution, simulation, bad_moves
                )
            except TypeError:
                # If we couldn't find any valid move, return current solution.
                # print("Search space empty, returning solution. ")
                return solution

            if candidate == solution:
                # If we can't add new move to solution, return it.
                return solution

            # Transform current teams into new teams.
            current_teams = self.transform(solution, simulation)
            new_teams = self.transform(candidate, simulation)
            # Calculate objective function value.
            old_f = self.get_f(
                current_teams, self.add_path_to_network(solution, self.network_graph)
            )
            new_f = self.get_f(
                new_teams, self.add_path_to_network(candidate, self.network_graph)
            )
            """
            print(solution)
            print(candidate)
            print(old_f)
            print(new_f)
            print(bad_moves)
            for node in self.graph.items():
                print(node)
            print()
            """
            if new_f > old_f:
                # If objective value of the new team is larger than current team setting, update the solution.
                solution = candidate
            else:
                # Record this move as bad cause its objective is lower than current solution.
                if _id not in bad_moves.keys():
                    bad_moves[_id] = []
                bad_moves[_id].append(target)

            # Prevent getting stuck in local maxima.
            if random.random() < self.epsilon:
                print("Returning solution cause epsilon reached. ")
                return solution

    def add_valid_move(
        self, solution: Dict, simulation, bad_moves: Dict
    ) -> Tuple[Dict, int, int]:
        """Search for a valid move.

        Args:
            solution: The current solution that we want to add a new move to.
            simulation: The simulation system.
            bad_moves: Pairs of the ids of workers and the ids of team leaders that we don't want to add the workers to.

        Returns:
            A tuple consists of a valid candidate we found, the new worker id, and the target team leader id.
        """
        # Get all team leader ids.
        leader_ids = [team.leader_id for team in simulation.teams]
        # Transform current teams into the teams using given solution.
        current_teams = self.transform(solution, simulation)

        # Shuffle them to maintain randomness.
        random.shuffle(current_teams)
        random.shuffle(leader_ids)
        for i in range(len(current_teams)):
            for _id in current_teams[i]:
                # For all workers.
                if _id not in leader_ids:
                    # Leader cannot move
                    for leader_id in leader_ids:
                        for team in current_teams:
                            # Get the target team ids
                            if leader_id in team:
                                target_team = team
                                break
                        for team in simulation.teams:
                            if leader_id in team.get_worker_ids():
                                next_team = team
                        # Rules:
                        # Must be a new team
                        # Must be within team size range
                        # Must not be a bad move that we already found
                        # If it is in the solution, then new team must be different from current solution
                        if (
                            leader_id not in current_teams[i]
                            and len(current_teams[i]) > Team.min_team_size
                            and len(target_team) < Team.max_team_size
                            and (
                                _id not in bad_moves.keys()
                                or leader_id not in bad_moves[_id]
                            )
                            and (
                                _id not in solution.keys() or leader_id != solution[_id]
                            )
                            and not simulation.hackathon.workers[_id].should_stay()
                            and next_team.get_new_worker_similarity(
                                simulation.hackathon.workers[_id]
                            )
                            > self.homophily_threshold
                        ):
                            candidate = copy.deepcopy(solution)
                            candidate[_id] = leader_id
                            return candidate, _id, leader_id

    def add_path_to_network(self, candidate, network_graph) -> Dict[int, Set[int]]:
        """Add all moves in candidate solution to network graph, i.e. build the edges between the moves.

        Args:
            candidate: The candidate solution that we found.
            network_graph: The network graph containing all edges between nodes.

        Returns:
            A new network graph.
        """
        graph = copy.deepcopy(network_graph)
        for _id, target in candidate.items():
            graph[_id].add(target)
            graph[target].add(_id)
        return graph

    def transform(self, candidate, simulation) -> List[List]:
        """Transform the teams to a new list of teams using candidate rotation.

        Args:
            candidate: The candidate solution that we found.
            simulation: The simulation system.

        Returns:
            A list of new teams containing their new team members.
        """
        team_ids = [list(team) for team in simulation.get_team_ids()]
        for _id, target in candidate.items():
            for i in range(len(team_ids)):
                if _id in team_ids[i]:
                    # Remove it from old team.
                    team_ids[i].remove(_id)
                if target in team_ids[i]:
                    # Add it to the new team.
                    team_ids[i].append(_id)
        return team_ids

    def get_outcomes(self, simulation) -> List[Tuple]:
        """Get outcome for each team and sort them in descending order.

        Args:
             simulation: The simulation system.

        Returns:
            All outcome with their corresponding teams, sorted in descending order.
        """
        all_outcome = []
        for team in simulation.teams:
            outcome = team.get_outcome()
            all_outcome.append((outcome, team))
        return sorted(all_outcome, reverse=True, key=lambda pair: pair[0])

    def record_results(self, simulation, all_outcome) -> None:
        """Record results for each worker in this round. Update tie strengths for each pair of workers in this round.

        Args:
            simulation: The simulation system.
            all_outcome: All outcome with their corresponding teams, sorted in descending order.
        """
        for team in simulation.teams:
            ids = team.get_worker_ids()
            for worker in team.team_members:
                for _id in ids:
                    if _id != worker.id:
                        worker.record_teammate(simulation.current_round, _id)
                        self.network_graph[worker.id].add(_id)
                        # Logistic function to increase tie strength
                        self.graph[worker.id][_id] = self.sigmoid(
                            self.graph[worker.id][_id]
                        )
                for _id in self.graph[worker.id].keys():
                    if _id not in ids:
                        # Dampening factor to decrease tie strength
                        self.graph[worker.id][_id] = (
                            self.lam * self.graph[worker.id][_id]
                        )

    def sigmoid(self, x) -> float:
        """Sigmoid (logistic) function used to increase tie strength between workers.

        Args:
            x: Old tie strength between workers.

        Returns:
            New tie strength after applied sigmoid function.
        """
        return 1 / (1 + math.exp(-self.k * (x - 0.2)))

    def get_network_efficiency(self, graph) -> float:
        """Get the network efficiency of this graph.

        Args:
            graph: The graph representation of the tie strength between workers.

        Returns:
            The total network efficiency of this graph.
        """
        # For all node in the graph, find the shortest path to every other nodes using Dijkstra algorithm,
        # then add the inverse of the path to network efficiency.
        # In the end, divide it by the total number of paths in the graph, i.e. n * (n - 1).
        # Note we add each path twice, so n * (n - 1) here.
        network_efficiency = 0
        for start in graph.keys():
            for target, path in self.dijkstra(start, graph).items():
                if start != target:
                    network_efficiency += 1 / path
        return network_efficiency / (len(graph.keys()) * (len(graph.keys()) - 1))

    def dijkstra(self, start, graph) -> Dict[int, float]:
        """Dijkstra algorithm to find the minimum path from start node.

        Args:
            start: Start node.
            graph: The Dict[int, Dict[int, float]] representation of a graph.

        Returns:
            A dict containing the shortest paths from start node to every nodes.
        """
        visited = {}
        weights = {}
        for key in graph.keys():
            visited[key] = False
            weights[key] = math.inf
        queue = []
        weights[start] = 0
        hq.heappush(queue, (0, start))
        while len(queue) > 0:
            g, u = hq.heappop(queue)
            visited[u] = True
            for v in graph[u]:
                if not visited[v]:
                    f = g + 1
                    if f < weights[v]:
                        weights[v] = f
                        hq.heappush(queue, (f, v))
        return weights

    def get_team_tie_strength(self, ids: List[int]) -> float:
        """Get tie strength within a team.

        Args:
            ids: A list of workers' ids in the team.

        Returns:
            The average tie strength of this team.
        """
        if len(ids) == 1:
            return 0
        tie_strength = 0.0
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                tie_strength += self.graph[ids[i]][ids[j]]
        return tie_strength / (len(ids) * (len(ids) - 1) / 2)

    def get_f(self, teams, graph) -> float:
        """Get the objective function.

        Args:
            teams: Lists of ids of the team members.
            graph: The Dict[int, Dict[int, float]] representation of a graph.

        Returns:
            The objective value based on tie strength and network efficiency.
        """
        tie_strength = 0.0
        for team in teams:
            tie_strength += self.get_team_tie_strength(team)
        # print("NE: " + str(self.get_network_efficiency(graph)))
        # print("TS: " + str(tie_strength))
        return (1 - self.alpha) * self.get_network_efficiency(
            graph
        ) + self.alpha * tie_strength * 0.005
