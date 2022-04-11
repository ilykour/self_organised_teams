import copy
import pickle
from typing import Set, Tuple, List

import matplotlib
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.stats import beta

from Algorithm import SOTAlgorithm, HIVEAlgorithm, HybridHive, Algorithm
from Hackathon import *
from Team import Team

# str-Algorithm pairs of all algorithms used in the simulation.
algorithm_by_name = {
    "sot": SOTAlgorithm,
    "hive": HIVEAlgorithm,
    "hybrid_hive": HybridHive
}

# matplotlib parameters of all algorithms - to keep graphs consistent.
plt_parameters = {
    "color": {
        "c_sot": "orange",
        "r_sort": "black",
        "hive": "blue",
        "hybrid_hive": "crimson"
    },
    "marker": {
        "c_sot": "v",
        "r_sort": "o",
        "hive": "s",
        "hybrid_hive": "*"
    },
    "linestyle": {
        "c_sot": "v",
        "r_sort": "dashed",
        "hive": "dashdot",
        "hybrid_hive": "dotted"
    }
}

"""
The values are the range we vary when we run the simulation, i.e. the x axis of the figures. 

Risk levels: 
[-5.0, -4.7,..., -2.3) and (2.0, 2.3,..., 5.0]
From high risk level (-5.0) to low risk level (5.0). 

Homophily thresholds: 
[1.0, 1.3,..., 4.0]
From low homophily to high homophily. 

Population sizes:
[20, 30,..., 100]
From small population size to large population size. 
"""
risk_levels = [float("{:.1f}".format(x)) for x in list(np.arange(5.0, 1.9, -0.3)) + list(np.arange(-2.3, -5, -0.3))]
homophily_thresholds = [float("{:.1f}".format(x)) for x in list(np.arange(1.0, 4.1, 0.3))]
population_sizes = [20, 30, 40, 50, 60, 70, 80, 90, 100]


class SimulationSystem:
    """Simulation system to run different team formation algorithm under different scenario.

    Class attributes:
         round: The total number of rounds that this system has to run for.
         current_round: The number of current round.
         hackathon: The hackathon containing all workers.
         teams: All formed teams in this system.
         free_workers: All workers that are not in any team in current round.
         quality_history: The best quality in each round.
         avg_quality_history: The average quality of each team in each round.
         worst_quality_historyL The worst quality of each team in each round.
         algorithm: The algorithm used to form team.
    """
    round: int
    current_round: int
    hackathon: Hackathon
    teams: Set[Team]
    free_workers: Dict  # id-Worker pair
    quality_history: Dict[int, float]  # round-best quality pair
    avg_quality_history: Dict[int, float]  # round-average quality pair
    worst_quality_history: Dict[int, float]  # round-worst quality pair
    algorithm: Algorithm
    info_to_console: bool

    def __init__(self,
                 _round: int,
                 _hackathon: Hackathon,
                 algorithm_name: str,
                 param_dict: Dict[str, Any],
                 info_to_console: bool) -> None:
        """Default constructor.

        Args:
            _round: The number of rounds specified for the system.
            _hackathon: The hackathon containing all workers.
            algorithm_name: The name of algorithm that will be used.
            param_dict: Dictionary containing all parameters that the algorithm must use.
        """
        self.round = _round
        self.hackathon = _hackathon
        self.current_round = 0
        self.teams = set()
        self.free_workers = {}
        self.quality_history = {}
        self.avg_quality_history = {}
        self.worst_quality_history = {}
        self.algorithm = algorithm_by_name[algorithm_name]()
        self.algorithm.set_param(param_dict)
        self.info_to_console = info_to_console

    def run(self) -> None:
        """Run the system for *round times. """
        for _ in range(self.round):
            self.run_one_round()

    def run_one_round(self) -> None:
        """Run the system for one round. """
        self.current_round += 1

        # Get new teams in this round.
        self.teams = self.algorithm.get_teams(self)

        # Get outcomes for each team.
        all_outcomes = self.algorithm.get_outcomes(self)

        # Record quality from the best team.
        self.quality_history[self.current_round] = all_outcomes[0][0]
        self.avg_quality_history[self.current_round] = sum([outcome[0] for outcome in all_outcomes]) / len(all_outcomes)
        self.worst_quality_history[self.current_round] = all_outcomes[len(all_outcomes) - 1][0]

        # Record outcome history and teammate history for each worker in each team.
        self.algorithm.record_results(self, all_outcomes)

        # Print the useful info in this round to console.
        if self.info_to_console:
            self.print_info(all_outcomes)

    def add_new_team(self, *args) -> None:
        """Add new teams to the system.

        Args:
            *args: One or more teams to be added to the system.
        """
        for team in args:
            self.teams.add(team)

    def get_team_ids(self) -> List[List[int]]:
        """Get all ids in each team in the system.

        Returns:
            A 2-D list containing all ids, sorted by the smallest id value in each team.
        """
        return sorted([sorted(ids) for ids in [team.get_worker_ids() for team in self.teams]], key=lambda ids: ids[0])

    def print_info(self, all_outcomes: List[Tuple]) -> None:
        """Print all useful info in this round to console.

        Args:
            all_outcomes: The outcomes for each team.
        """
        print("Current round: " + str(self.current_round))
        print("Teams in this round: " + str(self.get_team_ids()))
        print("Free workers in this round: " + str(list(self.free_workers.keys())))
        print("Best quality: " + str(self.quality_history[self.current_round]) + " from team: " + str(
            all_outcomes[0][1].get_worker_ids()))
        print("Total quality: " + str(sum(list(self.quality_history.values()))))
        print("AVG quality: " + str(self.avg_quality_history[self.current_round]) + "\n")


def run_system_one_restart(system_parameters: Dict, algorithm_names: Set):
    """Run the system for one restart using ALL algorithms. One restart means running system from beginning for *round rounds.

    Args:
        system_parameters: All parameters used to run the system.
        algorithm_names: The names of algorithms that need to run.
    """
    # Make a DEEP copy of first hackathon to ensure the systems accept identical hackathon.
    hackathon_hive = Hackathon(_id=0, x=system_parameters["x"], risk=system_parameters["risk"])
    hackathon_hybrid_hive = copy.deepcopy(hackathon_hive)
    hackathon_c_sot = copy.deepcopy(hackathon_hive)
    hackathon_r_sort = copy.deepcopy(hackathon_hive)

    # algorithm name-SimulationSystem pair
    # {"hive": SimulationSystem}
    results = {}

    hive_algo_param = {"k": system_parameters["k"], "lam": system_parameters["lam"],
                       "alpha": system_parameters["alpha"], "epsilon": system_parameters["epsilon"]}
    hive = SimulationSystem(_round=system_parameters["_round"],
                            _hackathon=hackathon_hive,
                            algorithm_name="hive",
                            param_dict=hive_algo_param,
                            info_to_console=system_parameters["info_to_console"])
    if "hive" in algorithm_names:
        hive.run()
        results["hive"] = hive

    hybrid_hive_algo_param = {"k": system_parameters["k"], "lam": system_parameters["lam"],
                              "alpha": system_parameters["alpha"], "epsilon": system_parameters["epsilon"],
                              "homophily_threshold": system_parameters["homophily_threshold"]}
    hybrid_hive = SimulationSystem(_round=system_parameters["_round"],
                                   _hackathon=hackathon_hybrid_hive,
                                   algorithm_name="hybrid_hive",
                                   param_dict=hybrid_hive_algo_param,
                                   info_to_console=system_parameters["info_to_console"])
    if "hybrid_hive" in algorithm_names:
        hybrid_hive.run()
        results["hybrid_hive"] = hybrid_hive

    sot_algo_param_1 = {"homophily_threshold": system_parameters["homophily_threshold"], "team_priority": False}
    c_sot = SimulationSystem(_round=system_parameters["_round"],
                             _hackathon=hackathon_c_sot,
                             algorithm_name="sot",
                             param_dict=sot_algo_param_1,
                             info_to_console=system_parameters["info_to_console"])
    if "c_sot" in algorithm_names:
        c_sot.run()
        results["c_sot"] = c_sot

    sot_algo_param_2 = {"homophily_threshold": system_parameters["homophily_threshold"], "team_priority": True}
    r_sort = SimulationSystem(_round=system_parameters["_round"],
                             _hackathon=hackathon_r_sort,
                             algorithm_name="sot",
                             param_dict=sot_algo_param_2,
                             info_to_console=system_parameters["info_to_console"])
    if "r_sort" in algorithm_names:
        r_sort.run()
        results["r_sort"] = r_sort

    return results


def run_system(system_parameters: Dict, restart: int, time_per_restart: int, algorithm_names: Set, ANOVA_test: bool):
    """Run the simulation system.

    Args:
        system_parameters: All parameters used to run the system.
        restart: The number of restart times to reduce randomness.
        time_per_restart: The number of run times in each restart.
        algorithm_names: The names of algorithms that need to run.
        ANOVA_test: Whether do ANOVA test to validate results.
    """
    all_results = {}
    for name in algorithm_names:
        all_results[name] = {"Best": [], "Average": [], "Worst": []}

    for _ in range(restart * time_per_restart):
        results = run_system_one_restart(system_parameters, algorithm_names)
        for name in algorithm_names:
            all_results[name]["Best"].append(sum(list(results[name].quality_history.values())))
            all_results[name]["Average"].append(sum(list(results[name].avg_quality_history.values())))
            all_results[name]["Worst"].append(sum(list(results[name].worst_quality_history.values())))

    for name in algorithm_names:
        # Take the average of every time_per_restart restarts
        # Before: "Best": [2, 3, 2, 1, 2, ..., 3] 15 values, time_per_restart = 3
        # After: "Best": [2.33, 1.66, ..., ] 5 values
        all_results[name]["Best"] = list(
            np.mean(np.array(all_results[name]["Best"]).reshape(-1, time_per_restart), axis=1))
        all_results[name]["Average"] = list(
            np.mean(np.array(all_results[name]["Average"]).reshape(-1, time_per_restart), axis=1))
        all_results[name]["Worst"] = list(
            np.mean(np.array(all_results[name]["Worst"]).reshape(-1, time_per_restart), axis=1))
    print("All results: ")
    for k, v in all_results.items():
        print(k + " with mean of " + str(np.mean(v["Best"])))
        print(v)

    if ANOVA_test:
        names = list(algorithm_names)
        f, p = stats.f_oneway(all_results[names[0]]["Best"], all_results[names[1]]["Best"],
                              all_results[names[2]]["Best"])
        print("ANOVA test results: ")
        print("F-value: " + str(f))
        print("P-value: " + str(p))

    return all_results


def get_cmap(n, name='hsv'):
    """Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name."""
    return plt.cm.get_cmap(name, n)


def get_beta_distribution_graph():
    """Draw the beta distribution graph with different risk levels. """
    x = np.arange(0.0, 1.001, 0.01)
    color_map = get_cmap(len(risk_levels))
    for i, b in enumerate(risk_levels):
        if b > 0:
            y = beta.pdf(x, 2, b)
        else:
            y = list(reversed(beta.pdf(x, 2, -b)))
        plt.plot(x, y, color=color_map(i), label="beta=" + str(b))
    plt.xlabel("Random variable")
    plt.ylabel("Probability")
    plt.grid(linestyle="--")
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left', prop={'size': 17})
    plt.show()


def run_with_different_risk_level(system_parameters: Dict, restart: int, time_per_restart: int, algorithm_names: Set):
    print("Risky levels: ")
    print(risk_levels)
    results_by_risk = {}
    # Run the system for each risk level.
    for risk in risk_levels:
        system_parameters["risk"] = risk
        print("Current risk level: " + str(risk))
        all_results = run_system(system_parameters, restart, time_per_restart, algorithm_names, ANOVA_test=False)
        results_by_risk[risk] = all_results
    # Save results in the disk.
    with open("./results/Quality results of different risk levels.pkl", "wb") as f:
        pickle.dump(results_by_risk, f, pickle.HIGHEST_PROTOCOL)
        print("Risky level data saved. ")


def load_results_with_different_risk_level(result_type: str, algorithm_names: Set):
    with open("./results/Quality results of different risk levels.pkl", "rb") as f:
        results_by_risk = pickle.load(f)
    for risk in risk_levels:
        # Get the results and average them by the round number.
        results_by_risk[risk] = [np.mean(results_by_risk[risk]["c_sot"][result_type]) / system_parameters["_round"],
                                 np.mean(results_by_risk[risk]["r_sort"][result_type]) / system_parameters["_round"],
                                 np.mean(results_by_risk[risk]["hive"][result_type]) / system_parameters["_round"]]

    c_sot_results = [result[0] for result in results_by_risk.values()]
    r_sort_results = [result[1] for result in results_by_risk.values()]
    hive_results = [result[2] for result in results_by_risk.values()]

    x = []
    for key in list(results_by_risk.keys()):
        if key < 0:
            x.append(key + 4)
        else:
            x.append(key)
    plt.plot(x, c_sot_results, label="C-SOT",
             color=plt_parameters["color"]["c_sot"], marker=plt_parameters["marker"]["c_sot"])
    plt.plot(x, r_sort_results, label="R-SOT",
             color=plt_parameters["color"]["r_sort"], marker=plt_parameters["marker"]["r_sort"],
             linestyle=plt_parameters["linestyle"]["r_sort"])
    plt.plot(x, hive_results, label="Hive", color=plt_parameters["color"]["hive"],
             marker=plt_parameters["marker"]["hive"], linestyle=plt_parameters["linestyle"]["hive"])
    plt.xlabel(r"Risk level ($\beta$)")
    plt.ylabel(result_type + " quality")
    _ = plt.xticks(np.arange(-1, 5.1, 0.5),
                   reversed([float("{:.1f}".format(x)) for x in
                             list(np.arange(5.0, 1.9, -0.5)) + list(np.arange(-2.5, -5.1, -0.5))]))
    plt.grid(linestyle="--")
    plt.legend()
    plt.show()


def run_with_different_homophily(system_parameters: Dict, restart: int, time_per_restart: int, algorithm_names: Set):
    print("Homophily thresholds: ")
    print(homophily_thresholds)
    results_by_homophily = {}
    for homophily in homophily_thresholds:
        system_parameters["homophily_threshold"] = homophily
        print("Current homophily threshold: " + str(homophily))
        all_results = run_system(system_parameters, restart, time_per_restart, algorithm_names, ANOVA_test=False)
        results_by_homophily[homophily] = all_results
    with open("./results/Quality results of different homophily thresholds.pkl", "wb") as f:
        pickle.dump(results_by_homophily, f, pickle.HIGHEST_PROTOCOL)


def load_results_with_different_homophily(result_type: str, algorithm_names: Set):
    with open("./results/Quality results of different homophily thresholds.pkl",
              "rb") as f:
        results_by_homophily = pickle.load(f)
    for homophily in homophily_thresholds:
        results_by_homophily[homophily] = [
            np.mean(results_by_homophily[homophily]["c_sot"][result_type]) / system_parameters["_round"],
            np.mean(results_by_homophily[homophily]["r_sort"][result_type]) / system_parameters["_round"],
            np.mean(results_by_homophily[homophily]["hive"][result_type]) / system_parameters["_round"]]

    c_sot_results = [result[0] for result in results_by_homophily.values()]
    r_sort_results = [result[1] for result in results_by_homophily.values()]
    hive_results = [result[2] for result in results_by_homophily.values()]

    plt.plot(results_by_homophily.keys(), c_sot_results, label="C-SOT",
             color=plt_parameters["color"]["c_sot"], marker=plt_parameters["marker"]["c_sot"])
    plt.plot(results_by_homophily.keys(), r_sort_results, label="R-SOT",
             color=plt_parameters["color"]["r_sort"], marker=plt_parameters["marker"]["r_sort"],
             linestyle=plt_parameters["linestyle"]["r_sort"])
    plt.plot(results_by_homophily.keys(), hive_results, label="Hive", color=plt_parameters["color"]["hive"],
             marker=plt_parameters["marker"]["hive"], linestyle=plt_parameters["linestyle"]["hive"])
    plt.xlabel("Homophily threshold")
    plt.ylabel(result_type + " quality")
    plt.grid(linestyle="--")
    plt.legend()
    plt.show()


def run_with_different_population(system_parameters: Dict, restart: int, time_per_restart: int, algorithm_names: Set):
    print("Population sizes: ")
    print(population_sizes)
    results_by_size = {}
    for population in population_sizes:
        system_parameters["x"] = population
        print("Current population size: " + str(population))
        all_results = run_system(system_parameters, restart, time_per_restart, algorithm_names, ANOVA_test=False)
        results_by_size[population] = all_results
    with open("./results/Quality results of different population sizes.pkl", "wb") as f:
        pickle.dump(results_by_size, f, pickle.HIGHEST_PROTOCOL)


def load_results_with_different_population(result_type: str, algorithm_names: Set):
    with open("./results/Quality results of different population sizes.pkl", "rb") as f:
        results_by_size = pickle.load(f)
    for population in population_sizes:
        results_by_size[population] = [
            np.mean(results_by_size[population]["c_sot"][result_type]) / system_parameters["_round"],
            np.mean(results_by_size[population]["r_sort"][result_type]) / system_parameters["_round"]]
    c_sot_results = [result[0] for result in results_by_size.values()]
    r_sort_results = [result[1] for result in results_by_size.values()]

    plt.plot(population_sizes, c_sot_results, label="C-SOT",
             color=plt_parameters["color"]["c_sot"], marker=plt_parameters["marker"]["c_sot"])
    plt.plot(population_sizes, r_sort_results, label="R-SOT",
             color=plt_parameters["color"]["r_sort"], marker=plt_parameters["marker"]["r_sort"],
             linestyle=plt_parameters["linestyle"]["r_sort"])
    plt.xlabel("Population size")
    plt.ylabel(result_type + " quality")
    plt.grid(linestyle="--")
    plt.legend()
    plt.show()


# Research question 2: compare the hybrid approach with top-down and bottom-up approaches.
def run_rq_2(system_parameters: Dict, restart: int, time_per_restart: int, algorithm_names: Set):
    all_results = run_system(system_parameters, restart, time_per_restart, algorithm_names, ANOVA_test=False)
    print(all_results)
    with open("./results/RQ2_results.pkl", "wb") as f:
        pickle.dump(all_results, f, pickle.HIGHEST_PROTOCOL)
        print("RQ2 data saved. ")


def load_results_rq_2(algorithm_names: Set):
    with open("./results/RQ2_results.pkl", "rb") as f:
        all_results = pickle.load(f)
    boxplot_data = {"Best": {}, "Average": {}, "Worst": {}}
    for key in boxplot_data.keys():
        boxplot_data[key]["Hive"] = [val / system_parameters["_round"] for val in all_results["hive"][key]]
        boxplot_data[key]["C-SOT"] = [val / system_parameters["_round"] for val in all_results["c_sot"][key]]
        boxplot_data[key]["R-SOT"] = [val / system_parameters["_round"] for val in all_results["r_sort"][key]]
        boxplot_data[key]["HiveHybrid"] = [val / system_parameters["_round"] for val in all_results["hybrid_hive"][key]]
    for key, val in boxplot_data.items():
        draw_graph_rq_2(boxplot_data[key], key)


def draw_graph_rq_2(boxplot_data: Dict, result_type: str):
    means = [np.mean(val) for val in boxplot_data.values()]
    stds = [np.std(val) for val in boxplot_data.values()]
    fig, ax = plt.subplots()
    bp = ax.boxplot(boxplot_data.values(), widths=0.1, notch=True, patch_artist=True)
    ax.set_xticklabels(boxplot_data.keys())
    for patch, color in zip(bp["boxes"], ["blue", "orange", "black", "tomato"]):
        patch.set_facecolor(color)
    for i, line in enumerate(bp["medians"]):
        x, y = line.get_xydata()[1]
        text = "  mean={:.3f}\n  std={:.3f}".format(means[i], stds[i])
        ax.annotate(text, xy=(x, y), fontsize=16)
    # ax.legend([bp["boxes"][0]], loc="upper right")
    plt.ylabel(result_type + " quality")
    plt.grid(linestyle="--")
    plt.show()


if __name__ == "__main__":
    system_parameters = {
        "x": 20,
        "_round": 2,
        "k": 8.0,
        "lam": 0.8,
        "alpha": 0.5,
        "epsilon": 0.000002,
        "homophily_threshold": 2.8,
        "risk": 2,
        "info_to_console": True
    }
    restart = 30
    runtime_per_restart = 6
    with_hive = {"c_sot", "r_sort", "hive"}
    without_hive = {"c_sot", "r_sort"}

    # Parameter settings for matplotlib
    cmfont = font_manager.FontProperties(fname=matplotlib.get_data_path() + '/fonts/ttf/cmr10.ttf')
    font = {'family': 'serif',
            'serif': cmfont.get_name(),
            'size': 22}
    matplotlib.rc('font', **font)
    matplotlib.rcParams['mathtext.fontset'] = 'cm'
    matplotlib.rcParams['axes.unicode_minus'] = False

    # Beta distribution graph
    get_beta_distribution_graph()

    # Run RQ2.1 - risk level
    run_with_different_risk_level(system_parameters, restart, runtime_per_restart, algorithm_names=without_hive)
    load_results_with_different_risk_level(result_type="Best", algorithm_names=without_hive)
    load_results_with_different_risk_level(result_type="Average", algorithm_names=without_hive)
    load_results_with_different_risk_level(result_type="Worst", algorithm_names=without_hive)

    # Run RQ2.2 - population size
    run_with_different_population(system_parameters, restart, runtime_per_restart, algorithm_names=without_hive)
    load_results_with_different_population(result_type="Best", algorithm_names=without_hive)
    load_results_with_different_population(result_type="Average", algorithm_names=without_hive)
    load_results_with_different_population(result_type="Worst", algorithm_names=without_hive)

    # Run RQ2.2 - homophily threshold
    run_with_different_homophily(system_parameters, restart, runtime_per_restart, algorithm_names=without_hive)
    load_results_with_different_homophily(result_type="Best", algorithm_names=without_hive)
    load_results_with_different_homophily(result_type="Average", algorithm_names=without_hive)
    load_results_with_different_homophily(result_type="Worst", algorithm_names=without_hive)

    # Run RQ1 - comprehensive comparison
    run_rq_2(system_parameters, restart, runtime_per_restart, {"c_sot", "r_sort", "hive", "hybrid_hive"})
    load_results_rq_2({"c_sot", "r_sort", "hive", "hybrid_hive"})
