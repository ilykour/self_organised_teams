# ABM Simulator

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#license">License</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project
In this simulation, we investigate different ways that crowd teams can be formed through three team formation models namely bottom-up, top-down, and hybrid.


By simulating an open collaboration scenario such as a hackathon, we observe that:
* bottom-up model forms the most competitive teams with the highest teamwork quality
* bottom-up approaches are particularly suitable for populations with high-risk appetites
* bottom-up approaches are particularly suitable for populations with high degrees of homophily

Our study highlights the importance of integrating worker agency in algorithm-mediated team formation systems.


<p align="right">(<a href="#top">back to top</a>)</p>



### Built With

* [Python](https://www.python.org/)

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

### Prerequisites
This simulator requires python >= 3.7.0. 
If you do not have it on your local machine, follow the links provided below:
  ```sh
  Mac:https://www.python.org/downloads/macos/
  Windows: https://www.python.org/downloads/windows/
  Linux: sudo apt-get install python3.7.0
  ```

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/ilykour/self_organised_teams.git
   ```
2. Move to the directory
   ```sh
   cd self_organised_teams
   ```
3. Check that you have an up-to-date python version
   ```sh
   python --version
   ```
   if not, download the latest one.
4. Activate the virtual environment (Git bash)
   ```sh
   source venv/bin/activate
   ```
5. Run the simulator
   ```sh
   python SimulationSystem.py
   ```
6. To exit the simulator press Crt+C

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage
Simulation system to run different team formation algorithm under different scenario.

### Parameters
```
system_parameters = {
        "x": 20,
        "_round": 10,
        "k": 8.0,
        "lam": 0.8,
        "alpha": 0.5,
        "epsilon": 0.000002,
        "homophily_threshold": 2.8,
        "risk": 2,
        "info_to_console": True
    }
    restart = 3
    runtime_per_restart = 6
    with_hive = {"sot_1", "sot_2", "hive"}
    without_hive = {"sot_1", "sot_2"}
```

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the GNU AFFERO GENERAL PUBLIC LICENSE. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>
