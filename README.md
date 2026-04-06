# MDP-ProbLog

> A Python 3 framework to represent and solve infinite-horizon MDPs using Probabilistic Logic Programming.

## Install

```bash
pip install mdpproblog
```

Or from source:

```bash
git clone https://github.com/hector-aviles/mdp-problog
cd mdp-problog/
pip install -e .
```

## Commands

| Command | Description |
|---------|-------------|
| `list` | Print all built-in example models |
| `show` | Print the domain and instance files |
| `solve` | Run Value Iteration and print V\* and π\* |
| `simulate` | Solve, then simulate the optimal policy from every state |

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `-m DOMAIN INSTANCE` | — | Domain and instance files to load |
| `-x EXAMPLE` | — | Select a built-in example (see `list`) |
| `-g GAMMA` | 0.9 | Discount factor |
| `-e EPSILON` | 0.1 | Convergence error bound |
| `-t TRIALS` | 100 | Simulation trials per state |
| `-z HORIZON` | 50 | Simulation horizon |
| `-b BACKEND` | auto | ProbLog compilation backend |
| `-o DIR` | — | Export results to CSV files in DIR |
| `-H` | off | Track V_k history per iteration → `convergence.csv` (requires `-o`) |
| `-Q` | off | Compute Q\*(s,a) after convergence → `q_values.csv` (requires `-o`) |
| `-v / -vv / -vvv` | — | Verbosity: INFO / DEBUG / TRACE |

## Example

```bash
$ mdp-problog solve -x sysadmin1
```

```
Value(running(c1,0)=0, running(c2,0)=0, running(c3,0)=0) = 16.829
Value(running(c1,0)=1, running(c2,0)=0, running(c3,0)=0) = 19.171
...
Value(running(c1,0)=1, running(c2,0)=1, running(c3,0)=1) = 25.607

Policy(running(c1,0)=0, running(c2,0)=0, running(c3,0)=0) = reboot(c1)
...
Policy(running(c1,0)=1, running(c2,0)=1, running(c3,0)=1) = reboot(none)

>> Value iteration converged in 0.031sec after 40 iterations.
>> Average time per iteration = 0.001sec.
```

Export the full MDP structures and Q-table to CSV:

```bash
$ mdp-problog solve -x sysadmin1 -o output/ -H -Q
```

## License

Copyright (c) 2016-2017 Thiago Pereira Bueno All Rights Reserved.

MDPProbLog is free software: you can redistribute it and/or modify it
under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or (at
your option) any later version.

MDPProbLog is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser
General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with MDPProbLog. If not, see http://www.gnu.org/licenses/.
