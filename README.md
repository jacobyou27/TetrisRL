# TetrisRL: Placement-Based Tetris with Heuristic Search, Genetic Tuning, and Lookahead

This project began as a conventional reinforcement learning Tetris idea using low-level actions, but evolved into a placement-based formulation after it became clear that primitive control was too hard and sample-inefficient for the project goals. The final system uses a custom Gymnasium-compatible Tetris environment, a handcrafted afterstate heuristic, a genetic algorithm (GA) to tune heuristic weights, short-horizon lookahead at evaluation time, and simulator optimizations that make repeated search practical.

## What It Does

This project reframes Tetris as a placement-selection problem instead of a joystick-control problem. Rather than choosing low-level actions like move, rotate, and drop, the agent evaluates valid final placements for the current piece, scores the resulting afterstates using structural board features, and selects the best move.

The final pipeline combines:

- a custom placement-based Tetris environment,
- a handcrafted structural heuristic baseline,
- GA tuning of the heuristic weights,
- one-piece lookahead at evaluation time,
- and simulator profiling / optimization for faster repeated search.

The central question of the project is whether structured domain knowledge plus automated tuning can outperform or out-explain more generic RL approaches on a highly structured game.

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run GA tuning

Example GA run:

```bash
python -m src.agents.tune_heuristic_ga --population 24 --elite 4 --generations 24 --episodes-per-individual 4 --max-steps 300 --lookahead-depth 0 --workers 12 --verbosity 2 --plot --save-name ga_overnight_l0_v3
```

### 3. Evaluate the best saved weights with lookahead

Example long-horizon evaluation:

```bash
python -m src.agents.heuristic_agent --weights-json runs/ga/ga_overnight_l0_v3/best_weights.json --lookahead-depth 1 --lookahead-weight 1.0 --episodes 20 --max-steps 1000 --plot
```

### 4. Render one episode

```bash
python -m src.agents.heuristic_agent --weights-json runs/ga/ga_overnight_l0_v3/best_weights.json --lookahead-depth 1 --lookahead-weight 1.0 --episodes 1 --max-steps 1000 --render --delay-ms 60
```

### 5. Run the benchmark / profiler

```bash
python -m src.scripts.benchmark_simulator
python -m src.scripts.profile_simulator
```

See `SETUP.md` for more detailed environment setup, troubleshooting, and notes on the expected repository layout.

## Video Links

- **Demo Video:** [Watch the demo](videos/demo.mp4)
- **Technical Walkthrough:** [Watch the technical walk through](videos/tech_walk.mp4)

## Evaluation

The final evaluation emphasizes **gameplay-relevant metrics** rather than only shaped reward. Reported metrics include:

- total reward,
- total score,
- total lines cleared,
- episode length,
- top-out rate,
- and line-clear mix (singles, doubles, triples, tetrises).

### Representative Results

#### Best GA result during tuning (lookahead off during search)

- Generation: **16**
- Fitness: **173.21**
- Mean lines: **117.75**
- Mean score: **1341.50**
- Mean singles / doubles / triples / tetrises: **68.0 / 14.5 / 4.25 / 2.0**
- Top-out rate: **0.0%**

This result reflects the strongest GA individual found during training under the final tuned objective.

#### Long-horizon evaluation (1000-step cap) of saved best weights

| Eval setting | Mean reward | Mean score | Mean lines | Mean episode length | Top-out rate | Singles | Doubles | Triples | Tetrises |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Lookahead weight **0.25** | 7221.36 | 2231.44 | 384.40 | 966.25 | 5.0% | 278.05 | 44.75 | 4.75 | 0.65 |
| Lookahead weight **0.50** | 7864.55 | 2517.36 | 389.20 | 979.00 | 5.0% | 269.90 | 49.40 | 4.90 | 1.45 |
| Lookahead weight **1.00** | 8175.54 | 2496.02 | 398.40 | 1000.00 | 0.0% | 267.65 | 56.05 | 5.35 | 0.65 |

### Interpretation

These results show a meaningful tradeoff between:

- **stable long-horizon play** (higher reward, lower top-out rate, longer survival), and
- **more aggressive big-clear behavior** (more triples/tetrises in some settings, but sometimes less stability).

The best overall stable model used **lookahead weight 1.0** during evaluation, while lower lookahead weights produced less stable behavior and more top-outs.

### Simulator Throughput

Representative benchmark results on the optimized simulator:

- **No-lookahead rollout throughput:** about **31.0 steps/sec**
- **One-piece lookahead rollout throughput:** about **1.4 steps/sec**

This large compute gap motivated the final workflow: **tune weights with greedy evaluation for speed, then use lookahead only during final evaluation and demonstration.**

## Project Goal / Motivation

Tetris is a difficult sequential decision problem, but it is also highly structured: board geometry, holes, wells, line clears, and stack shape matter directly. Early in the project I explored the standard RL framing of learning low-level control in a falling-piece environment. That approach was harder to train and less interpretable than expected.

I therefore pivoted to a placement-based abstraction where the system chooses among valid final placements and evaluates the resulting board states directly. This let me:

- make the action space much easier to reason about,
- build interpretable heuristic policies,
- tune those policies automatically with a GA,
- and evaluate the tradeoff between safe long-horizon play and larger line clears.

## Approach / System Design

### 1. Custom Placement-Based Tetris Environment

The project uses a custom Gymnasium-compatible environment built around final piece placements rather than primitive controller actions. The observation includes:

- the current board,
- the current piece,
- the next piece,
- and an action mask over valid placement slots.

The environment wraps a placement engine that enumerates valid placements, simulates the resulting boards, tracks score and lines cleared, and exposes gameplay metrics for evaluation.

### 2. Afterstate Heuristic Agent

The first strong baseline is a handcrafted heuristic agent that evaluates candidate placements using structural board features such as:

- aggregate height,
- holes,
- bumpiness,
- max height,
- row transitions,
- column transitions,
- rows with holes,
- hole depth,
- cumulative wells,
- and eroded piece cells.

The heuristic was later extended with a thresholded max-height penalty so that a clean mid-height stack is not punished too aggressively while still penalizing truly dangerous boards.

### 3. Genetic Algorithm Tuning

The GA tunes the heuristic automatically. Each GA individual corresponds to a weight vector for:

- line-clear scores (single / double / triple / tetris),
- structural board-feature weights,
- and a lookahead coefficient (used only at evaluation time when lookahead is enabled).

Each individual is evaluated across multiple seeded episodes and scored using a weighted fitness function based on gameplay performance.

### 4. Lookahead Evaluation

At evaluation time the agent can use one-piece lookahead. Instead of scoring only the immediate afterstate, it can also score the best follow-up with the next piece and combine that future value with the immediate board score. This improves stability but is much more computationally expensive than greedy evaluation.

### 5. Simulator Optimization

Because GA tuning and lookahead require many repeated rollouts, the project includes simulator profiling and optimization work focused on:

- candidate enumeration,
- feature extraction,
- action-mask handling,
- and rollout throughput.

## Project Evolution / Design Decisions

### Early vision

The project started as a fairly standard RL game-playing idea:

- use a Gymnasium-compatible Tetris environment,
- inspect the observations and actions,
- pick an algorithm like DQN or PPO,
- train a model to play from environment observations,
- and hopefully get gameplay that looked like real Tetris.

At that stage, the mental model was still controller-level play: the agent would observe the board and falling piece, then choose low-level actions like move left, move right, rotate, and drop.

### Setup and environment sanity

Before any major design pivot, a lot of work went into practical setup and inspection:

- Python / virtual environment setup,
- PyTorch + CUDA verification,
- VS Code environment configuration,
- rendering checks,
- smoke tests,
- random action runs,
- and explicit observation debugging.

This phase was important because it exposed the real structure of the environment: the observation was a dict, not just a flat vector, and the board / active-piece representation was more complex than a simple 10x20 matrix.

### Real Tetris logic

Early on, I also considered whether the project should model:

- modern rotation systems,
- T-spins,
- kick tables,
- movement reachability,
- and controller-accurate piece access to placements.

This quickly revealed a major complexity problem. If the project tried to model full modern Tetris movement, the agent would have to learn not only which final board states are good, but also:

- controller timing,
- detailed rotation behavior,
- reachability constraints,
- and special movement-based placements like tucks or overhang entries.

That would make the problem dramatically harder.

### Simplification: not aiming for modern guideline Tetris

A crucial design decision was to not make the project a full modern Tetris simulator. T-spins, SRS, kick logic, and controller-faithful movement were treated as out of scope. That removed a large amount of complexity and let the project focus on the central decision problem:

> Which final placement should the agent choose?

### Pivot from movement-learning to placement-learning

Instead of learning step-by-step joystick control, the system would enumerate valid final placements and let the agent choose one directly. This changed the project from low-level controller learning into a more structured decision-making process over candidate placements.

### Search-based placement generation vs controller reachability

Once the project became placement-based, there was still a question of how to generate valid placements:

- use BFS / graph search over piece states from spawn, or
- compute final placements directly from piece geometry and collision.

Direct geometry-based placement generation was much simpler and much faster, even though it intentionally dropped some controller-specific placement types such as tucks and overhang-entry tricks. However, speed and clean action semantics mattered more for this project than full controller realism.

### Custom placement core

After the pivot, the project built a custom placement core with:

- a 20x10 board,
- tetromino orientation and rotation tables,
- canonical placement slots,
- direct drop computation,
- candidate generation,
- and resulting-board simulation.

### Heuristic baseline

The next major step was the heuristic evaluator. Once candidate placements could be scored using board features like holes, height, and bumpiness, the resulting agent immediately looked more intentional and substantially stronger than the random-valid baseline.

### Final direction: GA + lookahead + optimized simulator

The final project direction became:

1. use the custom placement-based simulator,
2. generate valid candidate placements quickly,
3. score afterstates using rich structural features,
4. tune those weights with a GA,
5. add one-piece lookahead at evaluation time,
6. and optimize simulator throughput so large-scale repeated evaluation is practical.

That final direction is very different from the original PPO idea, and much better matched to the actual structure of the problem.

## Key Results / Highlights

- A custom placement-based Tetris environment replaced primitive-action control with final placement selection, making the task more tractable and interpretable.
- A handcrafted structural heuristic immediately outperformed a random valid-action baseline.
- Genetic tuning found stronger and more interpretable weights than manual tuning alone.
- One-piece lookahead improved long-horizon stability in evaluation, but was about **22x slower** than greedy evaluation in the benchmark.
- The final system achieved strong long-horizon play while surfacing distinct strategies ranging from safer line-throughput play to more aggressive large-clear behavior.

## Repository Structure

This README assumes the cleaned final repo keeps only the files needed for the final project:

```text
TetrisRL/
├── README.md
├── SETUP.md
├── ATTRIBUTION.md
├── requirements.txt
├── src/
│   ├── envs/
│   │   ├── placement_core.py
│   │   └── placement_env.py
│   ├── agents/
│   │   ├── random_valid_agent.py
│   │   ├── heuristic_core.py
│   │   ├── heuristic_agent.py
│   │   └── tune_heuristic_ga.py
│   └── scripts/
│       ├── benchmark_simulator.py
│       ├── placement_simulator.py
│       ├── profile_simulator.py
│       └── watch_saved_model.py
├── tests/
│   └── smoke_test_placement_env.py
└── runs/
    └── [generated outputs: GA history, plots, saved best weights]
```

## Why PPO Was Not the Final Method

A PPO branch was explored during the project as a serious comparison direction, but the final project did not center on PPO. The main reason was that generic masked-action RL was much harder to train efficiently than the placement-based abstraction, while also producing less interpretable policies. The final system therefore prioritizes choosing the right representation and search procedure over relying on a black-box policy learner.

## Future Work

Possible next steps include:

- adding stronger controller-reachability constraints,
- exploring more tetris-oriented or risk-aware fitness objectives,
- revisiting PPO or imitation learning as a secondary comparison,
- and extending the simulator toward more modern Tetris mechanics such as SRS or T-spin-aware logic.
