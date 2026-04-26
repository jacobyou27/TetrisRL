# ATTRIBUTION.md

## Overview

This project was developed as a solo course project. I used a combination of:

- my own design decisions and implementation,
- course materials and the course project handout,
- public references on Tetris AI and reinforcement learning,
- and AI development tools for drafting, debugging, refactoring, and documentation assistance.

I remained responsible for the overall system design, project direction, experiment selection, interpretation of results, and final code/documentation choices.

## External Sources and References

The following sources informed the project direction, design choices, and literature context:

- **The Game of Tetris in Machine Learning** — survey of Tetris in ML, useful for positioning Tetris as a benchmark and comparing handcrafted, genetic, and RL approaches.
- **Approximate Dynamic Programming Finally Performs Well in the Game of Tetris** — useful background on policy-space search and why direct controller optimization can outperform naive value-based approaches.
- **Code My Road: Tetris AI – The (Near) Perfect Bot** — practical reference for strong handcrafted board features.
- **Lucky’s Notes: Coding a Tetris AI using a Genetic Algorithm** — useful as a practical recipe for placement-based evaluation and weight search.
- **MeatFighter’s Tetris AI writeups** — useful for board-state reasoning and risk management ideas.
- Additional public Tetris RL / GA repositories and papers consulted during project planning and literature review.

These sources influenced the motivation and framing of the project, but the final environment, heuristic, GA setup, evaluation workflow, and documentation were adapted to the specific goals of this project.

## AI Tool Usage

I used AI development tools (including ChatGPT and GitHub Copilot) in a collaborative role, not as a substitute for understanding.

### How AI tools were used

AI assistance was used for:

- brainstorming project directions,
- comparing Tetris AI approaches from public references,
- discussing the tradeoffs between primitive-action RL and placement-based search,
- generating draft code for helper functions and refactors,
- debugging benchmark/profiler scripts,
- refining logging and evaluation output,
- drafting documentation structure,
- and revising README / setup / attribution text.

### What remained my responsibility

I personally made the key project decisions, including:

- pivoting from primitive-action RL to placement-based reasoning,
- choosing the final project scope,
- selecting which generated code to keep or reject,
- deciding how to define reward and evaluation metrics,
- implementing the different agents and testers,
- choosing the GA-based final direction,
- interpreting all results,
- and verifying the final writeup and claims against the rubric.

Any AI-assisted code that was retained was reviewed, tested, and often modified before use. I did not rely on generated code without inspecting and integrating it into the broader system myself.

## Human Collaboration

This was a **solo project**. No partner contributed code or documentation.
