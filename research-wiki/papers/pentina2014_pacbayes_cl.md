---
type: paper
node_id: paper:pentina2014_pacbayes_cl
title: "A PAC-Bayesian Bound for Lifelong Learning"
authors: ["Anastasia Pentina", "Christoph H. Lampert"]
year: 2014
venue: ICML 2014
external_ids:
  arxiv: null
  doi: null
  s2: null
tags: [PAC-Bayes, lifelong-learning, continual-learning, theory, hierarchical]
relevance: core
origin_skill: research-wiki
created_at: 2026-04-24T00:00:00Z
updated_at: 2026-04-24T00:00:00Z
---

# One-line thesis

Derives a PAC-Bayesian generalization bound for lifelong learning by assuming tasks share a common hyperposterior (meta-prior), yielding a hierarchical bound with a fixed hyper-distribution.

## Problem / Gap

Classical PAC-Bayes bounds apply to a single task; no principled theory existed for learning across a sequence of related tasks.

## Method

Two-level PAC-Bayes: a hyper-prior over task priors, updated once after all tasks are observed (fixed hyperposterior). Uses McAllester-style bound giving log(m/δ) sample complexity.

## Key Results

- First PAC-Bayes bound for lifelong learning.
- Shows that sharing information across tasks via the hyperposterior reduces individual task sample complexity.

## Assumptions

- Tasks are i.i.d. draws from a meta-distribution.
- Fixed hyper-distribution: does not allow the hyperposterior to evolve sequentially with task index.

## Limitations / Failure Modes

- Fixed hyperposterior assumption is violated in continual learning where tasks arrive sequentially and the learner cannot revisit past data.
- Does not account for cross-task interference (catastrophic forgetting).
- No explicit drift penalty for hyperposterior evolution.
- log(m) term appears in bound due to McAllester union-bound technique.

## Reusable Ingredients

- Two-level hierarchical PAC-Bayes structure: model-level KL + hyper-level KL.
- Serves as the baseline classical hierarchical bound that our Theorem 3.1 extends.

## Open Questions

- How to remove the i.i.d. task assumption to handle truly sequential CL?

## Claims

## Connections

<!-- AUTO-GENERATED -->
- **←extends** paper:nguyen2025_pacbayes_cumulative
- **←inspired_by** idea:001

## Relevance to This Project

This is the classical hierarchical PAC-Bayes paper that our Theorem 3.1 extends. The key distinctions: (1) we allow time-varying hyperposterior; (2) we add explicit drift penalty for cross-task interference; (3) we use supermartingale/DV technique that removes the log(m) term. Cited in rebuttal as prior work to differentiate from.
