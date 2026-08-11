---
type: paper
node_id: paper:fr_lora_cikm2025
title: "FR-LoRA: Fisher Regularized LoRA for Multilingual Continual Learning"
authors: ["Sayanta Adhikari", "Sanjay Agrawal", "Vivek Sembium"]
year: 2025
venue: CIKM 2025
external_ids:
  arxiv: null
  doi: "10.1145/3746252.3761531"
  s2: null
tags: [Fisher, LoRA, continual-learning, multilingual, NLP, stability]
relevance: related
origin_skill: research-wiki
created_at: 2026-04-24T00:00:00Z
updated_at: 2026-04-24T00:00:00Z
---

# One-line thesis

FR-LoRA applies Fisher-based regularization to LoRA fine-tuning for multilingual continual learning, achieving the best retention of prior knowledge with minimal inference overhead.

## Problem / Gap

Multilingual CL with LoRA suffers from forgetting across languages; standard LoRA lacks stability mechanisms.

## Method

Fisher-regularized LoRA penalty applied to LoRA parameters across sequential multilingual tasks. No flatness component.

## Key Results

- Best forgetting reduction among baselines.
- Strong stability; achieves best plasticity-stability tradeoff in multilingual setting.

## Limitations / Failure Modes

- NLP/multilingual focus; not demonstrated on vision PECL benchmarks.
- No flatness component → plasticity may not be explicitly optimized.

## Relevance to This Project

Related competitor to idea:002. Covers Fisher-in-LoRA for CL in NLP; idea:002 targets vision PECL with AS^(1) flatness added.
