<h1>Enhancing Multi-Agent Learning in Real-World Interactive Environments through Process Reward Decomposition</h1>

This is the official implementation of CollabUIAgents.

# Overview

!(./assets/model.png)

We propose CollabUIAgents, a two-stage multi-agent learning framework for interactive environments. In the first stage, the base model is adapted to the environment using curriculum learning on multi-level instruction data. In the second stage, a novel process reward decomposition strategy is introduced during reinforcement learning, allowing rewards to be distributed at both the agent and conversation round levels. This granular feedback fosters collaborative awareness among agents without predefined roles and improves learning efficacy. Experimental results show that our method significantly enhances the performance of multi-agent systems based on open-source models, achieving notable improvements both within and across domains, while also exhibiting strong cross-environment generalization capabilities. Moreover, our best-performing systems achieve results on par with or exceed those of the strong closed-source models, while maintaining the flexibility to be integrated with prompt-based multi-agent systems for future research.