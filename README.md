# Fine-tune a tiny DeepSeek model to solve the Taxi Game using Group Relative Policy Optimization (GRPO)


GRPO is a RL algorithm first introduced by the [DeepSeek Math Paper](https://arxiv.org/abs/2402.03300) and is one of the key innovations contributing to the reasoning capabilities of [R1](https://arxiv.org/abs/2501.12948).

Our goal is to test GRPO on a relatively small LLM to determine whether we can fine-tune a small model to solve our custom complex reasoning task — one that it previously could not solve and without any labeled data.

For this we leverage the [taxi environment](https://gymnasium.farama.org/environments/toy_text/taxi/) from gymnasium.
