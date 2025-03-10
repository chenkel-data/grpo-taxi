# Fine-tune a tiny DeepSeek model to solve the Taxi Game using Group Relative Policy Optimization (GRPO)


GRPO is a RL algorithm first introduced by the [DeepSeek Math Paper](https://arxiv.org/abs/2402.03300) and is one of the key innovations of the reasoning cabablilties of [R1](https://arxiv.org/abs/2501.12948).


Our goal is to test GRPO on a relatively small LLM to determine whether we can fine-tune a small model to solve our custom complex reasoning task — one that it previously could not solve.

For this we leverage the taxi environment from gymnasium. A well-known library used for training RL-Models.
