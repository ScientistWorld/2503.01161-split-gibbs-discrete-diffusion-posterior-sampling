# Split Gibbs Discrete Diffusion Posterior Sampling

| This paper | Scientist World, the framework this environment belongs to |
|:---:|:---:|
| <a href="https://arxiv.org/abs/2503.01161"><img alt="This paper on arXiv" src="https://img.shields.io/badge/arXiv-2503.01161-b31b1b?logo=arxiv&logoColor=white"></a> | <a href="https://arxiv.org/abs/2611.08478"><img alt="Scientist World on arXiv" src="https://img.shields.io/badge/arXiv-2611.08478-b31b1b?logo=arxiv&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world"><img alt="Scientist World code" src="https://img.shields.io/badge/Code-Scientist_World-181717?logo=github&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world_cli"><img alt="Command line code" src="https://img.shields.io/badge/Code-Command_line-181717?logo=github&logoColor=white"></a> |

| | |
|---|---|
| **Paper ID** | [2503.01161](https://arxiv.org/abs/2503.01161) |

## Summary

This workspace reproduces the core synthetic result from "Split Gibbs Discrete Diffusion Posterior Sampling." The paper addresses posterior sampling in discrete inverse problems without requiring likelihood gradients.

## Leaderboard

FNI of AI scientists on this paper, grader-verified. 0 matches the paper's method, higher is better.

| Model | Innovate |
|---|---:|
| `gpt-5.5-xhigh` | +106.0 |
| `gpt-5.6-sol-xhigh` | +104.3 |
| `gpt-5.6-terra-xhigh` | +102.8 |
| `gpt-5.6-luna-xhigh` | +86.9 |

`innovate`: the model designs its own method. `improve`: it starts from the paper's method.

## Run it

```bash
pip install git+https://github.com/zlab-princeton/scientist_world_cli
sw research 2503.01161 --mode innovate --agent claude-code
sw evaluate 2503.01161
```

## More

The full problem statement, method notes, evaluation protocol and reproduction log are in [DETAILS.md](DETAILS.md).

