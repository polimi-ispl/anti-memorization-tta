# Mitigating data replication in text-to-audio generative diffusion models through anti-memorization guidance
[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2509.14934) *Francisco Messina, Francesca Ronchini, Luca Comanducci, Paolo Bestagini, Fabio Antonacci*

This repository accompains the the paper *[Mitigating data replication in text-to-audio generative diffusion models through anti-memorization guidance](https://arxiv.org/abs/2509.14934)*. It builds on Stability AI’s Stable Audio Open 1.0 and adds *Anti-Memorization Guidance (AMG)* during sampling. 
  -  In the folder _code_ you can find the code used to perform the experiments and also the plots included in the paper.
  -  In the folder _docs_ you can find the additional material, which is also nicely presented in the accompanying webpage 
References to the paper and base model are at the end of this document.


## Citation

If you use this code in academic work, please cite:

```
@inproceedings{messina2026mitigating,
  title={Mitigating data replication in text-to-audio generative diffusion models through anti-memorization guidance},
  author={Messina, Francisco and Ronchini, Francesca and Comanducci, Luca and Bestagini, Paolo and Antonacci, Fabio},
  booktitle={ICASSP 2026-2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={15742--15746},
  year={2026},
  organization={IEEE}
}
```

Stable Audio Open:

```
@inproceedings{evans2025stable,
  title={Stable audio open},
  author={Evans, Zach and Parker, Julian D and Carr, CJ and Zukowski, Zack and Taylor, Josiah and Pons, Jordi},
  booktitle={ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2025},
  organization={IEEE}
}
```

---

For questions or reproducibility details (e.g., exact `c1/c2/c3` and scheduling configurations used for the paper experiments), you can inspect `amg_infer.py` in this repository, the AMG logic within `stable_audio_tools/inference/amg_generation.py`, and the reference paper.

