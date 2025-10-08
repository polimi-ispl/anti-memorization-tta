# Mitigating data replication in text-to-audio generative diffusion models through anti-memorization guidance
[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2509.14934) *Francisco Messina, Francesca Ronchini, Luca Comanducci, Paolo Bestagini, Fabio Antonacci*

This repository accompains the the paper *[Mitigating data replication in text-to-audio generative diffusion models through anti-memorization guidance](https://arxiv.org/abs/2509.14934)*. It builds on Stability AI’s Stable Audio Open 1.0 and adds *Anti-Memorization Guidance (AMG)* during sampling. 
  -  In the folder _code_ you can find the code used to perform the experiments and also the plots included in the paper.
  -  In the folder _docs_ you can find the additional material, which is also nicely presented in the accompanying webpage 
References to the paper and base model are at the end of this document.


## Citation

If you use this code in academic work, please cite:

```
@misc{
	messina2025mitigatingdatareplicationtexttoaudio,
	title={Mitigating data replication in text-to-audio generative diffusion models through anti-memorization guidance}, 
	author={Francisco Messina and Francesca Ronchini and Luca Comanducci and Paolo Bestagini and Fabio Antonacci},
	year={2025},
	eprint={2509.14934},
	archivePrefix={arXiv},
	primaryClass={eess.AS},
	url={https://arxiv.org/abs/2509.14934}, 
}
```

Stable Audio Open:

```
@misc{
	evans2024stableaudioopen,
	title={Stable Audio Open}, 
	author={Zach Evans and Julian D. Parker and CJ Carr and Zack Zukowski and Josiah Taylor and Jordi Pons},
	year={2024},
	eprint={2407.14358},
	archivePrefix={arXiv},
	primaryClass={cs.SD},
	url={https://arxiv.org/abs/2407.14358}, 
}
```

---

For questions or reproducibility details (e.g., exact `c1/c2/c3` and scheduling configurations used for the paper experiments), you can inspect `amg_infer.py` in this repository, the AMG logic within `stable_audio_tools/inference/amg_generation.py`, and the reference paper.

