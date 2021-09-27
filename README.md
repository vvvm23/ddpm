# Denoising Diffusion Probabilistic Models (WIP)
PyTorch implementation of "Denoising Diffusion Probabilistic Models" and DPPM
improvements from "Improved Denoising Diffusion Probabilistic Models".

> Currently a work-in-progress. Stay tuned!

### TODO:
- [ ] Forward diffusion
    - [X] Linear noise scheduling
    - [X] Cosine noise scheduling
    - [ ] (Other schedules?)
    - [ ] Tractable computations for loss calculation
- [ ] Implement UNet architecture
    - [X] Main model structure
    - [X] Multi-headed self-attention at certain resolutions
    - [X] Improved time embedding injection (Appendix A)
    - [ ] Learning variance (Section 3.1)
    - [ ] Class conditioning
    - [ ] EMA parameter updates
- [ ] Training script
    - [X] Main script
    - [X] Simple loss computation
    - [ ] Hybrid loss computation
    - [ ] Auxiliary denoising loss (from D3PM)
- [ ] Sample generation

### References

*Denoising Diffusion Probabilistic Models*
> Jonathan Ho, Ajay Jain, Pieter Abbeel

```
@misc{ho2020denoising,
      title={Denoising Diffusion Probabilistic Models}, 
      author={Jonathan Ho and Ajay Jain and Pieter Abbeel},
      year={2020},
      eprint={2006.11239},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

*Improved Denoising Diffusion Probabilistic Models*
> Alex Nichol, Prafulla Dhariwal

```
@misc{nichol2021improved,
      title={Improved Denoising Diffusion Probabilistic Models}, 
      author={Alex Nichol and Prafulla Dhariwal},
      year={2021},
      eprint={2102.09672},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

*Structured Denoising Diffusion Models in Discrete State-Spaces*
> Jacob Austin, Daniel D. Johnson, Jonathan Ho, Daniel Tarlow, Rianne van den Berg

```
@misc{austin2021structured,
      title={Structured Denoising Diffusion Models in Discrete State-Spaces}, 
      author={Jacob Austin and Daniel D. Johnson and Jonathan Ho and Daniel Tarlow and Rianne van den Berg},
      year={2021},
      eprint={2107.03006},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
