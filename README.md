# Denoising Diffusion Probabilistic Models (WIP)
> PyTorch implementation of "Denoising Diffusion Probabilistic Models" (DPPM)
> and DPPM improvements from "Improved Denoising Diffusion Probabilistic
> Models".

The original paper can be found [here](https://arxiv.org/abs/2006.11239).

OpenAI released a (claimed) improvement upon DDPM, which is incorporated in
this repo. Their paper can be found [here](https://arxiv.org/abs/2102.09672).

My aim is to provide an implementation that incorporates elements from both
papers whilst remaining (relatively) simple to understand.

> Currently a work-in-progress. Stay tuned!

## Installation
`TODO: pip install instructions`

`TODO: build instructions`

## Usage

### DDPM Model
`TODO: instructions on using the model as a standalone module`

### Training
`TODO: instructions on using the training script`

### Generation
`TODO: instructions on using sample generation script`

## Modifications
`TODO: note any deviations from the original works`

## Samples
`TODO: add some nice samples from the trained model`

## Checkpoints
`TODO: add trained checkpoints`

---

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
    - [X] EMA parameter updates
    - [ ] Learning variance (Section 3.1)
    - [ ] Class conditioning
- [ ] Training script
    - [X] Main script
    - [X] Simple loss computation
    - [ ] Hybrid loss computation
    - [ ] Auxiliary denoising loss (from D3PM)
- [ ] Sample generation
- [ ] Misc
    - [ ] Nice README :)
    - [ ] Nice docstrings
    - [ ] Nice Samples
    - [ ] PyPi library
    - [ ] Trained Checkpoints

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
