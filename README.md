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
    - [ ] Main model structure
    - [ ] Multi-headed self-attention at certain resolutions
    - [ ] Improved time embedding injection (Appendix A)
    - [ ] Learning variance (Section 3.1)
    - [ ] Class conditioning
    - [ ] EMA parameter updates
- [ ] Training script
    - [ ] Simple loss computation
    - [ ] Hybrid loss computation
    - [ ] Auxiliary denoising loss (from D3PM)
- [ ] Sample generation
