# Mitigating multi-bias in Vision-Language Model with embedding vector redirection

Team9. Taehyeok Ha, Kyoosik Lee, Hyeonbeen Park 

This repository is extended research of [Debiasing Vision-Language Models via Biased Prompts](https://github.com/chingyaoc/debias_vl)

## Environment

The code has only been tested with the below environment and this can set by `requirements.txt`

- python=3.6
- torch=1.10.1
- PIL
- diffusers
- scikit-learn
- clip
- transformers

## Procedure

1. Create a python virtual environment using `venv` in current folder, e.g., `python -m venv myenv`

2. Activate the virtual environment in current folder `source myenv/bin/activate`

3. Install the specified packages, run `pip install -r requirements.txt`

4. Move to `./generative`

5. Run following or user customized commend, below commends is example

```sh
    python main.py --cls Nurse --debias-method singleGender --lam 0 --preprompt A
    python main.py --cls Florist --debias-method singleRace --lam 500 --preprompt A
    python main.py --cls HollywoodActor --debias-method singleGender --lam 500 --preprompt A
    python main.py --cls Nurse --debias-method pair --lam 500 --preprompt A
    python main.py --cls Florist --debias-method multiple --multiple-param simple --lam 500 --preprompt A
    python main.py --cls HollywoodActor --debias-method multiple --multiple-param composite --lam 500 --preprompt A
```

- This is detail explanation of flags
  - `--cls`: Select the target class that user what to test for any job
    - e.g., Florist, HollywoodActor, Doctor
  - `--lam`: Hyperparameter lambda of debiasing algorithm
    - high lambda value means high level debiasing
  - `--debias_method`: Choice multi-vector or single-vector
    - `singleRace`
    - `singleGender`
    - `multiple --multiple-param composite`
    -  `multiple --multiple-param simple`
    - `pair`
  - `--preprompt`: Type of prompt senteces
    - `A`: "A photo of a"
    - `B`: "This is a"
    - `C`: "Photo cropped face of a"

6. Experiments results(Image, ) will be shown in generative folder

## Research

### Introduction

abc

The code aims to remove the gender bias of [Stable Diffusion v2.1](https://huggingface.co/stabilityai/stable-diffusion-2-1).


### Related Work

abc

<center><img src="https://github.com/Deep-Learning-Team-9/debias_vl/assets/105369662/094b1181-62bd-4184-a3d6-bd2497dcab2d" width="70%" height="70%"/></center>

### Experiments

abc

### Results


<center><img src="https://github.com/Deep-Learning-Team-9/debias_vl/assets/105369662/01d16f5b-eb22-44c4-83e9-e394cfe55f26" width="70%" height="70%"/></center>
<center><img src="https://github.com/Deep-Learning-Team-9/debias_vl/assets/105369662/46697a20-26d2-43cb-99e1-9dde937c545e" width="70%" height="70%"/></center>
<center><img src="https://github.com/Deep-Learning-Team-9/debias_vl/assets/105369662/f16f371d-6da7-412b-86be-a32524b0280d" width="70%" height="70%"/></center>


<center><img src="https://github.com/Deep-Learning-Team-9/debias_vl/assets/105369662/0da0a58c-2a67-4e2e-86db-1ccebff451d9" width="70%" height="70%"/></center>






### Conclusions

We present an extended method for removing multi-bias method

1. In general, our method remove the bias for gender and race at the same time
2. Visualize the embedding vector space and explain why the quality of the image is lower

than before
