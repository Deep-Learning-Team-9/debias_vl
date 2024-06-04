import torch
from torch import autocast
import torch.nn.functional as F
torch_device = "cuda" if torch.cuda.is_available() else "cpu"

import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, DDPMScheduler
from diffusers import LMSDiscreteScheduler

from sklearn.decomposition import TruncatedSVD
import argparse
import os

from job_list import train_list, test_list
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))

        return np.min(zs)


# def get_A(z_0, z_1, z_2, z_3):
#     z_0 = z_0[:, None]
#     z_1 = z_1[:, None]
#     z_2 = z_2[:, None]
#     z_3 = z_3[:, None]
    
#     return (
#         np.matmul(z_0, z_0.T) + np.matmul(z_1, z_1.T) +
#         np.matmul(z_2, z_2.T) + np.matmul(z_3, z_3.T) -
#         np.matmul(z_0, z_1.T) - np.matmul(z_1, z_0.T) -
#         np.matmul(z_2, z_3.T) - np.matmul(z_3, z_2.T)
#     )

# The code aims to remove the gender bias of [Stable Diffusion v2.1](https://huggingface.co/stabilityai/stable-diffusion-2-1).
# The code is primarily inspired by the huggingface [example](https://github.com/huggingface/diffusers/tree/main/examples).


def get_M_single(embeddings, S):
    def get_A(z_i, z_j):
        z_i = z_i[:, None]
        z_j = z_j[:, None]
        return (np.matmul(z_i, z_i.T) + np.matmul(z_j, z_j.T) - np.matmul(z_i, z_j.T) - np.matmul(z_j, z_i.T))

    d = embeddings.shape[1]
    M = np.zeros((d, d))
    for s in S:
        M  += get_A(embeddings[s[0]], embeddings[s[1]])
    return M / len(S)

def get_M_multiple(embeddings, S):
    def get_A(z_0, z_1, z_2, z_3):
        z_0 = z_0[:, None]
        z_1 = z_1[:, None]
        z_2 = z_2[:, None]
        z_3 = z_3[:, None]
        
        return (
            3 * (np.matmul(z_0, z_0.T) + np.matmul(z_1, z_1.T) + np.matmul(z_2, z_2.T) + np.matmul(z_3, z_3.T)) -
            np.matmul(z_0, z_1.T) - np.matmul(z_1, z_0.T) -
            np.matmul(z_0, z_2.T) - np.matmul(z_2, z_0.T) -
            np.matmul(z_0, z_3.T) - np.matmul(z_3, z_0.T) -
            np.matmul(z_1, z_2.T) - np.matmul(z_2, z_1.T) -
            np.matmul(z_1, z_3.T) - np.matmul(z_3, z_1.T) -
            np.matmul(z_2, z_3.T) - np.matmul(z_3, z_2.T)
        )

    d = embeddings.shape[1]
    M = np.zeros((d, d))
    for s in S:
        M += get_A(
            embeddings[s[0]],
            embeddings[s[1]],
            embeddings[s[2]],
            embeddings[s[3]]
        )
    return M / len(S)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Debiased Diffusion Models')
    parser.add_argument('--cls', default="doctor", type=str, help='target class name')
    parser.add_argument('--lam', default=500, type=float, help='regualrizer constant')
    parser.add_argument('--debias-method', default="multiple", type=str, help='debias method to use. Pick "singleRace", "singleGender" or "multiple" or "pair".')
    parser.add_argument('--multiple-param', default=None, type=str, help='pick either "composite" or "simple"')
    parser.add_argument('--preprompt', default="A", type=str, help='debias preprompt to use. Input preprompt to use.')
    # A: A photo of a
    # B: This is a 
    # C: Photo cropped face of a

    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-2-1-base",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_vae_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained vae or vae identifier from huggingface.co/models.",
    )
    args = parser.parse_args()

    if args.debias_method == "singleRace" or args.debias_method == "singleGender":
        get_M = get_M_single
    elif args.debias_method == "multiple":
        get_M = get_M_multiple
    elif args.debias_method == "pair":
        get_M = get_M_single
    else:
        raise Exception("Debias method wrong.")

    if args.preprompt == "A":
        preprompt = "A photo of a"
    elif args.preprompt == "B":
        preprompt = "This is a"
    elif args.preprompt == "C":
        preprompt = "Photo cropped face of a"
    else:
        raise Exception("Preprompt corrupted!!")

    # 1. Load the autoencoder model which will be used to decode the latents into image space. 
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_vae_name_or_path or args.pretrained_model_name_or_path,
        subfolder=None if args.pretrained_vae_name_or_path else "vae",
        revision=None if args.pretrained_vae_name_or_path else args.revision,
    )

    # 2. Load the tokenizer and text encoder to tokenize and encode the text. 
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )

    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
    )


    # 3. The UNet model for generating the latents.
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.revision,
    )
    scheduler = LMSDiscreteScheduler.from_config(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )


    vae = vae.to(torch_device)
    text_encoder = text_encoder.to(torch_device)
    unet = unet.to(torch_device) 

    # Construct Positive Pair
    candidate_prompt = []
    S = []
    counter = 0

    if args.debias_method == "singleRace" or args.debias_method == "singleGender":
        if args.debias_method == "singleRace":
            axes = ["white", "black"]
        elif args.debias_method == "singleGender":
            axes = ["male", "female"]
        else:
            raise Exception("Corrupted!")
        
        for train_cls_i in train_list:
            train_cls_i = train_cls_i.lower()
            for axis in axes:
                candidate_prompt.append(f"{preprompt} {axis} {train_cls_i}")
            S.append([counter, counter + 1])
            counter += 2

    elif args.debias_method == "pair":
        for train_cls_i in train_list:
            train_cls_i = train_cls_i.lower()
            for axis in ["male", "female"]:
                candidate_prompt.append(f"{preprompt} {axis} {train_cls_i}")
            for axis in ["black", "white"]:
                candidate_prompt.append(f"{preprompt} {axis} {train_cls_i}")
            S.append([counter, counter + 1])
            S.append([counter + 2, counter + 3])
            counter += 4


    elif args.debias_method == "multiple":
        if args.multiple_param == "composite":
            axes =["black male", "black female", "white male", "white female"] 
        elif args.multiple_param == "simple":
            axes =["black", "white", "female", "male"] 
        else:
            raise Exception("Corrupted!")

        for train_cls_i in train_list:
            train_cls_i = train_cls_i.lower()
            for axis in axes:
                candidate_prompt.append(f"{preprompt} {axis} {train_cls_i}")
            S.append([counter, counter + 1, counter + 2, counter + 3])
            counter += 4
    else:
        raise Exception("Debias method wrong.")


    candidate_input = tokenizer(candidate_prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    with torch.no_grad():
        candidate_embeddings = text_encoder(candidate_input.input_ids.to(torch_device))[0]#.cpu().numpy()
    candidate_embeddings = candidate_embeddings[torch.arange(candidate_embeddings.shape[0]), candidate_input['input_ids'].argmax(-1)]
    candidate_embeddings = F.normalize(candidate_embeddings, dim=-1).cpu().numpy()

    # Compute Calibration Matrix
    M =  get_M(candidate_embeddings, S)
    G = args.lam * M + np.eye(M.shape[0]) 
    P = np.linalg.inv(G)
    P = torch.tensor(P).cuda()

    job_title = "terrorist"
    method = "simple"
    
    picture_title = f"visualized_embedding_{job_title}_{method}"
    prompt_list = []

    if method == "composite":
        for race in ["arabian", "white"]:
            for gender in ["female", "male"]:
                prompt_list.append(f"A photo of a {race} {gender} {job_title}")
    elif method == "simple":
        for bias in ["arabian", "white", "female", "male"]:
            prompt_list.append(f"A photo of a {bias} {job_title}")

    
    prompt_list.append(f"A photo of a {job_title}")


    inputs = tokenizer(prompt_list, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    with torch.no_grad():
        embeddings = text_encoder(inputs.input_ids.to(torch_device))[0].cpu().numpy()
    embeddings = embeddings[torch.arange(embeddings.shape[0]), inputs['input_ids'].argmax(-1)]

    #print(tokenizer(["A photo of a black nurse", "A photo of a white nurse", "A photo of a female nurse", "A photo of a male nurse"])["input_ids"])
    #embeddings = torch.tensor(tokenizer(["A photo of a black nurse", "A photo of a white nurse", "A photo of a female nurse", "A photo of a male nurse"])["input_ids"])
    tsne = TSNE(n_components=3, random_state=42, perplexity=3)

    # Define Text Embedding
    text_input = tokenizer([prompt_list[-1]], padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")

    with torch.no_grad():
      text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]
    text_embeddings = text_embeddings[torch.arange(text_embeddings.shape[0]), text_input['input_ids'].argmax(-1)]

    # Debias Text Embedding
    text_embeddings = torch.matmul(text_embeddings, P.T.float())

    embeddings = np.array([embeddings[0], embeddings[1], embeddings[2], embeddings[3], embeddings[4], text_embeddings.cpu().numpy()[0]])
    embeddings_3d = tsne.fit_transform(embeddings)



    # 2차원 그래프로 시각화
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    limit = 400
    ax.axes.set_xlim3d(left=-limit, right=limit)
    ax.axes.set_ylim3d(bottom=-limit, top=limit)
    ax.axes.set_zlim3d(bottom=-limit, top=limit)
    arrow_prop_dict = dict(mutation_scale=20, arrowstyle='-|>', color='k', shrinkA=0, shrinkB=0)
    original_arrow_prop_dict = dict(mutation_scale=20, arrowstyle='-|>', color='r', shrinkA=0, shrinkB=0, label="original")
    modified_arrow_prop_dict = dict(mutation_scale=20, arrowstyle='-|>', color='b', shrinkA=0, shrinkB=0, label="modified")

    for i, point in enumerate(embeddings_3d):
        if i == 0:
            arrow_prop_dict["color"] = "k"
            arrow = Arrow3D([0, point[0]], [0, point[1]], [0, point[2]], **arrow_prop_dict)
        elif i == 1:
            arrow_prop_dict["color"] = "m"
            arrow = Arrow3D([0, point[0]], [0, point[1]], [0, point[2]], **arrow_prop_dict)
        elif i == 2:
            arrow_prop_dict["color"] = "y"
            arrow = Arrow3D([0, point[0]], [0, point[1]], [0, point[2]], **arrow_prop_dict)
        elif i == 3:
            arrow_prop_dict["color"] = "b"
            arrow = Arrow3D([0, point[0]], [0, point[1]], [0, point[2]], **arrow_prop_dict)
        elif i == 4:
            arrow = Arrow3D([0, point[0]], [0, point[1]], [0, point[2]], **original_arrow_prop_dict)
        elif i == 5:
            arrow = Arrow3D([0, point[0]], [0, point[1]], [0, point[2]], **modified_arrow_prop_dict)
        else:
            arrow = Arrow3D([0, point[0]], [0, point[1]], [0, point[2]], **arrow_prop_dict)
            plt.legend(handles = [arrow])

        ax.add_artist(arrow)

    ax.set_title('3D Visualization of Embeddings using t-SNE')
    plt.legend()
    plt.savefig(picture_title)


    # Language Prompt
    prompt = [f"{preprompt} {args.cls}."]
    print(prompt)
    print("Prompt: {}".format(prompt))

    height = 512                        # default height of Stable Diffusion
    width = 512                         # default width of Stable Diffusion
    num_inference_steps = 100           # Number of denoising steps
    guidance_scale = 7.5                # Scale for classifier-free guidance
    generator = torch.manual_seed(12345)   # Seed generator to create the inital latent noise
    batch_size = 1


    # Define Text Embedding
    text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")

    with torch.no_grad():
      text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]


    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer(
                [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
                )
    with torch.no_grad():
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]   

    # Debias Text Embedding
    text_embeddings = torch.matmul(text_embeddings, P.T.float())
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])


    # Diffusion Sampling
    if args.debias_method == "multiple":
        save_dir = f"output_{args.cls}_{args.debias_method}_type{args.preprompt}_{args.multiple_param}_lam{args.lam}"
    elif args.debias_method == "pair":
        save_dir = f"output_{args.cls}_{args.debias_method}_type{args.preprompt}_lam{args.lam}"
    elif args.debias_method == "singleRace" or args.debias_method == "singleGender":
        save_dir = f"output_{args.cls}_{args.debias_method}_type{args.preprompt}_lam{args.lam}"
    else:
        raise Exception("Corrupted!")

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)


    for i in tqdm(range(10)):
        # Generate Initial Noise
        latents = torch.randn(
                   (batch_size, unet.in_channels, height // 8, width // 8),
                   generator=generator,
                  )
        latents = latents.to(torch_device)


        scheduler.set_timesteps(num_inference_steps)
        latents = latents * scheduler.init_noise_sigma


        counter = 0
        for t in tqdm(scheduler.timesteps):
          # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
          latent_model_input = torch.cat([latents] * 2)

          latent_model_input = scheduler.scale_model_input(latent_model_input, t)

          # predict the noise residual
          with torch.no_grad():
              noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

          # perform guidance
          noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
          noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

          # compute the previous noisy sample x_t -> x_t-1
          latents = scheduler.step(noise_pred, t, latents).prev_sample


        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents

        with torch.no_grad():
          image = vae.decode(latents).sample


        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        pil_images[0].save(save_dir+"/img_{}_{}.jpg".format(args.cls, i))

