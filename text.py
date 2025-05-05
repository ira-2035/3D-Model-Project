import torch
from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import decode_latent_mesh

def process_text(prompt, output_model_path):
    # Load Shap-E model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    xm = load_model('transmitter', device=device)
    model = load_model('text300M', device=device)
    diffusion = diffusion_from_config(load_config('diffusion'))

    # Generate 3D model from text prompt
    latent = sample_latents(
        batch_size=1,
        model=model,
        diffusion=diffusion,
        guidance_scale=30.0,
        model_kwargs=dict(texts=[prompt]),
        progress=True,
        clip_denoised=True,
        use_fp16=True,
        use_karras=True,
        karras_steps=64,
        sigma_min=1e-3,
        sigma_max=160,
        s_churn=0,
    )

    # Decode and save 3D model
    mesh = decode_latent_mesh(xm, latent).tri_mesh()
    mesh.export(output_model_path)

    # Visualize the 3D model
    scene = pyrender.Scene()
    mesh_node = pyrender.Mesh.from_trimesh(mesh)
    scene.add(mesh_node)
    viewer = pyrender.Viewer(scene, use_raymond_lighting=True)

process_text("A small toy car", "output_model.obj")
