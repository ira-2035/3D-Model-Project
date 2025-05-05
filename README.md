# 3D-Model-Project
This prototype allows you to generate simple 3D mesh models (.obj or .stl) from either a photo (with a single object) or a short text prompt (e.g., "a small toy car"). It demonstrates practical AI integration using open-source models for 3D generation.
🖼️ Image Input: Accepts .jpg or .png of a single object, removes the background, and generates a basic 3D model.

✍️ Text Prompt: Uses a pre-trained open-source model to convert descriptive text into a 3D mesh.

💾 Outputs: Downloads a 3D file (.obj) and optionally visualizes it using a 3D viewer.

🧠 Models Used: OpenAI’s Shap-E for both image- and text-based 3D generation.
1. Clone & Setup Environment
bash
Copy
Edit
git clone https://github.com/openai/shap-e.git
cd shap-e
pip install -e .

cd ..
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
2. Install Requirements
bash
Copy
Edit
pip install torch rembg trimesh pyrender matplotlib
🖼️ Usage
🔹 Convert Image to 3D Model
python
Copy
Edit
from image_to_3d import process_image
process_image("input_image.jpg", "outputs/model_from_image.obj")
🔹 Convert Text Prompt to 3D Model
python
Copy
Edit
from text_to_3d import process_text
process_text("A red toy car", "outputs/model_from_text.obj")
📦 Output
.obj file of the generated mesh

Optional real-time 3D visualization using PyRender

📚 Dependencies
torch

rembg – Background removal

trimesh – 3D mesh handling

pyrender – 3D visualization

Shap-E – Text/image to 3D model generation

🧠 Thought Process
Image Preprocessing: Rembg is used for accurate background removal.

Text-to-3D: Leveraged OpenAI's Shap-E model to avoid building a 3D generator from scratch.

Mesh Handling: Used trimesh to export and optionally view models.

Visualization: Optional, but helpful for validation.
