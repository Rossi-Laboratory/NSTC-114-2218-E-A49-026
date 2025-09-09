import time
import threading
import gradio as gr
import bpy
import warnings
import os
import yaml
from openai import OpenAI
import subprocess
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=UserWarning)

# Load environment variables from .env file in parent directory
load_dotenv(dotenv_path="../.env")

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
)

def generate_yaml_configuration(user_input):
    """Use ChatGPT to generate YAML config"""
    prompt = (
        f"The following are the system requirements:\n{user_input}\n\n"
        f"Please generate the corresponding configuration.yaml file content and ensure the following conditions:\n"
        f"- The file starts with `---` to indicate the beginning of a YAML file.\n"
        f"- Ensure all key-value pairs have valid values and proper indentation.\n"
        f"- Format example:\n"
        f"  ---\n"
        f"  key: value # example comment\n\n"
        f"Provide configuration based on requirements. If a parameter is not specified in the user's requirements, set it to the default value (no change).\n"
        f"I have written a Range for each parameter. If the user's requirement is not within that range, use the closest value within the range.\n"
        f"For example: If the user says they want four bedrooms, but bedroom_count range: 1-3, then bedroom_count in configuration.yaml should be 3\n"
        f"# 1. Shortest path to entrance\n"
        f"# * rg[Semantics.Hallway].all(\n"
        f"#   lambda r: rooms[Semantics.Entrance]\n"
        f"#   .related_to(r, cl.Traverse())\n"
        f"#   .count()\n"
        f"#   .in_range(0, 1, mean=0.8)\n"
        f"# )\n"
        f"hallway_entrance_mean: 0.8 # range: 0.1-1.0\n"
        f"# 2. Typical room areas\n"
        f"# rooms[Semantics.Kitchen].sum(\n"
        f"#   lambda r: (r.area() / 20).log().hinge(0, 0.4).pow(2)\n"
        f"# )\n"
        f"kitchen_area: 20 # range: 1 - 1000\n"
        f"bedroom_area: 40 # range: 1 - 1000\n"
        f"living_room_area: 40 # range: 1 - 1000\n"
        f"dining_room_area: 20 # range: 1 - 1000\n"
        f"bathroom_area: 8 # range: 1 - 1000\n"
        f"# 3. Room aspect ratios\n"
        f"# + sum(\n"
        f"#   rooms[tag].sum(lambda r: r.aspect_ratio().log())\n"
        f"#   for tag in [\n"
        f"#       Semantics.Kitchen,\n"
        f"#       Semantics.Bedroom,\n"
        f"#       Semantics.LivingRoom,\n"
        f"#       Semantics.DiningRoom,\n"
        f"#   ]\n"
        f"# ).minimize(weight=50.0)\n"
        f"aspect_ratio: 50.0 # range: 10-2000\n"
        f"# 4. Room convexity\n"
        f"# + rooms[-Semantics.Hallway]\n"
        f"# .sum(lambda r: r.convexity().log())\n"
        f"# .minimize(weight=5.0)\n"
        f"convexity: 5.0\n"
        f"# 5. Wall simplicity\n"
        f"# + rooms[-Semantics.Hallway]\n"
        f"# .sum(lambda r: (r.n_verts() - 6).clip(0).pow(1.5))\n"
        f"# .minimize(weight=1.0)\n"
        f"wall_simplicity: 1.0 # range: 1-2000\n"
        f"# 6. Room count\n"
        f"bedroom_count: 2 # range: 1-3\n"
        f"def public_bathroom_via_hallway(r):return (rooms[Semantics.Bathroom].related_to(rooms[Semantics.Hallway].related_to(r, cl.Traverse()), cl.Traverse()).count()>= variables[\"public_bathroom_via_hallway_count\"])\n"
        f"def public_bathroom_via_living_room(r):return (rooms[Semantics.Bathroom].related_to(rooms[Semantics.LivingRoom].related_to(r, cl.Traverse()), cl.Traverse()).count()>= variables[\"public_bathroom_via_living_room_count\"])\n"
        f"public_bathroom_via_hallway_count: 1 # range: 1-3\n"
        f"public_bathroom_via_living_room_count: 1 # range: 1-3\n\n"
        "Ensure the YAML format is correct and adapts to the requirement description. Please only respond with the YAML content, no text explanations, and no ```yaml markup."
    )

    response = client.chat.completions.create(model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Our project's purpose is to allow users to customize the overall layout and room layout of houses, focusing on spatial configuration (excluding objects and furniture placement). Users can modify layout constraints in home.py through prompt input, including hard constraints and soft constraints, to achieve specific layout requirements. Users will input specific requirements, for example: 'Generate a house with four bedrooms and two bathrooms, with bedrooms and bathrooms in the same area, and the dining room and kitchen connected.' You need to modify the parameters in configuration.yaml accordingly based on such requirements to accurately reflect the desired layout."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=300,
        temperature=0.8
    )
    yaml_content = response.choices[0].message.content

    # Validate generated YAML
    try:
        yaml.safe_load(yaml_content)
    except yaml.YAMLError as e:
        raise ValueError(f"Generated YAML is invalid: {e}")

    return yaml_content

def run_project():
    """Run the Infinigen project to generate floor layout"""
    print("Generating floor layout...")
    
    # Get the parent directory (infinigen root)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    infinigen_root = os.path.dirname(current_dir)
    
    command = (
        f"cd {infinigen_root} && "
        "python -W ignore -m infinigen_examples.generate_indoors "
        "--seed 0 --task coarse --output_folder outputs/indoors/coarse "
        "-g no_objects.gin overhead.gin "
        "-p compose_indoors.terrain_enabled=False"
    )

    try:
        subprocess.run(command, shell=True, check=True)
        print("Floor layout generated and saved as scene.blend")
    except subprocess.CalledProcessError as e:
        print(f"Infinigen crashed, but continuing. Error message: {e}")

def translate_blend_to_glb():
    """Convert .blend file to .glb for web viewing"""
    # Get the parent directory (infinigen root)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    infinigen_root = os.path.dirname(current_dir)
    
    blend_file = os.path.join(infinigen_root, "outputs/indoors/coarse/scene.blend")
    output_glb = os.path.join(infinigen_root, "outputs/indoors/coarse/output_scene.glb")

    if not os.path.exists(blend_file):
        return "Error: scene.blend was not generated."

    # Use Blender CLI to translate .blend to .glb
    blender_command = [
        "/Applications/Blender.app/Contents/MacOS/Blender",
        "--background",
        "--python-expr",
        f"""
import bpy
bpy.ops.wm.open_mainfile(filepath='{blend_file}')
bpy.ops.export_scene.gltf(filepath='{output_glb}', export_format='GLB')
bpy.ops.wm.quit_blender()
"""
    ]

    try:
        subprocess.run(blender_command, check=True)
        return output_glb
    except subprocess.CalledProcessError as e:
        return f"Error converting blend to glb: {str(e)}"

def generate_3d_model(user_input):
    """Main function to generate 3D model from user input"""
    # Generate YAML configuration using AI
    yaml_content = generate_yaml_configuration(user_input)
    
    # Save the generated YAML to the web_interface directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_file = os.path.join(current_dir, "configuration.yaml")
    with open(yaml_file, "w") as file:
        file.write(yaml_content)
    
    # Copy YAML to the constraints directory for Infinigen to use
    infinigen_root = os.path.dirname(current_dir)
    constraints_yaml = os.path.join(infinigen_root, "infinigen_examples/constraints/configuration.yaml")
    with open(constraints_yaml, "w") as file:
        file.write(yaml_content)
    
    # Run the project to generate the 3D scene
    run_project()

    # Convert to GLB and return path
    return translate_blend_to_glb()

# Define the Gradio interface
demo = gr.Interface(
    fn=generate_3d_model,
    inputs=gr.Textbox(
        lines=5,
        placeholder="Please enter your room layout requirements:\n"
                    "Example: I want a house with two bedrooms, each with a private bathroom, "
                    "a kitchen, and a dining room. The living room should be 1.3 times larger than the bedrooms, "
                    "and the bedrooms should be the same size as the kitchen and dining room.\n"
    ),
    outputs=gr.Model3D(scale=2),
    title="Infinigen Web Interface - 3D Indoor Layout Generator",
    description="Generate custom indoor layouts using natural language descriptions. "
                "The system uses AI to interpret your requirements and generate corresponding 3D floor plans. "
                "Maximum of 3 bedrooms and up to 10 total rooms."
)

# Launch the Gradio interface
if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    print("🏠 Starting Infinigen Web Interface...")
    print("📝 Make sure you have set up your environment correctly (see README.md)")
    print("🔑 Ensure your OpenAI API key is configured")
    print("🎨 Blender must be installed at /Applications/Blender.app/ (macOS)")
    print()
    demo.launch(share=True)
