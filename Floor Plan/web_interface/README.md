# Infinigen Web Interface

A user-friendly web interface for generating custom indoor layouts using natural language descriptions.

## Features

- **Natural Language Input**: Describe your desired layout in English or Chinese
- **AI-Powered Configuration**: Automatic YAML generation based on your requirements  
- **Real-time 3D Preview**: View generated layouts directly in your browser
- **Customizable Parameters**: Support for room count, size, connectivity, and spatial relationships

## Prerequisites

### System Requirements
- **Python 3.10+**
- **Blender** (for 3D model conversion)
- **OpenAI API Key** (configured in .env file)
- **Complete Infinigen installation** (see Installation.md)

## Environment Setup

### 1. Install Infinigen (Required)
First, follow the complete Infinigen installation guide from [Installation.md](../docs/Installation.md):

```bash
# Clone the repository (v1.0.0)
git clone --branch v1.0.0 https://github.com/princeton-vl/infinigen.git
cd infinigen

# Create conda environment
conda create --name infinigen python=3.10
conda activate infinigen

# Install system dependencies (see Installation.md for platform-specific commands)
# Ubuntu example:
sudo apt-get install wget cmake g++ libgles2-mesa-dev libglew-dev libglfw3-dev libglm-dev zlib1g-dev

# Install Infinigen with all dependencies (including web interface dependencies)
pip install -e ".[dev]"
```

### 2. Configure OpenAI API Key
Copy the example environment file and configure your API key:
```bash
cp .env.example .env
```

Edit `.env` and replace with your actual API key:
```bash
OPENAI_API_KEY="your-openai-api-key-here"
OPENAI_BASE_URL="https://api.openai.com/v1"
```

## Usage

### Quick Start
1. **Navigate to Infinigen root directory**:
   ```bash
   cd /path/to/infinigen
   ```

2. **Launch the web interface**:
   ```bash
   python web_interface/app.py
   ```

3. **Access the interface**:
   - Open your browser and go to the displayed URL (typically `http://127.0.0.1:7860`)
   - The interface will also provide a public URL for sharing

### Using the Interface

1. **Enter Requirements**: Describe your desired room layout in the text box
2. **Generate**: Click submit to process your request
3. **View Results**: The 3D model will appear in the viewer once generation is complete

### Example Inputs

**English:**
```
I want a house with two bedrooms, each with a private bathroom, a kitchen, 
and a dining room. The living room should be 1.3 times larger than the bedrooms, 
and the bedrooms should be the same size as the kitchen and dining room.
```

## Configuration Parameters

The system supports various layout parameters:

- **Room Areas**: Kitchen, bedroom, living room, dining room, bathroom (1-1000 sqm)
- **Room Count**: Bedrooms (1-3), bathrooms (1-3)
- **Spatial Relationships**: Connectivity between rooms
- **Layout Constraints**: Aspect ratios, convexity, wall simplicity

## Output Files

Generated files are saved to:
- **3D Scene**: `outputs/indoors/coarse/scene.blend`
- **Web Model**: `outputs/indoors/coarse/output_scene.glb`
- **Configuration**: `web_interface/configuration.yaml`

## Development

### Modifying the Interface
- Edit `app.py` to customize the Gradio interface
- Update prompts in `generate_yaml_configuration()` for different AI behavior
- Adjust Infinigen parameters in `run_project()`


## License

This web interface follows the same license as the main Infinigen project.
