# **Feedpak Builder Pipeline**

This pipeline processes stems (audio files) and Guitar Pro (.gp5) files, running them through transcription, pitch tracking, and dynamic time warping (DTW) to generate a packaged .feedpak archive.

## **1\. Environment & Setup**

Because this pipeline relies on machine learning models (WhisperX, CREPE) alongside audio processing tools, setting up the environment requires a few prerequisites.

### **System Requirements**

> * **Windows Developer Mode:** Developer Mode should be enabled in Windows.  
> * **CUDA / GPU:** A CUDA-compatible NVIDIA GPU is highly recommended. The scripts default to \--device cuda.

### **Step 1: Install FFmpeg via winget**

FFmpeg is required for audio processing operations. Install it easily via Windows Package Manager:

> 1. Open PowerShell and run:  
>    PowerShell  
>    winget install Gyan.FFmpeg

> 2. Close and reopen PowerShell, then run ffmpeg \-version to confirm the installation works.

### **Step 2: Hugging Face Token & Gated Model Access**

process\_vocals.py uses whisperx for speaker diarization, which relies on gated Pyannote models hosted on Hugging Face.

> 1. Create or log in to your account at [Hugging Face](https://huggingface.co/).  
> 2. **Accept Model Conditions (Crucial):** Visit each of the following gated model pages and accept their user conditions:  
   * [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)  
   * [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)  
> 3. Generate an Access Token via **User Settings $\\rightarrow$ Access Tokens** (a *Read* token is sufficient).  
> 4. Save this token to pass into your environment later.

### **Step 3: Create & Activate a Virtual Environment**

Open PowerShell, navigate to your project directory, and initialize a virtual environment:

PowerShell  
\# Create virtual environment named 'env'  
python \-m venv env

\# Activate the virtual environment  
.\\env\\Scripts\\Activate.ps1

**Note:** If PowerShell throws a script execution error, run Set-ExecutionPolicy \-ExecutionPolicy RemoteSigned \-Scope Process in your session first.

### **Step 4: Install CUDA-Enabled PyTorch & Dependencies**

Standard pip install torch often installs CPU-only binaries. Install PyTorch with explicit CUDA support first, then install the remaining project dependencies:

PowerShell  
\# 1\. Install PyTorch with CUDA 12.4 support  
pip install torch torchvision torchaudio \--index-url https://download.pytorch.org/whl/cu124

\# 2\. Install ML & Vocal dependencies  
pip install whisperx torchcrepe pyannote.audio

\# 3\. Install GP5 parsing & Audio/DTW dependencies  
pip install pyguitarpro librosa scipy numpy

\# 4\. Install Utility & Manifest dependencies  
pip install jsonschema pyyaml soundfile

### **Step 5: Verify CUDA Availability**

Run this quick snippet to ensure PyTorch recognizes your GPU:

PowerShell  
python \-c "import torch; print('CUDA Available:', torch.cuda.is\_available()); print('GPU:', torch.cuda.get\_device\_name(0) if torch.cuda.is\_available() else 'None')"

> * **Expected Output:** CUDA Available: True and the name of your NVIDIA GPU.

## **2\. Setting Environment Variables**

The pipeline expects your Hugging Face access token to be set prior to execution. How you supply this depends on your terminal environment:

> * **PowerShell:**  
>   PowerShell  
>   $env:HF\_TOKEN="hf\_your\_actual\_token\_here"

> * **Command Prompt (CMD):**  
>   DOS  
>   set HF\_TOKEN=your\_token\_here

## **3\. Expected Folder Structure**

Your input directory should contain your target song folder with the required audio stems, Guitar Pro file, and metadata.

Plaintext  
my\_test\_song\\  
├── song.gp5              \# Requires exactly one .gp5 file (or pyguitarpro supported format)  
├── vocals.ogg            \# Required for vocals/pitch processing  
├── drums.ogg             \# Required for DTW timing alignment  
├── bass.ogg              \# Optional  
├── guitar.ogg            \# Optional  
├── full.ogg              \# Optional (Full mixdown)  
└── metadata.json         \# Required for manifest generation

*Note: Stems can be in .ogg, .wav, or .flac format.*

### **Sample metadata.json**

Place a metadata.json file inside your song folder so the builder can generate the final manifest:

JSON  
{  
  "title": "Test Song",  
  "artist": "Test Artist",  
  "album": "Test Album",  
  "year": 2024,  
  "genres": \["test"\]  
}

## **4\. Running the Pipeline**

### **Automatic Processing (Recommended)**

The build\_feedpak.py script acts as an orchestrator. It automatically executes process\_vocals.py and process\_gp\_alignment.py as subprocesses, converts their JSON outputs, generates the manifest.yaml, and packages everything into a .feedpak archive.  
**In PowerShell:**

PowerShell  
python build\_feedpak.py my\_test\_song output\_folder \--device cuda \--hf-token $env:HF\_TOKEN

**In Command Prompt (CMD):**

DOS  
python build\_feedpak.py my\_test\_song output\_folder \--device cuda \--hf-token %HF\_TOKEN%

### **Manual Step-by-Step Processing (Debugging)**

If you need to debug specific scripts individually, you can run them manually:  
**1\. Process Vocals (Script 1):**

PowerShell  
python process\_vocals.py my\_test\_song\\vocals.ogg \--out my\_test\_song\\intermediate\_vocals.json \--device cuda \--hf-token $env:HF\_TOKEN

**2\. Process Guitar Pro Alignment (Script 2):**

PowerShell  
python process\_gp\_alignment.py my\_test\_song\\song.gp5 my\_test\_song\\drums.ogg \--out my\_test\_song\\intermediate\_arrangements.json

**3\. Package Final Archive (Script 3):**

PowerShell  
python build\_feedpak.py my\_test\_song output\_folder \--device cuda \--hf-token $env:HF\_TOKEN  
