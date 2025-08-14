import streamlit as st
import requests
import json
import tempfile
import mido
from mido import Message, MidiFile, MidiTrack, MetaMessage
import re
from pathlib import Path
from io import BytesIO
import torch
import torch.nn as nn
from transformers import T5Tokenizer
from model.transformer_model import Transformer
from model.remi_tokenizer import REMITokenizer
from huggingface_hub import hf_hub_download
import pickle
import os
import numpy as np


# Streamlit UI Configuration
st.set_page_config(
    page_title="Trumpet MIDI Generator",
    page_icon="🎺",
    layout="wide",  # full-width layout for better visual space
    initial_sidebar_state="expanded"
)

# Try to import audio libraries with fallback
try:
    from midi2audio import FluidSynth
    from pydub import AudioSegment
    AUDIO_ENABLED = True
except ImportError:
    AUDIO_ENABLED = False
    st.warning("Audio libraries not available. MIDI playback will be disabled.")

# Configuration
OLLAMA_URL = "http://localhost:11434"
HUGGINGFACE_REPO = "amaai-lab/text2midi"

# ----------------------------------------
# MIDI Utility Functions
# ----------------------------------------
TEMPO = 120

NOTE_MAP = {
    "C": 0, "C#": 1, "DB": 1,
    "D": 2, "D#": 3, "EB": 3,
    "E": 4, "F": 5, "F#": 6, "GB": 6,
    "G": 7, "G#": 8, "AB": 8,
    "A": 9, "A#": 10, "BB": 10,
    "B": 11
}

# Trumpet-specific note ranges
TRUMPET_RANGE = {
    'low': 52,   # E3
    'high': 84,  # C6
    'comfortable_low': 58,  # Bb3
    'comfortable_high': 77   # F5
}

def note_name_to_midi(note):
    """Convert note name to MIDI number"""
    match = re.match(r"([A-Ga-g][#b]?)(\d)", note)
    if not match:
        raise ValueError(f"Invalid note: {note}")
    pitch, octave = match.groups()
    pitch = pitch.upper().replace('b', 'B')
    if pitch not in NOTE_MAP:
        raise ValueError(f"Invalid pitch: {pitch}")
    midi_number = NOTE_MAP[pitch] + (int(octave) + 1) * 12
    return midi_number

def midi_to_note_name(midi_note):
    """Convert MIDI number to note name"""
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (midi_note // 12) - 1
    note = note_names[midi_note % 12]
    return f"{note}{octave}"

def filter_notes_for_trumpet(midi_notes, range_type='full'):
    """Filter MIDI notes to trumpet range"""
    if range_type == 'comfortable':
        low, high = TRUMPET_RANGE['comfortable_low'], TRUMPET_RANGE['comfortable_high']
    else:
        low, high = TRUMPET_RANGE['low'], TRUMPET_RANGE['high']

    return [note for note in midi_notes if low <= note <= high]

def json_to_midi(json_data, instrument=56, tempo=TEMPO):
    """Convert JSON format to MIDI file"""
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    # Add tempo
    track.append(MetaMessage('set_tempo', tempo=mido.bpm2tempo(tempo), time=0))
    # Set instrument (program change)
    track.append(Message('program_change', program=instrument, time=0))

    ticks_per_beat = mid.ticks_per_beat
    for note, duration in json_data:
        try:
            note_num = note_name_to_midi(note)
            # Filter to trumpet range
            if TRUMPET_RANGE['low'] <= note_num <= TRUMPET_RANGE['high']:
                ticks = int(duration * ticks_per_beat)
                track.append(Message('note_on', note=note_num, velocity=64, time=0))
                track.append(Message('note_off', note=note_num, velocity=64, time=ticks))
            else:
                st.warning(f"Note {note} (MIDI {note_num}) is outside trumpet range, skipping.")
        except Exception as e:
            st.error(f"Error parsing note {note}: {e}")
            continue

    return mid

def midi_to_mp3(midi_obj, soundfont_path=None):
    """Convert MIDI to MP3 audio"""
    if not AUDIO_ENABLED:
        return None

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as mid_file:
        midi_obj.save(mid_file.name)
        wav_path = mid_file.name.replace(".mid", ".wav")
        mp3_path = mid_file.name.replace(".mid", ".mp3")

    try:
        # Use provided soundfont or default
        if soundfont_path and os.path.exists(soundfont_path):
            fs = FluidSynth(soundfont_path)
        else:
            # Try to use a default soundfont
            fs = FluidSynth()

        fs.midi_to_audio(mid_file.name, wav_path)
        sound = AudioSegment.from_wav(wav_path)
        sound.export(mp3_path, format="mp3")

        with open(mp3_path, "rb") as f:
            return BytesIO(f.read())
    except Exception as e:
        st.error(f"Audio conversion failed: {e}")
        return None
    finally:
        # Cleanup
        for path in [mid_file.name, wav_path, mp3_path]:
            if os.path.exists(path):
                os.unlink(path)

# ----------------------------------------
# Text-to-MIDI Model Functions
# ----------------------------------------

@st.cache_resource
def load_text2midi_model():
    """Load the pre-trained text-to-MIDI model"""
    try:
        # Determine device
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'

        st.info(f"Loading model on device: {device}")

        # Download model files
        model_path = hf_hub_download(repo_id=HUGGINGFACE_REPO, filename="pytorch_model.bin")
        tokenizer_path = hf_hub_download(repo_id=HUGGINGFACE_REPO, filename="vocab_remi.pkl")

        # Load the REMI tokenizer
        with open(tokenizer_path, "rb") as f:
            r_tokenizer = pickle.load(f)

        # Create model
        vocab_size = len(r_tokenizer) if hasattr(r_tokenizer, '__len__') else 5000
        model = Transformer(vocab_size, 768, 8, 2048, 18, 1024, False, 8, device=device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        # Load T5 tokenizer
        t5_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")

        return model, r_tokenizer, t5_tokenizer, device
    except Exception as e:
        st.error(f"Failed to load text-to-MIDI model: {e}")
        return None, None, None, None

def generate_midi_from_text(prompt, model, r_tokenizer, t5_tokenizer, device, max_len=2000, temperature=1.0):
    """Generate MIDI from text prompt using the pre-trained model"""
    try:
        # Tokenize input text
        inputs = t5_tokenizer(prompt, return_tensors='pt', padding=True, truncation=True)
        input_ids = nn.utils.rnn.pad_sequence(inputs.input_ids, batch_first=True, padding_value=0)
        input_ids = input_ids.to(device)
        attention_mask = nn.utils.rnn.pad_sequence(inputs.attention_mask, batch_first=True, padding_value=0)
        attention_mask = attention_mask.to(device)

        # Generate MIDI tokens
        with torch.no_grad():
            output = model.generate(input_ids, attention_mask, max_len=max_len, temperature=temperature)
            output_list = output[0].tolist()

        # Decode to MIDI
        if hasattr(r_tokenizer, 'decode'):
            generated_midi = r_tokenizer.decode(output_list)
        else:
            # Fallback: create a simple MIDI file
            generated_midi = create_fallback_midi(output_list)

        return generated_midi
    except Exception as e:
        st.error(f"MIDI generation failed: {e}")
        return None

def create_fallback_midi(tokens):
    """Create a fallback MIDI file when tokenizer decode fails"""
    midi_file = MidiFile()
    track = MidiTrack()
    midi_file.tracks.append(track)

    # Add basic setup
    track.append(MetaMessage('set_tempo', tempo=mido.bpm2tempo(120), time=0))
    track.append(Message('program_change', program=56, time=0))  # Trumpet

    # Convert tokens to notes (simplified)
    ticks_per_beat = midi_file.ticks_per_beat
    current_time = 0

    for i, token in enumerate(tokens[:50]):  # Limit to first 50 tokens
        note = 60 + (token % 25)  # Map to reasonable note range
        if TRUMPET_RANGE['low'] <= note <= TRUMPET_RANGE['high']:
            duration = ticks_per_beat // 4  # Quarter note duration
            track.append(Message('note_on', note=note, velocity=64, time=current_time))
            track.append(Message('note_off', note=note, velocity=64, time=duration))
            current_time = 0

    return midi_file

# ----------------------------------------
# Ollama Model Functions
# ----------------------------------------

def get_ollama_models():
    """Get available Ollama models"""
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        response.raise_for_status()
        return [m['name'] for m in response.json().get("models", [])]
    except:
        return []

def query_ollama_model(model_name, user_prompt):
    """Query Ollama model for JSON-based MIDI generation"""
    system_prompt = """
You are a music assistant that generates monophonic trumpet exercises using a specific JSON format.

Each exercise must be output in the following structure:

[
  ["C4", 0.5],
  ["D4", 0.25],
  ["E4", 1]
]

Format rules:
- Each element is a pair: [note_name, duration]
- Use standard English note names (e.g., "Bb3", "F#4", "C5")
- The notes must be monophonic (one at a time)
- Use only notes playable on a standard Bb trumpet (range: E3 to C6)
- Duration values in beats: 0.125 = eighth, 0.25 = quarter, 0.5 = half, 1 = whole, 2 = breve
- Focus on trumpet-appropriate exercises with proper valve combinations

Only output the JSON array. Do not include explanation, markdown, or other formatting.
"""

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "stream": False
    }

    try:
        response = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=30)
        response.raise_for_status()
        response_json = response.json()
        return response_json["message"]["content"]
    except Exception as e:
        raise Exception(f"Ollama query failed: {e}")

def safe_parse_json(text):
    """Safely parse JSON from model output"""
    try:
        # Try to extract JSON array from text
        match = re.search(r"\[\s*\[.*?\]\s*\]", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return json.loads(text)
    except Exception as e:
        st.error(f"JSON parsing error: {e}")
        return None

# ----------------------------------------
# Streamlit UI
# ----------------------------------------

def main():
    st.set_page_config(
        page_title="Trumpet MIDI Generator",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🎺 Advanced Trumpet MIDI Generator")
    st.markdown("Generate trumpet exercises and melodies using multiple AI approaches")

    # Sidebar configuration
    st.sidebar.header("Configuration")

    # Generation method selection
    generation_method = st.sidebar.selectbox(
        "Generation Method",
        ["Text-to-MIDI (Hugging Face)", "JSON-based (Ollama)", "Both"]
    )

    # Trumpet-specific settings
    st.sidebar.subheader("Trumpet Settings")
    note_range = st.sidebar.selectbox(
        "Note Range",
        ["Full Range (E3-C6)", "Comfortable Range (Bb3-F5)"],
        index=1
    )

    default_tempo = st.sidebar.slider("Tempo (BPM)", 60, 200, 120)

    # Audio settings
    if AUDIO_ENABLED:
        enable_audio = st.sidebar.checkbox("Enable Audio Playback", True)
        soundfont_file = st.sidebar.file_uploader(
            "Upload Soundfont (.sf2) - Optional",
            type=['sf2'],
            help="Upload a custom soundfont for better trumpet sound"
        )
    else:
        enable_audio = False
        soundfont_file = None

    # Main content area
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("Input")

        # Prompt suggestions
        trumpet_prompts = [
            "A simple warm-up exercise for trumpet with ascending scales",
            "A beginner trumpet exercise focusing on stepwise motion",
            "A trumpet arpeggio exercise in the key of Bb major",
            "A valve combination practice for trumpet with repeated notes",
            "A trumpet exercise with rhythmic variation and syncopation",
            "A melodic trumpet phrase with expressive dynamics",
            "A technical etude for intermediate trumpet players",
            "A jazz-style trumpet lick with blue notes",
            "A classical trumpet fanfare in the key of C major",
            "A lyrical trumpet melody suitable for solo performance"
        ]

        selected_prompt = st.selectbox(
            "Choose a suggested prompt:",
            trumpet_prompts,
            index=0
        )

        user_prompt = st.text_area(
            "Or enter your own prompt:",
            value=selected_prompt,
            height=100,
            help="Describe the type of trumpet exercise or melody you want to generate"
        )

        # Advanced settings
        with st.expander("Advanced Settings"):
            if generation_method in ["Text-to-MIDI (Hugging Face)", "Both"]:
                max_length = st.slider("Max MIDI Length", 500, 3000, 1500)
                temperature = st.slider("Temperature", 0.1, 2.0, 1.0, 0.1)

            if generation_method in ["JSON-based (Ollama)", "Both"]:
                # Ollama model selection
                ollama_models = get_ollama_models()
                if ollama_models:
                    selected_ollama_models = st.multiselect(
                        "Select Ollama Models:",
                        ollama_models,
                        default=ollama_models[:1] if ollama_models else []
                    )
                else:
                    st.warning("No Ollama models available. Make sure Ollama is running.")
                    selected_ollama_models = []

        # Generate button
        if st.button("🎵 Generate Trumpet MIDI", type="primary"):
            if not user_prompt.strip():
                st.error("Please enter a prompt")
                return

            with col2:
                st.header("Generated Results")

                # Text-to-MIDI Generation
                if generation_method in ["Text-to-MIDI (Hugging Face)", "Both"]:
                    st.subheader("🤗 Text-to-MIDI Model Results")

                    with st.spinner("Loading text-to-MIDI model..."):
                        model, r_tokenizer, t5_tokenizer, device = load_text2midi_model()

                    if model is not None:
                        with st.spinner("Generating MIDI from text..."):
                            try:
                                generated_midi = generate_midi_from_text(
                                    user_prompt, model, r_tokenizer, t5_tokenizer, device,
                                    max_len=max_length if 'max_length' in locals() else 1500,
                                    temperature=temperature if 'temperature' in locals() else 1.0
                                )

                                if generated_midi:
                                    # Display MIDI info
                                    st.success("MIDI generated successfully!")

                                    # Create download button
                                    midi_bytes = BytesIO()
                                    generated_midi.save(midi_bytes)
                                    midi_bytes.seek(0)

                                    st.download_button(
                                        "📁 Download MIDI",
                                        midi_bytes.read(),
                                        "generated_trumpet.mid",
                                        "audio/midi"
                                    )

                                    # Audio playback
                                    if enable_audio and AUDIO_ENABLED:
                                        with st.spinner("Converting to audio..."):
                                            soundfont_path = None
                                            if soundfont_file:
                                                # Save uploaded soundfont temporarily
                                                with tempfile.NamedTemporaryFile(delete=False, suffix=".sf2") as tmp:
                                                    tmp.write(soundfont_file.read())
                                                    soundfont_path = tmp.name

                                            audio_data = midi_to_mp3(generated_midi, soundfont_path)
                                            if audio_data:
                                                st.audio(audio_data, format="audio/mp3")

                                            # Cleanup
                                            if soundfont_path and os.path.exists(soundfont_path):
                                                os.unlink(soundfont_path)

                                    # Display MIDI analysis
                                    analyze_midi(generated_midi, "Text-to-MIDI")

                            except Exception as e:
                                st.error(f"Failed to generate MIDI: {e}")
                    else:
                        st.error("Failed to load text-to-MIDI model")

                # JSON-based Generation (Ollama)
                if generation_method in ["JSON-based (Ollama)", "Both"]:
                    st.subheader("🦙 Ollama Model Results")

                    if 'selected_ollama_models' in locals() and selected_ollama_models:
                        for model_name in selected_ollama_models:
                            st.write(f"**Model: {model_name}**")

                            try:
                                with st.spinner(f"Generating with {model_name}..."):
                                    output = query_ollama_model(model_name, user_prompt)

                                # Display raw output
                                with st.expander(f"Raw output from {model_name}"):
                                    st.code(output, language="json")

                                # Parse and convert to MIDI
                                parsed = safe_parse_json(output)
                                if parsed:
                                    try:
                                        midi = json_to_midi(parsed, instrument=56, tempo=default_tempo)

                                        # Download button
                                        midi_bytes = BytesIO()
                                        midi.save(midi_bytes)
                                        midi_bytes.seek(0)

                                        st.download_button(
                                            f"📁 Download MIDI ({model_name})",
                                            midi_bytes.read(),
                                            f"trumpet_{model_name.replace(':', '_')}.mid",
                                            "audio/midi",
                                            key=f"download_{model_name}"
                                        )

                                        # Audio playback
                                        if enable_audio and AUDIO_ENABLED:
                                            audio_data = midi_to_mp3(midi)
                                            if audio_data:
                                                st.audio(audio_data, format="audio/mp3")

                                        # Analysis
                                        analyze_midi(midi, model_name)

                                    except Exception as e:
                                        st.error(f"Failed to generate MIDI: {e}")
                                else:
                                    st.warning("Could not parse output as valid JSON")

                            except Exception as e:
                                st.error(f"Error with model {model_name}: {e}")

                            st.divider()
                    else:
                        st.warning("No Ollama models selected or available")

def analyze_midi(midi_file, source):
    """Analyze and display MIDI file information"""
    with st.expander(f"📊 MIDI Analysis ({source})"):
        try:
            # Count tracks and messages
            num_tracks = len(midi_file.tracks)
            total_messages = sum(len(track) for track in midi_file.tracks)

            # Analyze notes
            notes = []
            for track in midi_file.tracks:
                for msg in track:
                    if hasattr(msg, 'type') and msg.type == 'note_on' and msg.velocity > 0:
                        notes.append(msg.note)

            if notes:
                min_note = min(notes)
                max_note = max(notes)
                unique_notes = len(set(notes))

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Tracks", num_tracks)
                    st.metric("Total Messages", total_messages)

                with col2:
                    st.metric("Unique Notes", unique_notes)
                    st.metric("Note Range", f"{midi_to_note_name(min_note)} - {midi_to_note_name(max_note)}")

                with col3:
                    trumpet_notes = [n for n in notes if TRUMPET_RANGE['low'] <= n <= TRUMPET_RANGE['high']]
                    trumpet_percentage = (len(trumpet_notes) / len(notes)) * 100 if notes else 0
                    st.metric("Trumpet Range %", f"{trumpet_percentage:.1f}%")

                # Note distribution
                if len(set(notes)) <= 20:  # Only show if reasonable number of unique notes
                    note_counts = {}
                    for note in notes:
                        note_name = midi_to_note_name(note)
                        note_counts[note_name] = note_counts.get(note_name, 0) + 1

                    st.bar_chart(note_counts)
            else:
                st.info("No notes found in MIDI file")

        except Exception as e:
            st.error(f"Analysis failed: {e}")

if __name__ == "__main__":
    main()
