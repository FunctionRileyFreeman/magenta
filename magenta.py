import magenta.music as mm
from magenta.models.music_vae import TrainedModel
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_midi_data(file_path):
    """Load MIDI data from file."""
    try:
        return mm.note_sequence_io.midi_file_to_note_sequence(file_path)
    except Exception as e:
        logger.error(f"Failed to load MIDI data from '{file_path}': {e}")

def train_model(sequences):
    """Train the model using the given MIDI sequences."""
    try:
        model = TrainedModel()
        model.train(sequences)
        return model
    except Exception as e:
        logger.error(f"Failed to train model: {e}")

def generate_music(model, length=64, temperature=0.5):
    """Generate new music using the trained model."""
    try:
        return model.generate(length=length, temperature=temperature)
    except Exception as e:
        logger.error(f"Failed to generate music: {e}")

def save_generated_music(sequence, file_path):
    """Save the generated music to a MIDI file."""
    try:
        mm.sequence_proto_to_midi_file(sequence, file_path)
        logger.info(f"Generated music saved to '{file_path}'")
    except Exception as e:
        logger.error(f"Failed to save generated music to '{file_path}': {e}")

def main():
    try:
        # Load MIDI data for training
        midi_data = load_midi_data('path_to_your_midi_data')

        # Split MIDI data into training sequences
        sequences = mm.sequences_lib.split_note_sequence_on_time_changes(midi_data)

        # Train the model
        trained_model = train_model(sequences)

        # Generate new music
        generated_sequence = generate_music(trained_model)

        # Save the generated music to a MIDI file
        save_generated_music(generated_sequence, 'generated_music.mid')

    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
