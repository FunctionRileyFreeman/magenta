import magenta.music as mm
from magenta.models.music_vae import TrainedModel
import logging
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_midi_data(file_path):
    """Load MIDI data from file."""
    try:
        return mm.note_sequence_io.midi_file_to_note_sequence(file_path)
    except Exception as e:
        logger.error(f"Failed to load MIDI data from '{file_path}': {e}")
        raise

def train_model(sequences, epochs=10, batch_size=64):
    """Train the model using the given MIDI sequences."""
    try:
        model = TrainedModel()
        model.train(sequences, epochs=epochs, batch_size=batch_size)
        return model
    except Exception as e:
        logger.error(f"Failed to train model: {e}")
        raise

def generate_music(model, length=64, temperature=0.5):
    """Generate new music using the trained model."""
    try:
        return model.generate(length=length, temperature=temperature)
    except Exception as e:
        logger.error(f"Failed to generate music: {e}")
        raise

def save_generated_music(sequence, file_path):
    """Save the generated music to a MIDI file."""
    try:
        mm.sequence_proto_to_midi_file(sequence, file_path)
        logger.info(f"Generated music saved to '{file_path}'")
    except Exception as e:
        logger.error(f"Failed to save generated music to '{file_path}': {e}")
        raise

def main(args):
    try:
        # Load MIDI data for training
        midi_data = load_midi_data(args.input_file)

        # Split MIDI data into training sequences
        sequences = mm.sequences_lib.split_note_sequence_on_time_changes(midi_data)

        # Train the model
        trained_model = train_model(sequences, args.epochs, args.batch_size)

        # Generate new music
        generated_sequence = generate_music(trained_model, args.length, args.temperature)

        # Save the generated music to a MIDI file
        save_generated_music(generated_sequence, args.output_file)

    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="AI Music Composer")
    parser.add_argument("input_file", type=str, help="Input MIDI file for training")
    parser.add_argument("output_file", type=str, help="Output MIDI file for generated music")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (default: 10)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training (default: 64)")
    parser.add_argument("--length", type=int, default=64, help="Length of generated music (default: 64)")
    parser.add_argument("--temperature", type=float, default=0.5, help="Temperature for music generation (default: 0.5)")
    args = parser.parse_args()

    # Run the main function with parsed arguments
    main(args)
