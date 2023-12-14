import argparse
import logging

from pollen_detector import PollenDetector


class PollenDetectionCLI:

    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Process PNG image stacks and detect pollen grains.")
        self.args = None

        # Add arguments

        # Model directory path
        self.parser.add_argument("--model-dir", "-m", type=str, dest="model_dir_path", nargs='?', default=None,
                                 required=True, help="Full path of the directory containing the model files.")
        # Crop images directory path
        self.parser.add_argument("--crops-dir", "-c", type=str, dest="crops_dir_path", nargs='?', default=None,
                                 required=True, help="Full path of the directory containing the cropped image files.")

        # Detections output directory path
        self.parser.add_argument("--detections-dir", "-d", type=str, dest="detections_dir_path", nargs="?",
                                 default="detections", required=False,
                                 help="Full path of the directory to store the detection results.")

        # Verbose
        self.parser.add_argument("--verbose", "-v", action="store_true", dest="verbose", default=False,
                                 help="Display more details.")

    def parse_args(self):
        self.args = self.parser.parse_args()

    def print_args(self):
        if self.args.verbose:
            print("\nArguments\tValues")
            print("======================")
            for key, value in self.args.__dict__.items():
                print(key, ":", value)
            print("======================\n")


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)-7s : %(name)s - %(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Pollen Detection CLI")
    cli = PollenDetectionCLI()
    cli.parse_args()
    cli.print_args()

    pollen_detector = PollenDetector(cli.args.model_dir_path, cli.args.crops_dir_path, cli.args.detections_dir_path)
    pollen_detector.generate_dbinfo()
    pollen_detector.initialize_data()
    pollen_detector.initialize_model()
    pollen_detector.process_crop_images()

    logger.info("Stopping Pollen Detection CLI")
