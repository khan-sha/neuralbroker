__version__ = "0.4.1"
__author__ = "NeuralBroker contributors"

# Expose core submodules — avoid circular imports by only exposing needed modules
from . import config, types, detect, autoconfig

