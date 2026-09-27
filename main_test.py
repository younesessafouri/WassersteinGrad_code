# from py4castxai.utils_test.main_utils import *

from py4castxai.utils_test.main_utils import *

# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import sys
    import importlib

    mfai_pytorch = importlib.import_module("mfai.pytorch")

    # Create an alias so that 'mfai.torch' points to it
    sys.modules["mfai.torch"] = mfai_pytorch
    if len(sys.argv) < 2:
        print("Usage: python main_utils.py <xai_config_path>")
        sys.exit(1)
    
    run_experiment(sys.argv[1])