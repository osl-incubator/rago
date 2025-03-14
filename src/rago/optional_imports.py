import importlib, sys, subprocess
from typing import Any

def optional_import(package_name: str) -> Any:
    """
    Attempts to import a package. 
    If missing, prompts user to install it.
    If install does not work at the time, returns a stub that raises ImportError.
    """
    try:
        return importlib.import_module(package_name)
    except ImportError:
        prompt = input(
            f"'{package_name}' is not installed. Install now? (y/n): "
        ).strip().lower()

        if prompt == 'y':
            try:  # Automate installation of the required package using pip
                subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            except Exception as e: # Return exception if an error arises
                print(f"Failed to install {package_name}: {e}")
        else:
            print(f"Installation skipped. You can run 'pip install {package_name}' at anytime to install this package.")
        
        class MissingModuleStub:
            def __getattr__(self, item):
               raise ImportError(                
                   f"'{package_name}' is not installed. You can install it with 'pip install {package_name}' or get all dependencies with 'pip install rago[all]'"
               ) 
        return MissingModuleStub()