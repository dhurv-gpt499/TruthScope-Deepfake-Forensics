import sys
import pickle
import traceback
import os
import pathlib

if os.name == 'nt':
    pathlib.PosixPath = pathlib.WindowsPath

# Add project root to sys path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.tools.registry import _TOOL_MANIFEST

def run():
    try:
        if len(sys.argv) < 3:
            raise ValueError("Usage: python subprocess_worker.py <tool_name> <pickled_input_file>")
        
        tool_name = sys.argv[1]
        input_file = sys.argv[2]
        
        # Load input data
        with open(input_file, 'rb') as f:
            input_data = pickle.load(f)
            
        # Find which tool class
        module_path, class_name = None, None
        for label, m_path, c_name, _, _, _ in _TOOL_MANIFEST:
            if label == tool_name:
                module_path = m_path
                class_name = c_name
                break
                
        if not module_path:
            raise ValueError(f"Unknown tool name: {tool_name}")
            
        # Import and instantiate tool
        import importlib
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        tool_instance = cls()
        if hasattr(tool_instance, 'setup'):
            tool_instance.setup()
            
        result = tool_instance.execute(input_data)
        
        # Write back result
        output_file = input_file + ".out"
        with open(output_file, 'wb') as f:
            pickle.dump(result, f)
            
    except Exception as e:
        # Pass exception string back
        output_file = sys.argv[2] + ".out" if len(sys.argv) >= 3 else "error.out"
        with open(output_file, 'wb') as f:
            pickle.dump(e, f)

if __name__ == "__main__":
    run()
