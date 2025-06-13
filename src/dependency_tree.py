import graphviz
from pathlib import Path
import subprocess
import re
import pyan
import os

def generate_call_graph():
    """
    Generate a call graph using pyan3 as a library.
    Requires: pip install pyan3
    """
    print("Analyzing code to generate call graph...")
    
    try:
        # Create directories if they don't exist
        os.makedirs('logs/latest_sim', exist_ok=True)
        
        # Get absolute paths to the files
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        files = [
            os.path.join(base_dir, "src/run_base_sim.py"),
            os.path.join(base_dir, "src/base_sim.py")
        ]
        
        # Verify files exist
        for file in files:
            if not os.path.exists(file):
                print(f"File not found: {file}")
                return False
                
        print(f"Analyzing files: {files}")
        
        # Create the graph using pyan's library function
        graph = pyan.create_callgraph(
            files,
            format="dot",
            grouped=True,
            colored=True,
            annotated=True
        )
        
        if not graph:
            print("Error: pyan returned empty graph")
            return False
            
        # Save the dot content
        dot_path = os.path.join(base_dir, 'call_graph.dot')
        with open(dot_path, 'w') as f:
            f.write(graph)
        
        # Now use dot to convert to PNG
        png_path = os.path.join(base_dir, 'call_graph.png')
        cmd_dot = [
            "dot",
            "-Tpng",
            "-Gnewrank=true",
            dot_path,
            "-o",
            png_path
        ]
        
        result_dot = subprocess.run(cmd_dot, capture_output=True, text=True)
        if result_dot.returncode == 0:
            print("Call graph generated successfully!")
            return True
        else:
            print(f"Error running dot: {result_dot.stderr}")
            return False
            
    except Exception as e:
        print(f"Error generating call graph: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Generate and save the call graph"""
    if generate_call_graph():
        print("Call graph has been saved to call_graph.png")
    else:
        print("Failed to generate call graph")

if __name__ == "__main__":
    main() 