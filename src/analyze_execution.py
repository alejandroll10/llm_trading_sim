from pycallgraph import PyCallGraph, Config
from pycallgraph.output import GraphvizOutput
import os

def parse_graphviz_plain(input_file, output_file):
    """Parse the plain graphviz format and convert to readable text"""
    nodes = {}
    edges = []
    
    # First pass - collect nodes
    with open(input_file, 'r') as f:
        for line in f:
            parts = line.strip().split(' ')
            if parts[0] == 'node':
                name = parts[1]
                # Find the label part (everything between first and last quote)
                label_start = line.index('"') + 1
                label_end = line.rindex('"')
                label = line[label_start:label_end].replace('\\n', '\n')
                
                nodes[name] = {
                    'label': label,
                    'calls': 0,
                    'time': 0.0,
                    'called_by': set(),
                    'calls_to': set()
                }
                
                # Extract calls and time from label
                for info in label.split('\n'):
                    if info.startswith('calls: '):
                        nodes[name]['calls'] = int(info.split(': ')[1])
                    elif info.startswith('time: '):
                        time_str = info.split(': ')[1].split('s')[0]  # Get just the number part
                        nodes[name]['time'] = float(time_str)
            
            elif parts[0] == 'edge':
                source = parts[1]
                target = parts[2]
                edges.append((source, target))
                
                # Record relationships
                if source in nodes and target in nodes:
                    nodes[source]['calls_to'].add(target)
                    nodes[target]['called_by'].add(source)

    # Write formatted output
    with open(output_file, 'w') as f:
        f.write("Execution Call Graph Analysis\n")
        f.write("===========================\n\n")
        
        # Sort nodes by execution time
        sorted_nodes = sorted(nodes.items(), key=lambda x: x[1]['time'], reverse=True)
        
        # Print summary statistics
        total_calls = sum(node['calls'] for node in nodes.values())
        total_time = sum(node['time'] for node in nodes.values())
        f.write(f"Summary:\n")
        f.write(f"- Total function calls: {total_calls}\n")
        f.write(f"- Total execution time: {total_time:.4f}s\n")
        f.write(f"- Number of unique functions: {len(nodes)}\n\n")
        
        # Print detailed function information
        f.write("Function Details (sorted by execution time):\n")
        f.write("----------------------------------------\n\n")
        
        for name, data in sorted_nodes:
            if data['time'] > 0.001:  # Only show functions that took meaningful time
                f.write(f"Function: {name}\n")
                f.write(f"- Calls: {data['calls']}\n")
                f.write(f"- Total time: {data['time']:.4f}s\n")
                f.write(f"- Average time per call: {data['time']/data['calls']:.6f}s\n")
                
                if data['called_by']:
                    f.write("- Called by:\n")
                    for caller in sorted(data['called_by']):
                        f.write(f"  * {caller}\n")
                
                if data['calls_to']:
                    f.write("- Calls to:\n")
                    for callee in sorted(data['calls_to']):
                        f.write(f"  * {callee}\n")
                
                f.write("\n")

def analyze_simulation_execution():
    """
    Create both visual and text-based call graphs of the simulation execution.
    """
    print("Analyzing simulation execution...")
    
    # Create output directory if it doesn't exist
    os.makedirs('logs/latest_sim', exist_ok=True)
    
    # Configure the graphviz outputs
    graphviz_png = GraphvizOutput(
        output_file='logs/latest_sim/execution_graph.png',
        output_type='png',
        font_size=16,
        group=True,
        groups={
            'simulation': 'simulation.*',
            'market': 'market.*',
            'agents': 'agents.*',
            'services': 'services.*'
        }
    )
    
    graphviz_plain = GraphvizOutput(
        output_file='logs/latest_sim/execution_graph_raw.txt',
        output_type='plain',
        font_size=16,
        group=True,
        groups={
            'simulation': 'simulation.*',
            'market': 'market.*',
            'agents': 'agents.*',
            'services': 'services.*'
        }
    )
    
    # Run the simulation with both outputs
    config = Config(
        include_stdlib=False,
        max_depth=10,
        verbose=True
    )
    
    with PyCallGraph(output=[graphviz_png, graphviz_plain], config=config):
        # Import here to avoid tracking the imports
        from run_base_sim import main
        main()
    
    # Parse the plain output into readable format
    parse_graphviz_plain(
        'logs/latest_sim/execution_graph_raw.txt',
        'logs/latest_sim/execution_graph_analysis.txt'
    )

if __name__ == "__main__":
    analyze_simulation_execution() 