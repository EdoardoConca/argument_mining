import os
from collections import defaultdict, Counter
import networkx as nx
import gravis as gv

def count_arguments_in_folder(folder_path):
    """
    Counts the number of arguments in each .ann file in a folder.
    
    Args:
        folder_path (str): Path to the folder containing .ann files.
    
    Returns:
        dict: A dictionary mapping file names to the number of valid arguments.
    """
    def parse_ann_file(file_path):
        """
        Parses a .ann file to extract components and relations.

        Args:
            file_path (str): Path to the .ann file.

        Returns:
            dict: Parsed components and relations.
        """
        components = {}
        relations = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if parts[0].startswith('T'):  # Argumentative components
                    comp_id, comp_type_offset, comp_text = parts
                    comp_type = comp_type_offset.split()[0]
                    components[comp_id] = comp_type
                elif parts[0].startswith('R'):  # Relations
                    rel_id, rel_type_args = parts[:2]
                    rel_type, arg1, arg2 = rel_type_args.split()
                    arg1_id = arg1.split(':')[1]
                    arg2_id = arg2.split(':')[1]
                    relations.append((rel_type, arg1_id, arg2_id))
        
        return components, relations
    
    def count_arguments(components, relations):
        """
        Counts the number of valid arguments in a single .ann file.

        Args:
            components (dict): Dictionary of component IDs to types.
            relations (list): List of relations (type, source, target).

        Returns:
            int: Number of valid arguments.
        """
        claim_arguments = defaultdict(set)  # Maps claims to the set of premises supporting/attacking them
        
        for rel_type, source, target in relations:
            source_type = components.get(source, None)
            target_type = components.get(target, None)
            
            # Only count Premise → Claim (or Major-Claim) relationships
            if source_type == 'Premise' and target_type in {'Claim', 'Major-Claim'}:
                claim_arguments[target].add(source)
        
        # Count the number of unique arguments
        return len(claim_arguments)
    
    arguments_per_file = {}
    
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.ann'):
            filename = file_name.split('.')[0]
            file_path = os.path.join(folder_path, file_name)
            components, relations = parse_ann_file(file_path)
            arguments_per_file[filename] = count_arguments(components, relations)
    
    return arguments_per_file



def count_arguments_and_build_graph(relations):
    """
    Counts valid arguments and constructs a directed graph from relations.

    Rules for counting arguments:
        - A premise supports/attacks a claim: one argument.
        - Multiple premises supporting/attacking the same claim count as a single argument.
        - Claims supporting/attacking claims, or claims supporting/attacking premises, are not valid arguments.
        - Multiple claims supporting/attacking a single claim are not valid arguments.

    Args:
        relations (dict): A dictionary where keys are tuples of source and target components,
                          and values are lists of relation types (e.g., ['Relation']).

    Returns:
        int: The number of valid arguments.
        networkx.DiGraph: The directed graph representing the relations.
    """
    graph = nx.DiGraph()  # Directed graph
    claim_arguments = {}  # Track claims to ensure multiple premises count as one argument

    for (source, target), relation_types in relations.items():
        if len(source) == 5:
            source_type, _, _, _, source_id = source
            target_type, _, _, _, target_id = target
        else:
            source_type, source_id = source
            target_type, target_id = target

        # Add nodes to the graph
        graph.add_node(source_id, type=source_type)
        graph.add_node(target_id, type=target_type)

        # Add edges if there is a 'Relation'
        if 'Relation' in relation_types:
            graph.add_edge(source_id, target_id, relation=relation_types[0], color='black')

            # Check validity for forming arguments
            if source_type == 'Premise' and target_type == 'Claim':
                if target_id not in claim_arguments:
                    claim_arguments[target_id] = set()
                claim_arguments[target_id].add(source_id)

    # Count unique arguments based on claims
    argument_count = len(claim_arguments)

    return argument_count, graph


def plot_graph_with_gravis(graph):
    """
    Plots the directed graph using gravis with node colors for Premise and Claim.

    Args:
        graph (networkx.DiGraph): The directed graph to plot.
    """
    # Prepare edges with colors and strengths
    edges = [
        (u, v, graph[u][v].get('relation', 1), graph[u][v].get('color', 'black'))
        for u, v in graph.edges
    ]

    # Prepare nodes with colors based on their type
    node_colors = {
        'Premise': 'green',
        'Claim': 'blue',
        'Undefined': 'gray'  # Default color if type is missing
    }

    # Create a new directed graph
    g = nx.DiGraph()
    
    # Add edges with attributes
    for source, target, strength, color in edges:
        g.add_edge(source, target, strength=strength, color=color)
    
    # Add nodes with assigned colors
    for node in graph.nodes:
        node_type = graph.nodes[node].get('type', 'Undefined')
        g.add_node(node, color=node_colors.get(node_type, 'gray'))
    
    # Create gravis figure and display without edge labels
    fig = gv.d3(
        g,
        show_edge_label=False  
    )
    fig.display(inline=True)




