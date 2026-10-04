from multiprocessing import Pool, cpu_count
import os
import torch
from tqdm import tqdm
from eap.graph import Graph


def vectorize_circuit(graph):
    mask = graph.real_edge_mask
    in_graph = graph.in_graph.clone()[mask].bool()
    scores = graph.scores.clone()[mask].float()
    return scores * in_graph.float()

def vectorize_circuit_binary(graph):
    edge = graph.in_graph.clone()
    edge = edge.int()[graph.real_edge_mask]

    return edge

def process_single_circuit(args):
    """Process a single circuit index"""
    i, m, topn = args
    
    path = f'saves/circuit/difficulty/tofu_Llama-3.2-1B-Instruct_forget10_{m}/{i}.json'

    try:
        g = Graph.from_json(path)
    except:
        print(args)
        raise
    
    g.apply_greedy(topn)
    
    q = vectorize_circuit_binary(g)

    return (i, q)

def parallel_process_circuits(m, n_cores=64):
    """Main function to process circuits in parallel"""
    ori = 'original'
    topn = 75
    
    # Prepare arguments for parallel processing
    args_list = [(i, m, topn) for i in range(4000)]
    
    
    # Process in parallel
    print(f"Processing with {n_cores} cores...")
    with Pool(n_cores) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_single_circuit, args_list, chunksize=4),
            total=len(args_list),
            desc="Processing circuits"
        ))
    
    # Filter out None values (skipped files)
    results.sort(key=lambda x: x[0])
    
    # Filter out None values and extract tensors
    res = [r[1] for r in results if r is not None]
    res = torch.vstack(res)

    # Save results
    output_path = f'saves/circuit/difficulty/tofu_Llama-3.2-1B-Instruct_forget10_{m}/bin_greedy_{topn}.pt'
    torch.save(res, output_path)
    print(f"Processed {len(res)} circuits")
    return res

# Usage
if __name__ == '__main__':
    m = 'GradDiff'  # Set your model name
    results = parallel_process_circuits(m, n_cores=16)  # Use 8 cores, or None for all