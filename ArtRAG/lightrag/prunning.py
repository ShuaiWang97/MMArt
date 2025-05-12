import asyncio
import json
import logging
from typing import List, Dict
from PIL import Image
import numpy as np
import re
from .prompt_art import GRAPH_FIELD_SEP, PROMPTS

logger = logging.getLogger(__name__)


async def summarize_description(text, use_model_func):
    """
    Summarizes a long description using the configured LLM.
    Uses use_model_func for compatibility with the existing async pipeline.
    """
    if len(text.split()) < 30:  # If already short, return as is
        return text

    prompt = f"Summarize the following text withon 30 words, keeping only the most relevant information:\n\n{text}"
    
    response = await use_model_func(prompt)

    try:
        summary = response.strip()
    except Exception as e:
        logger.warning(f"Summarization failed: {e}")
        summary = text  # Fall back to the original text if summarization fails

    return summary

# Normalize scores to 0-1 scale for fair combination
def min_max_normalize(scores):
    values = np.array(list(scores.values()), dtype=np.float32)
    if values.max() - values.min() > 0:
        values = (values - values.min()) / (values.max() - values.min())  # Min-max normalization
    else:
        values[:] = 1  # If all values are identical, set to 1
    return {key: value for key, value in zip(scores.keys(), values)}

def softmax_normalize(scores_dict):
    """
    Applies softmax normalization to a dictionary of scores.
    Ensures smoother differentiation, making top scores more prominent.
    """
    scores = np.array(list(scores_dict.values()), dtype=np.float32)

    # Avoiding overflow issues by subtracting max before exponentiation
    exp_scores = np.exp(scores - np.max(scores))
    softmax_scores = exp_scores / np.sum(exp_scores)

    return {key: score for key, score in zip(scores_dict.keys(), softmax_scores)}

def extract_metadata(text):
    match = re.search(r"Metadata:\s*(.*)", text)
    if match:
        return match.group(1).strip()
    return None  # Keyword not found


async def rerank_nodes_with_vlm(
    image_path: str,
    query_text: Dict[str, str],
    nodes: List[Dict[str, str]],
    use_model_func,
):
    """
    Uses a Vision-Language Model (VLM) to perform listwise reranking of nodes.
    Returns nodes in a ranked order.
    """

    # Build the query text
    painting_metadata= extract_metadata(query_text)

    # Build listwise ranking prompt
    ranking_prompt = PROMPTS["rerank_entities"]
    ranking_prompt = ranking_prompt.format(
        Metadata=painting_metadata, entities=[f"{i+1}. {node['entity_name']}: {node['description']}" for i, node in enumerate(nodes)]
    )
    # Call the VLM with the listwise ranking prompt
    response = await use_model_func(ranking_prompt, query_image_path=image_path)

    try:
        # Extract ranked order from the model's response
        ranked_indices = [int(num.strip()) - 1 for num in response.split(',') if num.strip().isdigit()]
        ranked_nodes = [nodes[i] for i in ranked_indices if 0 <= i < len(nodes)]
    except Exception as e:
        logger.warning(f"Failed to parse VLM ranking response: {e}")
        ranked_nodes = nodes  # If ranking fails, return the original list

    # Create a dictionary to store VLM rank scores (higher is better)
    vlm_rank_scores = {node["entity_name"]: len(nodes) - i for i, node in enumerate(ranked_nodes)}

    return ranked_nodes, vlm_rank_scores


async def dual_passage_rerank(
    image_path: str,
    painting_metadata: Dict[str, str],
    nodes: List[Dict[str, str]],
    use_model_func,
    knowledge_graph_inst,
    topn=5
):
    """
    Performs dual-passage reranking by considering:
    1. Vision-Language Model (VLM) listwise ranking
    2. Node degree from the knowledge graph (graph importance)

    Returns a final reranked list combining both scores equally.
    """
    if not nodes:
        return []

        # Summarize long descriptions
    for node in nodes:
        node["description"] = await summarize_description(node.get("description", "UNKNOWN"), use_model_func)
        if "source_id" in node:
            del node["source_id"]

    # Step 1: Get VLM listwise ranking scores
    vlm_ranked_nodes, vlm_rank_scores = await rerank_nodes_with_vlm(image_path, painting_metadata, nodes, use_model_func)
    # Step 2: Compute node degree ranking scores
    degree_scores = {node["entity_name"]: node["rank"] for node in nodes}

    vlm_rank_scores = softmax_normalize(vlm_rank_scores)
    degree_scores = softmax_normalize(degree_scores)

    # Step 3: Combine rankings with equal weight (0.5 * VLM + 0.5 * Node Degree)
    final_scores = {
        node["entity_name"]: 0.5 * vlm_rank_scores.get(node["entity_name"], 0) +
                             0.5 * degree_scores.get(node["entity_name"], 0)
        for node in nodes
    }

    # Step 4: Sort nodes based on final combined score
    final_reranked_nodes = sorted(nodes, key=lambda x: final_scores[x["entity_name"]], reverse=True)[0:topn]
    inter_edges = await find_interconnected_edges([node["entity_name"] for node in final_reranked_nodes], knowledge_graph_inst)

    return final_reranked_nodes, inter_edges




async def find_interconnected_edges(all_node_names, knowledge_graph_inst):
    """
    Fetch all edges between the combined set of nodes.
    """
    all_edges = set()
    # check if there is are """ in node_name, if not add it to node name, like 'HIGH RENAISSANCE' to '"HIGH RENAISSANCE"'
    all_node_names = [f'"{node_name}"' if '"' not in node_name else node_name for node_name in all_node_names]    

    for node_name in all_node_names:
        # Check if there is are """ in node_name, if not add it to node name, like 'HIGH RENAISSANCE' to '"HIGH RENAISSANCE"'
        # if '"' not in node_name:
        #     node_name = f'"{node_name}"'
        node_edges = await knowledge_graph_inst.get_node_edges(node_name)
        all_edges.update([tuple(sorted(e)) for e in node_edges])
    all_edges = list(all_edges)
    all_edges_pack = await asyncio.gather(
        *[knowledge_graph_inst.get_edge(e[0], e[1]) for e in all_edges]
    )
    all_edges_data = [
        {"src_tgt": k, **v}
        for k, v in zip(all_edges, all_edges_pack)
        if v is not None and k[0] in all_node_names and k[1] in all_node_names
    ]
    return all_edges_data
