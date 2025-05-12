import asyncio
import json
import re
from typing import Union
from collections import Counter, defaultdict
import warnings
import pdb
from .utils import (
    logger,
    clean_str,
    compute_mdhash_id,
    decode_tokens_by_tiktoken,
    encode_string_by_tiktoken,
    is_float_regex,
    list_of_list_to_csv,
    pack_user_ass_to_openai_messages,
    split_string_by_multi_markers,
    truncate_list_by_token_size,
    generate_context_sections
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    TextChunkSchema,
    QueryParam,
)
from .prompt_art import GRAPH_FIELD_SEP, PROMPTS
from .prunning import (
    find_interconnected_edges,
    dual_passage_rerank)


def chunking_by_token_size(
    content: str, overlap_token_size=128, max_token_size=1024, tiktoken_model="gpt-4o"
):
    tokens = encode_string_by_tiktoken(content, model_name=tiktoken_model)
    results = []
    for index, start in enumerate(
        range(0, len(tokens), max_token_size - overlap_token_size)
    ):
        chunk_content = decode_tokens_by_tiktoken(
            tokens[start : start + max_token_size], model_name=tiktoken_model
        )
        results.append(
            {
                "tokens": min(max_token_size, len(tokens) - start),
                "content": chunk_content.strip(),
                "chunk_order_index": index,
            }
        )
    return results


async def _handle_entity_relation_summary(
    entity_or_relation_name: str,
    description: str,
    global_config: dict,
) -> str:
    """
    Summarize the entity or relation description
    """
    use_llm_func: callable = global_config["llm_model_func"]
    llm_max_tokens = global_config["llm_model_max_token_size"]
    tiktoken_model_name = global_config["tiktoken_model_name"]
    summary_max_tokens = global_config["entity_summary_to_max_tokens"]

    tokens = encode_string_by_tiktoken(description, model_name=tiktoken_model_name)
    if len(tokens) < summary_max_tokens:  # No need for summary
        return description
    prompt_template = PROMPTS["summarize_entity_descriptions"]
    use_description = decode_tokens_by_tiktoken(
        tokens[:llm_max_tokens], model_name = tiktoken_model_name
    )
    context_base = dict(
        entity_name=entity_or_relation_name,
        description_list=use_description.split(GRAPH_FIELD_SEP),
    )
    use_prompt = prompt_template.format(**context_base)
    logger.debug(f"Trigger summary: {entity_or_relation_name}")
    summary = await use_llm_func(use_prompt, max_tokens=summary_max_tokens)
    return summary


async def _handle_single_entity_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    if len(record_attributes) < 4 or record_attributes[0] != '"entity"':
        return None
    # add this record as a node in the G
    entity_name = clean_str(record_attributes[1].upper())
    if not entity_name.strip():
        return None
    entity_type = clean_str(record_attributes[2].upper())
    entity_description = clean_str(record_attributes[3])
    entity_source_id = chunk_key
    return dict(
        entity_name=entity_name,
        entity_type=entity_type,
        description=entity_description,
        source_id=entity_source_id,
    )


async def _handle_single_relationship_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    if len(record_attributes) < 5 or record_attributes[0] != '"relationship"':
        return None
    # add this record as edge
    source = clean_str(record_attributes[1].upper())
    target = clean_str(record_attributes[2].upper())
    edge_description = clean_str(record_attributes[3])

    edge_keywords = clean_str(record_attributes[4])
    edge_source_id = chunk_key
    weight = (
        float(record_attributes[-1]) if is_float_regex(record_attributes[-1]) else 1.0
    )
    return dict(
        src_id=source,
        tgt_id=target,
        weight=weight,
        description=edge_description,
        keywords=edge_keywords,
        source_id=edge_source_id,
    )


async def _merge_nodes_then_upsert(
    entity_name: str,
    nodes_data: list[dict],
    knwoledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    """
    Merge the nodes data and upsert to the graph
    """
    already_entitiy_types = []
    already_source_ids = []
    already_description = []

    already_node = await knwoledge_graph_inst.get_node(entity_name)
    if already_node is not None:
        already_entitiy_types.append(already_node["entity_type"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_node["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_node["description"])

    entity_type = sorted(
        Counter(
            [dp["entity_type"] for dp in nodes_data] + already_entitiy_types
        ).items(),
        key=lambda x: x[1],
        reverse=True,
    )[0][0]
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in nodes_data] + already_description))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in nodes_data] + already_source_ids)
    )
    description = await _handle_entity_relation_summary(
        entity_name, description, global_config
    )
    node_data = dict(
        entity_type=entity_type,
        description=description,
        source_id=source_id,
    )
    await knwoledge_graph_inst.upsert_node(
        entity_name,
        node_data=node_data,
    )
    node_data["entity_name"] = entity_name
    return node_data


async def _merge_edges_then_upsert(
    src_id: str,
    tgt_id: str,
    edges_data: list[dict],
    knwoledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    already_weights = []
    already_source_ids = []
    already_description = []
    already_keywords = []

    if await knwoledge_graph_inst.has_edge(src_id, tgt_id):
        already_edge = await knwoledge_graph_inst.get_edge(src_id, tgt_id)
        already_weights.append(already_edge["weight"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_edge["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_edge["description"])
        already_keywords.extend(
            split_string_by_multi_markers(already_edge["keywords"], [GRAPH_FIELD_SEP])
        )

    weight = sum([dp["weight"] for dp in edges_data] + already_weights)
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in edges_data] + already_description))
    )
    keywords = GRAPH_FIELD_SEP.join(
        sorted(set([dp["keywords"] for dp in edges_data] + already_keywords))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in edges_data] + already_source_ids)
    )
    for need_insert_id in [src_id, tgt_id]:
        if not (await knwoledge_graph_inst.has_node(need_insert_id)):
            await knwoledge_graph_inst.upsert_node(
                need_insert_id,
                node_data={
                    "source_id": source_id,
                    "description": description,
                    "entity_type": '"UNKNOWN"',
                },
            )
    description = await _handle_entity_relation_summary(
        (src_id, tgt_id), description, global_config
    )
    await knwoledge_graph_inst.upsert_edge(
        src_id,
        tgt_id,
        edge_data=dict(
            weight=weight,
            description=description,
            keywords=keywords,
            source_id=source_id,
        ),
    )

    edge_data = dict(
        src_id=src_id,
        tgt_id=tgt_id,
        description=description,
        keywords=keywords,
    )

    return edge_data


async def extract_entities(
    chunks: dict[str, TextChunkSchema],
    knwoledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    global_config: dict,
) -> Union[BaseGraphStorage, None]:
    """
    
    """
    use_llm_func: callable = global_config["llm_model_func"]
    entity_extract_max_gleaning = global_config["entity_extract_max_gleaning"]

    ordered_chunks = list(chunks.items())

    entity_extract_prompt = PROMPTS["entity_extraction"]
    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(PROMPTS["DEFAULT_ENTITY_TYPES"]),
    )
    continue_prompt = PROMPTS["entiti_continue_extraction"]
    if_loop_prompt = PROMPTS["entiti_if_loop_extraction"]

    already_processed = 0
    already_entities = 0
    already_relations = 0

    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):
        nonlocal already_processed, already_entities, already_relations
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]
        hint_prompt = entity_extract_prompt.format(**context_base, input_text=content)
        final_result = await use_llm_func(hint_prompt)

        history = pack_user_ass_to_openai_messages(hint_prompt, final_result)
        for now_glean_index in range(entity_extract_max_gleaning):
            glean_result = await use_llm_func(continue_prompt, history_messages=history)

            history += pack_user_ass_to_openai_messages(continue_prompt, glean_result)
            final_result += glean_result
            if now_glean_index == entity_extract_max_gleaning - 1:
                break


            if_loop_result: str = await use_llm_func(
                if_loop_prompt, history_messages=history
            )
            if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
            if if_loop_result != "yes":
                break

        records = split_string_by_multi_markers(
            final_result,
            [context_base["record_delimiter"], context_base["completion_delimiter"]],
        )

        maybe_nodes = defaultdict(list)
        maybe_edges = defaultdict(list)
        for record in records:
            record = re.search(r"\((.*)\)", record)
            if record is None:
                continue
            record = record.group(1)
            record_attributes = split_string_by_multi_markers(
                record, [context_base["tuple_delimiter"]]
            )
            if_entities = await _handle_single_entity_extraction(
                record_attributes, chunk_key
            )
            if if_entities is not None:
                maybe_nodes[if_entities["entity_name"]].append(if_entities)
                continue

            if_relation = await _handle_single_relationship_extraction(
                record_attributes, chunk_key
            )
            if if_relation is not None:
                maybe_edges[(if_relation["src_id"], if_relation["tgt_id"])].append(
                    if_relation
                )
        already_processed += 1
        already_entities += len(maybe_nodes)
        already_relations += len(maybe_edges)
        now_ticks = PROMPTS["process_tickers"][
            already_processed % len(PROMPTS["process_tickers"])
        ]
        print(
            f"{now_ticks} Processed {already_processed} chunks, {already_entities} entities(duplicated), {already_relations} relations(duplicated)\r",
            end="",
            flush=True,
        )
        return dict(maybe_nodes), dict(maybe_edges)

    # use_llm_func is wrapped in ascynio.Semaphore, limiting max_async callings
    results = await asyncio.gather(
        *[_process_single_content(c) for c in ordered_chunks]
    )
    print()  # clear the progress bar
    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)
    for m_nodes, m_edges in results:
        for k, v in m_nodes.items():
            maybe_nodes[k].extend(v)
        for k, v in m_edges.items():
            maybe_edges[tuple(sorted(k))].extend(v)
    all_entities_data = await asyncio.gather(
        *[
            _merge_nodes_then_upsert(k, v, knwoledge_graph_inst, global_config)
            for k, v in maybe_nodes.items()
        ]
    )
    all_relationships_data = await asyncio.gather(
        *[
            _merge_edges_then_upsert(k[0], k[1], v, knwoledge_graph_inst, global_config)
            for k, v in maybe_edges.items()
        ]
    )
    if not len(all_entities_data):
        logger.warning("Didn't extract any entities, maybe your LLM is not working")
        return None
    if not len(all_relationships_data):
        logger.warning(
            "Didn't extract any relationships, maybe your LLM is not working"
        )
        return None

    if entity_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                "content": dp["entity_name"] + dp["description"],
                "entity_name": dp["entity_name"],
            }
            for dp in all_entities_data
        }
        await entity_vdb.upsert(data_for_vdb)

    if relationships_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["src_id"] + dp["tgt_id"], prefix="rel-"): {
                "src_id": dp["src_id"],
                "tgt_id": dp["tgt_id"],
                "content": dp["keywords"]
                + dp["src_id"]
                + dp["tgt_id"]
                + dp["description"],
            }
            for dp in all_relationships_data
        }
        await relationships_vdb.upsert(data_for_vdb)

    return knwoledge_graph_inst


async def local_query(
    input_: Union[str, dict],
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    # relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
) -> str:
    context = None
    use_model_func = global_config["llm_model_func"]

    kw_prompt_temp = PROMPTS["keywords_extraction"]
    if isinstance(input_, dict):
        query = input_["text"]
        query_image = input_["image"]
    else:
        query = input_
        
    kw_prompt = kw_prompt_temp.format(query=query)
    result = await use_model_func(kw_prompt)

    try:
        keywords_data = json.loads(result)
        keywords = keywords_data.get("keywords", [])
        keywords = ", ".join(keywords)
    except json.JSONDecodeError:
        try:
            result = (
                result.replace(kw_prompt[:-1], "")
                .replace("user", "")
                .replace("model", "")
                .strip()
            )
            result = "{" + result.split("{")[1].split("}")[0] + "}"

            keywords_data = json.loads(result)
            keywords = keywords_data.get("keywords", [])
            keywords = ", ".join(keywords)
        # Handle parsing error
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            return PROMPTS["fail_response"]
    if keywords:
        rerank_context, beforererank_context = await _build_local_query_context(
            query,
            query_image,
            keywords,
            knowledge_graph_inst,
            entities_vdb,
            text_chunks_db,
            query_param,
            use_model_func,
        )

    # if query_param.only_need_context:
    #     return context
    # if context is None:
    #     return PROMPTS["fail_response"]

    # sys_prompt_temp = PROMPTS["rag_response"]
    if query_param.data_type == "SemArtv2" or query_param.data_type == "Artpedia" or query_param.data_type == "ExpArt":
        if query_param.shot_number == 1:
            sys_prompt_temp = PROMPTS["rag_SemArtv2_1-shot_incontext_response"]
            # system_images = ["./painting_examples/The Harvesters.png"]
        elif query_param.shot_number == 2:
            sys_prompt_temp = PROMPTS["rag_SemArtv2_2-shot_incontext_response"]
            # system_images = ["./painting_examples/The Harvesters.png", "./painting_examples/Still Life with Flowers.png"]
        elif query_param.shot_number == 3:
            sys_prompt_temp = PROMPTS["rag_SemArtv2_3-shot_incontext_response"]
        elif query_param.shot_number == 0:
            sys_prompt_temp = PROMPTS["zero-shot_response"]
            # system_images = ["./painting_examples/The Harvesters.png", "./painting_examples/Still Life with Flowers.png", "./painting_examples/Madonna and Child with Saints.png"]

    elif query_param.data_type == "SemArtv1-content":
        sys_prompt_temp = PROMPTS["rag_SemArtv1-content_incontext_response"]
    elif query_param.data_type == "SemArtv1-context":
        sys_prompt_temp = PROMPTS["rag_SemArtv1-context_incontext_response"]
    sys_prompt = sys_prompt_temp.format(
        context_data=rerank_context, response_type=query_param.response_type
    )
    
    if isinstance(input_, dict):
        # remove the "Painting Concepts" from generation
        query = input_["text"].split("Painting Concepts", 1)[0]
        query_image = input_["image"]
        # Use model functions to generate response
        response = await use_model_func(
            query,
            system_prompt=sys_prompt,
            query_image_path = query_image,
        )

        if query_param.fewshot_type=="MM_fewshot":
            response = await use_model_func(
                query,
                system_prompt=sys_prompt,
                query_image_path = query_image,
                system_image_paths = system_images,
            )
    else:
        # remove the "Painting Concepts" from generation
        query = input_.split("Painting Concepts", 1)[0]
        # Use model functions to generate response
        response = await use_model_func(
            query,
            system_prompt=sys_prompt,
        )

    if len(response) > len(sys_prompt):
        response = (
            response.replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )
    return response, beforererank_context, rerank_context


async def _build_local_query_context(
    query_text,
    query_image,
    keywords, 
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    use_model_func
):
    """
    Build the local query context, including entities and relationships, and rerank them based on the visual-language model (VLM) scores.
    """
    results = await entities_vdb.query(keywords, top_k=query_param.top_k)
    if not len(results):
        return None
    node_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_node(r["entity_name"]) for r in results]
    )
    if not all([n is not None for n in node_datas]):
        logger.warning("Some nodes are missing, maybe the storage is damaged")
    node_degrees = await asyncio.gather(
        *[knowledge_graph_inst.node_degree(r["entity_name"]) for r in results]
    )
    node_datas = [
        {**n, "entity_name": k["entity_name"], "rank": d}
        for k, n, d in zip(results, node_datas, node_degrees)
        if n is not None
    ]
 
    node_datas, use_relations = await ndoes_expension(
        node_datas, query_param, knowledge_graph_inst
    )
    logger.info(
        f"Local query uses {len(node_datas)} entites, {len(use_relations)} relations"
    )
    # Generate context sections
    entities_context, relations_context = generate_context_sections(node_datas, use_relations)

    # Rerank the nodes based on visual-language model (VLM) scores.
    reranked_nodes, reranked_edges = await dual_passage_rerank(query_image, query_text, node_datas, use_model_func,knowledge_graph_inst)
    rerank_entities_context, rerank_relations_context = generate_context_sections(reranked_nodes, reranked_edges)
    logger.info(
        f"After reranking Local query uses {len(reranked_nodes)} entites, {len(reranked_edges)} relations"
    )

    before_rerank_context = f"""
        -----Entities-----
        ```csv
        {entities_context}
        ```             
        -----Relationships-----
        ```csv  
        {relations_context}
        ```
        """
    
    after_rerank_context = f""" 
        -----Entities-----      
        ```csv      
        {rerank_entities_context}
        ```
        -----Relationships-----
        ```csv
        {rerank_relations_context}
        ```
        """
    return after_rerank_context, before_rerank_context


async def ndoes_expension(
    node_datas: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
    top_k=5,
):
    """
    1. Find the most related edges, based on edge degree and weight.
    2. then from the entities and go one-hop further to find the new nodes
    3. finally return new sets of nodes and all their interconnected edges 
    """
    all_related_edges = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas]
    )
    all_edges = set()
    for this_edges in all_related_edges:
        all_edges.update([tuple(sorted(e)) for e in this_edges])
    all_edges = list(all_edges)
    all_edges_pack = await asyncio.gather(
        *[knowledge_graph_inst.get_edge(e[0], e[1]) for e in all_edges]
    )
    all_edges_degree = await asyncio.gather(
        *[knowledge_graph_inst.edge_degree(e[0], e[1]) for e in all_edges]
    )
    all_edges_data = [
        {"src_tgt": k, "rank": d, **v}
        for k, v, d in zip(all_edges, all_edges_pack, all_edges_degree)
        if v is not None
    ]
    all_edges_data = sorted(
        all_edges_data, key=lambda x: (x["rank"], x["weight"]), reverse=True
    )
    all_edges_data = all_edges_data[:top_k]
    all_edges_data = truncate_list_by_token_size(
        all_edges_data,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_global_context,
    )
    # Collect new nodes from the expanded edges
    new_node_names = set()
    for edge in all_edges_data:
        new_node_names.add(edge["src_tgt"][0])
        new_node_names.add(edge["src_tgt"][1])

    # Fetch the new nodes' data
    new_node_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_node(node_name) for node_name in new_node_names]
    )
    node_degrees = await asyncio.gather(
        *[knowledge_graph_inst.node_degree(node_name) for node_name in new_node_names]
    )
    new_node_datas = [
        {"entity_name": k, **n, "rank": d}
        for k, n, d in zip(new_node_names, new_node_datas, node_degrees)
        if n is not None
    ]

    # Combine original nodes and new nodes, avoiding duplicates
    all_node_names = {node["entity_name"] for node in node_datas}
    all_node_datas = node_datas + [
        node for node in new_node_datas if node["entity_name"] not in all_node_names
    ]
    all_node_names.update(node["entity_name"] for node in new_node_datas)

    # Fetch all edges between the combined set of nodes
    all_edges_data = await find_interconnected_edges(all_node_names, knowledge_graph_inst)
    
    return all_node_datas, all_edges_data


# async def _find_most_related_entities_from_relationships(
#     edge_datas: list[dict],
#     query_param: QueryParam,
#     knowledge_graph_inst: BaseGraphStorage,
# ):
#     entity_names = set()
#     for e in edge_datas:
#         entity_names.add(e["src_id"])
#         entity_names.add(e["tgt_id"])

#     node_datas = await asyncio.gather(
#         *[knowledge_graph_inst.get_node(entity_name) for entity_name in entity_names]
#     )

#     node_degrees = await asyncio.gather(
#         *[knowledge_graph_inst.node_degree(entity_name) for entity_name in entity_names]
#     )
#     node_datas = [
#         {**n, "entity_name": k, "rank": d}
#         for k, n, d in zip(entity_names, node_datas, node_degrees)
#     ]

#     node_datas = truncate_list_by_token_size(
#         node_datas,
#         key=lambda x: x["description"],
#         max_token_size=query_param.max_token_for_local_context,
#     )

#     return node_datas



async def naive_query(
    input_,
    chunks_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
):
    use_model_func = global_config["llm_model_func"]

    if isinstance(input_, dict):
        query = input_["text"]
        query_image = input_["image"]
    else:
        query = input_
    results = await chunks_vdb.query(query, top_k=query_param.top_k)
    
    if not len(results):
        return PROMPTS["fail_response"]
    chunks_ids = [r["id"] for r in results]
    chunks = await text_chunks_db.get_by_ids(chunks_ids)

    maybe_trun_chunks = truncate_list_by_token_size(
        chunks,
        key=lambda x: x["content"],
        max_token_size=query_param.max_token_for_text_unit,
    )
    logger.info(f"Truncate {len(chunks)} to {len(maybe_trun_chunks)} chunks")
    section = "\n".join([f"{i+1}. -------New Chunk------\n{c['content']}" for i, c in enumerate(maybe_trun_chunks)])
    if query_param.only_need_context:
        return section
    sys_prompt_temp = PROMPTS["naive_rag_response"]
    sys_prompt = sys_prompt_temp.format(
        content_data=section, response_type=query_param.response_type
    )

    if isinstance(input_, dict):
        # Use model functions to generate response
        response = await use_model_func(
            query,
            system_prompt=sys_prompt,
            query_image_path = query_image,
        )
    else:
        response = await use_model_func(
            query,
            system_prompt=sys_prompt,
        )

    if len(response) > len(sys_prompt):
        response = (
            response[len(sys_prompt) :]
            .replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )

    return response, section


async def no_rag(
    input_,
    query_param: QueryParam,
    global_config: dict,
):
    """
    Return results without any retrived augmented generation and using put query to LLM and return the response
    """
    use_model_func = global_config["llm_model_func"]
    
    sys_prompt_temp = PROMPTS["no_rag_response"]
    sys_prompt = sys_prompt_temp.format(
        response_type=query_param.response_type
    )

    if isinstance(input_, dict):
        query = input_["text"]
        image = input_["image"]

        # Use model functions to generate response
        response = await use_model_func(
            query,
            system_prompt=sys_prompt,
            query_image_path = image,
        )
    else:
        query = input_
        # Use model functions to generate response
        response = await use_model_func(
            query,
            system_prompt=sys_prompt,
        )



    if len(response) > len(sys_prompt):
        response = (
            response
            .replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )

    return response