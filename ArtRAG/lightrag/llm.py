import os
import copy
import json
import aioboto3
import numpy as np
from typing import List, Optional
import ollama
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from .base import BaseKVStorage
from .utils import compute_args_hash, wrap_embedding_func_with_attrs
import base64
import pdb
from functools import lru_cache
from PIL import Image

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info


from openai import (
    AsyncOpenAI,
    APIConnectionError,
    RateLimitError,
    Timeout,
    APITimeoutError,
    AsyncAzureOpenAI,
)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, Timeout)),
)
async def openai_complete_if_cache(
    model,
    prompt,
    system_prompt=None,
    history_messages=[],
    base_url=None,
    api_key=None,
    query_image_path: Optional[str] = None,
    system_image_paths: Optional[str] = None,
    **kwargs,
) -> str:
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    openai_async_client = (
        AsyncOpenAI() if base_url is None else AsyncOpenAI(base_url=base_url)
    )
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages = []
    # pdb.set_trace()
    if system_prompt:
        system_message = {"role": "user", "content": system_prompt}
        if system_image_paths:  # Assuming system_image_paths is a list of image paths
            images_content = []
            for image_path in system_image_paths:
                base64_image = encode_image(image_path)
                images_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}})
            
            system_message["content"] = [
                {"type": "text", "text": system_prompt},
                *images_content  # Unpack the list of image contents
            ]
        messages.append(system_message)
    messages.extend(history_messages)
    user_message = {"role": "user", "content": prompt}
    """
    To be done on image-text pair incontext learning
    """
    if query_image_path:
        base64_image = encode_image(query_image_path)
        user_message["content"] = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png; base64, {base64_image}"}}
        ]
    
    messages.append(user_message)

    if hashing_kv is not None:
        args_hash = compute_args_hash(model, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]

    response = await openai_async_client.chat.completions.create(
        model=model, messages=messages, **kwargs
    )

    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": response.choices[0].message.content, "model": model}}
        )
    return response.choices[0].message.content


class BedrockError(Exception):
    """Generic error for issues related to Amazon Bedrock"""


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, max=60),
    retry=retry_if_exception_type((BedrockError)),
)
async def bedrock_complete_if_cache(
    model,
    prompt,
    system_prompt=None,
    history_messages=[],
    aws_access_key_id=None,
    aws_secret_access_key=None,
    aws_session_token=None,
    **kwargs,
) -> str:
    os.environ["AWS_ACCESS_KEY_ID"] = os.environ.get(
        "AWS_ACCESS_KEY_ID", aws_access_key_id
    )
    os.environ["AWS_SECRET_ACCESS_KEY"] = os.environ.get(
        "AWS_SECRET_ACCESS_KEY", aws_secret_access_key
    )
    os.environ["AWS_SESSION_TOKEN"] = os.environ.get(
        "AWS_SESSION_TOKEN", aws_session_token
    )

    # Fix message history format
    messages = []
    for history_message in history_messages:
        message = copy.copy(history_message)
        message["content"] = [{"text": message["content"]}]
        messages.append(message)

    # Add user prompt
    messages.append({"role": "user", "content": [{"text": prompt}]})

    # Initialize Converse API arguments
    args = {"modelId": model, "messages": messages}

    # Define system prompt
    if system_prompt:
        args["system"] = [{"text": system_prompt}]

    # Map and set up inference parameters
    inference_params_map = {
        "max_tokens": "maxTokens",
        "top_p": "topP",
        "stop_sequences": "stopSequences",
    }
    if inference_params := list(
        set(kwargs) & set(["max_tokens", "temperature", "top_p", "stop_sequences"])
    ):
        args["inferenceConfig"] = {}
        for param in inference_params:
            args["inferenceConfig"][inference_params_map.get(param, param)] = (
                kwargs.pop(param)
            )

    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    if hashing_kv is not None:
        args_hash = compute_args_hash(model, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]

    # Call model via Converse API
    session = aioboto3.Session()
    async with session.client("bedrock-runtime") as bedrock_async_client:
        try:
            response = await bedrock_async_client.converse(**args, **kwargs)
        except Exception as e:
            raise BedrockError(e)

        if hashing_kv is not None:
            await hashing_kv.upsert(
                {
                    args_hash: {
                        "return": response["output"]["message"]["content"][0]["text"],
                        "model": model,
                    }
                }
            )

        return response["output"]["message"]["content"][0]["text"]


@lru_cache(maxsize=1)
def initialize_hf_model(model_name):
    if "VL" in model_name:
        # Initialize vision-language model
        min_pixels = 256*28*28
        max_pixels = 1280*28*28
        processor = AutoProcessor.from_pretrained(model_name,min_pixels=min_pixels, max_pixels=max_pixels)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            # attn_implementation="flash_attention_2",
        )
        return model, processor
    else:
        # Initialize regular language model
        hf_tokenizer = AutoTokenizer.from_pretrained(
            model_name, device_map="auto"
        )
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto"
        )
        if hf_tokenizer.pad_token is None:
            hf_tokenizer.pad_token = hf_tokenizer.eos_token
        return hf_model, hf_tokenizer

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(
        (RateLimitError, APIConnectionError, APITimeoutError)
    ),
)

# def resize_if_needed(image, max_size=512):
#     width, height = image.size
#     if width > max_size or height > max_size:
#         # Keep aspect ratio
#         if width > height:
#             new_width = max_size
#             new_height = int(height * (max_size/width))
#         else:
#             new_height = max_size
#             new_width = int(width * (max_size/height))
#         image = image.resize((new_width, new_height))
#     return image

async def hf_model_if_cache(
    model,
    prompt,
    system_prompt=None,
    history_messages=[],
    query_image_path: Optional[str] = None,
    **kwargs,
) -> str:
    model_name = model
    model, processor = initialize_hf_model(model_name)
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
            
    messages.extend(history_messages)
    
    if query_image_path:
        image = Image.open(query_image_path).convert('RGB')
        # image = resize_if_needed(image)
        content = [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt}
        ]
        messages.append({"role": "user", "content": content})
    else:
        messages.append({"role": "user", "content": prompt})

    kwargs.pop("hashing_kv", None)

    if "VL" in model_name:
        # Process for vision-language model
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, _ = process_vision_info(messages)
        print("image_inputs", image_inputs)
        inputs = processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(model.device)
        
        output = model.generate(
            **inputs,
            max_new_tokens=512,
            num_return_sequences=1,
            early_stopping=True
        )
        
        output_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, output)
        ]
        response_text = processor.batch_decode(
            output_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
    else:
        # Process for regular language model
        input_prompt = ""
        try:
            input_prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            try:
                ori_message = copy.deepcopy(messages)
                if messages[0]["role"] == "system":
                    messages[1]["content"] = (
                        "<system>"
                        + messages[0]["content"]
                        + "</system>\n"
                        + messages[1]["content"]
                    )
                    messages = messages[1:]
                    input_prompt = processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
            except Exception:
                len_message = len(ori_message)
                for msgid in range(len_message):
                    input_prompt = (
                        input_prompt
                        + "<"
                        + ori_message[msgid]["role"]
                        + ">"
                        + ori_message[msgid]["content"]
                        + "</"
                        + ori_message[msgid]["role"]
                        + ">\n"
                    )

        input_ids = processor(
            input_prompt, return_tensors="pt", padding=True, truncation=True
        ).to(model.device)
        inputs = {k: v.to(model.device) for k, v in input_ids.items()}
        output = model.generate(
            **input_ids, max_new_tokens=128, num_return_sequences=1, early_stopping=True
        )
        response_text = processor.decode(
            output[0][len(inputs["input_ids"][0]):], skip_special_tokens=True
        )

    return response_text

# async def hf_model_if_cache(
#     model,
#     prompt,
#     system_prompt=None,
#     history_messages=[],
#     query_image_path: Optional[str] = None,
#     **kwargs,
# ) -> str:
#     model_name = model
#     model, processor = initialize_hf_model(model_name)
    
#         # Prepare batch of messages
#     batch_messages = []
    
#     # Handle single prompt/image case for backward compatibility
#     if isinstance(prompts, str):
#         prompts = [prompts]
#     if isinstance(query_image_paths, str):
#         query_image_paths = [query_image_paths]
    
#     # Create message for each prompt-image pair
#     for i, prompt in enumerate(prompts):
#         message = []
        
#         # Add system prompt if provided
#         if system_prompt:
#             message.append({"role": "system", "content": system_prompt})
        
#         # Add history messages if provided
#         if history_messages:
#             message.extend(history_messages)
        
#         # Add current prompt with image if available
#         if query_image_paths and i < len(query_image_paths):
#             image = Image.open(query_image_paths[i]).convert('RGB')
#             content = [
#                 {"type": "image", "image": image},
#                 {"type": "text", "text": prompt}
#             ]
#             message.append({"role": "user", "content": content})
#         else:
#             message.append({"role": "user", "content": prompt})
        
#         batch_messages.append(message)

#     kwargs.pop("hashing_kv", None)

#     if "VL" in model_name:
#         # Process for vision-language model
        
#         # Apply chat template to all messages
#         texts = [
#             processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
#             for msg in batch_messages
#         ]
        
#         # Process vision inputs
#         image_inputs, video_inputs = process_vision_info(batch_messages)
        
#         # Prepare batch inputs
#         inputs = processor(
#             text=texts,
#             images=image_inputs,
#             videos=video_inputs,
#             padding=True,
#             return_tensors="pt"
#         )
#         inputs = inputs.to(model.device)
        
#         # Batch inference
#         output = model.generate(
#             **inputs,
#             max_new_tokens=512,
#             num_return_sequences=1,
#             early_stopping=True
#         )
        
#         # Process output
#         output_trimmed = [
#             out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, output)
#         ]
#         response_text = processor.batch_decode(
#             output_trimmed,
#             skip_special_tokens=True,
#             clean_up_tokenization_spaces=False
#         )[0]
        
#     else:
#         # Process for regular language model
#         input_prompt = ""
#         try:
#             input_prompt = processor.apply_chat_template(
#                 messages, tokenize=False, add_generation_prompt=True
#             )
#         except Exception:
#             try:
#                 ori_message = copy.deepcopy(messages)
#                 if messages[0]["role"] == "system":
#                     messages[1]["content"] = (
#                         "<system>"
#                         + messages[0]["content"]
#                         + "</system>\n"
#                         + messages[1]["content"]
#                     )
#                     messages = messages[1:]
#                     input_prompt = processor.apply_chat_template(
#                         messages, tokenize=False, add_generation_prompt=True
#                     )
#             except Exception:
#                 len_message = len(ori_message)
#                 for msgid in range(len_message):
#                     input_prompt = (
#                         input_prompt
#                         + "<"
#                         + ori_message[msgid]["role"]
#                         + ">"
#                         + ori_message[msgid]["content"]
#                         + "</"
#                         + ori_message[msgid]["role"]
#                         + ">\n"
#                     )

#         input_ids = processor(
#             input_prompt, return_tensors="pt", padding=True, truncation=True
#         ).to(model.device)
#         inputs = {k: v.to(model.device) for k, v in input_ids.items()}
#         output = model.generate(
#             **input_ids, max_new_tokens=512, num_return_sequences=1, early_stopping=True
#         )
#         response_text = processor.decode(
#             output[0][len(inputs["input_ids"][0]):], skip_special_tokens=True
#         )

#     return response_text

async def ollama_model_if_cache(
    model, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    kwargs.pop("max_tokens", None)
    kwargs.pop("response_format", None)

    ollama_client = ollama.AsyncClient()
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    if hashing_kv is not None:
        args_hash = compute_args_hash(model, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]

    response = await ollama_client.chat(model=model, messages=messages, **kwargs)

    result = response["message"]["content"]

    if hashing_kv is not None:
        await hashing_kv.upsert({args_hash: {"return": result, "model": model}})

    return result


async def gpt_4o_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await openai_complete_if_cache(
        "gpt-4o",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def gpt_4o_mini_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await openai_complete_if_cache(
        "gpt-4o-mini",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def bedrock_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await bedrock_complete_if_cache(
        "anthropic.claude-3-haiku-20240307-v1:0",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def hf_model_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    model_name = kwargs["hashing_kv"].global_config["llm_model_name"]
    return await hf_model_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def ollama_model_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    model_name = kwargs["hashing_kv"].global_config["llm_model_name"]
    return await ollama_model_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


@wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=8192)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, Timeout)),
)
async def openai_embedding(
    texts: list[str],
    model: str = "text-embedding-3-small",
    base_url: str = None,
    api_key: str = None,
) -> np.ndarray:
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    openai_async_client = (
        AsyncOpenAI() if base_url is None else AsyncOpenAI(base_url=base_url)
    )
    response = await openai_async_client.embeddings.create(
        model=model, input=texts, encoding_format="float"
    )
    return np.array([dp.embedding for dp in response.data])


# @wrap_embedding_func_with_attrs(embedding_dim=1024, max_token_size=8192)
# @retry(
#     stop=stop_after_attempt(3),
#     wait=wait_exponential(multiplier=1, min=4, max=10),
#     retry=retry_if_exception_type((RateLimitError, APIConnectionError, Timeout)),  # TODO: fix exceptions
# )
async def bedrock_embedding(
    texts: list[str],
    model: str = "amazon.titan-embed-text-v2:0",
    aws_access_key_id=None,
    aws_secret_access_key=None,
    aws_session_token=None,
) -> np.ndarray:
    os.environ["AWS_ACCESS_KEY_ID"] = os.environ.get(
        "AWS_ACCESS_KEY_ID", aws_access_key_id
    )
    os.environ["AWS_SECRET_ACCESS_KEY"] = os.environ.get(
        "AWS_SECRET_ACCESS_KEY", aws_secret_access_key
    )
    os.environ["AWS_SESSION_TOKEN"] = os.environ.get(
        "AWS_SESSION_TOKEN", aws_session_token
    )

    session = aioboto3.Session()
    async with session.client("bedrock-runtime") as bedrock_async_client:
        if (model_provider := model.split(".")[0]) == "amazon":
            embed_texts = []
            for text in texts:
                if "v2" in model:
                    body = json.dumps(
                        {
                            "inputText": text,
                            # 'dimensions': embedding_dim,
                            "embeddingTypes": ["float"],
                        }
                    )
                elif "v1" in model:
                    body = json.dumps({"inputText": text})
                else:
                    raise ValueError(f"Model {model} is not supported!")

                response = await bedrock_async_client.invoke_model(
                    modelId=model,
                    body=body,
                    accept="application/json",
                    contentType="application/json",
                )

                response_body = await response.get("body").json()

                embed_texts.append(response_body["embedding"])
        elif model_provider == "cohere":
            body = json.dumps(
                {"texts": texts, "input_type": "search_document", "truncate": "NONE"}
            )

            response = await bedrock_async_client.invoke_model(
                model=model,
                body=body,
                accept="application/json",
                contentType="application/json",
            )

            response_body = json.loads(response.get("body").read())

            embed_texts = response_body["embeddings"]
        else:
            raise ValueError(f"Model provider '{model_provider}' is not supported!")

        return np.array(embed_texts)


async def hf_embedding(texts: list[str], tokenizer, embed_model) -> np.ndarray:
    input_ids = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True
    ).input_ids
    with torch.no_grad():
        outputs = embed_model(input_ids)
        embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.detach().numpy()


async def ollama_embedding(texts: list[str], embed_model, **kwargs) -> np.ndarray:
    """
    Deprecated in favor of `embed`.
    """
    embed_text = []
    ollama_client = ollama.Client(**kwargs)
    for text in texts:
        data = ollama_client.embeddings(model=embed_model, prompt=text)
        embed_text.append(data["embedding"])

    return embed_text


if __name__ == "__main__":
    import asyncio

    async def main():
        result = await gpt_4o_mini_complete("How are you?")
        print(result)

    asyncio.run(main())
