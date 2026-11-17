"""
rag.py
======
Retrieval-Augmented Generation (RAG).

Pipeline:
    consulta -> recuperação híbrida -> reranking -> seleção de contexto
             -> LLM -> resposta

Cada item do corpus é uma unidade de contexto; são enviados à LLM os Top-5
documentos após o reranking, com seus doc_ids. Duas condições são avaliadas:
  (A) sem RAG : a LLM recebe apenas a pergunta;
  (B) com RAG : a LLM recebe a pergunta + Top-5 documentos (com doc_ids).

A LLM é um modelo causal aberto (Qwen3, quantizado em 4 bits), executado na GPU via HuggingFace
Transformers. Recomenda-se temperatura 0 e seed fixa (reprodutibilidade).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import config


class TransformersLLM:
    """
    LLM local usando HuggingFace Transformers.

    Carrega um modelo causal quantizado (4 bits, NF4) do Hugging Face Hub na GPU e gera
    respostas com o PyTorch já instalado — sem instalar programas extras nem
    login (modelos públicos são baixados anonimamente na primeira execução).
    """

    def __init__(
        self,
        model: str = None,
        temperature: float = None,
        max_tokens: int = None,
        seed: int = None,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.model_name = model or config.LLM_MODEL
        self.temperature = temperature if temperature is not None else config.LLM_TEMPERATURE
        self.max_tokens = max_tokens if max_tokens is not None else config.LLM_MAX_TOKENS
        self.seed = seed if seed is not None else config.LLM_SEED

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA indisponível: o LLM quantizado (bitsandbytes) exige GPU NVIDIA "
                "com PyTorch CUDA instalado."
            )

        print(f"[LLM] Carregando {self.model_name} em 4 bits (NF4) na GPU ...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quant,
            dtype=torch.float16,
            device_map={"": 0},
            low_cpu_mem_usage=True,
        )
        self.model.eval()

        if self.seed is not None:
            torch.manual_seed(self.seed)

    def _build_inputs(self, prompt: str, system: Optional[str] = None):
        """Aplica o template de chat do modelo (se existir) ou o prompt bruto."""
        if system:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ]
        else:
            messages = [{"role": "user", "content": prompt}]

        if hasattr(self.tokenizer, "apply_chat_template"):
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        else:
            text = prompt

        return self.tokenizer(text, return_tensors="pt")

    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        """Gera uma resposta a partir do prompt (greedy se temperatura 0)."""
        import torch

        inputs = self._build_inputs(prompt, system).to(self.model.device)
        gen_kwargs = {
            "max_new_tokens": self.max_tokens,
            "do_sample": self.temperature > 0,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if self.temperature > 0:
            gen_kwargs["temperature"] = self.temperature

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        # Decodifica apenas os tokens gerados (depois do prompt).
        prompt_len = inputs["input_ids"].shape[1]
        generated = outputs[0][prompt_len:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()


# Construção de prompts
def build_prompt_no_rag(query: str) -> str:
    """
    Prompt da condição (A): a LLM recebe apenas a pergunta, sem acesso à
    coleção NFCorpus.
    """
    return (
        f"Responda à seguinte pergunta de forma direta e objetiva, em "
        f"{config.LLM_LANGUAGE}.\n\n"
        f"Pergunta: {query}\n\n"
        f"Resposta:"
    )


def build_prompt_rag(query: str, context_docs: List[Dict[str, str]]) -> str:
    """
    Prompt da condição (B): a LLM recebe a pergunta + Top-5 documentos com
    seus identificadores. Instrui a responder com base no contexto, citar as
    fontes (doc_ids) e indicar quando não houver evidência suficiente.
    """
    context_blocks = []
    for i, doc in enumerate(context_docs, start=1):
        context_blocks.append(
            f"[{i}] doc_id: {doc['doc_id']}\n    texto: {doc['text']}"
        )
    context_str = "\n".join(context_blocks)

    return (
        f"Você é um assistente que responde perguntas usando APENAS o contexto "
        f"fornecido abaixo. Responda em {config.LLM_LANGUAGE}.\n\n"
        f"Instruções:\n"
        f"1. Baseie sua resposta exclusivamente no contexto fornecido.\n"
        f"2. Indique, ao final, quais doc_ids foram usados como fonte.\n"
        f"3. Se o contexto não tiver evidência suficiente, diga explicitamente "
        f"que não há informação suficiente.\n\n"
        f"Contexto (documentos recuperados):\n{context_str}\n\n"
        f"Pergunta: {query}\n\n"
        f"Resposta:"
    )


def build_context(
    ranked_docs: List[tuple], corpus: Dict[str, str], top_k: int = None
) -> List[Dict[str, str]]:
    """
    Constrói a lista de contexto (Top-k documentos) a partir de um ranking
    [(doc_id, score)] já reranqueado, no formato esperado pelo prompt RAG.
    """
    top_k = top_k or config.RAG_TOP_K
    context = []
    for doc_id, _score in ranked_docs[:top_k]:
        context.append({"doc_id": doc_id, "text": corpus.get(doc_id, "")})
    return context
