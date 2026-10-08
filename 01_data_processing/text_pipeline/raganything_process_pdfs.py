#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
raganything_process_pdfs.py

功能：
- 单个 PDF 或一个文件夹批量 PDF
- 解析（MinerU/Docling） -> 多模态内容分析（图/表/公式） -> 写入 RAGAnything/LightRAG 存储
- 可选：处理后立即 query

依赖：
pip install raganything
（建议：pip install 'raganything[all]' 以获得更多格式支持）
"""

import os
import sys
import argparse
import asyncio
from pathlib import Path
from typing import Optional, List

from raganything import RAGAnything, RAGAnythingConfig
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    return v if v not in (None, "") else default


def build_rag(
    working_dir: str,
    parser_name: str,
    parse_method: str,
    api_key: str,
    base_url: Optional[str],
    llm_model: str,
    vision_model: str,
    embed_model: str,
    embed_dim: int,
    embed_max_tokens: int,
    enable_image: bool,
    enable_table: bool,
    enable_equation: bool,
) -> RAGAnything:
    """
    按官方 README 的用法示例构造 RAGAnything：
    - llm_model_func 使用 openai_complete_if_cache
    - vision_model_func 兼容 messages / 单图 image_data / 纯文本回退
    - embedding_func 使用 openai_embed + EmbeddingFunc
    """
    config = RAGAnythingConfig(
        working_dir=working_dir,
        parser=parser_name,         # "mineru" or "docling"
        parse_method=parse_method,  # "auto" or "ocr" or "txt"
        enable_image_processing=enable_image,
        enable_table_processing=enable_table,
        enable_equation_processing=enable_equation,
    )

    def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        return openai_complete_if_cache(
            llm_model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )

    def vision_model_func(prompt, system_prompt=None, history_messages=[],
                          image_data=None, messages=None, **kwargs):
        # 1) VLM-enhanced query：如果直接传 messages，就原样走
        if messages:
            return openai_complete_if_cache(
                vision_model,
                "",
                system_prompt=None,
                history_messages=[],
                messages=messages,
                api_key=api_key,
                base_url=base_url,
                **kwargs,
            )
        # 2) 单图：image_data 是 base64
        elif image_data:
            msg_list = [
                {"role": "system", "content": system_prompt} if system_prompt else None,
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
                    ],
                },
            ]
            msg_list = [m for m in msg_list if m is not None]
            return openai_complete_if_cache(
                vision_model,
                "",
                system_prompt=None,
                history_messages=[],
                messages=msg_list,
                api_key=api_key,
                base_url=base_url,
                **kwargs,
            )
        # 3) 没有图：回退到纯文本 LLM
        else:
            return llm_model_func(prompt, system_prompt, history_messages, **kwargs)

    embedding_func = EmbeddingFunc(
        embedding_dim=embed_dim,
        max_token_size=embed_max_tokens,
        func=lambda texts: openai_embed(
            texts,
            model=embed_model,
            api_key=api_key,
            base_url=base_url,
        ),
    )

    rag = RAGAnything(
        config=config,
        llm_model_func=llm_model_func,
        vision_model_func=vision_model_func,
        embedding_func=embedding_func,
    )
    return rag


async def process_one_pdf(
    rag: RAGAnything,
    pdf_path: str,
    output_dir: str,
    parse_method: Optional[str] = None,
):
    pdf_path = str(Path(pdf_path).resolve())
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    await rag.process_document_complete(
        file_path=pdf_path,
        output_dir=output_dir,
        parse_method=parse_method or "auto",
    )


async def process_folder_pdfs(
    rag: RAGAnything,
    folder_path: str,
    output_dir: str,
    recursive: bool = True,
    max_workers: int = 4,
):
    folder_path = str(Path(folder_path).resolve())
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    await rag.process_folder_complete(
        folder_path=folder_path,
        output_dir=output_dir,
        file_extensions=[".pdf"],
        recursive=recursive,
        max_workers=max_workers,
    )


async def run_query(rag: RAGAnything, question: str, mode: str = "hybrid"):
    # 纯文本 query（对已入库内容检索+生成）
    result = await rag.aquery(question, mode=mode)
    return result


def parse_args():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--pdf", type=str, help="单个 PDF 路径")
    g.add_argument("--pdf_dir", type=str, help="包含多个 PDF 的文件夹路径")

    p.add_argument("--output_dir", type=str, default="./output", help="解析中间产物输出目录（图片/文本等）")
    p.add_argument("--working_dir", type=str, default="./rag_storage", help="LightRAG/RAGAnything 存储目录（索引/图/向量等）")

    p.add_argument("--parser", type=str, default=_env("PARSER", "mineru"), choices=["mineru", "docling"])
    p.add_argument("--parse_method", type=str, default=_env("PARSE_METHOD", "auto"), choices=["auto", "ocr", "txt"])

    p.add_argument("--api_key", type=str, default=_env("OPENAI_API_KEY"), help="OpenAI API Key（或兼容网关的 key）")
    p.add_argument("--base_url", type=str, default=_env("OPENAI_BASE_URL"), help="可选：OpenAI-compatible Base URL")

    p.add_argument("--llm_model", type=str, default="gpt-4o-mini")
    p.add_argument("--vision_model", type=str, default="gpt-4o")
    p.add_argument("--embed_model", type=str, default="text-embedding-3-large")
    p.add_argument("--embed_dim", type=int, default=3072)
    p.add_argument("--embed_max_tokens", type=int, default=8192)

    p.add_argument("--no_image", action="store_true", help="禁用图像处理")
    p.add_argument("--no_table", action="store_true", help="禁用表格处理")
    p.add_argument("--no_equation", action="store_true", help="禁用公式处理")

    p.add_argument("--recursive", action="store_true", help="pdf_dir 递归扫描子目录")
    p.add_argument("--max_workers", type=int, default=4, help="批处理并发 worker 数")

    p.add_argument("--query", type=str, default=None, help="处理完成后立刻发起一次 query")
    p.add_argument("--query_mode", type=str, default="hybrid", choices=["hybrid", "local", "global", "naive"])

    return p.parse_args()


async def main():
    args = parse_args()

    if not args.api_key:
        raise RuntimeError("缺少 --api_key 或环境变量 OPENAI_API_KEY")

    rag = build_rag(
        working_dir=args.working_dir,
        parser_name=args.parser,
        parse_method=args.parse_method,
        api_key=args.api_key,
        base_url=args.base_url,
        llm_model=args.llm_model,
        vision_model=args.vision_model,
        embed_model=args.embed_model,
        embed_dim=args.embed_dim,
        embed_max_tokens=args.embed_max_tokens,
        enable_image=not args.no_image,
        enable_table=not args.no_table,
        enable_equation=not args.no_equation,
    )

    # 可选：MinerU 安装检查（仅在你用 mineru 时有意义）
    if args.parser == "mineru":
        ok = rag.check_parser_installation()
        if not ok:
            raise RuntimeError("MinerU 未正确安装/配置：rag.check_parser_installation() 返回 False")

    if args.pdf:
        await process_one_pdf(rag, args.pdf, args.output_dir, parse_method=args.parse_method)
    else:
        await process_folder_pdfs(
            rag,
            args.pdf_dir,
            args.output_dir,
            recursive=args.recursive,
            max_workers=args.max_workers,
        )

    if args.query:
        ans = await run_query(rag, args.query, mode=args.query_mode)
        print("\n===== QUERY RESULT =====\n")
        print(ans)


if __name__ == "__main__":
    asyncio.run(main())
