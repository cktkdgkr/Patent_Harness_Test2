"""
Anthropic API wrapper for the FTO Harness.

Provides a unified interface for LLM calls across all agents,
with mock mode support for testing without API keys.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import yaml


@dataclass
class LLMResponse:
    content: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    raw_response: dict | None = None


def load_config(config_path: str | None = None) -> dict:
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class LLMClient:
    def __init__(self, config: dict | None = None, mock_mode: bool = False):
        self.config = config or load_config()
        self.mock_mode = mock_mode or self.config.get("mock_mode", {}).get("enabled", False)
        self.model = self.config.get("llm", {}).get("model", "claude-sonnet-4-6")
        self.max_tokens = self.config.get("llm", {}).get("max_tokens", 4096)
        self.temperature = self.config.get("llm", {}).get("temperature", 0.0)
        self._client = None

    def _get_client(self):
        if self._client is None:
            if self.mock_mode:
                return None
            try:
                import anthropic
                self._client = anthropic.Anthropic()
            except Exception as e:
                raise RuntimeError(
                    f"Failed to initialize Anthropic client. "
                    f"Set ANTHROPIC_API_KEY or enable mock_mode in config.yaml. Error: {e}"
                )
        return self._client

    def query(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        if self.mock_mode:
            return self._mock_query(system_prompt, user_message)

        client = self._get_client()
        response = client.messages.create(
            model=self.model,
            max_tokens=max_tokens or self.max_tokens,
            temperature=temperature if temperature is not None else self.temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        return LLMResponse(
            content=response.content[0].text,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            raw_response=None,
        )

    def query_json(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: int | None = None,
    ) -> dict:
        response = self.query(
            system_prompt=system_prompt + "\n\nRespond ONLY with valid JSON, no markdown fences.",
            user_message=user_message,
            max_tokens=max_tokens,
        )
        text = response.content.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
        return json.loads(text)

    def _mock_query(self, system_prompt: str, user_message: str) -> LLMResponse:
        msg_lower = user_message.lower()
        if "extract" in system_prompt.lower() and "sequence" in system_prompt.lower():
            mock_content = json.dumps({
                "sequences_found": [
                    {
                        "seq_id": "SEQ ID NO:1",
                        "sequence": "MKTLLLTLVVVTLVLSSPCILSQPVLTQPPSVSAAPGQRVTISC",
                        "location": "page 5, paragraph 2",
                        "confidence": 0.85,
                    }
                ],
                "accession_numbers": ["P12345", "Q67890"],
                "enzyme_names": ["lipase B", "Candida antarctica lipase"],
                "ec_numbers": ["EC 3.1.1.3"],
            })
        elif "claim" in system_prompt.lower():
            mock_content = json.dumps({
                "claims": [
                    {
                        "claim_number": 1,
                        "text": "An isolated polypeptide having at least 90% sequence identity to SEQ ID NO:1",
                        "sequence_references": ["SEQ ID NO:1"],
                        "identity_thresholds": ["90%"],
                    }
                ]
            })
        elif "search" in system_prompt.lower() or "web" in system_prompt.lower():
            mock_content = json.dumps({
                "search_clues": {
                    "accession_numbers": ["P12345"],
                    "enzyme_names": ["lipase B"],
                    "ec_numbers": ["EC 3.1.1.3"],
                    "organism": "Candida antarctica",
                },
                "search_queries": [
                    "P12345 uniprot protein sequence",
                    "Candida antarctica lipase B amino acid sequence",
                ],
            })
        else:
            mock_content = json.dumps({
                "status": "mock_response",
                "message": "This is a mock LLM response for testing.",
            })
        return LLMResponse(
            content=mock_content,
            model="mock-model",
            input_tokens=0,
            output_tokens=0,
        )
