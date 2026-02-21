"""Ollama API client for classifier prompts."""

import json
import sys

import requests


OLLAMA_URL = "http://localhost:11434/api/chat"


def call_ollama(model, system_prompt, user_prompt, temperature=0.0, think=True, timeout=120):
	options = {"temperature": temperature}
	if not think:
		options["think"] = False
	payload = {
		"model": model,
		"messages": [
			{"role": "system", "content": system_prompt},
			{"role": "user", "content": user_prompt},
		],
		"stream": False,
		"format": "json",
		"options": options,
	}
	try:
		response = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
		response.raise_for_status()
		content = response.json()["message"]["content"]
		return json.loads(content)
	except (requests.RequestException, json.JSONDecodeError, KeyError) as error:
		print(f"  error: {error}", file=sys.stderr)
		return None
