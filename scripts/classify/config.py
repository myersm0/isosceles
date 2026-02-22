"""Default configuration for the classify pipeline."""

task_defaults = {
	"littre": {
		"type": "dictionary",
	},
	"compare": {
		"type": "compare",
	},
	"lemma": {
		"type": "llm",
		"model": "mistral-small3.2:24b",
		"think": True,
	},
	"tense": {
		"type": "morph_llm",
		"model": "qwen3:8b",
		"think": True,
		"ensemble": [
			{"model": "gemma3:12b", "think": True},
		],
	},
	"aux": {
		"type": "llm",
		"model": "qwen3:8b",
		"think": True,
	},
	"que": {
		"type": "llm",
		"model": "gemma3:12b",
		"think": True,
	},
	"adjadv": {
		"type": "llm",
		"model": "qwen3:8b",
		"think": True,
	},
}

all_tasks = list(task_defaults.keys())
