# generate llms.txt for dpytest
[group('llmstxt')]
llmstxt-dpytest:
    uv run llmstxt-architect \
    --urls https://dpytest.readthedocs.io/en/latest/modules/index.html \
    https://dpytest.readthedocs.io/en/latest/modules/index.html \
    --max-depth 3 \
    --llm-name claude-3-7-sonnet-latest \
    --llm-provider anthropic \
    --project-dir docs/ai_docs/dpytest

# update llms.txt for dpytest
[group('llmstxt')]
llmstxt-dpytest-update:
    uv run llmstxt-architect \
    --existing-llms-file docs/ai_docs/dpytest/llms.txt \
    --max-depth 3 \
    --llm-name claude-3-7-sonnet-latest \
    --llm-provider anthropic \
    --update-descriptions-only \
    --project-dir docs/ai_docs/dpytest
