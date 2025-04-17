# generate llms.txt for dpytest
[group('llmstxt')]
llmstxt-dpytest:
    uv run llmstxt-architect \
    --urls https://dpytest.readthedocs.io/en/latest/index.html \
    https://dpytest.readthedocs.io/en/latest/tutorials/index.html \
    https://dpytest.readthedocs.io/en/latest/modules/index.html \
    https://dpytest.readthedocs.io/en/latest/modules/backend.html \
    https://dpytest.readthedocs.io/en/latest/modules/callbacks.html \
    https://dpytest.readthedocs.io/en/latest/modules/factories.html \
    https://dpytest.readthedocs.io/en/latest/modules/runner.html \
    https://dpytest.readthedocs.io/en/latest/modules/state.html \
    https://dpytest.readthedocs.io/en/latest/modules/utils.html \
    https://dpytest.readthedocs.io/en/latest/modules/verify.html \
    https://dpytest.readthedocs.io/en/latest/modules/websocket.html \
    https://dpytest.readthedocs.io/en/latest/tutorials/using_pytest.html \
    --max-depth 5 \
    --llm-name claude-3-7-sonnet-latest \
    --llm-provider anthropic \
    --project-dir docs/ai_docs/dpytest

# update llms.txt for dpytest
[group('llmstxt')]
llmstxt-dpytest-update:
    uv run llmstxt-architect \
    --existing-llms-file docs/ai_docs/dpytest/llms.txt \
    --max-depth 5 \
    --llm-name claude-3-7-sonnet-latest \
    --llm-provider anthropic \
    --update-descriptions-only \
    --project-dir docs/ai_docs/dpytest

# generate llms.txt for langgraph
[group('llmstxt')]
llmstxt-langgraph:
    uv run llmstxt-architect \
    --urls https://langchain-ai.github.io/langgraph/concepts/ \
    https://langchain-ai.github.io/langgraph/how-tos/ \
    https://langchain-ai.github.io/langgraph/tutorials/workflows/ \
    https://langchain-ai.github.io/langgraph/tutorials/introduction/ \
    https://langchain-ai.github.io/langgraph/tutorials/langgraph-platform/local-server/ \
    https://python.langchain.com/docs/get_started/introduction/ \
    https://python.langchain.com/docs/modules/ \
    https://python.langchain.com/docs/use_cases/ \
    --max-depth 5 \
    --llm-name claude-3-7-sonnet-latest \
    --llm-provider anthropic \
    --project-dir docs/ai_docs/langgraph

# update llms.txt for langgraph
[group('llmstxt')]
llmstxt-langgraph-update:
    uv run llmstxt-architect \
    --existing-llms-file docs/ai_docs/langgraph/llms.txt \
    --max-depth 5 \
    --llm-name claude-3-7-sonnet-latest \
    --llm-provider anthropic \
    --update-descriptions-only \
    --project-dir docs/ai_docs/langgraph

# generate llms.txt for langchain
[group('llmstxt')]
llmstxt-langchain:
    uv run llmstxt-architect \
    --urls https://python.langchain.com/docs/get_started/introduction/ \
    https://python.langchain.com/docs/modules/ \
    https://python.langchain.com/docs/use_cases/ \
    --max-depth 5 \
    --llm-name claude-3-7-sonnet-latest \
    --llm-provider anthropic \
    --project-dir docs/ai_docs/langchain

# update llms.txt for langchain
[group('llmstxt')]
llmstxt-langchain-update:
    uv run llmstxt-architect \
    --existing-llms-file docs/ai_docs/langchain/llms.txt \
    --max-depth 5 \
    --llm-name claude-3-7-sonnet-latest \
    --llm-provider anthropic \
    --update-descriptions-only \
    --project-dir docs/ai_docs/langchain

# generate llms.txt for discord
[group('llmstxt')]
llmstxt-discord:
    uv run llmstxt-architect \
    --urls https://discordpy.readthedocs.io/en/stable/index.html \
    https://discordpy.readthedocs.io/en/stable/intro.html \
    https://discordpy.readthedocs.io/en/stable/quickstart.html \
    https://discordpy.readthedocs.io/en/stable/logging.html \
    https://discordpy.readthedocs.io/en/stable/discord.html \
    https://discordpy.readthedocs.io/en/stable/intents.html \
    https://github.com/Rapptz/discord.py/tree/v2.5.2/examples \
    https://discordpy.readthedocs.io/en/stable/faq.html \
    https://discordpy.readthedocs.io/en/stable/genindex.html \
    https://discordpy.readthedocs.io/en/stable/ext/commands/index.html \
    https://discordpy.readthedocs.io/en/stable/ext/tasks/index.html \
    https://discordpy.readthedocs.io/en/stable/interactions/api.html \
    https://discordpy.readthedocs.io/en/stable/ext/commands/api.html \
    https://discordpy.readthedocs.io/en/stable/ext/tasks/index.html \
    https://discordpy.readthedocs.io/en/stable/migrating.html \
    --max-depth 5 \
    --llm-name claude-3-7-sonnet-latest \
    --llm-provider anthropic \
    --project-dir docs/ai_docs/discord

# update llms.txt for discord
[group('llmstxt')]
llmstxt-discord-update:
    uv run llmstxt-architect \
    --existing-llms-file docs/ai_docs/discord/llms.txt \
    --max-depth 5 \
    --llm-name claude-3-7-sonnet-latest \
    --llm-provider anthropic \
    --update-descriptions-only \
    --project-dir docs/ai_docs/discord
