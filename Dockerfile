# claude-openai-proxy in a container.
#
# The claude CLI needs your Claude Code credentials. Those live in ~/.claude on
# the host (created when you run `claude` and log in once). Mount that directory
# into the container at runtime — do NOT bake credentials into the image.
#
#   docker build -t claude-proxy .
#   docker run --rm -p 8088:8088 \
#       -v "$HOME/.claude:/root/.claude" \
#       claude-proxy
#
# Then point clients at http://localhost:8088/v1
FROM node:22-slim

# Python is needed for the proxy (stdlib only); node carries the claude CLI.
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && npm install -g @anthropic-ai/claude-code

WORKDIR /opt/claude-proxy
COPY claude_openai_proxy.py .

ENV CLAUDE_PROXY_HOST=0.0.0.0 \
    CLAUDE_PROXY_PORT=8088 \
    CLAUDE_PROXY_DEFAULT_MODEL=sonnet \
    CLAUDE_PROXY_TIMEOUT=600

EXPOSE 8088
CMD ["python3", "claude_openai_proxy.py"]
