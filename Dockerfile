# VerifyBench image: the harness, the HumanEval mini suite and pytest for the
# graders, running as a non-root user. The base image is pinned by digest so a
# rebuild months from now gets the same bytes (refresh with `docker pull
# python:3.12-slim` and `docker image inspect --format '{{index .RepoDigests 0}}'`).
FROM python:3.12-slim@sha256:f77ac9e44ae96ef2c90b8053ea08c31f8be030f824196b0ae4db6d462c84e51f

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    VERIFYBENCH_RESULTS_DIR=/app/results \
    VERIFYBENCH_LOG_FORMAT=text

RUN useradd --create-home --uid 1000 --shell /usr/sbin/nologin app \
    && mkdir -p /app/results \
    && chown -R app:app /app

WORKDIR /app

# Dependency metadata first so the pip layer is reused while the code changes.
COPY pyproject.toml README.md LICENSE ./
COPY human_eval ./human_eval
COPY codeeval ./codeeval
COPY data ./data

# An editable install keeps /app as the project root, which is where the demo
# finds data/tasks/humaneval_mini.jsonl; pytest is what the graders run under.
RUN pip install --no-compile --editable . "pytest>=8.3" \
    && chown -R app:app /app

USER app

ENTRYPOINT ["verifybench"]
CMD ["demo"]
