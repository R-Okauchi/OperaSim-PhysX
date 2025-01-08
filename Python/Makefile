.PHONY: dev

_fmt_check:
	poetry run ruff format . --check --diff

_fix_check:
	poetry run ruff check .

_mypy:
	poetry run mypy ./

lint:
	make -j _fmt_check _fix_check _mypy

_ruff_format_apply:
	poetry run ruff format

fmt:
	poetry run ruff format .

fix:
	poetry run ruff check . --fix

test:
	export DATABASE_URL=postgresql://test_user:test_password@localhost:5433/test_db && \
	poetry run prisma db push --schema ./prisma_schemas/schema.prisma && \
	poetry run pytest -v

dev:
	poetry run uvicorn api.main:app --reload

prisma_generate:
	poetry run prisma generate --schema ./prisma_schemas/schema.prisma

openapi:
	poetry run python ./api/openapi.py > ./api/openapi.json