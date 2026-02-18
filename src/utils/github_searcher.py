import json
import urllib

from config import config
from rag.code_splitter import CodeSplitter
from rag.loader import RepositoryLoader
from rag.vector_store import CodeVectorStore
from utils.vector_store_singleton import get_vector_store


class GitHubSearcher:
    def __init__(self) -> None:
        pass

    def index_repositories(self, query: str, max_repos: int = 3) -> str:
        token = config.environment.GITHUB_ACCESS_TOKEN
        if not token:
            return (
                "Error: GITHUB_ACCESS_TOKEN is not set in .env — cannot search GitHub."
            )

        repos = self._get_repos(max_repos=max_repos, token=token, query=query)

        loader = RepositoryLoader()
        splitter = CodeSplitter()
        store = get_vector_store()

        count = len(repos)
        summary_lines = [
            f"Indexing {count} repositor{'y' if count == 1 else 'ies'} for '{query}':"
        ]

        for repo in repos:
            full_name = repo["full_name"]
            owner = full_name.split("/", 1)[0]
            stars = repo.get("stargazers_count", 0)
            repository_info = self._store_repo_info(
                full_name, owner, stars, loader, splitter, store
            )
            summary_lines.append(repository_info)

        return "\n".join(summary_lines)

    def _get_repos(self, max_repos: int, token: str, query: str):
        max_repos = min(max_repos, 5)

        search_url = (
            "https://api.github.com/search/repositories"
            f"?q={urllib.parse.quote(query)}+language:python"
            f"&sort=stars&order=desc&per_page={max_repos}"
        )
        req = urllib.request.Request(
            search_url,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read())

        repos = data.get("items", [])
        if not repos:
            return f"No repositories found for query: {query!r}"

    def _store_repo_info(
        self,
        full_name: str,
        owner: str,
        stars: str,
        loader: RepositoryLoader,
        splitter: CodeSplitter,
        store: CodeVectorStore,
    ):
        try:
            loader.load_repository(repository_name=full_name, creator=owner)
            docs = loader.get_repository_documents()
            chunks = splitter.split(docs)
            store.add_documents(chunks)
            return (
                f"  + {full_name} ({stars:,} stars) — {len(docs)} files,"
                + "{len(chunks)} chunks"
            )
        except Exception as exc:
            return f"  - {full_name} — failed: {exc}"
