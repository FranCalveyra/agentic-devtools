from langchain_community.document_loaders import GithubFileLoader

from config import config


class RepositoryLoader:
    """
    RepositoryLoader is responsible for loading Python files
    from a specified GitHub repository.

    This class utilizes the GithubFileLoader from langchain_community to fetch files
    from a given GitHub repository, specifically targeting Python (.py) files by default
    The loader requires a valid GitHub access token and repository details to perform
    its operation.

    Attributes:
        ACCESS_TOKEN (str): GitHub access token, sourced from environment configuration.
        loader (GithubFileLoader): An instance of GithubFileLoader
        that performs the file loading.

    Methods:
        __init__():
            Initializes the RepositoryLoader and sets the GitHub access token.

        load_repository(repository_name: str, creator: str):
            Prepares the loader to fetch Python files from the specified repository
            and owner/creator.

        get_repository_documents():
            Loads and returns all Python files (as Document objects) from the previously
            specified repository.
    """

    def __init__(self) -> None:
        self.ACCESS_TOKEN = config.environment.GITHUB_ACCESS_TOKEN

    # TODO: add the possibility to analyze further types
    def load_repository(self, repository_name: str, creator: str):
        """
        Loads a repository using its name and the creator name.
        It only loads the repository's Python files.

        Args:
            - `repository_name`: repository's name, with the given form: owner/repo_name
            - `creator`: creator's name.

        Returns:
            - None
        """
        self.loader = GithubFileLoader(
            repo=repository_name,
            access_token=self.ACCESS_TOKEN,
            creator=creator,
            include_prs=False,
            github_api_url="https://api.github.com",
            file_filter=lambda file_path: file_path.endswith(".py"),
        )

    def get_repository_documents(self):
        """
        Gets all Python files inside the repository.
        Returns:
            - List[Document]: a list of documents with the following structure:
                - page_content: literal content
                - metadata: a dictionary containing the document's me
        """
        return self.loader.load()
