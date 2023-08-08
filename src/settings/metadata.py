import tomllib

_project_file = open("pyproject.toml", "rb")
_project_toml = tomllib.load(_project_file)["project"]

PROJECT_NAME: str = _project_toml["name"]
DEPENDENCY_LIST: list[str] = _project_toml["dependencies"]
