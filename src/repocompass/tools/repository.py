from pathlib import Path

IGNORED_DIRECTORY_NAMES: frozenset[str] = frozenset(
    {
        ".git",
        ".venv",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        "node_modules",
        "dist",
        "build",
    }
)

def list_tree(
    repo_path: Path,
    max_depth: int = 3,
    max_entries: int = 200,
) -> list[str]:
    '''List repository entries as relative POSIX-style paths.
    
    Args:
        repo_path: Path to the repository root directory.
        max_depth: Maximum traversal depth. The repository root is depth 0.
        max_entries: Maximum number of entries to return
    Returns:
        A stable list of relative paths. Directory paths end with ``/``,
        and the repository root itself is excluded.
    '''
        
    if not repo_path.exists():
        raise FileNotFoundError(f"Repository path does not exist: {repo_path}")
    
    if not repo_path.is_dir():
        raise NotADirectoryError(f"Repository path is not a directory: {repo_path}")
    
    if max_depth < 0:
        raise ValueError(f"Max_depth must be non-negative, current max_depth is {max_depth}")
    
    if max_entries <= 0:
        raise ValueError(f"Max_entries must be positive, current max_entries is {max_entries}")
    
    
    current_depth = 0
    entries =  []
    _walk(repo_path, repo_path, entries, current_depth, max_depth, max_entries)
    
    return entries

def _walk(
    dir_path: Path,
    repo_path: Path,
    entries: list[str],
    current_depth: int=0,
    max_depth: int=3,
    max_entries: int=200,
    ) -> None:
    '''Recursive helper of list tree'''
    if current_depth >= max_depth or len(entries) >= max_entries:
        return
    
    for item in sorted(dir_path.iterdir()):
        
        if len(entries) >= max_entries:
            return

        if not item.is_symlink() and item.is_dir():
            
            if item.name in IGNORED_DIRECTORY_NAMES:
                continue
 
            entries.append(item.relative_to(repo_path).as_posix() + "/")
            child_depth = current_depth + 1
            
            if child_depth < max_depth:
                _walk(item, repo_path, entries, child_depth, max_depth, max_entries)
                
            continue

        entries.append(item.relative_to(repo_path).as_posix())
        
    return
    
    