"""Gold patch location derivation tests without retaining patch bodies."""

import pytest
from pydantic import ValidationError

from repocompass.evaluation import (
    MAX_GOLD_PATCH_BYTES,
    GoldPatchLocalization,
    derive_gold_patch_localization,
)


def test_patch_derivation_maps_hunks_to_existing_and_added_python_symbols() -> None:
    source = """\
class Router:
    def retry(self, request):
        return request

    def existing(self):
        return None
"""
    patch = """\
diff --git a/src/pkg.py b/src/pkg.py
index 1111111..2222222 100644
--- a/src/pkg.py
+++ b/src/pkg.py
@@ -1,6 +1,10 @@ class Router:
 class Router:
     def retry(self, request):
-        return request
+        return self._retry(request)
+
+    def new_method(self):
+        return None
<BLANK-CONTEXT>
     def existing(self):
         return None
diff --git a/docs/guide.md b/docs/guide.md
index 3333333..4444444 100644
--- a/docs/guide.md
+++ b/docs/guide.md
@@ -1 +1 @@
-old
+new
""".replace("<BLANK-CONTEXT>", " ")

    result = derive_gold_patch_localization(
        patch,
        base_sources={"src/pkg.py": source},
    )

    assert result.files == ("docs/guide.md", "src/pkg.py")
    assert {(item.path, item.qualified_name) for item in result.symbols} == {
        ("src/pkg.py", "src.pkg.Router.retry"),
        ("src/pkg.py", "src.pkg.Router.new_method"),
    }
    assert result.python_file_count == 1
    assert result.ast_source_file_count == 1
    assert "patch" not in type(result).model_fields
    assert patch not in result.model_dump_json()


def test_deleted_and_new_files_use_the_repository_path_and_patch_definitions() -> None:
    patch = """\
diff --git a/src/old.py b/src/old.py
deleted file mode 100644
--- a/src/old.py
+++ /dev/null
@@ -1 +0,0 @@
-VALUE = 1
diff --git a/src/new.py b/src/new.py
new file mode 100644
--- /dev/null
+++ b/src/new.py
@@ -0,0 +1,2 @@
+def created():
+    return True
"""

    result = derive_gold_patch_localization(patch, base_sources={})

    assert result.files == ("src/new.py", "src/old.py")
    assert tuple((item.path, item.qualified_name) for item in result.symbols) == (
        ("src/new.py", "src.new.created"),
    )
    assert result.python_file_count == 2
    assert result.ast_source_file_count == 0


def test_derivation_is_content_addressed_and_rejects_tampering() -> None:
    patch = """\
diff --git a/src/a.py b/src/a.py
--- a/src/a.py
+++ b/src/a.py
@@ -1 +1 @@
-A = 1
+A = 2
"""
    result = derive_gold_patch_localization(patch, base_sources={})
    changed = result.model_copy(update={"python_file_count": 2})

    with pytest.raises(ValidationError, match="python file count|derivation_id"):
        GoldPatchLocalization.model_validate_json(changed.model_dump_json())


@pytest.mark.parametrize(
    "header",
    (
        "diff --git a/../escape.py b/../escape.py",
        "diff --git a/src\\escape.py b/src\\escape.py",
        "diff --git /absolute.py /absolute.py",
    ),
)
def test_unsafe_or_malformed_diff_paths_fail_closed(header: str) -> None:
    patch = f"{header}\n--- a/src/a.py\n+++ b/src/a.py\n@@ -1 +1 @@\n-a\n+b\n"
    with pytest.raises(ValueError, match="diff path|normalized"):
        derive_gold_patch_localization(patch, base_sources={})


def test_patch_and_source_bounds_are_enforced() -> None:
    with pytest.raises(ValueError, match="patch exceeds"):
        derive_gold_patch_localization(
            "x" * (MAX_GOLD_PATCH_BYTES + 1),
            base_sources={},
        )

    patch = """\
diff --git a/src/a.py b/src/a.py
--- a/src/a.py
+++ b/src/a.py
@@ -1 +1 @@
-A = 1
+A = 2
"""
    with pytest.raises(ValueError, match="base source exceeds"):
        derive_gold_patch_localization(
            patch,
            base_sources={"src/a.py": "x" * 1_048_577},
        )
