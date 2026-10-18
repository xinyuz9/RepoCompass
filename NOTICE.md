# Third-party notices

RepoCompass's original code is MIT-licensed. Dependencies retain their respective licenses.

The UI build emits [bundled dependency notices](src/repocompass/dashboard/static/THIRD_PARTY_LICENSES.md).
Font licenses for JetBrains Mono and Manrope are included in the static distribution. Python
dependency notices remain with installed package metadata in the environment or container.

The `swe-bench-lite-localization-*` manifests and the cross-repository subset contain task
text and derived localization labels from [SWE-bench Lite](https://huggingface.co/datasets/SWE-bench/SWE-bench_Lite),
frozen at revision `b0dde1093fe417d83b7184254edf8199c1f0dff5`. Credit belongs to the
[SWE-bench authors and original issue contributors](https://github.com/SWE-bench/SWE-bench).
The upstream project's [MIT notice](licenses/SWE-bench-MIT.txt) is retained here.
Each case records the upstream repository, instance ID, and base commit. RepoCompass adds
file/symbol localization labels; these are not patch-resolution results or an official subset.
Upstream repository source and issue contributions retain their respective rights and licenses;
RepoCompass's MIT grant covers its original work, not a relicensing of third-party material.
