# Guide to release new version of the Micro Manager

The developer who is releasing a new version of the the Micro Manager is expected to follow this work flow:

The release of the `micro-manager` repository is made directly from a release branch called `micro-manager-v1.2.3`. This branch is mainly needed to help other developers with testing.

1. Create a branch called `micro-manager-v1.2.3` from the latest commit of the `develop` branch.

2. If it is a real release, [open a Pull Request `main` <-- `micro-manager-v1.2.3`](https://github.com/precice/micro-manager/compare/main...main) named after the version (i.e. `Release v1.2.3`) and briefly describe the new features of the release in the PR description.

3. Bump the version in the `CHANGELOG.md` and in `setup.py` on `micro-manager-v1.2.3`.

4. [Draft a new release](https://github.com/precice/micro-manager/releases/new) in the `Releases` section of the repository page in a web browser. The release tag needs to be the exact version number (i.e.`v1.2.3` or `v1.2.3rc1`, compare to [existing tags](https://github.com/precice/micro-manager/tags)). Use `@target:main`. Release title is also the version number (i.e. `v1.2.3` or `v1.2.3rc1`, compare to [existing releases](https://github.com/precice/micro-manager/tags)).

    * *Note:* If it is a pre-release then the option *This is a pre-release* needs to be selected at the bottom of the page. Use `@target:micro-manager-v1.2.3` for a pre-release, since we will never merge a pre-release into `main`.
    * Use the `Auto-generate release notes` feature.

    a) If a pre-release is made: Directly hit the "Publish release" button in your Release Draft. Now you can check the artifacts (e.g. release on [PyPI](https://pypi.org/project/micro-manager-precice/#history)) of the release. *Note:* As soon as a new tag is created github actions will take care of deploying the new version on PyPI using [this workflow](https://github.com/precice/micro-manager/actions?query=workflow%3A%22Upload+Python+Package%22).

    b) If this is a "real" release: As soon as one approving review is made, merge the release PR (from `micro-manager-v1.2.3`) into `main`.

5. Merge `main` into `develop` for synchronization of `develop`.

6. If everything is in order up to this point then the new version can be released by hitting the "Publish release" button in your Release Draft. This will create the corresponding tag and trigger [publishing the release to PyPI](https://github.com/precice/micro-manager/actions?query=workflow%3A%22Upload+Python+Package%22).

7. Add an empty commit on main via `git checkout main`, then `git commit --allow-empty -m "post-tag bump"`. Check that everything is in order via `git log`. Important: The `tag` and `origin/main` should not point to the same commit.
