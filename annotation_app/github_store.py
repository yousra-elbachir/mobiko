"""Optional GitHub-branch storage backend for annotation output.

Persists each ``<task>_<name>_annotated.json`` to a GitHub repo **branch** via
the REST API, so annotations survive restarts on ephemeral hosts (e.g. Streamlit
Community Cloud, whose disk is wiped on every reboot/sleep).

Design:
  - Code and data are kept apart. Annotations are written to a dedicated branch
    (default ``annotations``) that the deployment does NOT track, so saving never
    triggers a redeploy of the app.
  - Only the latest file is pushed; git history *is* the snapshot trail.
  - stdlib only (urllib) — no extra dependency.

Configuration comes from Streamlit secrets (a ``[github]`` table) or env vars:
    repo    "owner/name"      GITHUB_REPO    (required)
    token   fine-grained PAT  GITHUB_TOKEN   (required; Contents: read & write)
    branch  "annotations"     GITHUB_BRANCH  (optional, default: annotations)
    subdir  "" | "outputs"    GITHUB_SUBDIR  (optional path prefix in the repo)

If repo or token is missing, ``from_config`` returns None and the app just uses
local disk.
"""

import base64
import json
import os
import urllib.error
import urllib.parse
import urllib.request

_API = "https://api.github.com"


class GitHubBackend:
    def __init__(self, repo: str, token: str, branch: str = "annotations", subdir: str = ""):
        self.repo = repo.strip().strip("/")
        self.token = token.strip()
        self.branch = (branch or "annotations").strip()
        self.subdir = (subdir or "").strip().strip("/")
        self._sha: dict[str, str] = {}      # repo path -> last known blob sha
        self._branch_ready = False

    @classmethod
    def from_config(cls, cfg: dict | None) -> "GitHubBackend | None":
        if not cfg:
            return None
        repo, token = cfg.get("repo"), cfg.get("token")
        if not repo or not token:
            return None
        return cls(repo, token, cfg.get("branch", "annotations"), cfg.get("subdir", ""))

    # ------------------------------------------------------------------ #
    def _req(self, method: str, url: str, body: dict | None = None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Authorization", f"Bearer {self.token}")
        req.add_header("Accept", "application/vnd.github+json")
        req.add_header("X-GitHub-Api-Version", "2022-11-28")
        if data is not None:
            req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.status, json.loads(r.read().decode() or "{}")

    def _repo_path(self, basename: str) -> str:
        return f"{self.subdir}/{basename}" if self.subdir else basename

    def _contents_url(self, repo_path: str) -> str:
        return f"{_API}/repos/{self.repo}/contents/{urllib.parse.quote(repo_path)}"

    def _ensure_branch(self) -> None:
        """Create the target branch from the repo's default branch if it's missing."""
        if self._branch_ready:
            return
        try:
            self._req("GET", f"{_API}/repos/{self.repo}/branches/"
                             f"{urllib.parse.quote(self.branch)}")
            self._branch_ready = True
            return
        except urllib.error.HTTPError as e:
            if e.code != 404:
                raise
        _, repo = self._req("GET", f"{_API}/repos/{self.repo}")
        default = repo["default_branch"]
        _, ref = self._req("GET", f"{_API}/repos/{self.repo}/git/ref/"
                                  f"heads/{urllib.parse.quote(default)}")
        self._req("POST", f"{_API}/repos/{self.repo}/git/refs",
                  {"ref": f"refs/heads/{self.branch}", "sha": ref["object"]["sha"]})
        self._branch_ready = True

    def _fetch_sha(self, repo_path: str) -> str | None:
        url = self._contents_url(repo_path) + f"?ref={urllib.parse.quote(self.branch)}"
        try:
            _, data = self._req("GET", url)
            return data.get("sha")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            raise

    # ------------------------------------------------------------------ #
    def pull_latest(self, dst_path: str) -> bool:
        """Fetch the remote copy of ``basename(dst_path)`` into dst_path.
        Returns True if it existed remotely, False if not yet present."""
        repo_path = self._repo_path(os.path.basename(dst_path))
        url = self._contents_url(repo_path) + f"?ref={urllib.parse.quote(self.branch)}"
        try:
            _, data = self._req("GET", url)
        except urllib.error.HTTPError as e:
            if e.code == 404:      # branch or file not there yet
                return False
            raise
        content = base64.b64decode(data["content"])
        os.makedirs(os.path.dirname(os.path.abspath(dst_path)) or ".", exist_ok=True)
        with open(dst_path, "wb") as f:
            f.write(content)
        self._sha[repo_path] = data["sha"]
        return True

    def push_latest(self, src_path: str, message: str | None = None) -> None:
        """Commit the contents of src_path to the target branch (create/update)."""
        self._ensure_branch()
        repo_path = self._repo_path(os.path.basename(src_path))
        with open(src_path, "rb") as f:
            payload = base64.b64encode(f.read()).decode()
        body = {
            "message": message or f"Update {os.path.basename(src_path)}",
            "content": payload,
            "branch": self.branch,
        }
        if self._sha.get(repo_path):
            body["sha"] = self._sha[repo_path]
        try:
            _, data = self._req("PUT", self._contents_url(repo_path), body)
        except urllib.error.HTTPError as e:
            if e.code in (409, 422):        # stale/missing sha -> refetch once and retry
                sha = self._fetch_sha(repo_path)
                if sha:
                    body["sha"] = sha
                else:
                    body.pop("sha", None)
                _, data = self._req("PUT", self._contents_url(repo_path), body)
            else:
                raise
        self._sha[repo_path] = data["content"]["sha"]
