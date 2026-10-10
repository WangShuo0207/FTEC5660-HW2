#!/usr/bin/env python3
"""FTEC5660 HW2 student starter: build an agent that verifies CVs via MCP."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import re
from pathlib import Path
from typing import Any


MCP_URL = "https://ftec5660.ngrok.app/mcp"
MODEL_NAME = "deepseek-v4-flash"
THRESHOLD = 0.5


def load_env_file(path: Path = Path(".env")) -> None:
    """Load the simple KEY=VALUE entries used by this homework."""
    if not path.is_file():
        return
    import os

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def cv_files(folder: Path) -> list[Path]:
    """Return PDFs directly inside *folder*, sorted numerically (CV_2 before CV_10)."""

    def key(path: Path) -> tuple[int, str]:
        digits = "".join(ch for ch in path.stem if ch.isdigit())
        return (int(digits) if digits else math.inf, path.name)

    return sorted(
        (p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"),
        key=key,
    )


def cv_text(path: Path) -> str:
    """Convert one CV PDF to markdown text."""
    from markitdown import MarkItDown

    return MarkItDown(enable_plugins=False).convert(str(path)).text_content


async def load_mcp_tools() -> list[Any]:
    """Connect to the course MCP server and return its tools as LangChain tools."""
    from langchain_mcp_adapters.client import MultiServerMCPClient

    async def _connect() -> list[Any]:
        client = MultiServerMCPClient(
            {
                "social_graph": {
                    "transport": "http",
                    "url": MCP_URL,
                    "headers": {"ngrok-skip-browser-warning": "true"},
                }
            }
        )
        return await client.get_tools()

    return await _with_retry(_connect, tries=5, delay=3.0)


# ---------------------------------------------------------------------------
# Our Task 1 implementation: structured extraction -> entity resolution ->
# deterministic field-by-field comparison against the SocialGraph profiles.
# ---------------------------------------------------------------------------

from difflib import SequenceMatcher

import asyncio as _asyncio
from typing import Awaitable, Callable


async def _with_retry(coro_factory: Callable[[], Awaitable[Any]], tries: int = 3, delay: float = 1.5) -> Any:
    """Re-await an MCP tool call a few times when the shared ngrok tunnel drops."""
    last: Exception | None = None
    for i in range(tries):
        try:
            return await coro_factory()
        except Exception as exc:  # network/protocol errors from the tunnel
            last = exc
            if i < tries - 1:
                await _asyncio.sleep(delay * (i + 1))
    if last is not None:
        raise last


# ---------- small text / JSON helpers ----------

def _parse_mcp(result: Any) -> Any:
    """MCP tools return a list of content blocks; pull out the JSON text."""
    blocks = result if isinstance(result, list) else getattr(result, "content", [result])
    texts: list[str] = []
    for block in blocks:
        if isinstance(block, dict) and block.get("type") == "text":
            texts.append(block.get("text", "") or "")
        elif hasattr(block, "text"):
            texts.append(block.text or "")
    joined = "\n".join(t for t in texts if t)
    if not joined:
        return None
    try:
        return json.loads(joined)
    except Exception:
        m = re.search(r"[\[\{].*[\]\}]", joined, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
        return joined


def _norm(s: Any) -> str:
    if s is None:
        return ""
    s = str(s).lower()
    s = s.replace("&", " and ")
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _sim(a: Any, b: Any) -> float:
    return SequenceMatcher(None, _norm(a), _norm(b)).ratio()


def _loose_in(a: str, b: str) -> bool:
    """True if normalized strings overlap strongly (substring or high ratio)."""
    na, nb = _norm(a), _norm(b)
    if not na or not nb:
        return False
    return na in nb or nb in na or _sim(na, nb) > 0.82


# seniority ranks used to catch inflated job titles
_SENIORITY_RANK = {
    "": 2, "junior": 1, "entry": 1, "associate": 1, "analyst": 1,
    "mid": 2, "intermediate": 2, "manager": 2, "professional": 2,
    "senior": 3, "sr": 3, "lead": 3, "staff": 3, "principal": 3,
    "executive": 4, "director": 4, "head": 4, "chief": 4, "vp": 4,
    "vice": 4, "president": 4, "partner": 4, "c-level": 4,
}
_CV_SENIOR_WORDS = {
    "senior": 3, "sr": 3, "lead": 3, "staff": 3, "principal": 3,
    "manager": 2, "head": 4, "director": 4, "chief": 4, "vp": 4,
    "vice": 4, "junior": 1, "associate": 1, "staff": 3,
}

_DEGREE_GROUPS = {
    "bsc": "bachelor", "bs": "bachelor", "b": "bachelor", "bachelor": "bachelor",
    "sc": "bachelor",
    "msc": "master", "ms": "master", "m": "master", "master": "master",
    "ma": "master", "me": "master",
    "mba": "mba",
    "phd": "phd", "ph": "phd", "doctor": "phd", "doctorate": "phd", "dr": "phd",
}


def _degree_group(s: str) -> str:
    n = _norm(s)
    if not n:
        return ""
    if n in _DEGREE_GROUPS:
        return _DEGREE_GROUPS[n]
    for k, v in _DEGREE_GROUPS.items():
        if k in n.split():
            return v
    return n


# ---------- prompts ----------

_EXTRACT_SYS = (
    "You are a meticulous HR data-extraction assistant. Read the CV text and "
    "return ONLY a JSON object (no prose, no markdown fence) with this exact "
    "schema:\n"
    '{"name": str, "headline": str, "city": str, "country": str, '
    '"jobs": [{"company": str, "title": str, "start_year": int|null, '
    '"end_year": int|null, "is_current": bool}], '
    '"education": [{"school": str, "degree": str, "field": str, '
    '"graduation_year": int|null}], '
    '"skills": [str]}\n'
    "Rules: use the years exactly as written on the CV. Infer is_current from "
    "'Present'. Split a location like 'London, UK' into city='London' and "
    "country='UK'. If a field is absent use null or empty string. Never invent "
    "a value that is not in the CV."
)


async def _extract_claims(model, cv_text: str) -> dict:
    resp = await model.ainvoke(
        [
            {"role": "system", "content": _EXTRACT_SYS},
            {"role": "user", "content": cv_text[:8000]},
        ]
    )
    text = resp.content if hasattr(resp, "content") else str(resp)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}


# ---------- entity resolution ----------

def _industry_from_claim(claims: dict) -> str | None:
    """Headlines look like 'Finance Professional' / 'AI Professional' -> 'Finance'."""
    headline = _norm(claims.get("headline", ""))
    if not headline:
        return None
    words = headline.split()
    if words and words[-1] == "professional":
        ind = " ".join(words[:-1]).strip()
        return ind or None
    return headline or None


async def _find_person(toolmap: dict, claims: dict) -> dict | None:
    """Search LinkedIn for the right person and return their full profile.

    Widens the search incrementally (city+industry -> city -> anywhere) and
    scores every candidate by how many CV companies/schools it matches.
    """
    t_search = toolmap["search_linkedin_people"]
    t_profile = toolmap["get_linkedin_profile"]

    name = claims.get("name", "") or ""
    city = claims.get("city", "") or ""
    if not name:
        return None
    industry = _industry_from_claim(claims)

    cv_companies = [_norm(j.get("company", "")) for j in claims.get("jobs", [])]
    cv_schools = [_norm(e.get("school", "")) for e in claims.get("education", [])]

    searches = [
        (city or None, industry),
        (city or None, None),
        (None, None),
    ]

    seen: set[int] = set()
    best, best_score = None, -1.0

    for loc, ind in searches:
        try:
            kwargs: dict = {"q": name, "limit": 20}
            if loc:
                kwargs["location"] = loc
            if ind:
                kwargs["industry"] = ind
            data = _parse_mcp(await _with_retry(lambda: t_search.ainvoke(kwargs)))
        except Exception:
            data = None
        if not isinstance(data, list):
            continue

        new_candidates = [
            c for c in data
            if isinstance(c, dict) and c.get("id") not in seen
            and _sim(c.get("name", ""), name) >= 0.6
        ]
        for c in new_candidates:
            seen.add(c["id"])
            try:
                prof = _parse_mcp(
                    await _with_retry(lambda pid=c["id"]: t_profile.ainvoke({"person_id": pid}))
                )
            except Exception:
                continue
            if not isinstance(prof, dict) or "experience" not in prof:
                continue
            score = 0.0
            for exp in prof.get("experience", []):
                pc = _norm(exp.get("company", ""))
                if any(pc and _loose_in(pc, cc) for cc in cv_companies):
                    score += 3
            for edu in prof.get("education", []):
                ps = _norm(edu.get("school", ""))
                if any(ps and _loose_in(ps, cs) for cs in cv_schools):
                    score += 3
            if city and _loose_in(prof.get("city", ""), city.split(",")[0]):
                score += 1
            if industry and _norm(prof.get("industry", "")) == _norm(industry):
                score += 1
            if _sim(prof.get("name", ""), name) < 0.75:
                score -= 5
            if score > best_score:
                best, best_score = prof, score

        # stop early only when very confident (>=2 strong signals, e.g. company+school)
        if best is not None and best_score >= 7.0:
            break

    # Require at least one strong identity signal (a matching company OR school)
    if best is None or best_score < 3.0:
        return None
    return best


# ---------- field-by-field comparison ----------

def _compare(claims: dict, li: dict) -> list[str]:
    mismatches: list[str] = []

    # name
    if _sim(li.get("name", ""), claims.get("name", "")) < 0.7:
        mismatches.append(f"name: CV '{claims.get('name')}' vs LinkedIn '{li.get('name')}'")

    # city
    cv_city = _norm(claims.get("city", "").split(",")[0])
    li_city = _norm(li.get("city", ""))
    if cv_city and li_city and not _loose_in(cv_city, li_city):
        mismatches.append(f"city: CV '{claims.get('city')}' vs LinkedIn '{li.get('city')}'")

    # jobs
    li_exps = li.get("experience", []) or []
    for cj in claims.get("jobs", []) or []:
        cco = _norm(cj.get("company", ""))
        best, best_s = None, 0.0
        for le in li_exps:
            r = _sim(cco, _norm(le.get("company", ""))) if cco else 0
            if r > best_s:
                best_s, best = r, le
        if best is None or best_s < 0.5:
            if cco:
                mismatches.append(f"company '{cj.get('company')}' not found on LinkedIn")
            continue

        # start year
        csy = cj.get("start_year")
        lsy = best.get("start_year")
        if csy and lsy and int(csy) != int(lsy):
            mismatches.append(
                f"job start year at {best.get('company')}: CV {csy} vs LinkedIn {lsy}"
            )
        # end year
        cey = cj.get("end_year")
        ley = best.get("end_year")
        if cey and ley and int(cey) != int(ley):
            mismatches.append(
                f"job end year at {best.get('company')}: CV {cey} vs LinkedIn {ley}"
            )
        # title / seniority inflation
        cv_title_words = _norm(cj.get("title", "")).split()
        li_sen = _SENIORITY_RANK.get(_norm(best.get("seniority", "")), 2)
        li_title_words = set(_norm(best.get("title", "")).split())
        cv_base = [w for w in cv_title_words if w not in _CV_SENIOR_WORDS]
        # only treat seniority words as inflation if they are NOT already part
        # of the LinkedIn title (otherwise "Manager" vs "Manager" looks inflated)
        cv_sen_words = [
            _CV_SENIOR_WORDS[w] for w in cv_title_words
            if w in _CV_SENIOR_WORDS and w not in li_title_words
        ]
        base_ok = (not cv_base) or any(
            w in li_title_words or _sim(w, " ".join(li_title_words)) > 0.6 for w in cv_base
        )
        if not base_ok:
            mismatches.append(
                f"job title at {best.get('company')}: CV '{cj.get('title')}' "
                f"vs LinkedIn '{best.get('title')}'"
            )
        elif cv_sen_words and max(cv_sen_words) > li_sen + 0.5:
            # CV explicitly claims a seniority level above what LinkedIn records
            mismatches.append(
                f"inflated title at {best.get('company')}: CV '{cj.get('title')}' "
                f"vs LinkedIn '{best.get('title')}' ({best.get('seniority')})"
            )

    # education
    for ce in claims.get("education", []) or []:
        csc = _norm(ce.get("school", ""))
        best, best_s = None, 0.0
        for le in li.get("education", []) or []:
            r = _sim(csc, _norm(le.get("school", ""))) if csc else 0
            if r > best_s:
                best_s, best = r, le
        if best is None or best_s < 0.5:
            if csc:
                mismatches.append(f"school '{ce.get('school')}' not found on LinkedIn")
            continue
        cgy = ce.get("graduation_year")
        lgy = best.get("end_year")
        if cgy and lgy and int(cgy) != int(lgy):
            mismatches.append(
                f"graduation year at {best.get('school')}: CV {cgy} vs LinkedIn {lgy}"
            )
        # degree level upgrade (e.g. PhD vs BSc)
        cdg = _degree_group(ce.get("degree", ""))
        ldg = _degree_group(best.get("degree", ""))
        order = {"": 0, "bachelor": 1, "master": 2, "mba": 3, "phd": 4}
        if cdg and ldg and order.get(cdg, 1) - order.get(ldg, 1) >= 2:
            mismatches.append(
                f"degree at {best.get('school')}: CV '{ce.get('degree')}' "
                f"vs LinkedIn '{best.get('degree')}'"
            )

    # skills
    li_skills = [_norm(s.get("name", "")) for s in li.get("skills", []) or []]
    for sk in claims.get("skills", []) or []:
        n = _norm(sk)
        if not n:
            continue
        if not any(n in ls or ls in n or _sim(n, ls) > 0.85 for ls in li_skills):
            mismatches.append(f"skill '{sk}' not on LinkedIn")

    return mismatches


def _score_from_mismatches(mismatches: list[str], found_person: bool) -> float:
    if not found_person:
        return 0.35  # could not even resolve identity -> suspicious
    if not mismatches:
        return 0.92
    # one confirmed discrepancy is enough to fail; more discrepancies -> lower
    if len(mismatches) == 1:
        return 0.22
    if len(mismatches) == 2:
        return 0.12
    return 0.05


# ---------- public API required by the homework ----------

def build_agent(tools: list[Any]) -> Any:
    """Create the DeepSeek model and bundle it with the MCP tool map."""
    from langchain_deepseek import ChatDeepSeek

    model = ChatDeepSeek(model=MODEL_NAME, temperature=0, timeout=60)
    toolmap = {t.name: t for t in tools}
    return {"model": model, "tools": toolmap}


async def _score_one(agent: dict, name: str, cv_text: str, sem: asyncio.Semaphore) -> tuple[str, float]:
    async with sem:
        try:
            model = agent["model"]
            toolmap = agent["tools"]
            claims = await _extract_claims(model, cv_text)
            if not claims or not claims.get("name"):
                return name, 0.4
            li = await _find_person(toolmap, claims)
            if li is None:
                return name, 0.35
            mismatches = _compare(claims, li)
            return name, _score_from_mismatches(mismatches, found_person=True)
        except Exception as exc:  # never crash the whole run
            print(f"[warn] {name} failed: {exc!r}")
            return name, 0.5


async def score_cvs(agent: Any, cvs: dict[str, str]) -> dict[str, float | None]:
    sem = asyncio.Semaphore(3)
    tasks = [_score_one(agent, name, text, sem) for name, text in cvs.items()]
    pairs = await asyncio.gather(*tasks)
    return {name: score for name, score in pairs}


# Everything below is provided runner/scoring code. No edits are needed.

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def parse_score(value: Any) -> float | None:
    """Accept a float/int, or text containing exactly one number, in [0, 1]."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        score = float(value)
    else:
        text = str(getattr(value, "content", value))
        matches = _NUMBER_RE.findall(text)
        if len(matches) != 1:
            return None
        score = float(matches[0])
    if math.isnan(score) or not 0.0 <= score <= 1.0:
        return None
    return score


def read_ground_truth(folder: Path) -> dict[str, dict[str, Any]]:
    """Read labels (1 = valid CV, 0 = has discrepancy) and reasons from the test folder."""
    path = folder / "ground_truth.json"
    if not path.is_file():
        return {}
    return {
        name: entry if isinstance(entry, dict) else {"label": entry}
        for name, entry in json.loads(path.read_text(encoding="utf-8")).items()
    }


def correctness_text(score: float | None, expected: dict[str, Any] | None) -> str:
    """Return `correct`, or an expected/predicted mismatch explanation."""
    if score is None:
        return "incorrect: score is missing or not a number in [0, 1]"
    if expected is None:
        return "not graded: no ground truth for this CV"
    label = int(expected["label"])
    predicted = 1 if score > THRESHOLD else 0
    if predicted == label:
        return "correct"
    reason = f" ({expected['reason']})" if expected.get("reason") else ""
    return f"incorrect: expected {label}{reason}, predicted {predicted}"


def write_results(names: list[str], scores: dict[str, Any], truth: dict[str, dict[str, Any]]) -> tuple[Path, int]:
    """Write the required results.csv file and return how many CVs were correct."""
    output = Path("results.csv")
    correct = 0
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cv", "score", "correctness"])
        for name in names:
            score = parse_score(scores.get(name))
            verdict = correctness_text(score, truth.get(name))
            correct += verdict == "correct"
            writer.writerow([name, "" if score is None else f"{score:.4f}", verdict])
    return output, correct


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FTEC5660 HW2 on CV PDFs")
    parser.add_argument(
        "--cv-folder",
        required=True,
        type=Path,
        help="folder containing CV PDF files",
    )
    return parser.parse_args()


async def run(folder: Path) -> int:
    paths = cv_files(folder)
    if not paths:
        raise SystemExit(f"no PDF files found in {folder}")

    load_env_file()
    cvs = {path.name: cv_text(path) for path in paths}
    tools = await load_mcp_tools()
    agent = build_agent(tools)
    scores = await score_cvs(agent, cvs)
    if not isinstance(scores, dict):
        raise TypeError("score_cvs() must return a dictionary")

    truth = read_ground_truth(folder)
    output, correct = write_results(list(cvs), scores, truth)
    summary = f" Accuracy: {correct}/{len(cvs)}." if truth else ""
    print(f"Processed {len(cvs)} CV(s). Wrote {output}.{summary}")
    return 0


def main() -> int:
    args = parse_args()
    if not args.cv_folder.is_dir():
        raise SystemExit(f"not a folder: {args.cv_folder}")
    return asyncio.run(run(args.cv_folder))


if __name__ == "__main__":
    raise SystemExit(main())
