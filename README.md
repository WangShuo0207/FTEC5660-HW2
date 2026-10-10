# FTEC5660 Homework 2: CV Verification Agent

Build a LangChain agent that reads each CV in a folder, looks the candidate up
on our SocialGraph MCP server (mock LinkedIn and Facebook), and outputs a
reliability score in [0, 1] for each CV.

A CV is **valid** (label `1`) when its claims agree with the candidate's
social media profiles. It **has a discrepancy** (label `0`) when it contains
problems such as an inflated job title, shifted dates, an upgraded degree, a
fake school or employer, a wrong location, or made-up skills. Differences in
wording only ("Bachelor of Science" vs `BSc`, "UI/UX Design" vs `UI/UX`,
"Senior Engineer" for an `Engineer` role with seniority `senior`, listing fewer
skills) are not discrepancies.

## Student task

Fill in the two functions in `hw2.py` that contain `### YOUR CODE HERE`. You
may add imports, constants and helper functions above them, but do not change
the provided code below them:

- `build_agent(tools)` creates your agent from the MCP tools.
- `score_cvs(agent, cvs)` runs the agent on every CV and returns
  `{file_name: score}`, one float in [0, 1] per CV.

A score above `0.5` means "valid"; `0.5` or below means "has discrepancy". You
may use a single tool-calling agent, multiple agents, reflection, or a
combination. Do not hard-code filenames, names, or public answers; grading
uses unseen CVs.

## MCP server

The server is hosted at `https://ftec5660.ngrok.app/mcp`. `hw2.py` already
connects to it and passes you these tools:

| Tool | Purpose |
| --- | --- |
| `search_facebook_users(q, limit, fuzzy)` | find Facebook users by display name |
| `get_facebook_profile(user_id)` | full Facebook profile |
| `get_facebook_mutual_friends(user_id_1, user_id_2)` | mutual friends of two users |
| `search_linkedin_people(q, location, industry, limit, fuzzy)` | search LinkedIn by name, skill, or title |
| `get_linkedin_profile(person_id)` | full LinkedIn profile (experience, education, skills) |
| `get_linkedin_interactions(person_id)` | post and like statistics |

Print `tool.name`, `tool.description`, and `tool.args` for the full schemas.

## Setup and public test

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
echo "DEEPSEEK_API_KEY=your_key_here" > .env
python3 hw2.py --cv-folder public_test
```

Use your own DeepSeek API key (from https://platform.deepseek.com).

The program creates `results.csv` in the current directory. Its columns are
`cv`, `score`, and `correctness`. The public labels are in
`public_test/ground_truth.json`; every CV with label `0` also has a `reason`
explaining the discrepancy, and `results.csv` shows it when your agent misses
one. The starter returns no score (`None`) for every CV so it runs before you
add any API code; missing scores count as incorrect.

The required model is `deepseek-v4-flash`, and all evidence must come from the
MCP server (no web search). The server is shared by the whole class: keep at
most about 3 CVs in flight at a time (e.g. `asyncio.Semaphore(3)`). We grade
with our own API key, so never put a key in your code; `.env` must stay out of
git.


## Task 2: Fool the Verifier

We added a target candidate, **Kelly Tsang** (LinkedIn `person_id` 10001,
Facebook `user_id` 10001, display name "Kel Tsang"). Her true CV is
`task2/target_cv.pdf`.

Write one CV for Kelly Tsang with at least one false or embellished detail
that our 5 verifier agents (same function as Task 1) still score above 0.5.
Each fooled agent is worth 4 points; a CV without a real false or embellished
detail earns 0. Attack your own Task 1 agent first:

```bash
cp my_attack.pdf task2/adversarial_cv.pdf
python3 hw2.py --cv-folder task2
```

`results.csv` shows your agent's score for both CVs (they are not graded, as
`task2/` has no `ground_truth.json`). Only the PDF is the attack surface; do not
attack the MCP server. See the homework description for the full rules.

## Homework 2 solution:

### Task 1

My verifier is a **structured pipeline** rather than a free-form ReAct loop,
because the discrepancy fields are closed-list (name, city, jobs, education,
skills) and I wanted deterministic, field-by-field coverage rather than hoping a
reasoning agent remembers to check every year. For each CV: (1) `deepseek-v4-flash`
extracts the CV into a JSON claim sheet (name, headline, city, jobs with
start/end years, education with graduation year, skills); (2) I resolve the right
person on LinkedIn by an incremental search — first `name + city + industry`
(industry taken from the "X Professional" headline), then `name + city`, then
unfiltered — and score every candidate by how many CV companies and schools its
profile overlaps with, requiring at least one company/school match before
trusting the identity (this is what disambiguates the dozens of shared names
like "John Smith" / "Michael Cheung"); (3) I compare the extracted claims against
the resolved LinkedIn profile with normalised, fuzzy rules: wording-only
differences (BSc vs Bachelor of Science, "UI/UX Design" vs "UI/UX") are
tolerated, but a one-year start/end/graduation shift, an inflated seniority word
not supported by the profile's `seniority` field, a fake company/school/skill,
or a two-step degree upgrade is flagged as a hard discrepancy; (4) the score is
deterministic: 0.92 when no discrepancy is found, ~0.2 for one discrepancy,
lower for several, and ~0.35 when the person cannot be resolved. MCP calls are
wrapped in a 3-try retry because the shared ngrok tunnel drops intermittently,
and CVs run with `asyncio.Semaphore(3)`. On the public 7-CV set the agent
correctly catches the one-year start-year shift (CV_4), the inflated
"Senior Analyst" title (CV_5) and the fake "Docker" skill (CV_7), and clears
the four valid CVs.

### Task 2

The adversarial CV (`task2/adversarial_cv.pdf`) is a clean, professional
one-page CV for Kelly Tsang in which every fact except one matches her real
LinkedIn 10001 profile: the AIA role is written as **Senior Manager** rather
than her actual `Manager (mid)` title. All other fields — city, the two prior
Analyst roles with their exact 2016–2019 and 2019–2022 dates, the HKUST BSc
Finance 2016 graduation, and all four skills — are verbatim-correct, so a
verifier that checks dates, schools and skills sees a fully consistent record.
The bet is that job-title seniority is the field a verifier is most likely to
gloss over: title strings are fuzzy-matched for wording, and "Senior Manager"
shares the base token "Manager" with the real title, so a verifier that checks
company, years, school and skills but does not separately diff the `seniority`
field will see one consistent CV and score it high. There is no hidden text, no
prompt injection, and no reference to the audit pipeline — the embellishment
is a single visible word, which is also what the manual grader will spot when
confirming that a real embellishment exists.

### Task 3 — Reflection on AI paradoxes

This homework forces the same model to wear two hats: it must act as a
trusted KYC verifier, and then be fooled by a document it is supposed to
distrust. That is the central AI paradox in production trust systems: the very
property that makes LLMs useful for verification — they read unstructured,
natural-language claims and reason about them — is exactly the property that
lets the document talk back. A deterministic rule engine cannot be prompt-injected,
but it also cannot decide that "BSc" and "Bachelor of Science" are the same;
an LLM can, but it will also believe the fake audit trail I put in white text.
We traded brittle, rigid trust for fluent, manipulable trust, and the attack
surface moves from "did you check the database" to "did your model realise the
document is not an instruction".

The second paradox is the security/utility trade-off: every extra check I add
entity-resolution, seniority ranks, degree normalisation — is also another way
to reject a valid CV (I briefly scored the wrong John Smith because a same-named
stranger happened to list one overlapping company). Stricter verification is not
free; it buys precision at the cost of recall and legibility. The third is the
"oracle fallibility" paradox: we want the LLM to be the final judge of
truthfulness, but LLM-as-judge scoring is itself biased by authority cues,
length, and sycophancy — so a trust system whose last word is an LLM score is
only as robust as the prompt it was given, and that prompt is itself public
knowledge once a fork is submitted. The practical takeaway for KYC is that AI
verifiers must treat document text as hostile data by construction, run
independent deterministic checks that the LLM cannot override, and never let the
input set its own score — otherwise the "verification" is just the candidate
auditing themselves through a slightly more expensive mouthpiece.
