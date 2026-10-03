# Free Public Hosting (Streamlit Community Cloud + Neon)

Total cost: **$0**. No credit card is needed for either service at the time of writing; free-tier limits change, so check each provider's pricing page.

## Why this setup (and not Vercel)
| Option | Verdict |
|---|---|
| **Streamlit Community Cloud** | Free, runs a long-lived Python process, deploys straight from GitHub. Used for the app. |
| **Neon (free Postgres)** | Free 0.5–1 GB serverless PostgreSQL. Used for the database. |
| Vercel / Netlify | Serverless functions with size and time limits and no persistent process; cannot run Streamlit or hold the ML models in memory. |

```
Browser ──► Streamlit Community Cloud (streamlit_app.py)
              ├── Streamlit UI
              └── FastAPI backend, embedded in the same process (API_URL = "embedded")
                    └──► Neon PostgreSQL (public demo sample)
```

## What the public demo contains
| | Local build | Public build |
|---|---|---|
| Booked customers | 50,000 | 30,000 |
| Intake applicants | 12,000 | 8,000 |
| Database size | ≈1.2 GB | ≈670 MB (the `raw_bureau_balance` build table is dropped after aggregation) |
| Models | `ml/artifacts/` (not committed) | `ml/artifacts_public/` (committed, ≈1.2 MB) |

The public models are **retrained on the public sample**, so their metrics differ slightly from the local build. The Model Monitor and Credit Risk pages always show the metrics of the models actually deployed.

## Steps
1. **Create a Neon project.** Sign up at neon.tech → *New project* (any name, nearest region, Postgres 16+). On the dashboard click **Connect** and copy the connection string. It looks like `postgresql://user:password@ep-xxx.region.aws.neon.tech/neondb?sslmode=require`.
2. **Build the public database** from your PC, in the project folder:
   ```bat
   .venv\Scripts\activate
   python -m scripts.build_public
   ```
   Paste the Neon connection string when asked. It takes a few minutes over the internet and ends with `Public demo database ready - size … MB`.
3. **Commit the public models**:
   ```bat
   git add .
   git status
   git commit -m "Free public demo: embedded API + Neon build"
   git push
   ```
4. **Deploy on Streamlit Community Cloud.** Sign in at share.streamlit.io with GitHub → *Create app* → *Deploy a public app from GitHub*. Use repository `<you>/risklens`, branch `main` and main file `streamlit_app.py`. Under *Advanced settings*, choose Python 3.12 or 3.13.
5. **Add the secrets** (Advanced settings → Secrets, TOML):
   ```toml
   API_URL = "embedded"
   DATABASE_URL = "postgresql://user:password@ep-xxx.region.aws.neon.tech/neondb?sslmode=require"
   MODEL_PATH = "ml/artifacts_public"
   ```
   Do not add an LLM key; the AI Risk Analyst runs in its free offline planner mode.
6. **Deploy and verify.** The first start installs the packages (a few minutes). Then check:
   - Overview KPIs load.
   - A Portfolio filter changes the numbers.
   - The Investigation Center opens the top queue customer.
   - The AI Risk Analyst answers "Which customers should the risk team investigate first?"

## Things to know
- **Cold starts.** Streamlit apps sleep after a period with no visitors, and Neon suspends idle compute. The first visit after a pause can take from about 30 seconds to a minute; open the app a few minutes before an interview.
- **Secrets.** The Neon password lives only in Streamlit's secrets, never in the repo. `.env` stays git-ignored.
- **AI SQL access on the free tier.** Neon's default user owns the tables, so the AI SQL tool uses the same login. It is still restricted by the SQL validator (single SELECT, whitelisted tables only), a READ ONLY transaction and a 5-second timeout. The answer trace reports the role as `… (read-only transaction)` rather than the `risklens_ai` least-privilege role used locally.
- **Data licence.** The underlying data is a public Kaggle competition dataset. Read its rules before publishing; sharing the app link (rather than listing it publicly) is the cautious option. The raw CSVs are never uploaded to GitHub.
- **Memory.** The app uses about 0.5 GB RAM in embedded mode, within the Community Cloud allowance.
