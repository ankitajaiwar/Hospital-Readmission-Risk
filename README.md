# Predicting which hospital patients will be readmitted within 30 days

When a patient leaves the hospital, some of them end up back within a month. Hospitals
try to prevent this with follow-up calls, medication checks and early appointments,
but care teams can only reach so many people. I wanted to see how well a simple model
could point them to the patients who need the most help.

I built this project from start to finish: checking and cleaning the data, training
the model, testing it honestly, and putting it behind a small web service that returns
a risk score for a patient.

## The data

I used a public dataset of about 100,000 hospital stays by diabetic patients from 130
US hospitals between 1999 and 2008. It has no names or other identifying details.
([UCI dataset 296](https://archive.ics.uci.edu/dataset/296), free to use.)

Before training, I removed a few kinds of records:

- Patients who died or went to hospice, since they can't be readmitted.
- Repeat visits by the same patient. I kept only each patient's first visit, so the
  same person can't show up in both the data the model learns from and the data I
  test it on.

That left 69,987 patients. About 9% of them came back within 30 days.

## How I tested it

The data has no dates, but the visit IDs run roughly in time order. So I trained the
model on the earliest 60% of visits, fine-tuned it on the next 20%, and tested it on
the most recent 20%. This is closer to real life, where a model trained on last year's
patients is used on this year's, than mixing everything up at random.

## Results

On the most recent 13,998 patients:

| What I measured | Result | In plain terms |
|---|---|---|
| Ranking ability (AUROC) | 0.63 | If you pick one patient who was readmitted and one who wasn't, the model gives the readmitted one a higher score 63% of the time. 50% would be a coin flip. |
| Top 5% of patients by score | 21% were readmitted | That's about 3 times the normal rate of 7%. |
| Top 10% of patients by score | Caught 23% of all readmissions | A team calling 1 in 10 patients would reach about a quarter of the people who come back. |

![Results charts](docs/evaluation.png)

The full numbers are in [docs/REPORT.md](docs/REPORT.md).

## What I learned

**It's good at sorting patients by risk, not at predicting with certainty.** A score
of 0.63 is modest, but that's normal for this kind of billing-style data, which
doesn't include lab trends or doctors' notes. It's still useful: the patients it
ranks highest really are much more likely to come back. If each follow-up call costs
$250 and prevents 1 in 5 readmissions, the most value comes from calling roughly the
top 15% of patients. (Those costs are my guesses, and a hospital would plug in its
own.)

**The strongest warning signs** were where the patient went after discharge (home,
a nursing facility, and so on) and how many times they'd been in the hospital in the
past year.

**The model's risk estimates drifted over time.** Readmissions were less common in the
later records (about 7%) than the earlier ones. Because the model learned from earlier
patients, it overestimated risk for recent ones, predicting about 8.5% on average. In
a real hospital, I'd re-adjust the model on the most recent few months before using
it, and keep checking it every month.

**The way hospitals recorded information changed.** Insurance type and the admitting
doctor's specialty were filled in very differently in later records. Changes like this
usually come from new hospital systems or workflows, and they can quietly make a model
worse. I added an automatic check that flags these shifts.

**It didn't work equally well for everyone.** The model was noticeably worse at
picking out high-risk Black patients than white patients. At the top-10% cutoff it
caught 14% of readmissions among Black patients versus 32% among white patients. It
also missed most high-risk patients over 90. Race and sex were not used as inputs; I
only used them to check fairness. Before anyone relied on this, I'd work through the
gap with clinicians, look at which inputs drive it, and adjust.

## What's in the code

- **Data checks:** the pipeline stops with a clear message if the data has missing
  columns, impossible values or duplicate records.
- **Two models compared:** a simple logistic regression and a gradient-boosted model
  (scikit-learn). The better one is kept and its scores are adjusted so that a score
  of 0.2 means roughly a 20% chance of readmission.
- **A report** written automatically after every run.
- **A web service** (FastAPI): send it a patient's details and it returns a risk score.
  It runs in Docker.
- **19 automated tests** that run on every change through GitHub Actions. They use fake
  data in the same format, so they don't need the real file.

## Running it yourself

You need Python 3.10 or newer.

```bash
pip install -r requirements.txt
export PYTHONPATH=src          # Windows PowerShell: $env:PYTHONPATH = "src"
python -m pytest -q            # run the tests
python -m readmit.download     # download the data
python -m readmit.pipeline --data data/raw/diabetic_data.csv --out artifacts
uvicorn readmit.api:app --port 8000   # then open http://localhost:8000/docs
```

On Mac or Linux, `make test`, `make train`, `make serve` and `make docker` do the same.

## If I were doing this for a real hospital

1. Retrain on the hospital's own recent records, with real dates.
2. Run it quietly for a month or two, scoring patients without showing anyone, then
   check how its predictions compared with what actually happened.
3. Choose how many patients to flag together with the care team, based on how many
   calls they can make.
4. Track whether flagged patients actually got contacted, and whether readmissions
   went down.
5. Check every month for drift and for differences across patient groups.

## Limits

The data is old (1999–2008) and only covers diabetic patients. It can't see
readmissions to a different hospital. The cost figures are my own placeholders. This
is a demonstration of how I'd approach the problem, not a tool for patient care.

---
Ankita Singh, Ph.D. · [LinkedIn](https://www.linkedin.com/in/ankitasingh1292) · [Google Scholar](https://scholar.google.com/citations?user=xRtUSq0AAAAJ)
