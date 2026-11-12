# Walmart Naive Bayes Timesplit

Bayes' theorem and a Naive Bayes classifier on real Walmart weekly sales data.

The project answers three questions from a (fictional) regional operations manager:

1. How often does a store have a strong week, and is it more likely in holiday weeks, in November–December, or in cold weeks?
2. How good is the current rule "add extra staff only in holiday weeks"?
3. Can a simple Naive Bayes model pick strong weeks better than that rule?

## Project structure

```
.
├── data/
│   └── Walmart.csv          # not in the repo, download it (see below)
├── main.ipynb               # full analysis: Parts P, B and N
├── note.txt                 # 5-sentence note to the stakeholder
├── results.pdf              # filled-in result sheet 
├── .gitignore
└── README.md
```

## 1. Download the dataset

The dataset is the [Walmart Dataset on Kaggle](https://www.kaggle.com/datasets/yasserh/walmart-dataset): `Walmart.csv`, 6,435 rows (45 stores × 143 weeks, February 2010 to October 2012). It is not committed to this repository.

**Option A: from the website**

1. Open the Kaggle page above and sign in.
2. Click **Download** and unzip the archive.
3. Put `Walmart.csv` in a `data/` folder at the root of the project.

**Option B: with the Kaggle CLI**

```bash
pip install kaggle
# place your API token at ~/.kaggle/kaggle.json first (Kaggle → Settings → API → Create New Token)
kaggle datasets download -d yasserh/walmart-dataset -p data --unzip
```

## 2. Set up the environment

Tested with Python 3.9.

```bash
git clone <your-repo-url>
cd <repo-folder>

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install pandas numpy scikit-learn jupyter
```

## 3. Run the notebook

```bash
jupyter notebook main.ipynb
```

Then use **Kernel → Restart & Run All**. The notebook runs top to bottom and checks the data on load (6,435 rows, 45 stores, 143 weeks, 450 holiday store-weeks).

## Event definitions

One row = one store-week. The sample space is all 6,435 store-weeks.

| Event | Definition |
|---|---|
| S | Strong week: weekly sales ≥ 1.10 × that store's average weekly sales |
| H | Holiday week: `Holiday_Flag = 1` |
| Q | Holiday season: the week is in November or December |
| C | Cold week: temperature < 40 °F |
| F | Expensive fuel: fuel price ≥ $3.50 |
| J | High unemployment: unemployment ≥ 8 % |
| G | Large store: the store's average weekly sales ≥ $1,000,000 |

The Naive Bayes model trains on 2010–2011 and tests on 2012. Store averages for the model come from the training weeks only, so the test weeks stay unseen.

## Results

| Task | Quantity | Value |
|---|---|---|
| P1 | rows, stores, weeks, holiday store-weeks | 6,435 rows; 45 stores; 143 weeks; 450 holiday store-weeks |
| P2 | P(S), P(H), P(Q), P(C), P(F), P(J), P(G) | P(S) = 0.1265; P(H) = 0.0699; P(Q) = 0.1259; P(C) = 0.1537; P(F) = 0.4707; P(J) = 0.4696; P(G) = 0.4889 |
| P3 | P(S ∩ H), P(S ∩ Q), P(H ∩ Q), P(S ∩ C) | 0.0190 (122); 0.0517 (333); 0.0280 (180); 0.0287 (185) |
| B1 | P(S \| H), P(S \| Hᶜ) | 0.2711 (122/450); 0.1156 (692/5,985) |
| B2 | P(H \| S) by Bayes, and by counting | 0.1499 by Bayes; 0.1499 by counting (122/814) |
| B3 | holiday with the highest P(S \| g); group with the largest P(g \| S) | Thanksgiving, P(S \| g) = 0.8222; No holiday, P(g \| S) = 0.8501 |
| B4 | P(S \| Q), P(Q \| S) | 0.4111 (333/810); 0.4091 (333/814) |
| B5 | holiday rule: precision, recall; Nov–Dec rule: precision, recall | Holiday rule: 0.2711, 0.1499; Nov–Dec rule: 0.4111, 0.4091 |
| B6 | precision of the holiday rule if strong weeks doubled, if halved | Doubled (P(S) = 0.2530): 0.4652; halved (P(S) = 0.0632): 0.1478 |
| N3 | P(S \| profile a), P(S \| profile b) | 0.7939; 0.0301 |
| N6 | accuracy / precision / recall | Naive Bayes: 0.8682 / 0.0000 / 0.0000; Never strong: 0.8682 / 0.0000\* / 0.0000; Strong if holiday: 0.8455 / 0.2556 / 0.0902 |

\* "Never strong" flags no weeks, so its precision is undefined (0/0) and is reported as 0. Naive Bayes at the 0.5 threshold also flags no weeks, so its row is identical. Part B uses store averages from all 143 weeks; N3 and N6 use averages from the training years only, with P(S) = 0.1276 (574/4,500) in training and P(S) = 0.1318 (255/1,935) in test.

## Key Findings:

> Across the 45 stores, a store has a strong week (sales at least 10% above its own average) in 12.6% of weeks, or about 1 week in 8. Thanksgiving week matters most, with a strong week 82% of the time, and November–December as a whole raises the chance to 41%, compared with 9% in other months, while cold weeks help only a little (19%). The current holiday rule is right only 27% of the time it adds staff and catches just 15% of strong weeks, so it misses 85% of them; adding staff for all of November–December would be right 41% of the time and catch 41% of strong weeks. Our simple model did not beat the holiday rule: when tested on 2012, it predicted no strong weeks at all, while the holiday rule found 9% of them. The main risk is that the 2012 data ends in October, so the model was never tested on November–December, its strongest signal, and its results may look better or worse once a full holiday season is included.

## Data source

Walmart Dataset by yasserh on Kaggle. The people and requests in the scenario are fictional; the data is public.
