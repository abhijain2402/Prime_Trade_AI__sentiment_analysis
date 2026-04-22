# 📊 Bitcoin Market Sentiment & Trader Performance Analysis
### Hyperliquid × Fear & Greed Index — Data Science Assignment
> **Submitted for:** Primetrade.ai
> **Author:** Abhi Jain  
> **Dataset Period:** 2023 – 2024  
> **Total Trades Analysed:** 211,224 trades across multiple crypto assets

---

## 📁 Project Structure

```
├── main.ipynb                  # Main analysis notebook
├── fear_greed_index.csv        # Bitcoin Fear & Greed Index dataset
├── historical_data.csv         # Hyperliquid historical trader dataset
├── fear_greed_index.json       # JSON export of fear/greed data
├── historical_data.json        # JSON export of historical trades
└── README.md                   # This file
```

---

## 🎯 Objective

The goal of this project is to **explore the relationship between Bitcoin market sentiment (Fear & Greed Index) and trader performance on Hyperliquid**, one of the leading decentralised perpetual exchanges. By merging trade-level data with daily sentiment scores, we aim to uncover:

- How trader **profitability** shifts across different market sentiment phases
- How **win rates** vary between Fear, Neutral, Greed, and Extreme Greed conditions
- Whether **trade direction (Buy/Sell)** changes with market mood
- How **position sizing** is influenced by sentiment

---

## 📦 Datasets

### 1.Fear & Greed Index (`fear_greed_index.csv`)
| Column | Description |
|---|---|
| `timestamp` | Unix timestamp |
| `value` | Sentiment score (0–100) |
| `classification` | Label: Extreme Fear / Fear / Neutral / Greed / Extreme Greed |
| `date` | Human-readable date |



**Total records:** 2,644 daily entries (Feb 2018 – Apr 2025)

---

### 2.Historical Trader Data (`historical_data.csv`)
| Column | Description |
|---|---|
| `Account` | Trader wallet address |
| `Coin` | Asset traded (ETH, BTC, SEI, NTRN, etc.) |
| `Execution Price` | Trade execution price |
| `Size Tokens` | Position size in tokens |
| `Size USD` | Position size in USD |
| `Side` | BUY or SELL |
| `Timestamp IST` | Trade time (IST) |
| `Start Position` | Position size before this trade |
| `Direction` | Open Long / Close Long / Open Short / Close Short |
| `Closed PnL` | Profit or Loss when position closed (0 = still open) |
| `Fee` | Trading fee paid |

**Total records:** 211,224 trades

---

## 🛠️ Methodology

### Step 1 — Data Loading & Inspection
```python
fear_greed = pd.read_csv("fear_greed_index.csv")
historical_data = pd.read_csv("historical_data.csv")
# Dataset sizes: 2,644 sentiment records | 211,224 trade records
```

### Step 2 — Timestamp Normalisation
```python
fear_greed["timestamp"] = pd.to_datetime(fear_greed["timestamp"], unit="s")
historical_data["Timestamp"] = pd.to_datetime(historical_data["Timestamp"], unit="ms")
```
Both datasets were converted to datetime format and sorted chronologically.

### Step 3 — Merging Datasets (`merge_asof`)
```python
merged = pd.merge_asof(
    historical_data,
    fear_greed,
    on='time',
    direction='backward',
    tolerance=pd.Timedelta('1D')
)
```
A **backward-looking merge** was used so that each trade is tagged with the most recent available sentiment score (within a 1-day tolerance). Forward-fill was applied to handle any remaining gaps.

### Step 4 — Feature Engineering
New features created for analysis:
| Feature | Description |
|---|---|
| `profit` | Copy of Closed PnL |
| `win` | Boolean: profit > 0 |
| `hour / day / month / day_of_week` | Time-based features |
| `trade_size` | Copy of Size USD |
| `fee_ratio` | Fee / Trade Size |
| `abs_profit` | Absolute profit value |
| `is_buy` | 1 if BUY, 0 if SELL |
| `sentiment_score` | Numeric encoding: Fear=0, Neutral=1, Greed=2, Extreme Greed=3 |
| `account_total_profit` | Per-trader total profit |
| `account_trade_count` | Per-trader trade count |
| `account_avg_profit` | Per-trader average profit |
| `account_win_rate` | Per-trader win rate |

### Step 5 — Distribution of Trades by Sentiment
After merging and forward-fill:

| Sentiment | Trade Count |
|---|---|
| Fear | 160,832 (76.1%) |
| Neutral | 42,382 (20.1%) |
| Extreme Greed | 6,962 (3.3%) |
| Greed | 1,048 (0.5%) |

> ⚠️ **Note:** The dataset is heavily skewed toward Fear conditions. This should be considered when interpreting comparative results.

---

## 📊 Analysis & Visualisations

---

### Graph 1 — Average Profit by Market Sentiment

**Chart type:** Bar plot  
**X-axis:** Sentiment classification (Fear → Neutral → Greed → Extreme Greed)  
**Y-axis:** Average Closed PnL (USD)

```python
sns.barplot(data=merged, x="classification", y="profit", order=order)
```

#### 🔍 Key Insights

> The analysis reveals that **trader profitability is highly dependent on market sentiment**. The highest average profit is observed during **Neutral conditions**, indicating that stable, low-volatility markets provide the most favourable trading opportunities.
>
> **Greed phases show near-zero profitability**, suggesting that traders may be influenced by herd behaviour and enter positions at suboptimal prices — buying at peaks driven by crowd excitement.
>
> **Fear conditions yield moderate but more consistent returns**, supporting the effectiveness of cautious or contrarian strategies. When others are fearful, disciplined traders find better entry points.
>
> Overall, **extreme emotional states (both Fear and Greed) tend to reduce trading efficiency**, confirming the classic market wisdom: "Be fearful when others are greedy, and greedy when others are fearful."

> ⚠️ *The dataset skew toward Fear conditions may amplify this segment's apparent profitability.*

---

### Graph 2 — Win Rate by Market Sentiment

**Chart type:** Bar plot  
**X-axis:** Sentiment classification  
**Y-axis:** Win rate (% of profitable trades)

```python
win_rate = merged.groupby("classification")["win"].mean().reset_index()
win_rate['win_percent'] = win_rate['win'] * 100
sns.barplot(data=win_rate, x="classification", y="win_percent", order=order)
```

#### 🔍 Key Insights

> **Greed phases have the lowest win rate**, confirming they are the least favourable conditions for traders — not just in terms of average profit, but also in how frequently individual trades are profitable.
>
> **Neutral conditions again demonstrate the best balance** of high profitability and consistent win rate, making them the most reliable environment for structured trading.
>
> Interestingly, **Extreme Greed yields the highest win rate** among all sentiments. This may indicate that during strong bull market trends, the momentum is so powerful that even poorly timed entries eventually turn profitable. However, these individual wins may be smaller in magnitude.
>
> The divergence between Greed (low win rate) and Extreme Greed (high win rate) is a nuanced finding — it suggests there is a **critical threshold of market optimism** beyond which trend-following becomes more reliably profitable.

> ⚠️ *The dataset skew toward Fear conditions may influence the comparison.*

---

### Graph 3 — Buy vs Sell Distribution by Market Sentiment

**Chart type:** Stacked 100% bar chart  
**X-axis:** Sentiment classification  
**Y-axis:** Percentage (%) of BUY vs SELL trades

```python
buy_sell = pd.crosstab(merged['classification'], merged['Side'], normalize='index') * 100
buy_sell.plot(kind='bar', stacked=True, colormap='Set2')
```

#### 🔍 Key Insights

> Despite significant variation in **market sentiment and profitability outcomes**, the ratio of BUY to SELL trades remains **remarkably stable across all sentiment phases**.
>
> This is a crucial finding: **traders do not significantly alter their directional bias (long vs short) based on whether the market is in Fear or Greed**. The proportion of buy and sell orders stays consistent.
>
> This suggests that **differences in profitability across sentiment phases are driven more by trade quality and timing** — not by whether traders are going long or short more frequently.
>
> It may also indicate that the trader cohort in this dataset consists largely of **systematic or algorithmic traders** who maintain fixed strategies regardless of market mood, rather than retail traders who reactively change direction based on sentiment.

---

### Graph 4 — Trade Size Distribution by Market Sentiment

**Chart type:** Box plot (log scale)  
**X-axis:** Sentiment classification  
**Y-axis:** Trade Size in USD (log scale)

```python
sns.boxplot(data=merged, x='classification', y='trade_size')
plt.yscale('log')
```

#### 🔍 Key Insights

> The analysis of trade size distribution reveals that **median trade sizes remain relatively consistent across all sentiment phases**, indicating that typical position sizing does not significantly vary with market mood.
>
> However, **Fear conditions exhibit a markedly higher number of extreme outliers** — certain traders take disproportionately large positions during bearish markets, possibly attempting to capitalise on potential reversals or averaging down on losing positions.
>
> In contrast, **Extreme Greed does not show a corresponding spike in trade size**, indicating that heightened market optimism does not necessarily translate into more aggressive risk-taking in terms of position sizing.
>
> The **substantial variability across all sentiment categories** highlights **inconsistent risk management** among traders in the dataset. A log scale is used here to ensure extreme values do not distort the visual representation of typical behaviour.

---

## 💡 Overall Strategic Insights

| Finding | Implication |
|---|---|
| Neutral markets → highest avg profit | Trade most actively in calm, ranging markets |
| Greed markets → lowest win rate | Avoid chasing momentum during euphoria phases |
| Extreme Greed → high win rate (small gains) | Trend-following works, but manage size carefully |
| Fear markets → moderate consistent profit | Contrarian entries during Fear can be rewarding |
| Buy/Sell ratio is sentiment-independent | Systematic strategies outperform reactive ones |
| Fear → large trade size outliers | Some traders size up aggressively during downturns — high risk |

---

## 🚀 Recommended Trading Strategies

Based on the analysis:

1. **Neutral Market Strategy:** Deploy the highest capital during Neutral conditions. Win rates and average profits are both maximised here. Use range-trading or mean-reversion setups.

2. **Fear Contrarian Strategy:** Accumulate positions during Fear phases with moderate sizing. The data shows reasonable profitability — consistent with the "buy the fear" philosophy.

3. **Extreme Greed Trend Strategy:** Use tight, momentum-based entries during Extreme Greed. Win rate is high but manage risk carefully — profits per trade may be smaller.

4. **Avoid Pure Greed Phases:** The data clearly shows that trading performance deteriorates during Greed (not Extreme Greed). This is the most dangerous phase — avoid overtrading or chasing.

5. **Maintain Consistent Position Sizing:** Since buy/sell ratio doesn't change with sentiment, focus on trade quality and discipline rather than switching directional bias based on the index.

---

## ⚙️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.12 | Core language |
| Pandas | Data manipulation & merging |
| NumPy | Numerical operations |
| Matplotlib | Base visualisation |
| Seaborn | Statistical charts |
| Scikit-learn | (Imported for future ML extension) |
| Jupyter Notebook | Interactive analysis environment |

---

## 📌 Limitations & Notes

- The dataset is **heavily skewed toward Fear conditions** (76% of all trades). This must be accounted for when making cross-sentiment comparisons.
- `Closed PnL = 0` for open positions — these are included in the analysis but do not reflect actual realised outcomes.
- The merge uses a 1-day tolerance with backward direction — trades in early morning may be tagged with the previous day's sentiment score.
- Trader identity is anonymised via wallet addresses. No demographic or strategy-type information is available.

---

## 📬 Submission

Submitted via Google Form as required by the Primetrade.ai hiring team.  
**Deadline:** As specified in the assignment brief.

---

*This project was completed as part of the Round 0 screening assignment for Primetrade.ai's Data Science internship programme.*
