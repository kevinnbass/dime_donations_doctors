# Detailed Methodology

This document provides a comprehensive explanation of how we measure physician political contributions, written for both technical and non-technical audiences.

## Overview

We analyze Federal Election Commission (FEC) records to track how physicians' political donations have changed from 1980 to 2024. Our main finding is that physicians have shifted from predominantly Republican donors to predominantly Democratic donors over this period.

---

## Part 1: For General Readers

### Where does the data come from?

Every political donation over $200 in the United States must be reported to the Federal Election Commission. When you donate to a candidate, PAC, or party committee, the recipient files a report that includes:

- Your name
- Your address
- Your employer
- Your occupation (self-reported)
- The amount you gave
- The date

These records are public. The DIME database, created by Professor Adam Bonica at Stanford University, compiles all of these records from 1979 to the present and uses statistical techniques to link donations from the same person over time.

### How do we know someone is a physician?

We look at the **occupation** field that donors report to the FEC. When someone writes "Physician," "Doctor," "Surgeon," "Cardiologist," or similar terms, we count them as a physician.

Because people describe their jobs differently, we use five different definitions:

1. **Doctor/Physician Keyword**: The occupation contains the word "doctor" or "physician"
2. **MD/DO Credential**: The occupation includes "MD" or "DO"
3. **Medical Specialists**: The occupation mentions a specialty like "surgeon" or "cardiologist"
4. **CMS Medicare Match**: The donor appears in Medicare billing records as a physician
5. **Broad Rules**: A comprehensive set of patterns that catches more physicians

The good news: all five definitions tell the same story. The correlation between them is over 0.92, meaning it doesn't matter much which definition we use.

### How do we measure partisanship?

For each donation, we look at who received it:
- If the recipient is a Republican candidate or committee, it's a "Republican donation"
- If the recipient is a Democratic candidate or committee, it's a "Democratic donation"

We then calculate the percentage going to Republicans:

```
% Republican = Republican $ / (Republican $ + Democratic $)
```

A value of 65% means that for every $100 physicians donated, $65 went to Republicans and $35 went to Democrats.

### What did we find?

| Decade | % to Republicans | What this means |
|--------|------------------|-----------------|
| 1980s | ~65% | For every $100, ~$65 went to Republicans |
| 1990s | ~60% | Still majority Republican, but declining |
| 2000s | ~55% | Approaching even split |
| 2010s | ~35% | Democrats now receive majority |
| 2020s | ~20-30% | Democrats receive vast majority |

This is a remarkable shift. In just 40 years, physicians went from being one of the most Republican professional groups to one of the most Democratic.

---

## Part 2: Technical Details

### Data Sources

#### DIME Database

- **Source**: Stanford University, [data.stanford.edu/dime](https://data.stanford.edu/dime)
- **Coverage**: 1979-2024 federal election contributions
- **Size**: ~200 million contribution records
- **Donor linkage**: Probabilistic matching using name, address, employer, occupation

The DIME database links contributions to unique donor IDs (`bonica_cid`), allowing us to track individuals over time.

#### CMS Medicare Data

- **Source**: Centers for Medicare & Medicaid Services
- **Data**: Medicare Provider Utilization and Payment Data
- **Purpose**: Validates physician identification using ground-truth billing records
- **Limitation**: Only available through 2018 in our current linkage

### Physician Identification

#### Definition 1: Doctor/Physician Keyword

```sql
LOWER(occupation) LIKE '%doctor%' OR LOWER(occupation) LIKE '%physician%'
```

**Pros**: Simple, clear, unambiguous
**Cons**: Misses physicians who write "MD" or specialty only

#### Definition 2: MD/DO Credential

```sql
(occupation LIKE '% MD%' OR occupation LIKE '%,MD%' OR
 occupation LIKE '% DO%' OR UPPER(occupation) = 'MD' OR UPPER(occupation) = 'DO')
AND LOWER(occupation) NOT LIKE '%phd%'
```

**Pros**: Captures credential-only entries
**Cons**: Smaller sample, may miss those who don't list credentials

#### Definition 3: Medical Specialists

```sql
LOWER(occupation) LIKE '%surgeon%' OR LOWER(occupation) LIKE '%cardiologist%'
OR LOWER(occupation) LIKE '%anesthesiologist%' OR LOWER(occupation) LIKE '%radiologist%'
OR LOWER(occupation) LIKE '%oncologist%' OR LOWER(occupation) LIKE '%psychiatrist%'
-- ... additional specialties
```

**Pros**: High precision for specific specialties
**Cons**: Different specialties may have different politics

#### Definition 4: CMS Medicare Match

```sql
cms_medicare_active = TRUE
```

**Pros**: Ground-truth validation from billing records
**Cons**: Biased toward older physicians who bill Medicare; ends at 2018

#### Definition 5: Broad Physician Rules

Combines credentials, specialties, and keywords with explicit exclusions:

```sql
(
    -- Credentials
    occupation LIKE '% MD%' OR occupation LIKE '% DO%'
    -- Specialties
    OR LOWER(occupation) LIKE '%surgeon%'
    OR LOWER(occupation) LIKE '%cardiologist%'
    -- ... more patterns
    OR LOWER(occupation) LIKE '%physician%'
    OR LOWER(occupation) LIKE '%doctor%'
)
-- Exclusions for non-physicians
AND LOWER(occupation) NOT LIKE '%nurse%'
AND LOWER(occupation) NOT LIKE '%chiropract%'
AND LOWER(occupation) NOT LIKE '%dentist%'
AND LOWER(occupation) NOT LIKE '%veterinar%'
AND LOWER(occupation) NOT LIKE '%phd%'
```

**Pros**: Largest sample, most inclusive
**Cons**: May include edge cases

### Party Score Calculation

The DIME database assigns a `revealed_party` score to each donation based on recipient:

- **-1.0**: Pure Republican (donated to Republican candidates only)
- **+1.0**: Pure Democrat (donated to Democratic candidates only)
- **0.0**: Mixed or nonpartisan

We convert this to Republican share:

```python
rep_share = (1.0 - revealed_party) / 2.0
```

### Aggregation

For each election cycle and physician definition, we calculate:

```python
# Total donations in that cycle
n_donations = count(*)

# Weighted average party score
avg_party = sum(revealed_party * amount) / sum(amount)

# Republican share
rep_share = avg_party_to_rep_share(avg_party)

# Dollar breakdowns
rep_dollars = sum(amount WHERE revealed_party < 0)
dem_dollars = sum(amount WHERE revealed_party > 0)
```

### Robustness Checks

#### Cross-Definition Correlation

| Pool A | Pool B | Correlation |
|--------|--------|-------------|
| Doctor/Physician | MD/DO | 0.94 |
| Doctor/Physician | Specialists | 0.93 |
| Doctor/Physician | CMS Medicare | 0.95 |
| Doctor/Physician | Broad Rules | 0.97 |
| MD/DO | Specialists | 0.92 |

All correlations exceed 0.92, indicating robust findings.

#### Sample Size by Year

| Cycle | Doctor/Physician | Broad Rules |
|-------|------------------|-------------|
| 1980 | 1,100 | 2,200 |
| 1990 | 2,300 | 4,100 |
| 2000 | 17,000 | 24,500 |
| 2010 | 28,900 | 49,200 |
| 2020 | 190,000 | 340,000 |

Sample sizes have grown dramatically due to increased political engagement and improved FEC reporting.

### Limitations

1. **Self-reported occupation**: Donors choose how to describe their jobs. Some physicians may write vague descriptions we don't capture.

2. **Contribution-weighted**: We weight by dollar amount, not by donor. High-dollar donors have more influence on our statistics.

3. **Federal contributions only**: We don't include state or local races, which may differ.

4. **Missing small donations**: Donations under $200 don't require occupation reporting.

5. **Selection bias**: Physicians who donate are not representative of all physicians. More politically engaged or wealthier physicians are overrepresented.

### Replication

All analysis can be reproduced using:

1. DIME database (public, requires registration)
2. Code in this repository
3. Data dictionary above

For questions about replication, open an issue on this repository.
