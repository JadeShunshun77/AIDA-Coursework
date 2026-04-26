# Task 3 NanoGPT Experiment Summary

This document consolidates the experimental results you shared for **Task 3 (NanoGPT)** and the subsequent analysis of those results. It is written as a reusable markdown reference for report drafting.

The coursework requires a notebook that implements a character-level GPT pipeline from data preparation to inference, explains the transformer architecture, compares two values of one key hyperparameter using loss curves, and generates sample text from the best model for analysis. The report is assessed mainly on the quality of the methods, results, and discussion rather than coding style. fileciteturn20file0 fileciteturn20file9

---

## 1. Experimental scope covered here

This summary covers five parts of the shared evidence:

1. **Main model config and run logs**
2. **Train/validation loss history**
3. **Final summary tables**
4. **Loss curves for the main and additional experiments**
5. **Best-model text generation under different temperatures**

---

## 2. Shared base configuration

### Base configuration used for the main experiment

```python
BASE_CONFIG = dict(
    # Architecture
    n_embd        = 64,
    n_head        = 4,
    block_size    = 64,
    dropout       = 0.0,
    # Training
    batch_size    = 32,
    max_iters     = 5000,
    eval_interval = 500,
    eval_iters    = 50,
    lr            = 3e-4,
)
SEED = 42
```

### What changed in the main experiment

- **Model A**: `n_layer = 2`
- **Model B**: `n_layer = 4`

All other settings were kept fixed.

### Evaluation of the setup

This is a **fair and well-controlled main experiment** because only one key hyperparameter was changed: the number of transformer blocks (`n_layer`). That matches the Task 3 requirement to compare two values of a key hyperparameter using training loss curves. fileciteturn20file0

A good report sentence based on this setup is:

> To isolate the effect of model depth, all hyperparameters were fixed except the number of transformer blocks.

---

## 3. Main experiment: full run logs and loss history

## 3.1 Model A: `n_layer = 2`

- **Seed**: 42
- **Device**: CPU
- **Parameter count**: 112,193

### Loss history

| Iteration | Train Loss | Val Loss |
|---:|---:|---:|
| 0    | 4.4057 | 4.4057 |
| 500  | 2.5255 | 2.5317 |
| 1000 | 2.3893 | 2.3928 |
| 1500 | 2.2763 | 2.2883 |
| 2000 | 2.1886 | 2.2118 |
| 2500 | 2.1146 | 2.1413 |
| 3000 | 2.0504 | 2.0925 |
| 3500 | 2.0024 | 2.0722 |
| 4000 | 1.9554 | 2.0301 |
| 4500 | 1.9234 | 2.0036 |
| 5000 | 1.8881 | 1.9659 |

## 3.2 Model B: `n_layer = 4`

- **Seed**: 42
- **Device**: CPU
- **Parameter count**: 211,777

### Loss history

| Iteration | Train Loss | Val Loss |
|---:|---:|---:|
| 0    | 4.2840 | 4.2878 |
| 500  | 2.4779 | 2.4822 |
| 1000 | 2.3018 | 2.3194 |
| 1500 | 2.1787 | 2.1918 |
| 2000 | 2.0630 | 2.0984 |
| 2500 | 1.9832 | 2.0516 |
| 3000 | 1.9220 | 1.9925 |
| 3500 | 1.8707 | 1.9688 |
| 4000 | 1.8252 | 1.9390 |
| 4500 | 1.7913 | 1.9091 |
| 5000 | 1.7458 | 1.8859 |

---

## 4. Main experiment: result table and interpretation

### Final comparison table

| Model | Params | Train Loss | Val Loss |
|---|---:|---:|---:|
| Model A (`n_layer=2`) | 112,193 | 1.8881 | 1.9659 |
| Model B (`n_layer=4`) | 211,777 | 1.7458 | 1.8859 |

### Key findings

1. **The 4-layer model outperformed the 2-layer model on both training and validation loss.**
   - Train loss improvement: `1.8881 -> 1.7458` (down by **0.1423**)
   - Validation loss improvement: `1.9659 -> 1.8859` (down by **0.0800**)

2. **The advantage of the deeper model was consistent across training, not just at the final checkpoint.**
   The 4-layer model had lower validation loss at essentially every recorded evaluation point.

3. **The deeper model had a larger generalisation gap.**
   - 2-layer gap: `1.9659 - 1.8881 = 0.0778`
   - 4-layer gap: `1.8859 - 1.7458 = 0.1401`

4. **This indicates higher fitting capacity rather than severe overfitting.**
   The 4-layer model fit the training data more strongly, but validation loss still improved through to 5000 iterations. There was **no clear validation-loss rebound**, so there is no strong evidence of severe overfitting within the given training budget.

5. **The gain was real but modest relative to the increase in parameter count.**
   - Parameter count increased from **112,193** to **211,777**
   - This is close to a doubling of model size
   - Validation loss improved by only **0.08**

### Concise interpretation for report use

> Increasing depth from 2 to 4 transformer blocks improved both optimisation and validation performance under a fixed training budget. The deeper model consistently achieved lower losses throughout training, although the gain was modest relative to the near-doubling of parameter count.

### Cautious wording that is safe to use

- Safe:
  - “The 4-layer model performed better than the 2-layer model under the current setup.”
  - “The 4-layer model was selected as the better-performing configuration because it achieved the lowest validation loss.”
- Avoid writing too strongly:
  - “4 layers is the optimal depth.”
  - “Deeper models are always better.”
  - “The result is statistically significant.”

You only tested two depths under one seed, so the conclusion should remain local to **this setup**.

---

## 5. Final unified summary table

### Shared settings across all runs

- `seed = 42`
- `n_embd = 64`
- `n_head = 4`
- `block_size = 64`
- `max_iters = 5000`
- `lr = 0.0003`
- `batch_size = 32`

### Full summary table

| Type | Model | n_layer | dropout | Params | Train Loss | Val Loss | Gap |
|---|---|---:|---:|---:|---:|---:|---:|
| Main | Model A (`n_layer=2`) | 2 | 0.0 | 112,193 | 1.8881 | 1.9659 | 0.0778 |
| Main | Model B (`n_layer=4`) | 4 | 0.0 | 211,777 | 1.7458 | 1.8859 | 0.1401 |
| Additional | Additional (`dropout=0.0`) | 4 | 0.0 | 211,777 | 1.7458 | 1.8859 | 0.1401 |
| Additional | Additional (`dropout=0.2`) | 4 | 0.2 | 211,777 | 1.8991 | 1.9753 | 0.0762 |

---

## 6. Loss curves: what they show

## 6.1 Main experiment loss curves (`n_layer = 2 vs 4`)

### Observed pattern

- In the **training loss curve**, the 4-layer model remained below the 2-layer model almost from the start.
- In the **validation loss curve**, the 4-layer model was also consistently lower.
- Both validation curves were still trending downward at the final checkpoint.

### Interpretation

1. **The deeper model’s advantage is dynamic, not just final-point noise.**
   This strengthens the claim that the depth increase genuinely helped.

2. **The curves are smooth and stable.**
   There is no major instability, collapse, or late-stage divergence.

3. **The models may not have fully converged yet.**
   Since validation loss was still decreasing at 5000 iterations, a longer training budget might produce further improvements, especially for the deeper model.

### Strong report-ready conclusion

> The 4-layer model consistently achieved lower training and validation losses across the whole training process. This suggests that additional transformer blocks improved the model’s ability to form contextual representations for next-character prediction under the same optimisation settings.

This aligns with the course explanation that transformers stack multiple self-attention blocks to build richer contextual representations, and that autoregressive language modelling predicts the next token from preceding context. fileciteturn20file10

## 6.2 Additional experiment loss curves (`dropout = 0.0 vs 0.2`)

### Observed pattern

- In the **training loss curve**, `dropout=0.2` stayed above `dropout=0.0` throughout.
- In the **validation loss curve**, `dropout=0.2` also stayed above `dropout=0.0` throughout.
- However, the **gap** between training and validation loss was smaller for `dropout=0.2`.

### Interpretation

1. **Dropout had a clear regularisation effect.**
   The smaller train–validation gap suggests that it reduced overfitting pressure.

2. **But this regularisation did not improve final validation loss.**
   Although the gap shrank, both train and validation loss were worse.

3. **The most plausible interpretation is that dropout slowed optimisation more than it helped generalisation in this setup.**
   This is especially plausible because the model is relatively small and the training budget is limited.

### Strong report-ready conclusion

> In the 4-layer setting, dropout=0.2 reduced the train–validation gap but did not improve validation loss. This suggests that dropout provided regularisation, but under the current model size and training budget it weakened optimisation more than it improved generalisation.

### Useful extension sentence

> Therefore, dropout had a measurable regularisation effect, but it was not beneficial enough to replace the non-dropout model as the best configuration.

---

## 7. Best-model text generation under different temperatures

The best-performing model was **Model B (`n_layer=4`, `dropout=0.0`)**, selected because it achieved the lowest validation loss among the tested models.

Generation settings shared across samples:

- **Model**: Model B (`n_layer=4`)
- **Max new tokens**: 400
- **Temperatures tested**: `0.5`, `1.0`, `1.5`

The coursework explicitly asks for text generation from the best model and analysis of structural characteristics such as capitalization, grammar, and punctuation. fileciteturn20file0

## 7.1 Temperature = 0.5

### Quantitative indicators

- Characters: 401
- Words: 90
- Lines: 9
- Capitalised words: 9 (**10.0%**)
- Punctuation counts:
  - `.`: 0
  - `,`: 1
  - `!`: 0
  - `?`: 0
  - `:`: 1
  - `'`: 0
  - `\n`: 10
- High-frequency words (>5): `{'the': 20}`
- Speaker-format lines: 1
- Example speaker tag: `AUTINGEL:`

### Sample

```text
AUTINGEL:
That in the you crow sto stand to conter thee a the that
I do dreath be that the griens, the to the uncers
An fament me to the but lie the the to preat to contle
This to but the chortion the well the death
To so to the forter under this my dies the all the course
And the you the sue cont to to do the seee
To seet the speet for that of in his the courts the ring
To-h is on conted a succh
```

### Interpretation

- This is the **most conservative** sample.
- It already captures some **play-like surface structure**, especially the speaker label format (`AUTINGEL:`).
- However, it is **highly repetitive**, especially the repeated use of “the”.
- It resembles English at the local level, but semantic coherence is weak.

### Summary judgement

> Temperature 0.5 produced the most stable structure and strongest dialogue-like formatting, but also the highest repetition and lowest lexical diversity.

---

## 7.2 Temperature = 1.0

### Quantitative indicators

- Characters: 401
- Words: 70
- Lines: 12
- Capitalised words: 15 (**21.4%**)
- Punctuation counts:
  - `.`: 3
  - `,`: 11
  - `!`: 0
  - `?`: 0
  - `:`: 2
  - `'`: 4
  - `\n`: 13
- High-frequency words (>5): none
- Speaker-format lines: 0

### Sample

```text
KIDY LIDGHABET:
Than is chard'd, Stainn you court of in marced
An his comre of the made thegnly sore, that in'u
In sigh famenter: that noble leact, a whough--for not prest
not hear, thoughniop is, but Well should; not anot
He bungle loward
They to neet form an a queseerver'd wouch, by that,
Anco broth. Come, fasce'd of I to his.

First friaten,
So houghts biseass thereinty,-host, Froth.
did your 
```

### Interpretation

- This sample provides the **best balance between variety and readability**.
- Punctuation is richer and more natural-looking than at 0.5.
- Capitalisation is more varied and the output still loosely resembles dramatic dialogue.
- There are still many invented or distorted words, so the model has not learned robust word-level semantics or grammar.

### Summary judgement

> Temperature 1.0 gave the strongest balance between structural plausibility and lexical diversity, making it the most suitable sample for report discussion.

---

## 7.3 Temperature = 1.5

### Quantitative indicators

- Characters: 401
- Words: 54
- Lines: 14
- Capitalised words: 22 (**40.7%**)
- Punctuation counts:
  - `.`: 5
  - `,`: 19
  - `!`: 0
  - `?`: 1
  - `:`: 4
  - `'`: 7
  - `\n`: 15
- High-frequency words (>5): none
- Speaker-format lines: 0

### Sample

```text
Kitlo, far duising forger: Gung, StoWanquen'd.
 Evere urgaress likeg-Xaqure of kincially;
Cgrlent?

DRUKE:H, 'u
Iereie primen, me'tial noblezle:
I, a whough--for not prevatn, folfe,st
Ithoniracia, but Wen coditize not, Oo's;
Nown likeg,ut came, toon diff.'m aaga,
Isseeqesirob,; cony Ise wouth;-s bJuot. Comarve:
As'
Sobore on pacite. forfe,,-do,'ss, too hourss,
Os I
Bringtbo-high, Froth.Wdick  Len
```

### Interpretation

- This is the **most diverse but least coherent** sample.
- Surface variety is high, but many strings are essentially noise-like pseudo-words.
- Capitalisation and punctuation become much less controlled.
- The sample clearly demonstrates the trade-off between diversity and coherence at higher temperature.

### Summary judgement

> Temperature 1.5 increased diversity, but coherence and lexical stability deteriorated sharply.

---

## 8. Cross-temperature generation conclusions

### Main pattern

The samples show a clear **coherence–diversity trade-off**:

- **0.5**: more conservative, more repetitive, more stable formatting
- **1.0**: best balance between readability and diversity
- **1.5**: more diverse, but much noisier and less coherent

### Structural properties learned by the model

Across the samples, the best model appears to have learned:

- line-based dramatic formatting
- speaker-name style prefixes (at least partially)
- punctuation placement patterns
- English-like local character combinations

However, it still struggles with:

- stable word spelling
- global semantic coherence
- robust grammar across longer spans

### Strong report-ready conclusion

> The best model learned several Shakespeare-like surface patterns, including line breaks, dialogue-style formatting, and punctuation usage. However, generation remained largely local and character-driven, with limited long-range coherence. Lower temperature improved structural stability but increased repetition, while higher temperature improved diversity at the cost of readability.

---

## 9. Overall conclusions from all experiments

### Main conclusion

Among the tested configurations, **Model B (`n_layer=4`, `dropout=0.0`)** was the best-performing model because it achieved the lowest validation loss.

### Depth conclusion

Increasing depth from 2 to 4 transformer blocks:

- improved training loss
- improved validation loss
- produced a consistent advantage across the whole training process
- increased model capacity and train–validation gap

This means the deeper model was **better overall**, although the gain was moderate relative to the increase in parameter count.

### Dropout conclusion

Adding `dropout=0.2` in the 4-layer setting:

- reduced the train–validation gap
- demonstrated a regularisation effect
- did **not** reduce validation loss
- therefore did **not** produce a better final model

### Generation conclusion

The best model produced text that captured some Shakespeare-like formatting and punctuation patterns, but not stable long-range meaning. Temperature 1.0 produced the most balanced sample for qualitative discussion.

---

## 10. What this supports in your report

These results are already enough to support the key Task 3 report claims:

1. **Architecture explanation**
   - You implemented a character-level autoregressive transformer that predicts the next token from preceding context. fileciteturn20file0 fileciteturn20file10

2. **Main hyperparameter study**
   - You compared two values of one key hyperparameter (`n_layer`) using training and validation loss curves. fileciteturn20file0

3. **Additional attempt**
   - You tested dropout as a regularisation change and explained why it did not outperform the selected best model. fileciteturn20file0

4. **Text generation analysis**
   - You generated text from the best model and analysed structure, punctuation, capitalization, repetition, and coherence under different temperatures. fileciteturn20file0

---

## 11. Final concise paragraph you can adapt later

> The main experiment compared a 2-layer and a 4-layer character-level NanoGPT under identical training settings. The 4-layer model achieved consistently lower training and validation losses throughout training and was selected as the best-performing configuration, although its parameter count was nearly doubled and its train–validation gap was larger. An additional experiment with dropout showed that dropout=0.2 reduced the gap between training and validation loss, indicating regularisation, but did not improve final validation loss. Text generation from the best model showed that the model had learned several Shakespeare-like surface patterns, including dialogue-style line breaks and punctuation, but still produced many invented or unstable words. Temperature 1.0 provided the best balance between readability and diversity.

