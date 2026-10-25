# Five-minute demonstration script

Open `prototype/gait_review.html` (or the shared link) before starting.

## 1. What the data are (30 s)

"This is GaitRec: force-plate walking data from 1,667 patients at a workers' compensation rehabilitation centre in Austria. Each visit records several steps on both legs. I show the injured leg relative to the uninjured leg for three measures: weight accepted, braking and push-off."

## 2. One patient over time (90 s)

Search **1079**.

- "A knee patient, five visits over 36 days. Each dot is one visit, and the bar is that day's noise from repeated steps. The shaded band is where most knee patients sit in the same week."
- Point to the summary sentence: "The injured leg is taking more weight relative to the other leg, by 0.104 body weights. That is larger than the ±0.082 threshold, so it is more than ordinary variation."
- Point to the prompt: "It is flagged because loading is above the typical range for knee patients. That is a prompt to look, not a judgement."
- Point to the checklist: "Before anyone acts, the clinician would check walking speed, pain, footwear and treatment changes. The page states what it does not tell you."

## 3. Why the threshold matters (90 s)

Switch **Change threshold** to *Original: same-day noise only*.

"With only same-day noise, 457 patients get a prompt. 'Lower than first visit' alone hits 304, one in five."

Scroll to **Why visit-to-visit variation matters**.

"28% of visit-to-visit changes crossed a limit that should be crossed 5% of the time. Measurements move between visits for ordinary reasons. When I add that variation, estimated on half the patients, the limit works on the other half: 4.7%. And 'lower than first visit' drops to about one in eighteen."

Switch back to **Corrected**.

"There is a cost. When I added declines of known size to real patients, the corrected threshold caught only a quarter of 0.05 body-weight declines. Fewer false prompts mean more missed small declines. This is statistical calibration, not clinical detection, and which trade-off is right needs clinical input."

## 4. The prediction study (45 s)

"The project began with a frozen internal analysis plan, testing whether first-week gait predicts a later rehabilitation stay. It didn't: AUROC 0.49 against 0.50 for basic patient characteristics. A planted signal was recovered, so the pipeline works and the null is real."

## 5. Close (45 s)

"The code has 37 tests. A second implementation reproduces the reliability numbers, and every number in the report is checked automatically. No one outside the project has reviewed it yet. My next step is a test–retest study, and I have a draft protocol. I would value your view on that design and on how to decide what size of decline matters clinically."
