# Data

Download of the data is provided by Code.org. Please follow this link (will become ready soon) to download.

In total, there are 711,274 submissions, and 111,773 programs out of these submisions are unique. 
We only include the unique programs in this dataset.

We provide three sets of data:

**1. 500 Body/Tailed Balanced Dataset**

We sampled 250 programs randomly from the body and tail distribution of the student programs. This is the set where we use to compute
result for Table 3.

These two files are called:

```bash
body_sampled_broken_250.json
body_sampled_correct_250.json
tail_sampled_broken_250.json
tail_sampled_correct_250.json
```

**2. 500 Body/Tailed Sampled Dataset**

The previous dataset, we divide programs into correct/broken label first, and then sample 250 out of each label.
It doesn't reflect the actual distribution of the labels (because Bounce has more broken programs than correct programs).

Therefore, we sampled another set of programs (500 in total) for body and for tail distribution.

These data are used to produce Table 4 in the appendix.

```bash
body_sampled_broken_423.json
body_sampled_correct_77.json
tail_sampled_broken_464.json
tail_sampled_correct_36.json
```

**3. Full Unique Dataset**

We provide the full unique dataset in a `csv` file. The CSV file looks like:

| Program      | Distribution Label | Binary Error Label| Multi-Error Label | Submission Count |
| ----------- | ----------- | -----------| -----------| -----------|
| {"when ball hits paddle": ["bounce"], ...}      | Body       | Correct| [] | 140860 |
| {"when ball hits paddle": ["score point"], ...}   | Tail        | Broken | ["whenPaddle-illegal-incrementPlayerScore", ...] | 3955|
| ...  | ...        | ...  | ...  | ... |

There are 41 error multi-labels. Since this is not part of Play to Grade paper, we do not elaborate further.

Submission count indicates how many times this unique program has been submitted by students.

There are 111,773 unique programs in total in this dataset. 

Body/tail cutoff point is for the unique program to have more than 10 submissions. There are are 3,189 programs in Body, and the rest are in Tail.

The error labels (multi-label) are generated with reachability constraint:
- If there is no ball launched in the game play, and no action player/agent can perform to get any ball to launch, then we
only tag with one label "whenRun-noBallLaunch".
- If the player/agent cannot control the paddle, or the paddle goes the wrong direction, we unify them into one label "whenMove-error", and ignore all other labels.
- If the paddle does not bounce, this means we can't control the ball to reach any other potential bug state, we also tag programs with just 
one label "whenPaddle-noBounce".
  
**Note**

We corrected a data uniqueness check issue that made us under-counted the number of unique programs in our dataset. This does not affect
the binary prediction task and results showed in the paper, but it does affect the numbers (number of programs) that we report. The ArXiv report has been corrected, but not the NeurIPS version.