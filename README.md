# Play to Grade

<img src="https://github.com/windweller/play-to-grade/blob/main/images/thumbnail.png?raw=true" style='height: 295.6px; width: 175.4px;'>

A correct program:

<img src="https://media.giphy.com/media/LhC4oRHFahkxvryfQA/giphy.gif" style='height: 150px; width: 150px;'>

An incorrect program:

<img src="https://media.giphy.com/media/i8ITbB6QtNS67t9dk6/giphy.gif"  style='height: 150px; width: 150px;'>


We include our training and model code for the two environments: Car and Bounce.
We provide the Jupyter notebook for easy re-running of our experiment.

We should note that since we didn't fix any random seed for our experiments, the result you get
will be slightly different from what we report on the paper. Since we run
the experiment for 3 or 5 times, the difference shouldn't be significant.

## Installation

We recommend Python >= 3.7.

```
pip install -r requirements.txt
```

## Training

The training is relatively quick and is done in the Jupyter notebook inside each folder.

We do require at least one GPU for training (in Jupyter notebook, many function calls have "cuda=True").

- "Car Experiments.ipynb"
- "Bounce Experiment.ipynb"

## Bounce: Simulator

You can run our simulator by:

```
cd bounce
python bounce.py
``` 

This simulator is built to take in student programs, represented as JSON files. We include 10 reference programs in
`./bounce/programs/*.json`. The simulator allows you to actually play the game using your keyboard (arrow key).

Load in a JSON program and play it yourself! These are included in the `bounce.py`.

```
program = Program()
program.set_correct()

# program.load("programs/miss_paddle_no_launch_ball.json")
# program.load("programs/hit_goal_no_point.json")
# program.load("programs/empty.json")
# program.load("programs/multi_ball.json")
# program.load("programs/ball_through_wall.json")
# program.load("programs/goal_bounce.json")
# program.load("programs/multi_ball2.json")
# program.load("programs/paddle_not_bounce.json")

game = Bounce(program)
game.run()
```

Note that you need to have a monitor in order to play. Can't be played in a server environment.

## Bounce: Data

We are still in the process of obtaining permission from Code.org for data release.
 We include the training 10 programs in `./bounce/programs/*.json`. These 
programs' format is the same as the programs in the full dataset.

## Terminology in Code

We use SAE (Sequential AutoEncoder) to refer to "HoareLSTM" in paper, and we use "hoarelstm" in code to refer
to "Contrastive HoareLSTM" in paper.