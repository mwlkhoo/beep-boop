## Labs
#### Lab 4 - SIFT
* A lot of technical issues were encountered - we had trouble getting our laptops to connect to Anki.
* We also were not familiar with the "with..." statement in Python. After fixing that, Lab 4 was quite quick and easy to finish.
* We suspected that there were issues with our connection to Anki again, because our Anki didn't perform as expected, but the same code worked fine on David's Anki.

#### Lab 5 - CNN Model Training
* Issues with numpy arrays. I was not familiar with the syntex, but after lab5, I got more comfortable with working with numpy arrays, which helped a lot in later stages.
* I forgot to plot the confusion matrix. Searched on stackoverflow and found a really nice pandas library function to plot colorful confusion matrix. Adopted in my code.
* After reading the confusion matrix, I realized character Z always had problems. I later figured that it was because I didn’t shuffle the data set - since the files were arranged in chronological order, Z was always ranked at the end and thus was always in the validation set. It didn’t get trained properly. I then shuffled the data set and every character received a very high accuracy.

#### Lab 6 - Q Learning
* Parameters changed:
** Starting position facing line
** Angular velocity bigger
** Rewards for going straight and turning
** Look for bottom ⅙ pixels
** Make restarting reward 2x more negative <- very useful fix
** Add off-centered flag, if center_entry keeps being at 0/1/2/8/9/10 for more than 5 loops, punish agent <- not very helpful
** If instead of only punishing, we can reward our agent for turning in the right direction to correct off-center issues, and punish if doing anything else <- pretty helpful, big improvement in accuracy and smoothness, agent doesn’t slack off anymore :)
** Rewards converge to a steady value, which is a good thing.