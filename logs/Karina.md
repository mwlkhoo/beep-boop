I'm sorry if my grammar here sounds weird :/
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
* Starting position facing line
* Angular velocity bigger
* Rewards for going straight and turning
* Look for bottom ⅙ pixels
* Make restarting reward 2x more negative <- very useful fix
* Add off-centered flag, if center_entry keeps being at 0/1/2/8/9/10 for more than 5 loops, punish agent <- not very helpful
* If instead of only punishing, we can reward our agent for turning in the right direction to correct off-center issues, and punish if doing anything else <- pretty helpful, big improvement in accuracy and smoothness, agent doesn’t slack off anymore :)
* Rewards converge to a steady value, which is a good thing.

## Final Project
#### March 3
Miti suggestion: accelerometer to check whether anki gets stuck.
License Plate Locator - 
* tried to implement EAST algorithm, doesn’t work very well with anki’s perspective
* Was able to draw polylines that are tilted at an angle based on different perspectives
* Michelle was able to get crosswalk detection to work :)
* Tried different methods (edge detection, gray scale, B&W scale, etc.) for ease of locating the plate, failed
* Switched back to EAST for now, optimize it later

#### March 11
License Plate Locator Continued - 
* Even though the border has been thickened, SIFT still wasn’t able to recognize the plate. Features are too discrete to describe a complete rectangle
* Switched back to EAST
* Integrated existing EAST code with anki’s camera, worked pretty well if assumed only parking number and lhs of plate could be recognized
* Not very reliable at this point
* Code is slow because of all the for loops
* Trying to optimize by replacing for loops with numpy functions
* Success!!! :) using numpy instead of nested for loop / append saved us SO MUCH TIME. The delay introduced is so much less! Happy :)
* Integrating the license plate locator with the plate reader. One huge problem - image is too blurry. Also, we might need a properly trained model (with blurred image) for it. The accuracy right now seems really low. -> Retrain model and/or find ways to deblur image

#### March 12
Michelle successfully retrained the model with blurred images. I also changed the letters from black to grey to better represent the color of the plates from Anki’s perspective.

Was able to get the perspective transform matrix to work. Realized I was passing the wrong arguments to my function the entire time!!!! Again Michelle reminded me. Cheers to her. :) 

Still have lots of major problems:
* Image cropping has a lot of issues. The perspective transform function highly depends on how the image is cropped. This is still very unreliable at this point. Need to keep working on that.
* Because of the off-center issues of the cropped characters, the cnn model does not seem to be able to detect characters with a good accuracy. Will try to fix this issue.

#### March 27/28
Everything switched to simulation world because of coronavirus.
Several issues encountered:
* Previously when we were working with Anki Vector, we were coding in Python 3. However, ROS Melodic supports only Python 2, hence there were issues switching back to it because of some modules installation problems. Managed to fix that by replacing the broken /opt/ros/melodic folder with a new one (cheers to Michelle).
* I was using this imutils module in my code for license plate reading, and I thought it was only available in Python 3. Hence I was trying so hard to not switch back to 2 which made my cv_bridge not function. I then realized the same module is available in Python 2, so we code only in Python 2 now. Everything else is fine.
In the simulation world, license plate is a lot easier to work with:
* They are computer-generated, which matches with the trained model we have;
* I had to change some image cropping parameters to incorporate with the change in competition format.
Model Changes:
* Retrained CNN model with updated license plate. Changed the color from grey to black.
There are still problems in the code:
* Data processing is too slow. Need to find a way to make things faster. I realized it was way faster when working with Anki’s camera. Not sure if this is because of Python 2 or something else. Will keep working on it.
* Plate reading is not very stable. It sometimes sees half the characters with good accuracy. Sometimes it just sees everything perfectly. And this highly depends on the image cropping quality, which depends highly on the stability of the captured rectangles. Will keep working on it.
Called with Michelle to further discuss plans - I am going to keep working on the license plate reader, and she’s gonna try working with Q-Learning for the course navigation. We are going to first try to make it follow the path, then add conditions to it so that it stops and does what it’s supposed to when it sees pedestrians / plates.
Pedestrians: We are going to detect the red line. After seeing the red line, stop first, then proceed to either:
* count the number of white pixels / blue pixels (corresponds to the pair of jeans the 
pedestrian is wearing) present in the crosswalk area for pedestrian detection; 
or:
* use SIFT algorithm to detect the entire pedestrian figure.
We will see which one works better / easier as a whole. We figured that the CPU might not be powerful enough to process so much at once, so SIFT might be a little too much for this. We will see after the basic navigation is done.
Now that we are not using Anki anymore, there’s no immediate need for WiFi connection. I can thus switch back to my own laptop, which has a higher CPU capacity and hopefully can make things faster.

#### April 4
Starting to integrate things together:
* I integrated the plate detector and plate reader together. The function from plate_reader.py is called within the plate_detector.py, so I needed to import the class from plate_reader.py and create an instance of that class.
* Accuracy is still doing ok
* I haven’t had time to check if the plate reader works in the entire course. I have only been checking it with the first plate the robot sees. This is a task for tomorrow.
I also created a pedestrian.py to detect pedestrian after a crosswalk is detected:
* I updated main.py in anki_control from anki vector mode to simulation mode.
* I created a pedestrian detection script with SIFT algorithm to detect pedestrians. I took screenshots of the front and back sides of the pedestrian from the simulation, and used both as the model for SIFT. Since the pedestrian is such a small figure, there are only limited key points associated with it. SIFT algorithm couldn’t really draw matches from the robot capture to the pedestrian figure, but when I was debugging my code I figured that whenever the pedestrian is crossing the crosswalk, there will at least be 5 or more keypoints; and when it’s not crossing the road and is just waiting on the side, there will be maximum 2 key points. I used this as the indicator of whether there is a pedestrian crossing or not, and it’s been working so far! I am actually pretty surprised that SIFT can also be used in such case.
Something I still need to work on:
* So far everything integrated quite nicely, however the speed that the plate detector is working at is too slow. We will see how we can compromise performance over speed (i.e. how the scoring works).
Michelle and I talked about navigation earlier, and we concluded that it is now too late to implement Q-learning since it is going to take a while to figure what works and doesn’t, and it still needs time to train. So far Michelle hasn’t figured out how to combine essential files together, so we decided to just move on and use PID instead. Hopefully it can be implemented as quickly as it should be.

#### April 5
After testing the model for a while, I have concluded that EAST is not very stable in the simulation - unless a perfect angle is given, there is no guarantee that it can detect both parking number and plate at the same time. Instead of trying to force EAST to work and keep testing on predetermined parameters that change all the time, I decided to switch over to another method. I implemented (kinda resorted to) this:
* Use EAST to detect at least one text area. Odds are it either detects both of them, or only the parking number. Then draw a big box around the detected area. This way, we can have an if-statement to decide how big this box should be - the border of the box depends on what have been detected. This way, we can make sure that both parking and plate are bounded by the box that we manually created.
* We can then proceed to process the image. We first crop the robot capture around the box, then find corners of the black edges. This can be done with np.where and np.min/max. This way we can ensure that the corners of the plates are located no matter how twisted the angle is, so that we can use cv2.perspectiveTransform with a high accuracy. After this, proceed with the previous CNN trained license plate model to read the characters (this has been working quite nicely).
TODO’s:
* Check whether i can just use “PX” box to generate boxing -> pretty good! Need to double check tomorrow. However, pay extra attention to the first plate - it always needs some cool down time in order to get both parking and plate. If you don’t wait then it will always see the bottom part and throws you an error. (implement tomorrow!)
* Find corners using for-loops, similar to finding boxes -> if this is successful, the accuracy would be greatly improved.

#### April 6
Miti extended the competition deadline until after classes ended - good news since we have more time. However, we should still aim to finish implementing the majority of the stuff because we won’t have much time after final starts.
TODO’s:
* Finish implementing the corner locating algorithm, and rigorously test it. -> success! Numpy library helped a lot.
* Try to come up with solutions to make things run faster. -> loop counter
Problems:
* Sometimes it indeed sees the plate rather than the parking number (when boxes = 1). Need special treatment when this happens -> done, split the screen in half, if the box lands in the upper half -> parking, vice versa
Miti updated the sims world, which makes things a lot faster. I updated threshold for plate cropping, and it’s been working quite nicely. I also implemented everything I wanted to for the day. Now the robot can detect plates quite nicely, just need to integrate this with navigation. (right now i’m controlling the robot with the keyboard, which is biased).

#### April 7 - 19
* I have been integrating things on my side with Michelle's side. The loop was a bit too slow at first. We then spent a lot of time on tuning PID values and getting the agent to pass the crosswalk.
* I also added code which prompts the users (me and michelle) to type "Ready" before the simulation starts, and stop the timer and display results after the timer has reached 240 seconds (in simulation time). 
* License plate detection worked so much better after swithing to black edges detection. A couple of corrections were introduced in the code since the CNN model sometimes acted really weirdly. I have been observing patterns and found some corrections that have been proved to work consistently.
* After integrating everything together, we have just been tuning PID / tuning timing parameters for crosswalk & pedestrian detection. We succesfully got 4 plates, but we wanted one more. We spent the last day (sunday, april 19th) trying to get the 5th plate, and we did!!!! YAAAAY :)) 
* We would definitely have gone for the 6th plate (since we only used 100 something seconds to get 5), but considering that we have finals on Monday and all, we just went with 5. Afterall, 5 isn't too bad with 107 seconds :)