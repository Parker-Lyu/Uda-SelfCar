
### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps. First, I converted the images to grayscale, then I used a Gaussian_blur to get a good Canny result. Third , Cannny to detect all the lines. Fourth , get the important lines by using a ROI,the last step is to draw it on the image input.

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by using fitLine function.First,I draw lines on a empty image as descriped in the fourth step last paragraph.Second I get all the coordinates of  points' of the lines  into two group.Third,I fit the two group points into two lines and draw them.




### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be that to split all points into two group,I used a two layers loop which takes a lot computing power.



### 3. Suggest possible improvements to your pipeline

A possible improvement would be to find a easy way to split points into two group.

Another potential improvement could be to modify params so that the functions could suit for different  size of video.

The third potential improvement could be to modify the vertices in the ROI function,so that we would get the more accurate lanelines.


###4.By the way
I'm a Chinese and my English level is in general,so please use the easy-understand sentences when you write back.Thank you!!!