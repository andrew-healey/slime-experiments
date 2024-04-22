My implementation of the SLiMe (Segment Like Me), a pretty cool paper I read. I did only single class segmentation, not multi-class.

I also tried to add on instance segmentation, but it didn't work too well. The attention maps don't seem very "instance-focused", but maybe I didn't try hard enough.

Uses cross attention and self attention maps of Stable Diffusion to do few-shot segmentation. It learns a handful of weights and one Stable Diffusion language embedding vector per class.

i.e. from two labelled pictures of a dog, you can learn a vector that'll highlight dogs in a new image.

Given this labelled image (and one other):

![image](https://github.com/andrew-healey/slime-experiments/assets/26335275/35415d15-62cc-4c66-b6fa-ef0efe0b1c27)

It semi-precisely labels new images of that cat. Here are some of its test-set predictions. 

![image](https://github.com/andrew-healey/slime-experiments/assets/26335275/c22961f4-35e9-4b58-a717-a1aaed46facf)
