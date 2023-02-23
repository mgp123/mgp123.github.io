---
layout:     post
title:      Simpson GAN
date:       2022-01-30
summary:    Using StyleGAN2-ADA to generate images from The Simpsons (with low quality results)
---
<span style="font-size:0.8em;">I'm not an expert in the topic. May contain errors!</span>

----
### Introduction

The Simpsons have been around for a really long time. The show started airing at the start of the 90s and more than 30 years has passed since its premiere. At the time of writing there are about 700 episodes. Assuming 24 frames per second and 20 minutes per episodes, that's *20 millions frames* <sup>[1](#framesfootnote)</sup>. This is more than enough for a reasonable dataset. 

In the machine learning front, deep learning methods are becoming very good at generating images according to some given distribution. They are still far from perfect but gradual improvements are made every year. We try to train StyleGAN2-ADA using frames from Simpsons episodes.

How does StyleGAN2-ADA work? I have no idea (other than it is a GAN). Modern networks tend to have complicated tricks that are way beyond my understanding. The reasoning behind why some tricks (of the few I know about) work and some don't is mysterious to me. I suspect that, at least for part of them, [nobody knows](http://www.argmin.net/2017/12/05/kitchen-sinks/). 

Luckily, if we only care about getting images, there is a convenient script to do the training. Running a script doesn't provide any knowledge, insights or nice ideas. That being said, it's undeniably *fun*. Seeing a neural network trying to recreate images that look like they could be in The Simpsons is very entertaining. The results are somewhat disappointing but there are *a few* interesting/weird pictures that seemed good enough.

### Creating the dataset
First, we need to get the episodes and make some decisions on what to keep. Getting the episodes is easy, we just need to download collections from the internet. Being such a popular show, these are not hard to find. 

What should we keep? With multiple seasons, the show has changed its style a lot. The drawings from the first seasons used different color schemes and the character designs were still changing. Should we try to minimize the variation by keeping episodes from the same era? Making the dataset more homogenous might help to generate better results. We did not take this route and just picked random episodes from any season for the dataset. In hindsight, it probably would have helped to not have done that.

Another problem is handling the intro and outro sections of each episode. We'd like to remove those parts as they are not what we are interested in. Also, given that they change very little across episodes, they are going take a big proportion of the dataset if not removed. The exact moment and duration of these sections varies across episodes but they occurr at the extremes so we just need to take a conservative estimate about when to start and when to stop saving frames and we are good to go.


On top of that, the input dimensions for the network are 512x512 squares so we need to fit our frames into that format. The episodes have a height of 360 pixels but a varied width. The most recent episodes have a much wider format than 512 pixels so they can't fit into the squares. We can just do a center crop of the frames and pad with zeros when needed. Initially, we did something different. As the height is 360, there's space left that we could use for adding some of the remainder of the frame. So, when the image was wider than 512 pixels, we cut the left and right sides and move them above and below the image. 


<figure align="center"   display="inline-block">
  <img src="/images/simpsons/jigsaw_original.jpeg" height=135  style="margin-bottom:28.5px"/>
  <img src="/images/simpsons/jigsaw_frame.png" height=192 />
  <img src="/images/simpsons/jigsaw_crop.png" height=192 />

  <figcaption style="font-size:0.8em;"> 
    Initially, we moved slices of the image in order to fit in the 512x512 format. Later we went back to center cropping.
   </figcaption>
</figure>
This gave us more "usable image" per input. That is, a smaller fraction of the images are going to be the zero-padding instead of something useful. However, after some training iterations we went back to center cropping. We feared that the "non-locality" may somehow hurt the learning.

Lastly, as the training is done in google colab, the dataset needs to be relatively small so as not to take all the drive space.  We don't need every frame from every episode. A subset of episodes and a small samplerate is enough. Our final dataset is made up of 28k images (56k with mirroring).

### Sampling the generator
The first result is that training a large model in google colab is a *really bad idea*. It took *four months* of intermittent running just to reach 10% of the final training time. After a while we stopped using colab and switched to a different option. In the end, we trained the model for around 1514 kimg. Quality got worse near the end so we decided to keep an earlier model as the final version.

Here are some samples from the generator:
<figure align="center"   display="inline-block">
  <img src="/images/simpsons/samples/sample1.png" width=350 />
  <img src="/images/simpsons/samples/sample2.png" width=350 />
  <img src="/images/simpsons/samples/sample3.png" width=350 />
  <img src="/images/simpsons/samples/sample4.png" width=350 />
  <img src="/images/simpsons/samples/sample5.png" width=350 />
  <img src="/images/simpsons/samples/sample6.png" width=350 />
  <img src="/images/simpsons/samples/sample7.png" width=350 />
  <img src="/images/simpsons/samples/sample8.png" width=350 />

  <figcaption style="font-size:0.8em;"> 
    Some non cherry picked samples from the generator   
  </figcaption> 
</figure>

The general feel of the frames is there. Objects have black outlines and the color schemes seem reasonable. Backgrounds and characters definitely look strange but they do retain broad features like how the main characters are roughly drawn and things like trees and clouds. Still, it's easy to see that the image quality is bad. It's far from the level that it would fool anyone. 

StyleGAN2-ADA can produce very realistic results in datasets such as faces so: What went wrong? Why are we getting bad quality images? Some completely uninformed and untested hypotheses:

- <ins>The dataset is too diverse.</ins> These models benefit from the dataset consisting of (in some sense) similar images. When constructing a dataset, a general effort is made to eliminate unnecessary variety. As an example, in the faces dataset [FFHQ](https://github.com/NVlabs/ffhq-dataset) the authors crop the images and align the faces. Our simpsons dataset is most likely much more complex than FFHQ. Is it simply not currently possible to generate good samples from such a diverse dataset? Models like [BigGANs](https://arxiv.org/pdf/1809.11096.pdf) or [Diffusion Models](https://arxiv.org/pdf/2105.05233.pdf) do it. They can get high quality images using the ImageNet dataset so it can be done. Maybe StyleGAN2-ADA is to blame. It might just lack the flexibility needed for this kind of datasets.

- <ins>Blurriness in animation.</ins> It's common to find frames from older episodes that are a combination of nearby frames. When things used to be hand drawn, it made sense to use this kind of tricks to save the costs of having to do some extra drawings.

  <figure align="center"   display="inline-block">
    <img src="/images/simpsons/blurry.jpeg" />
    <figcaption style="font-size:0.8em;"> 
      A blurry frame from the dataset
    </figcaption> 
  </figure>

  While this shouldn't be the main reason our generative model failed, it's plausible that it had a negative effect on the final performance (specially on the eyes because of all the blurry blinking and eye movement).

- <ins>It needs more training.</ins> This one is hard to estimate. On the official [repo](https://github.com/NVlabs/stylegan2-ada-pytorch) it says:

  *In typical cases, 25000 kimg or more is needed to reach convergence, but the results are already quite reasonable around 5000 kimg. 1000 kimg is often enough for transfer learning, which tends to converge significantly faster*

  Considering that it took 4 GPU days to train to 1514 kimg, 25000 kimg is completely out of reach. However, we didn't train from scratch, instead resuming training from the ffhq512 model<sup>[2](#transfer-learning)</sup>. As we have said before, we trained the model a bit more but without getting any improvements. It's also true that during training the generator frequently got worse but improved later on, indicating that this is not a good criterion for deciding when to stop. So, does it need more training? I'm inclined to believe that it does *but that it won't make a huge difference anyway*. It might also lead to mode collapse (this was already beginning to be a problem).

It's also interesting to see what details the generator picked up. For instance, as we kept too much of the intro, it sometimes tries to imitate the opening credits that appear at the start of an episode (as you can see in one of the samples above). The pink background of some of the generated images is probably due to the pink walls of the house. 

It also learned to imitate different periods of the show. Look at the following samples. Notice how the first one uses darker colors, has a less sharp outline and two black stripes at the sides.
<figure align="center"   display="inline-block">
  <img src="/images/simpsons/samples/homer_old.png"  width=350/>
  <img src="/images/simpsons/samples/homer_new.png"  width=350/>
  <figcaption style="font-size:0.8em;"> 
    Homer samples from different eras 
  </figcaption> 
</figure>
Variants of that last image are common samples from the generator.  
<figure align="center"   display="inline-block">
  <img src="/images/simpsons/samples/homer1.png"  width=250/>
  <img src="/images/simpsons/samples/homer2.png"  width=250/>
  <img src="/images/simpsons/samples/homer3.png"  width=250/>

  <figcaption style="font-size:0.8em;"> 
    The generator ends up producing a lot of very similar samples
  </figcaption> 
</figure>

Occasionally it also produces background images. Here is one that looked specially nice. 
<figure align="center"   display="inline-block">
  <img src="/images/simpsons/samples/background.png"  />
  <figcaption style="font-size:0.8em;"> 
    A pretty generated background
  </figcaption> 
</figure>

Sometimes it mixes features of different characters. With enough luck, it would produce a "custom" character.
<figure align="center"   display="inline-block">
  <img src="/images/simpsons/samples/character.png"  />
  <figcaption style="font-size:0.8em;"> 
    A custom character sample.
  </figcaption> 
</figure>

Lastly here is a fun interpolation video <sup>[3](#video-generator)</sup>

<div  align="center" display="inline-block">
  <video  controls>
    <source src="/images/simpsons/video/out.mp4" type="video/mp4">
  </video>
</div>

### Autoencoders for videos. Maybe?
Generating frames is fun but what we truly want is to generate episodes. This would give us an infinite episode creator. You just run the program and get an episode with sound and everything. While that would be really interesting, that program is currently sitting quite comfortably in the science fiction realm. The technology is just not there yet.
 

Here is an idea that for generating videos, not whole episodes, only short videos with a little bit of movement and no sound. First we train an autoencoder on frames so that we get a low dimensional representation. Then we split the episodes into continuous shots. With the trained autoencoder, we encode each frame of the sequence with it. This will be used as training data fed into some model for sequence generation.

It's so simple that someone must have thought of it already. Is it going to work? Probably not.

We started training an autoencoder for this exact propose. We grabbed [this one](https://github.com/alexandru-dinu/cae) and made some modifications to fit to our needs. The original model works with 128x128 squares so we added some extra layers that combined each encoded patch into a 1024 vector. After around 12 hours of training, the performance stagnated and (as we were using a paid service) we ended up abandoning the idea. Here is how the reconstruction looked:

<figure align="center"   display="inline-block">
  <img src="/images/simpsons/autoencoder_original.png"/>
  <img src="/images/simpsons/autoencoder_frames.png"/>

  <figcaption style="font-size:0.8em;"> 
    The autoencoder reconstruction was not good enough to be useful   
  </figcaption> 
</figure>

Why did it fail? Probably the network's architecture (specifically the tweaks) were not good. Also a latent representation of 1024 might be too small to represent the whole image with enough accuracy.

### Footnotes

<a name="framesfootnote">1</a>: This is a bit of an oversimplification. Episodes might be shorter or using a different framerate. Also credits or black frames do not count.

<a name="transfer-learning">2</a>: How much did transfer learning help, considering that the datasets are so different?

<a name="video-generator">3</a>: Created using script from [dvschultz](https://github.com/dvschultz/stylegan2-ada-pytorch)

### References

- [StyleGAN2-ADA — Official PyTorch implementation](https://github.com/NVlabs/stylegan2-ada-pytorch)
- [Making Anime Faces With StyleGAN](https://www.gwern.net/Faces). 
  This post is filled with discussions, tricks and just interesting things in general. I wish I had seen it before I started training. It even recommends not using colab.
