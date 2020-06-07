![Cooperative Communication Networks](/media/ccn.png)

# Cooperative Communication Networks

WIP code demonstrating Cooperative Communication Networks — generative systems
that learn representations with no dataset. Rather than learning from data, CCNs
are  guided by the constraints of a given environment. CCNs are autoencoders
with latent spaces that can be shaped according to arbitrary constraints.

## Usage

You can run this locally by cloning the repo and installing the project as a
Python package with `pip install -e .` in the project directory.

`python ccn/main.py`. This will start training based on the config set in
`config.json`. By default, a snapshot of 4 random samples is saved to the
`gallery` folder at the end of every epoch.

An example snapshot may look like this:

![](/media/example_snapshot.png)

You can see that the top row is the generated samples, and the bottom row is the
samples after being passed through a noisy channel.

## Config

The configuration can be modified to run experiments. Three different
experiment types can be run.

`JUST_VISION` trains a CNN autoencoder to produce unique symbols, with a vocab
size equal to `NUM_SYMBOLS`.

`JUST_GRAPH` trains a graph autoencoder, which is run to pretrain the graph
autoencoder portion of the full pipeline. It encodes arithmetic syntax
trees as defined in `graph_data.py` 

`FULL` trains a CNN autoencoder nested inside of a graph autoencoder, which is
the main demonstration of this work. Syntax trees can be expressed as graphs
and passed through the autoencoder, and the latent space is visual.

## Samples

### `JUST_VISION` experiments

![](/media/cloud_vision_only_newaug_test_night_animation.gif)

![](/media/cloud_vision_only_color_animation.gif)

### `FULL` experiments

![](/media/cloud_full_test_animation.gif)

![](/media/cloud_full_color_2_animation.gif)

![](/media/cloud_6node_full_color_animation.gif)


## Contributing

If you have any questions about the code or run into any issues, please open an
issue.

## Current/Future Work

Below are some ideas for future work.

* **Adversarial games**. I'm working on adding a "Spy" generator that attempts to
learn the same representations as the original generator and is trained to
trick the decoder. Training would be similar to a GAN, but the dataset is
replaced with a collaborative generator.

* **Perceptual loss**. Implementing perceptual loss could encourage symbols to be
perceptually different from each other via some external metric, such as a
pre-trained CNN or something like [SimCLR](https://arxiv.org/abs/2002.05709).

## Disclaimer

I am not a formal ML researcher, and I currently don't have any claims about the
theoretical signifiance of this work. It is built off of work from
[Joel Simon](http://www.joelsimon.net/tablets.html) and


Regardless, I'm happy to share this code and respond to questions!

The code is MIT-licensed.
