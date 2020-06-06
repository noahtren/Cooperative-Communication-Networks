![Cooperative Communication Networks](ccn.png)

# Cooperative Communication Networks

WIP code of Cooperative Communication Networks — generative systems that learn
representations with no dataset, guided by the constraints of the environment.
CCNs are heavily inspired by information theory. They are essentially
autoencoders with latent spaces that can be shaped according to different
constraints.

## Usage

To run this locally, run `python gnet/main.py`. A training session will begin
according to the configuration set in `config.json`. By default, 4 snapshots
are saved to the `gallery` folder at the end of every epoch.

## Config

The configuration can be modified to run experiments. Three different
experiment types can be run, each with their own configuration.

`JUST_VISION` trains a CNN autoencoder to produce unique images, with a vocab
size equal to `NUM_SYMBOLS`.

`JUST_GRAPH` trains a graph autoencoder, which is run to pretrain the graph
autoencoder portion of the full pipeline. Currently encodes arithmetic syntax
trees as defined in `graph_data.py` 

`FULL` trains a CNN autoencoder nested inside of a graph autoencoder, which is
the main demonstration of this work. Syntax trees can be expressed as graphs
and passed through the autoencoder, and the latent space is visual.

## Samples

### `JUST_VISION` experiments

![](media/cloud_vision_only_newaug_test_night_animation.mp4)
![](media/cloud_vision_only_color_animation.mp4)

### `FULL` experiments

![](media/cloud_full_test_animation.mp4)
![](media/cloud_full_color_2_animation.mp4)
![](media/cloud_6node_full_color_animation.mp4)


## Contributing

If you have any questions about the code or run into any issues, please open an
issue.

## Current/Future Work

Below are some ideas for future work.

* Adversarial games. Working on adding a "Spy" generator that attempts to learn
the same representations as the original generator and thus trick the decoder.
Training would be similar to a GAN, but the "true" dataset is also a generative
model.

* Perceptual loss. Encouraging symbols to be perceptually different from each
other via some external metric, such as a pre-trained CNN.

## Disclaimer

I am not a formal ML researcher, and I currently don't have any claims about the
theoretical signifiance of this work. I'm happy to share the code and respond to
questions!
