# Interpretability-of-Temporal-Music-Activations-in-AudioGen

# Overview:

in recent works, researchers found neural networks trained on music appear to have learned implicit theories of musical structure through statistical learning alone[1]. In this paper feature activations are averaged over entire tracks which collapses the time dimension. However, music fundamentally changes through time. We propose analyzing feature activations over time to reveal structural and narrative features that track-level averaging cannot show us.

### **The Problem:**

the problem with the recent papers on Interpretable concepts in Large generative music models [1], is that the authors computed the mean activation across all time steps. Because the authors computed the average across all time steps, we lose information about when the activations occur in songs.

### **The Solution:**

instead of collapsing activations to a single scaler, instead we keep the activation time series for each feature and analyze its shape. for instance, different musical phenomena would have different characteristic temporal activation profiles:

**Static Properties:** Static properties are features that should remain active across a song. this includes:

- The Genre, Instrumentation, etc.

This type of profile is what is captured already in previous papers [1].

**Structural Markers:** structural markers are activations that activate to describe the ‘structure’ of a song, this may include:

- the intro, verse, chorus, bridge, etc.

these features should activate periodically and predictably based on where in the song we are listening.

**Narrative Arc:**  The Narrative Arc are features that activate when we ramp up, spike, or decay a narrative of a song. this includes:

- building tension, climatic release, fading out, etc.

**Local Events:** Local events are events that are brief and usually only occurs once. This includes:

- Drum fills, key changes, guitar solos, etc.

### The Process

**Dataset:** MTG-Jamendo

we need a dataset to train a SAE on longer audio files, otherwise the learned features may no capture those longer range patterns such as structural markers. not just 10 second clips [1]. we also need a dataset to evaluate and label features.

**Option A: Reuse Original SAE**

The plan is to reuse the original SAE [1], run the SAE over longer audio files, then analyze the temporal activation since the features learned on 10-second audio clips may still fire meaningfully on longer audio files. This helps reduce the cost from retraining a SAE.

**Option B: Train New SAE**

Train new SAEs on longer audio files from full length tracks this gives the SAE the opportunity to learn long range features. This is far more expensive, but could reveal new features that Nikhil Singh, et al. [1] couldn’t discover.

**Option C: Sliding window:**

instead of re-training an SAE, evaluate small chunks of the same track on the original authors SAE then stitching together the activations resulting from it. this means we wont have to fine tune or re train a SAE.

**Activation Extraction:**

for activation extraction we can follow the same process that Nikhil Singh did [1] except we do not average. for each track we can extract the full residual stream activation time series from MusicGens transformer layer.

then we can take the retrained (or new SAE) and encode each activation  vector to get the sparse feature activations for every time step. this will result in a matrix for each track [num_features × num_timesteps] rather than a single vector.

**Analysis (Core Contribution):**

the goal is to characterize the shape of these time series and cluster features based on their temporal behavior.

to do this we need to create four types of profiles:

- **Static:** High, sustained activations throughout the track.
    - detectable through low variance over time, and a high mean.
    - should math original paper
- **Structural/Periodic:** activations that oscillate predictably.
    - detectable via auto correlation or a Fourier analysis of the time series.
- **Narrative Arc:** Monotonically increasing, single spike, or decaying profiles
    - Detectable by fitting simple trend models or using change-point detection algorithms.
- **Local Event:** brief, isolated activation pulses
    - Detectable by high kurtosis in the time series (the activation is near zero most of the time but spikes sharply at specific moments.)

References:

[1] https://www.google.com/url?q=https://arxiv.org/abs/2505.18186&sa=D&source=docs&ust=1772811583918737&usg=AOvVaw2Ec7i8IAEYYMlgYCigYgGC