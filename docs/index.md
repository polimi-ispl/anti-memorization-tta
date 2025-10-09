
## Mitigating data replication in text-to-audio generative diffusion models through anti-memorization guidance
## Abstract
A persistent challenge in generative audio models is data replication, where the model unintentionally generates parts of its training data during inference. In this work, we address this issue in text-to-audio diffusion models by exploring the use of anti-memorization strategies. We adopt Anti-Memorization Guidance (AMG), a technique that modifies the sampling process of pre-trained diffusion models to discourage memorization. Our study explores three types of guidance within AMG, each designed to reduce replication while preserving generation quality. We use Stable Audio Open as our backbone, leveraging its fully open-source architecture and training dataset. Our comprehensive experimental analysis suggests that AMG significantly mitigates memorization in diffusion-based text-to-audio generation without compromising audio fidelity or semantic alignment. 

# Additional material
In the following 

## Example 1

### Audio
- Original
  
<audio controls style="width: 300px;"><source src="media/ID_1980/Audios/sound_1980.wav" type="audio/mpeg"></audio>

- Generated Without AMG
  <p align="center">
  <audio controls style="margin: 5px;">
    <source src="media/ID_1980/Audios/With_memorization/1980_generation_17.wav" type="audio/mpeg">
    Your browser does not support the audio element.
  </audio>
  <audio controls style="margin: 5px;">
    <source src="media/ID_1980/Audios/With_memorization/1980_generation_23.wav" type="audio/mpeg">
    Your browser does not support the audio element.
  </audio>
  <audio controls style="margin: 5px;">
    <source src="media/ID_1980/Audios/With_memorization/1980_generation_28.wav" type="audio/mpeg">
    Your browser does not support the audio element.
  </audio>
</p>

- Generated With AMG
  <p align="center">
  <audio controls style="margin: 5px;">
    <source src="media/ID_1980/Audios/With_AMG/1980_generation_2.wav" type="audio/mpeg">
    Your browser does not support the audio element.
  </audio>
  <audio controls style="margin: 5px;">
    <source src="media/ID_1980/Audios/With_AMG/1980_generation_7.wav" type="audio/mpeg">
    Your browser does not support the audio element.
  </audio>
  <audio controls style="margin: 5px;">
    <source src="media/ID_1980/Audios/With_AMG/1980_generation_9.wav" type="audio/mpeg">
    Your browser does not support the audio element.
  </audio>
</p>
Prompt:

