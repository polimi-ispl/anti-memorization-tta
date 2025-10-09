
## Mitigating data replication in text-to-audio generative diffusion models through anti-memorization guidance
## Abstract
A persistent challenge in generative audio models is data replication, where the model unintentionally generates parts of its training data during inference. In this work, we address this issue in text-to-audio diffusion models by exploring the use of anti-memorization strategies. We adopt Anti-Memorization Guidance (AMG), a technique that modifies the sampling process of pre-trained diffusion models to discourage memorization. Our study explores three types of guidance within AMG, each designed to reduce replication while preserving generation quality. We use Stable Audio Open as our backbone, leveraging its fully open-source architecture and training dataset. Our comprehensive experimental analysis suggests that AMG significantly mitigates memorization in diffusion-based text-to-audio generation without compromising audio fidelity or semantic alignment. 

# Additional material
In the following 

## Example 1
Prompt:

### Audio
- Original
  
<audio controls style="width: 300px;"><source src="media/ID_1980/Audios/sound_1980.wav" type="audio/mpeg"></audio>

- Generated Without AMG
  <p align="left">
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
  <p align="left">
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

### Spectrograms
- Original
  
  <img src="media/ID_1980/Spectrogram/sound_1980_spectrogram_original.pdf" alt="" width="400" style="margin: 5px;"/>
  
- Generated Without AMG
<p align="center">
  <img src="media/ID_1980/Spectrogram/sound_1980_spectrogram_NO_AMG_17.pdf" alt="Image 1" width="400" style="margin: 5px;"/>
  <img src="media/ID_1980/Spectrogram/sound_1980_spectrogram_NO_AMG_23.pdf" alt="Image 2" width="400" style="margin: 5px;"/>
  <img src="media/ID_1980/Spectrogram/sound_1980_spectrogram_NO_AMG_28.pdf" alt="Image 3" width="400" style="margin: 5px;"/>
</p>
- Generated With AMG
<p align="center">
  <img src="media/ID_1980/Spectrogram/sound_1980_spectrogram_AMG_2.pdf" alt="Image 1" width="400" style="margin: 5px;"/>
  <img src="media/ID_1980/Spectrogram/sound_1980_spectrogram_AMG_7.pdf" alt="Image 2" width="400" style="margin: 5px;"/>
  <img src="media/ID_1980/Spectrogram/sound_1980_spectrogram_AMG_9.pdf" alt="Image 3" width="400" style="margin: 5px;"/>
</p>

### Similarity Matrices
- Generated Without AMG
<p align="center">
  <img src="media/ID_1980/Similarity_matrix/1980_with_memorization_similarity_matrix_17.pdf" alt="Image 1" width="400" style="margin: 5px;"/>
  <img src="media/ID_1980/Similarity_matrix/1980_with_memorization_similarity_matrix_23.pdf" alt="Image 2" width="400" style="margin: 5px;"/>
  <img src="media/ID_1980/Similarity_matrix/1980_with_memorization_similarity_matrix_28.pdf" alt="Image 3" width="400" style="margin: 5px;"/>
</p>
- Generated With AMG
<p align="center">
  <img src="media/ID_1980/Similarity_matrix/1980_all_guidance_similarity_matrix_2.pdf" alt="Image 1" width="400" style="margin: 5px;"/>
  <img src="media/ID_1980/Similarity_matrix/1980_all_guidance_similarity_matrix_7.pdf" alt="Image 2" width="400" style="margin: 5px;"/>
  <img src="media/ID_1980/Similarity_matrix/1980_all_guidance_similarity_matrix_9.pdf" alt="Image 3" width="400" style="margin: 5px;"/>
</p>
