Accompanying website to the paper _Messina, Francisco, Francesca Ronchini, Luca Comanducci, Paolo Bestagini, and Fabio Antonacci. "Mitigating data replication in text-to-audio generative diffusion models through anti-memorization guidance." arXiv preprint arXiv:2509.14934 (2025)._


## Abstract
A persistent challenge in generative audio models is data replication, where the model unintentionally generates parts of its training data during inference. In this work, we address this issue in text-to-audio diffusion models by exploring the use of anti-memorization strategies. We adopt Anti-Memorization Guidance (AMG), a technique that modifies the sampling process of pre-trained diffusion models to discourage memorization. Our study explores three types of guidance within AMG, each designed to reduce replication while preserving generation quality. We use Stable Audio Open as our backbone, leveraging its fully open-source architecture and training dataset. Our comprehensive experimental analysis suggests that AMG significantly mitigates memorization in diffusion-based text-to-audio generation without compromising audio fidelity or semantic alignment. 

# Additional material
In the following we will report data relative to generations with or without Anti-Memorization Guidance (AMG) for five prompts, specifically for each of these we report:
- Original audio/prompt.
- Audio generated without and with AMG (three each).
- Spectrograms of the original audio and of the ones generated with and without AMG.
- Similarity matrices computed between original audio and the audio files generated with and without AMG. 

## Example 1 (ID 1980)
Prompt: _126bpm 4/4. 4 measures with a fill. recorded with a pair of Neumann TLM 103s into protools._ 

[Original freesound link](https://freesound.org/people/RHumphries/sounds/1980/)

### Audio
- Original
  
  <p align="left">
  <audio controls style="margin: 5px;">
    <source src="media/ID_1980/Audios/sound_1980.wav" type="audio/mpeg">
    Your browser does not support the audio element.
  </audio>
  </p>
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

## Example 2 (ID 4567)
Prompt: _This is loop 52 in a series of 135 loops that belong together. They all have a deep dubspace feel in 1 bar 4/4 at 60 bpm and belong to the "Convoloops pack 02 - bare bone dubspace 60 bpm" sample pack. They all have the same name: "convoluted bare bone loop 60 bpm"
with three numbers as suffix. The first of the three numbers indicates
a group of samples with a similar sound and feel. The most rhythmic
ones are these with 1 till 4 as last number in the suffix and these
with 5 till 8 are more effects. Finally number 9 as the last number in
the suffix is the start of the delay and/or reverb
tail of the previous loop. The second number separates variations in
pitch of the initial loop before any processing is applied. Of these
variations number 5 is more granular & experimental. All loops were
created using the microtonik VSTi.
I took the bare bones preset and convoluted it with some variations of
itself. After this process I added some more convolution with another
SIR, some diffusion, delay & chorus with the Fusion Reflector
ensemble from Native Instruments Reaktor, and some more character with (you never guess it) the excellent Character plugin from my TC powercore firewire. All this was done within Cubase SX.
After that I mastered these loops within Wavelab using several plugins:
fades, equalising, multiband compressing, limiting & dither._ 

[Original freesound link](https://freesound.org/people/Jovica/sounds/4567/)

### Audio
- Original
  
  <p align="left">
  <audio controls style="margin: 5px;">
    <source src="media/ID_4567/Audios/sound_4567.wav" type="audio/mpeg">
    Your browser does not support the audio element.
  </audio>
  </p>
- Generated Without AMG
  <p align="left">
  <audio controls style="margin: 5px;">
    <source src="media/ID_4567/Audios/With_memorization/4567_generation_10.wav" type="audio/mpeg">
    Your browser does not support the audio element.
  </audio>
  <audio controls style="margin: 5px;">
    <source src="media/ID_4567/Audios/With_memorization/4567_generation_16.wav" type="audio/mpeg">
    Your browser does not support the audio element.
  </audio>
  <audio controls style="margin: 5px;">
    <source src="media/ID_4567/Audios/With_memorization/4567_generation_5.wav" type="audio/mpeg">
    Your browser does not support the audio element.
  </audio>
</p>

- Generated With AMG
  <p align="left">
  <audio controls style="margin: 5px;">
    <source src="media/ID_4567/Audios/With_AMG/4567_generation_10.wav" type="audio/mpeg">
    Your browser does not support the audio element.
  </audio>
  <audio controls style="margin: 5px;">
    <source src="media/ID_4567/Audios/With_AMG/4567_generation_21.wav" type="audio/mpeg">
    Your browser does not support the audio element.
  </audio>
  <audio controls style="margin: 5px;">
    <source src="media/ID_4567/Audios/With_AMG/4567_generation_9.wav" type="audio/mpeg">
    Your browser does not support the audio element.
  </audio>
</p>

### Spectrograms
- Original
  
  <img src="media/ID_4567/Spectrogram/sound_4567_spectrogram_original.pdf" alt="" width="400" style="margin: 5px;"/>
  
- Generated Without AMG
<p align="center">
  <img src="media/ID_4567/Spectrogram/sound_4567_spectrogram_NO_AMG_10.pdf" alt="Image 1" width="400" style="margin: 5px;"/>
  <img src="media/ID_4567/Spectrogram/sound_4567_spectrogram_NO_AMG_16.pdf" alt="Image 2" width="400" style="margin: 5px;"/>
  <img src="media/ID_4567/Spectrogram/sound_4567_spectrogram_NO_AMG_5.pdf" alt="Image 3" width="400" style="margin: 5px;"/>
</p>
- Generated With AMG
<p align="center">
  <img src="media/ID_4567/Spectrogram/sound_4567_spectrogram_AMG_10.pdf" alt="Image 1" width="400" style="margin: 5px;"/>
  <img src="media/ID_4567/Spectrogram/sound_4567_spectrogram_AMG_21.pdf" alt="Image 2" width="400" style="margin: 5px;"/>
  <img src="media/ID_4567/Spectrogram/sound_4567_spectrogram_AMG_9.pdf" alt="Image 3" width="400" style="margin: 5px;"/>
</p>

### Similarity Matrices
- Generated Without AMG
<p align="center">
  <img src="media/ID_4567/Similarity_matrix/4567_with_memorization_similarity_matrix_10.pdf" alt="Image 1" width="400" style="margin: 5px;"/>
  <img src="media/ID_4567/Similarity_matrix/4567_with_memorization_similarity_matrix_16.pdf" alt="Image 2" width="400" style="margin: 5px;"/>
  <img src="media/ID_4567/Similarity_matrix/4567_with_memorization_similarity_matrix_5.pdf" alt="Image 3" width="400" style="margin: 5px;"/>
</p>
- Generated With AMG
<p align="center">
  <img src="media/ID_4567/Similarity_matrix/4567_all_guidance_similarity_matrix_10.pdf" alt="Image 1" width="400" style="margin: 5px;"/>
  <img src="media/ID_4567/Similarity_matrix/4567_all_guidance_similarity_matrix_21.pdf" alt="Image 2" width="400" style="margin: 5px;"/>
  <img src="media/ID_4567/Similarity_matrix/4567_all_guidance_similarity_matrix_9.pdf" alt="Image 3" width="400" style="margin: 5px;"/>
</p>


## Example 3 (ID 5131)
Prompt: _"ATTACK loop 140 bpm-00.wav" till "ATTACK loop 140 bpm-31.wav"
are all part of the "ATTACK LOOP 6" sample package and belong together
as they are all variations on the same 1 measure 4/4 140 bpm drumloop. The loop has a techno-trance
feel. The first four loops (00 till 03) contain some variations of the
pure drumloop, where 00 is the most minimal and 03 the fullest. All
other variations add other sound effects, some of them being sounds
with a certain pitch, mostly C. These loop are suitable for your trance
and techno productions. They were created using the Waldorf Attack VSTi within Cubase SX. Mastering (EQ, Stereo Enhancer, Multi-Band expand/compress/limit, dither, fades at start and/or end) done within Wavelab._ 

[Original freesound link](https://freesound.org/people/Jovica/sounds/5131/)

### Audio
- Original
  
  <p align="left">
  <audio controls style="margin: 5px;">
    <source src="media/ID_5131/Audios/sound_5131.wav" type="audio/mpeg">
    Your browser does not support the audio element.
  </audio>
  </p>
- Generated Without AMG
  <p align="left">
  <audio controls style="margin: 5px;">
    <source src="media/ID_5131/Audios/With_memorization/5131_generation_0.wav" type="audio/mpeg">
    Your browser does not support the audio element.
  </audio>
  <audio controls style="margin: 5px;">
    <source src="media/ID_5131/Audios/With_memorization/5131_generation_1.wav" type="audio/mpeg">
    Your browser does not support the audio element.
  </audio>
  <audio controls style="margin: 5px;">
    <source src="media/ID_5131/Audios/With_memorization/5131_generation_7.wav" type="audio/mpeg">
    Your browser does not support the audio element.
  </audio>
</p>

- Generated With AMG
  <p align="left">
  <audio controls style="margin: 5px;">
    <source src="media/ID_5131/Audios/With_AMG/5131_generation_4.wav" type="audio/mpeg">
    Your browser does not support the audio element.
  </audio>
  <audio controls style="margin: 5px;">
    <source src="media/ID_5131/Audios/With_AMG/5131_generation_45.wav" type="audio/mpeg">
    Your browser does not support the audio element.
  </audio>
  <audio controls style="margin: 5px;">
    <source src="media/ID_5131/Audios/With_AMG/5131_generation_6.wav" type="audio/mpeg">
    Your browser does not support the audio element.
  </audio>
</p>

### Spectrograms
- Original
  
  <img src="media/ID_5131/Spectrogram/sound_5131_spectrogram_original.pdf" alt="" width="400" style="margin: 5px;"/>
  
- Generated Without AMG
<p align="center">
  <img src="media/ID_5131/Spectrogram/sound_5131_spectrogram_NO_AMG_0.pdf" alt="Image 1" width="400" style="margin: 5px;"/>
  <img src="media/ID_5131/Spectrogram/sound_5131_spectrogram_NO_AMG_1.pdf" alt="Image 2" width="400" style="margin: 5px;"/>
  <img src="media/ID_5131/Spectrogram/sound_5131_spectrogram_NO_AMG_7.pdf" alt="Image 3" width="400" style="margin: 5px;"/>
</p>
- Generated With AMG
<p align="center">
  <img src="media/ID_5131/Spectrogram/sound_5131_spectrogram_AMG_4.pdf" alt="Image 1" width="400" style="margin: 5px;"/>
  <img src="media/ID_5131/Spectrogram/sound_5131_spectrogram_AMG_45.pdf" alt="Image 2" width="400" style="margin: 5px;"/>
  <img src="media/ID_5131/Spectrogram/sound_5131_spectrogram_AMG_6.pdf" alt="Image 3" width="400" style="margin: 5px;"/>
</p>

### Similarity Matrices
- Generated Without AMG
<p align="center">
  <img src="media/ID_5131/Similarity_matrix/5131_with_memorization_similarity_matrix_0.pdf" alt="Image 1" width="400" style="margin: 5px;"/>
  <img src="media/ID_5131/Similarity_matrix/5131_with_memorization_similarity_matrix_1.pdf" alt="Image 2" width="400" style="margin: 5px;"/>
  <img src="media/ID_5131/Similarity_matrix/5131_with_memorization_similarity_matrix_7.pdf" alt="Image 3" width="400" style="margin: 5px;"/>
</p>
- Generated With AMG
<p align="center">
  <img src="media/ID_5131/Similarity_matrix/5131_all_guidance_similarity_matrix_4.pdf" alt="Image 1" width="400" style="margin: 5px;"/>
  <img src="media/ID_5131/Similarity_matrix/5131_all_guidance_similarity_matrix_45.pdf" alt="Image 2" width="400" style="margin: 5px;"/>
  <img src="media/ID_5131/Similarity_matrix/5131_all_guidance_similarity_matrix_6.pdf" alt="Image 3" width="400" style="margin: 5px;"/>
</p>


## Example 4 (ID 5375)
Prompt: _Recorded direct with a Peavey Dynabass in passive mode, active mode EQ is nice but noisy as hell so I never use it. Ran the bass through my Zoom bass processor and played all notes on E string up to 16th fret then went the rest of the way up the strings and onto highest fret on G string._ 

[Original freesound link](https://freesound.org/people/NoiseCollector/sounds/5375/)

### Audio
- Original
  
  <p align="left">
  <audio controls style="margin: 5px;">
    <source src="media/ID_5375/Audios/sound_5375.wav" type="audio/mpeg">
    Your browser does not support the audio element.
  </audio>
  </p>
- Generated Without AMG
  <p align="left">
  <audio controls style="margin: 5px;">
    <source src="media/ID_5375/Audios/With_memorization/5375_generation_3.wav" type="audio/mpeg">
    Your browser does not support the audio element.
  </audio>
  <audio controls style="margin: 5px;">
    <source src="media/ID_5375/Audios/With_memorization/5375_generation_7.wav" type="audio/mpeg">
    Your browser does not support the audio element.
  </audio>
  <audio controls style="margin: 5px;">
    <source src="media/ID_5375/Audios/With_memorization/5375_generation_9.wav" type="audio/mpeg">
    Your browser does not support the audio element.
  </audio>
</p>

- Generated With AMG
  <p align="left">
  <audio controls style="margin: 5px;">
    <source src="media/ID_5375/Audios/With_AMG/5375_generation_12.wav" type="audio/mpeg">
    Your browser does not support the audio element.
  </audio>
  <audio controls style="margin: 5px;">
    <source src="media/ID_5375/Audios/With_AMG/5375_generation_26.wav" type="audio/mpeg">
    Your browser does not support the audio element.
  </audio>
  <audio controls style="margin: 5px;">
    <source src="media/ID_5375/Audios/With_AMG/5375_generation_28.wav" type="audio/mpeg">
    Your browser does not support the audio element.
  </audio>
</p>

### Spectrograms
- Original
  
  <img src="media/ID_5375/Spectrogram/sound_5375_spectrogram_original.pdf" alt="" width="400" style="margin: 5px;"/>
  
- Generated Without AMG
<p align="center">
  <img src="media/ID_5375/Spectrogram/sound_5375_spectrogram_NO_AMG_3.pdf" alt="Image 1" width="400" style="margin: 5px;"/>
  <img src="media/ID_5375/Spectrogram/sound_5375_spectrogram_NO_AMG_7.pdf" alt="Image 2" width="400" style="margin: 5px;"/>
  <img src="media/ID_5375/Spectrogram/sound_5375_spectrogram_NO_AMG_9.pdf" alt="Image 3" width="400" style="margin: 5px;"/>
</p>
- Generated With AMG
<p align="center">
  <img src="media/ID_5375/Spectrogram/sound_5375_spectrogram_AMG_12.pdf" alt="Image 1" width="400" style="margin: 5px;"/>
  <img src="media/ID_5375/Spectrogram/sound_5375_spectrogram_AMG_26.pdf" alt="Image 2" width="400" style="margin: 5px;"/>
  <img src="media/ID_5375/Spectrogram/sound_5375_spectrogram_AMG_28.pdf" alt="Image 3" width="400" style="margin: 5px;"/>
</p>

### Similarity Matrices
- Generated Without AMG
<p align="center">
  <img src="media/ID_5375/Similarity_matrix/5375_with_memorization_similarity_matrix_3.pdf" alt="Image 1" width="400" style="margin: 5px;"/>
  <img src="media/ID_5375/Similarity_matrix/5375_with_memorization_similarity_matrix_7.pdf" alt="Image 2" width="400" style="margin: 5px;"/>
  <img src="media/ID_5375/Similarity_matrix/5375_with_memorization_similarity_matrix_9.pdf" alt="Image 3" width="400" style="margin: 5px;"/>
</p>
- Generated With AMG
<p align="center">
  <img src="media/ID_5375/Similarity_matrix/5375_all_guidance_similarity_matrix_12.pdf" alt="Image 1" width="400" style="margin: 5px;"/>
  <img src="media/ID_5375/Similarity_matrix/5375_all_guidance_similarity_matrix_26.pdf" alt="Image 2" width="400" style="margin: 5px;"/>
  <img src="media/ID_5375/Similarity_matrix/5375_all_guidance_similarity_matrix_28.pdf" alt="Image 3" width="400" style="margin: 5px;"/>
</p>


