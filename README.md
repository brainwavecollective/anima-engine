# Anima

```
anima /ˈæn.ɪ.mə/

the soul, especially the irrational part of the soul as distinguished from the rational mind.
```
Anima extracts a living emotional state from raw text, and outputs a vector that can be broadcast to any downstream consumer.  

One example use is generating in-flight modifications to body language and changes in eye-color with the related [Reachy Mini Conversational App behavioral adapter](https://github.com/brainwavecollective/anima-reachy-conversation) for the [Reachy Mini](https://github.com/pollen-robotics/reachy_mini) robot.  To get a better sense of how the affect extraction engine works you can check out this [writeup explaining the approach](https://github.com/brainwavecollective/affection/blob/main/EMOTION_ENGINE.md).  

You can get a quick sense for how the engine converts text to an emotional vector by running:  
`uv run pytest tests/smoke.py`  

## Installation  

`pip install anima-engine`

## Acknowledgements  

This project uses the NRC Valence, Arousal, and Dominance (VAD) Lexicon (v2.1) created by Saif M. Mohammad at the National Research Council Canada.  
Project Homepage: http://saifmohammad.com/WebPages/nrc-vad.html  
Author Email: saif.mohammad@nrc-cnrc.gc.ca  

If you use this project in academic work, please cite:
Mohammad, Saif M. (2025). NRC VAD Lexicon v2: Norms for Valence, Arousal, and Dominance for over 55k English Terms. arXiv:2503.23547.

## About the Author  

Daniel Ritchie is an independent technologist and founder of the Brain Wave Collective.  
[LinkedIn](https://linkedin.com/in/danielritchie123)  
[Email](mailto:daniel@brainwavecollective.ai)  
