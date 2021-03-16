# Keyword spotter

A simple project on speech recognition.

Sebastian Thomas (datascience at sebastianthomas dot de)


In this project, we intend to recognize a keyword out of a list of ten given keywords.

It is an extension of the [introductory tutorial on speech command recognition from Tensorflow](https://www.tensorflow.org/datasets/catalog/speech_commands).

It uses the [speech_commands dataset](https://www.tensorflow.org/datasets/catalog/speech_commands) of Pete Warden, version 0.0.2. The dataset contains 105829 [WAV files](https://en.wikipedia.org/wiki/WAV), each of a duration of at most 1 second. Each file consists of a spoken command out of a list of 35 commands.

For demonstration purposes, a REST API was implemented. This was inspired by a [tutorial of Velardo](https://youtu.be/1rSNlrEzdL4) of his series [Deep Learning (Audio) Application: From Design to Deployment](https://www.youtube.com/playlist?list=PL-wATfeyAMNpCRQkKgtOZU_ykXc63oyzp).


## Content

Data mining, analysis, training and evaluation of the classifier:
- [Predictive analysis](predictive_analysis.ipynb)

Main development:
- [Preprocessor](common/preprocessing.py)
- [Keyword Spotter](keyword_spotter.py)

REST API:
- [Flask server](server.py)
- [Keyword resource](resources/keyword.py)
- [Test client](client.py)



## References

[Warden, Pete: Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition. arXiv:1804.03209, 2018.](https://arxiv.org/abs/1804.03209)

[Velardo, Valerio: Deep Learning (Audio) Application: From Design to Deployment. YouTube, 2020.](https://www.youtube.com/playlist?list=PL-wATfeyAMNpCRQkKgtOZU_ykXc63oyzp)