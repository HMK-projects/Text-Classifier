# Text-Classifier
<!-- [![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url] -->
[![LinkedIn][linkedin-shield]][linkedin-url]
[![Youtube][youtube-shield]][youtube-url]

<!-- DESCRIPTION -->
This repository contains the code for a machine learning(ML) based sentiment text classifier. This project is open-source.

I wanted to create this project as a way to improve my skills in MLOps and building ML Pipelines. This project is also a part of my [100 days of ML part 2](https://github.com/Harsh188/100-Days-of-ML-Pt2) challenge.

<!-- GETTING STARTED -->
## Getting Started

### Installation

### Tutorials

### Contributing

<!-- USAGE EXAMPLES -->
## Usage

### Development

To develop/maintain code use the following steps to setup your enviroment.

1. To build the docker dev image run the following command

```
docker-compose up
```

This command builds the docker image which can then be used to start up the container.

2. Next use the following command to start up the dev docker container.

```
docker run --gpus all -it --rm -p 8888:8888 -v $PWD:/text-classifier text-classifier_dev
```

Once the container is up and running use the following code to launch jupyter notebooks.

```
jupyter notebook --ip 0.0.0.0 --no-browser --allow-root
```

<!-- ROADMAP -->

<!-- CONTACT -->
## Contact

Harshith MohanKumar

email: harshithmohankumar@pesu.pes.edu 

<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
* [Img Shields](https://shields.io)
* [README-Template](Best-README-Template)

<!-- License -->
## License
[MIT License](https://github.com/HMK-projects/Text-Classifier/blob/main/LICENSE)

<!-- MARKDOWN LINKS & IMAGES --> 

[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/harsh188/
[youtube-shield]: https://img.shields.io/badge/YouTube-FF0000?style=for-the-badge&logo=youtube&logoColor=white
[youtube-url]: https://www.youtube.com/channel/UCFpda-r5V_aHpBVgYhm_JDA
