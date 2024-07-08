# To do : Add detailed readme file for installation guide

## Getting Started

To get a local copy up and running follow these simple steps.

### Prerequisites

You need to have ffmpeg installed on your system, you can download it from the official website: https://www.ffmpeg.org/download.html

* For linux:
```bash
apt-get -y update && apt-get -y install ffmpeg
pip3 install --upgrade pip setuptools
pip3 install ffmpeg-python
```

* for Windows
* Install FFMPEG from [here](https://www.ffmpeg.org/download.html). Make sure to add it to system path. You can refer to this awesome tutorial for downloading and installing ffmpeg on windows [youtube-tutorial](https://www.youtube.com/watch?v=jZLqNocSQDM).

### Installation

**1. Clone the repo**
```bash
git clone https://github.com/Aeidle/ML1-MiniProject.git
```
**2. Create a virtual environnement.**

```bash
# In windows
python -m venv venv
venv\Scripts\activate
```

```bash
# In linux
python -m venv venv
source venv/bin/activate
```

**Or even better use anaconda**

```bash
conda create --name venv python=3.9
conda activate venv
```

**3. Install the dependencies using pip :**
```bash 
pip  install dlib_installation/dlib-19.22.99-cp39-cp39-win_amd64.whl
pip install -r requirements.txt
```
<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Usage

To run the application, you can simply execute the main file:

```bash
flask run
```

and visit the url in your prowser

```bash
http://127.0.0.1:5000
```