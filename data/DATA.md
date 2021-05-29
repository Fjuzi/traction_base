# SCENT 360-DEGREE VIDEO DATASET

# Requirements

- [youtube-dl](https://youtube-dl.org/)  version = 2021.04.26.
- [ffmpeg](https://www.ffmpeg.org/) version = 4.x (2021.04.07)
- our downloader based on [activity-net]()


# Download
Download and crop videos from [DATASET.csv](https://github.com/Fjuzi/traction_base/blob/main/data/DATASET.csv).
```
python .\activitynet\Crawler\Kinetics\download.py <DATASET.csv> <OUTPUT_PATH>
```
DATASET.csv is the csv containing the videos. It has the following format: label, youtube_id, time_start, time_end, split
The script downloads the videos using youtube-dl in mp4 format, and then using ffmpeg converts them into EAC projection, libx264 video codec.
This dataset can be directly fed to the system, which will perform the furthr preprocessing in runtime.