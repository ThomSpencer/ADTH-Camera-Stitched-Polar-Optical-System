xhost +local:root

sudo modprobe -r v4l2loopback
sudo modprobe v4l2loopback video_nr=10 card_label=StitchedCam exclusive_caps=1

docker compose up -d --build && docker compose exec app /bin/bash