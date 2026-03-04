docker run --gpus all -it --rm \
  --shm-size=8g \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /media/mateusz/D/home/magisterka/data:/data \
  -v /media/mateusz/D/home/magisterka/mac-vo-adaptation:/home/macvo/workspace \
  --name macvo_master_thesis \
  macvo:latest