docker run --gpus all -it --rm \
  --shm-size=8g \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /home/mateusz/Desktop/magisterka/data:/data \
  -v /home/mateusz/Desktop/magisterka/mac-vo-adaptation:/home/macvo/workspace \
  --name macvo_master_thesis \
  macvo:latest